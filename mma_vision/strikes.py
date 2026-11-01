"""Stage 2: fighter keypoint series -> strike events.

Heuristic, zero-training strike detector. A strike is a limb-tip (wrist/ankle)
speed peak moving *toward* the opponent. It's "landed" when, at peak extension,
the tip is within `land_dist` torso-heights of a target zone AND/OR the
opponent's head snaps right after. Everything is scale-normalised by torso
height, so thresholds transfer across cameras.

Swap this module for a learned classifier later; the event schema stays the same.

Event columns:
  frame, t, fighter, limb, kind (punch|kick), target (head|body|leg), landed,
  speed (torso/frame), land_dist (torso), head_react (torso/frame), conf (0-1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrikeCfg
from .tracks import FighterSeries

LIMBS = {
    "l_wrist": ("l_shoulder", "punch"),
    "r_wrist": ("r_shoulder", "punch"),
    "l_ankle": ("l_hip", "kick"),
    "r_ankle": ("r_hip", "kick"),
}


def _speed(pos: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vel = np.vstack([np.zeros((1, 2)), np.diff(pos, axis=0)]) / scale[:, None]
    return vel, np.linalg.norm(vel, axis=1)


def _local_peaks(x: np.ndarray, mask: np.ndarray, cooldown: int) -> list[int]:
    """Indices of local maxima of x where mask is True, at least `cooldown` apart."""
    x = np.where(np.isnan(x), -np.inf, x)
    cand = [i for i in range(1, len(x) - 1) if mask[i] and x[i] >= x[i - 1] and x[i] >= x[i + 1]]
    cand.sort(key=lambda i: -x[i])
    picked: list[int] = []
    for i in cand:
        if all(abs(i - j) >= cooldown for j in picked):
            picked.append(i)
    return sorted(picked)


def detect_strikes(series: dict[str, FighterSeries], fps: float, cfg: StrikeCfg) -> pd.DataFrame:
    events: list[dict] = []
    names = list(series.keys())
    for me in names:
        opp = [n for n in names if n != me][0]
        A, B = series[me], series[opp]
        scale = A.scale
        opp_center = B.torso_center
        opp_targets = {"head": B.head, "body": B.torso_center, "leg": B.knee_center}
        _, opp_head_speed = _speed(B.head, B.scale)

        for limb, (anchor, kind) in LIMBS.items():
            pos = A.pt(limb)
            vel, speed = _speed(pos, scale)
            to_opp = opp_center - pos
            to_opp_n = to_opp / (np.linalg.norm(to_opp, axis=1, keepdims=True) + 1e-6)
            vel_n = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6)
            cos = np.sum(vel_n * to_opp_n, axis=1)
            ext = np.linalg.norm(pos - A.pt(anchor), axis=1) / scale

            vthr = cfg.vel_thresh if kind == "punch" else cfg.vel_thresh_kick
            mask = (speed > vthr) & (cos > cfg.toward_cos) & ~np.isnan(speed) & ~np.isnan(cos)
            lift = None
            if kind == "kick":
                # Walking moves an ankle toward the opponent too. A kick lifts the foot off the floor:
                # the kicking ankle must rise well above the other (standing) ankle. Image y grows downward.
                other = "r_ankle" if limb == "l_ankle" else "l_ankle"
                lift = (A.pt(other)[:, 1] - pos[:, 1]) / scale
            for p in _local_peaks(speed, mask, cfg.cooldown):
                if lift is not None:
                    lw = lift[p:min(len(lift), p + 7)]
                    if np.all(np.isnan(lw)) or np.nanmax(lw) < cfg.min_kick_lift:
                        continue
                # Impact ~ max extension shortly after the speed peak
                w_end = min(len(ext), p + 6)
                win = ext[p:w_end]
                if np.all(np.isnan(win)):
                    continue
                imp = p + int(np.nanargmax(win))
                tip = pos[imp]
                if np.any(np.isnan(tip)):
                    continue
                # Gate 1: the limb must actually be extended at impact (kills guard twitches / feints)
                min_ext = cfg.min_ext_punch if kind == "punch" else cfg.min_ext_kick
                if np.isnan(ext[imp]) or ext[imp] < min_ext:
                    continue
                # Gate 2: punches must go retracted -> extended; kicks must travel (a kicking leg stays
                # about the same length, so hip-ankle distance is a poor signal — use tip displacement).
                lo = max(0, p - cfg.ext_lookback)
                if kind == "punch":
                    base = ext[lo:p]
                    if base.size and not np.all(np.isnan(base)) and (ext[imp] - np.nanmin(base)) < cfg.min_ext_gain:
                        continue
                else:
                    prior = pos[lo:p]
                    if prior.size and not np.all(np.isnan(prior)):
                        travel = np.nanmax(np.linalg.norm(prior - tip, axis=1)) / scale[imp]
                        if travel < cfg.min_kick_travel:
                            continue
                dists = {}
                for tname, tpos in opp_targets.items():
                    if kind == "punch" and tname == "leg":
                        continue
                    d = np.linalg.norm(tip - tpos[imp]) / scale[imp]
                    if not np.isnan(d):
                        dists[tname] = d
                if not dists:
                    continue
                target = min(dists, key=dists.get)
                dmin = dists[target]

                r_end = min(len(opp_head_speed), imp + cfg.react_window + 1)
                base = np.nanmedian(opp_head_speed[max(0, imp - 10):imp]) if imp > 2 else 0.0
                react = float(np.nanmax(opp_head_speed[imp:r_end]) - (0 if np.isnan(base) else base)) if r_end > imp else 0.0

                landed = bool(dmin < cfg.land_dist or (dmin < 1.5 * cfg.land_dist and react > cfg.head_react_vel))
                # crude confidence: closeness + reaction + speed margin
                conf = float(np.clip(
                    0.5 * np.clip(1 - dmin / (2 * cfg.land_dist), 0, 1)
                    + 0.3 * np.clip(react / (2 * cfg.head_react_vel), 0, 1)
                    + 0.2 * np.clip((speed[p] - cfg.vel_thresh) / cfg.vel_thresh, 0, 1), 0, 1))
                if not landed and conf < cfg.min_miss_conf:
                    continue
                events.append({
                    "frame": int(imp), "t": imp / fps, "fighter": me, "limb": limb, "kind": kind,
                    "target": target, "landed": landed, "speed": float(speed[p]),
                    "land_dist": float(dmin), "head_react": react, "conf": conf,
                })
    cols = ["frame", "t", "fighter", "limb", "kind", "target", "landed", "speed", "land_dist", "head_react", "conf"]
    ev = pd.DataFrame(events, columns=cols).sort_values("frame").reset_index(drop=True)
    return ev


def label(ev_row) -> str:
    """Human label for an event, e.g. 'HEAD KICK ✓' or 'PUNCH (body) ✗'."""
    kind = ev_row["kind"].upper()
    tgt = ev_row["target"]
    if kind == "KICK":
        txt = {"head": "HEAD KICK", "body": "BODY KICK", "leg": "LEG KICK"}[tgt]
    else:
        txt = "PUNCH" if tgt == "head" else "BODY SHOT"
    return f"{txt} {'LANDED' if ev_row['landed'] else 'MISS'}"
