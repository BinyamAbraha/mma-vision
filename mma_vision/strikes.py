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

from . import KP
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


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment a-b (any NaN -> nan)."""
    if np.any(np.isnan(p)) or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.nan
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _target_dists(tip: np.ndarray, B: FighterSeries, fi: int, kind: str, scale: float) -> dict[str, float]:
    """Distance (torso units) from the striking tip to each target *region*, not a single point.

    head: face centre (minus a head radius)   body: shoulder-mid -> hip-mid segment (minus half torso width)
    leg: hip -> knee -> ankle segments of either leg (minus a thigh radius)
    """
    k = B.kp[fi]
    out: dict[str, float] = {}
    head = B.head[fi]
    if not np.any(np.isnan(head)):
        out["head"] = max(0.0, np.linalg.norm(tip - head) / scale - 0.18)
    sh = np.nanmean(k[[KP["l_shoulder"], KP["r_shoulder"]]], axis=0)
    hp = np.nanmean(k[[KP["l_hip"], KP["r_hip"]]], axis=0)
    d = _seg_dist(tip, sh, hp)
    if not np.isnan(d):
        out["body"] = max(0.0, d / scale - 0.25)
    if kind == "kick":
        ds = [_seg_dist(tip, k[KP[f"{s}_hip"]], k[KP[f"{s}_knee"]]) for s in ("l", "r")]
        ds += [_seg_dist(tip, k[KP[f"{s}_knee"]], k[KP[f"{s}_ankle"]]) for s in ("l", "r")]
        ds = [x for x in ds if not np.isnan(x)]
        if ds:
            out["leg"] = max(0.0, min(ds) / scale - 0.12)
    return out


def _zone(tip: np.ndarray, B: FighterSeries, fi: int, kind: str, scale: float) -> str:
    """head / body / leg by the tip's height relative to the opponent's shoulders and hips."""
    k = B.kp[fi]
    sh_y = np.nanmean(k[[KP["l_shoulder"], KP["r_shoulder"]], 1])
    hp_y = np.nanmean(k[[KP["l_hip"], KP["r_hip"]], 1])
    if np.isnan(sh_y) or np.isnan(hp_y):
        return "body"
    if tip[1] < sh_y + 0.12 * scale:          # at or above the shoulder line (chin included)
        return "head"
    if kind == "kick" and tip[1] > hp_y + 0.15 * scale:
        return "leg"
    return "body"


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
    """Two passes. (1) Per limb: find speed peaks toward the opponent and compute every feature we
    gate on. (2) In time order per fighter: apply the gates, relaxing the retraction gate when the
    fighter has just landed/thrown something (combinations don't fully retract between punches)."""
    cands: list[dict] = []
    names = list(series.keys())
    for me in names:
        opp = [n for n in names if n != me][0]
        A, B = series[me], series[opp]
        scale = A.scale
        opp_center = B.torso_center
        _, opp_head_speed = _speed(B.head, B.scale)
        # a head "reaction" faster than this is a keypoint jump, not a snap — ignore it
        opp_head_speed = np.where(opp_head_speed > cfg.max_head_react, np.nan, opp_head_speed)
        hip_dist = np.linalg.norm(A.hip_center - B.hip_center, axis=1) / ((A.scale + B.scale) / 2)
        # opponent body displacement (torso/frame): a landed kick/teep shoves the hips back; a miss doesn't
        _, opp_body_speed = _speed(B.hip_center, B.scale)
        opp_body_speed = np.where(opp_body_speed > cfg.max_head_react, np.nan, opp_body_speed)
        cuts = A.cuts if A.cuts is not None else np.zeros(len(scale), dtype=bool)
        cut_idx = np.where(cuts)[0]

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
            cool = cfg.cooldown if kind == "punch" else cfg.cooldown_kick
            for p in _local_peaks(speed, mask, cool):
                near_cut = bool(len(cut_idx)) and bool(np.min(np.abs(cut_idx - p)) <= cfg.cut_guard)
                nb = np.concatenate([speed[max(0, p - 2):p], speed[p + 1:p + 3]])
                support = int(np.sum(nb > cfg.support_frac * speed[p]))
                if lift is not None:
                    lw = lift[p:min(len(lift), p + 7)]
                    if np.all(np.isnan(lw)) or np.nanmax(lw) < cfg.min_kick_lift:
                        continue
                # Impact = closest approach to the opponent after the speed peak. (Max extension was
                # wrong for kicks: the speed peak happens while the leg is still chambered.)
                w_end = min(len(ext), p + (cfg.impact_window if kind == "punch" else cfg.impact_window_kick) + 1)
                win = np.linalg.norm(pos[p:w_end] - opp_center[p:w_end], axis=1)
                if np.all(np.isnan(win)):
                    continue
                imp = p + int(np.nanargmin(win))
                tip = pos[imp]
                if np.any(np.isnan(tip)) or np.isnan(ext[imp]):
                    continue
                lo = max(0, p - cfg.ext_lookback)
                base = ext[lo:p]
                gain = float(ext[imp] - np.nanmin(base)) if base.size and not np.all(np.isnan(base)) else np.nan
                prior = pos[lo:p]
                travel = float(np.nanmax(np.linalg.norm(prior - tip, axis=1)) / scale[imp]) if prior.size and not np.all(np.isnan(prior)) else np.nan
                # Ownership: is this tip really A's? Compare distance to A's own joint (elbow/knee) with
                # distance to B's nearest matching joints. A stolen glove sits on B's forearm.
                own_joint = limb.replace("wrist", "elbow").replace("ankle", "knee")
                d_own = np.linalg.norm(tip - A.pt(own_joint)[imp]) / scale[imp]
                opp_joints = ("l_elbow", "r_elbow") if kind == "punch" else ("l_knee", "r_knee")
                d_opp = np.nanmin([np.linalg.norm(tip - B.pt(j)[imp]) for j in opp_joints]) / scale[imp]
                owner_ratio = float(d_own / d_opp) if not (np.isnan(d_own) or np.isnan(d_opp)) and d_opp > 0 else np.nan

                dists = _target_dists(tip, B, imp, kind, scale[imp])
                if not dists:
                    continue
                # Target = the vertical zone the tip ends up in (head / body / leg), then the distance
                # to *that* zone decides landed vs. miss. Picking the nearest zone instead would call a
                # jab to the chin a "body shot" because the chin sits right above the shoulders.
                target = _zone(tip, B, imp, kind, scale[imp])
                # ...except a glove/foot touching the head sphere is a head shot regardless of height
                # (a crouched or falling opponent's shoulders sit at head height).
                if "head" in dists and dists["head"] < cfg.land_dist:
                    target = "head"
                if target not in dists:
                    target = min(dists, key=dists.get)
                dmin = dists[target]

                r_end = min(len(opp_head_speed), imp + cfg.react_window + 1)
                hb = np.nanmedian(opp_head_speed[max(0, imp - 10):imp]) if imp > 2 else 0.0
                react = float(np.nanmax(opp_head_speed[imp:r_end]) - (0 if np.isnan(hb) else hb)) if r_end > imp else 0.0
                bb = np.nanmedian(opp_body_speed[max(0, imp - 10):imp]) if imp > 2 else 0.0
                body_react = float(np.nanmax(opp_body_speed[imp:r_end]) - (0 if np.isnan(bb) else bb)) if r_end > imp else 0.0
                if np.isnan(body_react):
                    body_react = 0.0

                cands.append({
                    "frame": int(imp), "peak": int(p), "t": imp / fps, "fighter": me, "limb": limb, "kind": kind,
                    "range": float(hip_dist[imp]) if not np.isnan(hip_dist[imp]) else np.nan,
                    "target": target, "speed": float(speed[p]), "cos": float(cos[p]), "ext": float(ext[imp]),
                    "ext_gain": gain, "travel": travel, "land_dist": float(dmin), "head_react": react,
                    "owner_ratio": owner_ratio, "support": support, "body_react": body_react,
                    "near_cut": near_cut,
                })

    # ---- pass 2: gates, in time order, with combo context. Every candidate gets a verdict so
    # rejected ones can be inspected (candidates.csv) when tuning.
    events: list[dict] = []
    verdicts: list[dict] = []
    last_strike: dict[str, int] = {}          # fighter -> last accepted impact frame
    last_limb: dict[tuple, int] = {}          # (fighter, limb) -> last accepted impact frame

    def reject(c, why):
        v = dict(c); v["rejected_by"] = why; verdicts.append(v)

    for c in sorted(cands, key=lambda d: (d["frame"], -d["speed"])):
        kind, me, limb = c["kind"], c["fighter"], c["limb"]
        in_combo = me in last_strike and 0 < c["frame"] - last_strike[me] <= cfg.combo_window
        relax = cfg.combo_relax if in_combo else 1.0
        in_clinch = not np.isnan(c["range"]) and c["range"] < cfg.clinch_range
        vthr = cfg.vel_thresh if kind == "punch" else cfg.vel_thresh_kick
        if in_clinch:
            vthr *= cfg.clinch_speed_mult      # in the pocket every twitch is "near the target"

        # Gate -1: camera cut nearby — every joint jumped, nothing here is a strike
        if c["near_cut"]:
            reject(c, "scene_cut"); continue
        # Gate 0: a real strike accelerates over several frames; a keypoint teleport is a one-frame spike.
        if c["support"] < cfg.min_peak_support:
            reject(c, "teleport"); continue
        # Gate 0b: two strikes by one fighter can't land closer together than this (even a 1-2 is ~5 frames)
        if me in last_strike and 0 <= c["frame"] - last_strike[me] < cfg.min_gap_any:
            reject(c, "too_soon_any"); continue
        cool = cfg.cooldown if kind == "punch" else cfg.cooldown_kick
        if (me, limb) in last_limb and 0 <= c["frame"] - last_limb[(me, limb)] < cool:
            reject(c, "too_soon_same_limb"); continue
        # Gate 0c: speed (with clinch multiplier)
        if c["speed"] < vthr:
            reject(c, "slow_clinch" if in_clinch else "slow"); continue
        # Gate 1: limb actually extended at impact (kills guard twitches / feints). Hooks are bent.
        min_ext = (cfg.min_ext_punch if kind == "punch" else cfg.min_ext_kick) * relax
        if c["ext"] < min_ext:
            reject(c, "not_extended"); continue
        # Gate 2: the tip must have TRAVELLED (a body shot goes chin-guard -> ribs without extending
        # much, so retraction gain is the wrong instrument; displacement isn't).
        min_travel = (cfg.min_punch_travel if kind == "punch" else cfg.min_kick_travel) * relax
        if not np.isnan(c["travel"]) and c["travel"] < min_travel:
            reject(c, "no_travel"); continue

        dmin, react, speed = c["land_dist"], c["head_react"], c["speed"]
        react = 0.0 if np.isnan(react) else react
        if kind == "kick" and c["target"] in ("body", "leg"):
            # 2D can't tell a foot that stops short-but-offset-in-depth from one that lands. Physics can:
            # a landed kick/teep moves the opponent. Need a very tight distance OR a visible shove.
            shoved = c["body_react"] > cfg.body_react_vel
            landed = bool(dmin < cfg.land_dist_kick_tight or (dmin < 1.5 * cfg.land_dist and shoved))
        else:
            landed = bool(dmin < cfg.land_dist or (dmin < 1.5 * cfg.land_dist and react > cfg.head_react_vel))
        if not landed:
            if dmin > (cfg.max_miss_dist if kind == "punch" else cfg.max_miss_dist_kick):
                reject(c, "miss_too_far"); continue
            if speed < vthr * cfg.miss_speed_mult:
                reject(c, "miss_too_slow"); continue
        conf = float(np.clip(
            0.5 * np.clip(1 - dmin / (2 * cfg.land_dist), 0, 1)
            + 0.3 * np.clip(react / (2 * cfg.head_react_vel), 0, 1)
            + 0.2 * np.clip((speed - vthr) / vthr, 0, 1), 0, 1))
        if not landed and conf < cfg.min_miss_conf:
            reject(c, "low_conf_miss"); continue
        last_strike[me] = c["frame"]
        last_limb[(me, limb)] = c["frame"]
        c = dict(c); c["landed"] = landed; c["conf"] = conf; c["in_combo"] = in_combo; c["in_clinch"] = in_clinch
        events.append(c)
        v = dict(c); v["rejected_by"] = ""; verdicts.append(v)

    detect_strikes.last_candidates = pd.DataFrame(verdicts).sort_values("frame").reset_index(drop=True) if verdicts else pd.DataFrame()
    cols = ["frame", "t", "fighter", "limb", "kind", "target", "landed", "speed", "land_dist", "head_react", "conf",
            "ext", "ext_gain", "travel", "cos", "support", "body_react", "range", "in_combo", "in_clinch", "near_cut", "peak"]
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
