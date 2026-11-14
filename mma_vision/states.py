"""Fight state, knockdowns, stance. All from the keypoint series, nothing learned.

state per frame:   standing | clinch | ground
knockdowns:        events where a fighter's hips drop fast and the torso goes horizontal and stays there
stance per frame:  orthodox | southpaw | unknown, defined relative to the opponent in image space
                   (lead foot = the foot nearer the opponent; left foot lead = orthodox)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FIGHTERS, KP
from .tracks import FighterSeries


from .config import StateCfg


def _torso_angle(s: FighterSeries) -> np.ndarray:
    """Degrees from vertical of hip-mid -> shoulder-mid. 0 = upright, 90 = horizontal."""
    v = s.mid("l_shoulder", "r_shoulder") - s.mid("l_hip", "r_hip")
    return np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1]) + 1e-9))


def _iou_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, 0], b[:, 0]); iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2]); iy2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ua > 0, inter / ua, 0.0)


def _debounce(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep only True runs at least min_frames long."""
    out = np.zeros_like(mask, dtype=bool)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def fight_state(series: dict[str, FighterSeries], cfg: StateCfg = StateCfg()) -> pd.DataFrame:
    A, B = series["red"], series["blue"]
    F = len(A.scale)
    scale = (A.scale + B.scale) / 2
    dist = np.linalg.norm(A.hip_center - B.hip_center, axis=1) / scale
    iou = _iou_series(np.nan_to_num(A.box), np.nan_to_num(B.box))
    ang = {f: _torso_angle(series[f]) for f in FIGHTERS}

    ground = np.zeros(F, dtype=bool)
    for f in FIGHTERS:
        ground |= _debounce(np.nan_to_num(ang[f]) > cfg.ground_angle, cfg.ground_min_frames)
    clinch = _debounce(np.nan_to_num(dist, nan=9) < cfg.clinch_dist, cfg.clinch_min_frames) & (iou > cfg.clinch_iou)
    clinch = _debounce(clinch, cfg.clinch_min_frames) & ~ground

    state = np.where(ground, "ground", np.where(clinch, "clinch", "standing"))
    return pd.DataFrame({"frame": np.arange(F), "state": state, "hip_dist": dist, "box_iou": iou,
                         "red_torso_angle": ang["red"], "blue_torso_angle": ang["blue"]})


def knockdowns(series: dict[str, FighterSeries], fps: float, cfg: StateCfg = StateCfg()) -> pd.DataFrame:
    rows = []
    for f in FIGHTERS:
        s = series[f]
        y = s.hip_center[:, 1] / s.scale          # image y grows downward, so a drop is an increase
        ang = _torso_angle(s)
        cuts = s.cuts if s.cuts is not None else np.zeros(len(y), dtype=bool)
        last = -10**9
        for i in range(cfg.kd_window, len(y)):
            if i - last < cfg.kd_cooldown:
                continue
            if cuts[max(0, i - cfg.kd_window):i + 1].any():
                continue
            drop = y[i] - np.nanmin(y[i - cfg.kd_window:i])
            if np.isnan(drop) or drop < cfg.kd_drop:
                continue
            stay = ang[i:i + cfg.kd_stay_frames]
            if len(stay) < cfg.kd_stay_frames or np.nanmean(stay) < cfg.kd_angle:
                continue
            # must have been upright shortly before (so a fighter already on the ground can't re-trigger)
            before = ang[max(0, i - 45):max(0, i - cfg.kd_window)]
            if before.size and not np.all(np.isnan(before)) and np.nanmedian(before) > cfg.kd_angle:
                continue
            rows.append({"frame": i, "t": i / fps, "fighter": f, "drop": float(drop),
                         "torso_angle": float(np.nanmean(stay))})
            last = i
    return pd.DataFrame(rows, columns=["frame", "t", "fighter", "drop", "torso_angle"])


def stance(series: dict[str, FighterSeries], cfg: StateCfg = StateCfg()) -> pd.DataFrame:
    """Per frame per fighter: orthodox / southpaw / unknown, plus switch events."""
    out = []
    for f in FIGHTERS:
        s = series[f]; o = series["blue" if f == "red" else "red"]
        F = len(s.scale)
        axis = o.hip_center - s.hip_center                      # toward the opponent
        axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
        la = s.pt("l_ankle"); ra = s.pt("r_ankle")
        # how far forward (toward the opponent) each foot is, in torso units
        l_fwd = np.sum((la - s.hip_center) * axis, axis=1) / s.scale
        r_fwd = np.sum((ra - s.hip_center) * axis, axis=1) / s.scale
        sep = l_fwd - r_fwd
        raw = np.where(np.isnan(sep), "unknown",
              np.where(sep > cfg.stance_min_sep, "orthodox",
              np.where(sep < -cfg.stance_min_sep, "southpaw", "unknown")))
        # hysteresis: a new stance must hold for stance_min_frames before it counts
        st = np.array(["unknown"] * F, dtype=object)
        cur, cand, run = "unknown", None, 0
        for i in range(F):
            r = raw[i]
            if r == cur or r == "unknown":
                cand, run = None, 0
            else:
                if r == cand:
                    run += 1
                else:
                    cand, run = r, 1
                if run >= cfg.stance_min_frames:
                    cur, cand, run = r, None, 0
            st[i] = cur
        out.append(pd.DataFrame({"frame": np.arange(F), "fighter": f, "stance": st, "foot_sep": sep}))
    df = pd.concat(out, ignore_index=True)
    return df


def stance_switches(st: pd.DataFrame, fps: float) -> pd.DataFrame:
    rows = []
    for f, g in st.groupby("fighter"):
        g = g.sort_values("frame")
        prev = None
        for fr, s in zip(g["frame"], g["stance"]):
            if s != "unknown" and prev is not None and s != prev:
                rows.append({"frame": int(fr), "t": fr / fps, "fighter": f, "from": prev, "to": s})
            if s != "unknown":
                prev = s
    return pd.DataFrame(rows, columns=["frame", "t", "fighter", "from", "to"])


def summarize(states: pd.DataFrame, kds: pd.DataFrame, st: pd.DataFrame, switches: pd.DataFrame, fps: float) -> dict:
    n = len(states)
    share = states["state"].value_counts(normalize=True).to_dict()
    out = {"time_in_state": {k: float(share.get(k, 0.0)) for k in ("standing", "clinch", "ground")},
           "knockdowns": kds[["t", "fighter"]].to_dict("records"), "fighters": {}}
    for f in FIGHTERS:
        g = st[st.fighter == f]
        vc = g["stance"].value_counts(normalize=True).to_dict()
        sw = switches[switches.fighter == f]
        out["fighters"][f] = {
            "stance_orthodox_share": float(vc.get("orthodox", 0.0)),
            "stance_southpaw_share": float(vc.get("southpaw", 0.0)),
            "primary_stance": max(("orthodox", "southpaw"), key=lambda k: vc.get(k, 0.0)) if vc else "unknown",
            "stance_switches": int(len(sw)),
            "knockdowns_suffered": int((kds.fighter == f).sum()),
        }
    return out
