"""Per-frame joint and segment kinematics from the 2D keypoints.

Everything here is IMAGE-PLANE kinematics from a single broadcast camera. Angles are the
projected angles, speeds are in torso-heights per second (no metric calibration). A limb pointing
at the camera foreshortens: an elbow aimed at the lens reads as bent whatever it's doing. Treat
these as a consistent relative signal, not a measurement.

Output columns (per frame, per fighter), written to kinematics.csv:
  frame, t, fighter,
  elbow_l, elbow_r, knee_l, knee_r, hip_l, hip_r   joint angles in degrees (180 = straight)
  trunk_lean                                       torso angle from vertical, degrees (+ = leaning toward opponent)
  wrist_l_speed, wrist_r_speed, ankle_l_speed, ankle_r_speed, hip_speed   torso-heights / second
  wrist_l_accel, ...                               torso-heights / second^2 (smoothed)
  elbow_l_angvel, elbow_r_angvel, knee_l_angvel, knee_r_angvel            degrees / second
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import KP, FIGHTERS
from .tracks import FighterSeries

ANGLES = {
    "elbow_l": ("l_shoulder", "l_elbow", "l_wrist"),
    "elbow_r": ("r_shoulder", "r_elbow", "r_wrist"),
    "knee_l": ("l_hip", "l_knee", "l_ankle"),
    "knee_r": ("r_hip", "r_knee", "r_ankle"),
    "hip_l": ("l_shoulder", "l_hip", "l_knee"),
    "hip_r": ("r_shoulder", "r_hip", "r_knee"),
}
TIPS = ["l_wrist", "r_wrist", "l_ankle", "r_ankle"]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) for point arrays of shape (F,2). NaN where any point is missing."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / (n1 * n2 + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang[(n1 < 1e-6) | (n2 < 1e-6)] = np.nan
    return ang


def _deriv(x: np.ndarray, fps: float, smooth: int = 3) -> np.ndarray:
    """Central-difference derivative per second, lightly smoothed. NaN-aware."""
    s = pd.Series(x).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    d = np.full_like(s, np.nan)
    d[1:-1] = (s[2:] - s[:-2]) * fps / 2
    d[0], d[-1] = d[1], d[-2]
    return d


def compute(series: dict[str, FighterSeries], fps: float) -> pd.DataFrame:
    frames = []
    for f in FIGHTERS:
        s = series[f]
        F = len(s.scale)
        opp = series["blue" if f == "red" else "red"]
        out = {"frame": np.arange(F), "t": np.arange(F) / fps, "fighter": f}

        for name, (a, b, c) in ANGLES.items():
            out[name] = joint_angle(s.pt(a), s.pt(b), s.pt(c))
        for name in ("elbow_l", "elbow_r", "knee_l", "knee_r"):
            out[f"{name}_angvel"] = _deriv(out[name], fps)

        # trunk lean: angle of hip-mid -> shoulder-mid from vertical, signed toward the opponent
        sh = s.mid("l_shoulder", "r_shoulder")
        hp = s.mid("l_hip", "r_hip")
        trunk = sh - hp
        lean = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))  # 0 = upright, +ve = leaning to image right
        toward = np.sign(opp.hip_center[:, 0] - hp[:, 0])
        out["trunk_lean"] = lean * toward

        for tip in TIPS:
            pos = s.pt(tip)
            vel = np.vstack([np.full((1, 2), np.nan), np.diff(pos, axis=0)]) * fps / s.scale[:, None]
            speed = np.linalg.norm(vel, axis=1)
            key = tip.replace("l_", "", 1).replace("r_", "", 1)
            side = "l" if tip.startswith("l_") else "r"
            out[f"{key}_{side}_speed"] = speed
            out[f"{key}_{side}_accel"] = _deriv(speed, fps)
        hv = np.vstack([np.full((1, 2), np.nan), np.diff(s.hip_center, axis=0)]) * fps / s.scale[:, None]
        out["hip_speed"] = np.linalg.norm(hv, axis=1)
        frames.append(pd.DataFrame(out))
    return pd.concat(frames, ignore_index=True)


def summarize(kin: pd.DataFrame) -> dict:
    """Per-fighter kinematic summary for summary.json."""
    res = {}
    for f, g in kin.groupby("fighter"):
        q = lambda col, p=99: float(np.nanpercentile(g[col], p)) if g[col].notna().any() else float("nan")
        res[f] = {
            "peak_hand_speed": max(q("wrist_l_speed"), q("wrist_r_speed")),      # torso-heights/s, 99th pct
            "peak_foot_speed": max(q("ankle_l_speed"), q("ankle_r_speed")),
            "peak_elbow_angvel": max(q("elbow_l_angvel"), q("elbow_r_angvel")),  # deg/s
            "mean_elbow_angle": float(np.nanmean(g[["elbow_l", "elbow_r"]].to_numpy())),
            "mean_knee_angle": float(np.nanmean(g[["knee_l", "knee_r"]].to_numpy())),
            "mean_trunk_lean": float(np.nanmean(g["trunk_lean"])),
            "mean_hip_speed": float(np.nanmean(g["hip_speed"])),
            "units": "angles deg (180=straight), speeds torso-heights/s, image-plane 2D",
        }
    return res


def at_events(kin: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach the striking limb's speed and the striking joint's angle at impact to each event."""
    if events.empty:
        return events
    ev = events.copy()
    k = kin.set_index(["fighter", "frame"])
    speeds, angles = [], []
    for e in ev.itertuples():
        side = "l" if e.limb.startswith("l_") else "r"
        joint = "elbow" if e.kind == "punch" else "knee"
        tip = "wrist" if e.kind == "punch" else "ankle"
        try:
            row = k.loc[(e.fighter, int(e.frame))]
            win = kin[(kin.fighter == e.fighter) & kin.frame.between(int(e.peak), int(e.frame))]
            speeds.append(float(np.nanmax(win[f"{tip}_{side}_speed"])) if len(win) else np.nan)
            angles.append(float(row[f"{joint}_{side}"]))
        except KeyError:
            speeds.append(np.nan); angles.append(np.nan)
    ev["tip_speed_tps"] = speeds       # torso-heights per second, peak during the strike
    ev["joint_angle_at_impact"] = angles
    return ev
