"""Turn the long-form tracks DataFrame into clean per-fighter keypoint arrays.

Handles: missing frames, low-confidence keypoints, short gaps (interpolated),
EMA smoothing, and a per-fighter scale (torso height) so every threshold in the
pipeline is expressed in "torso heights" and works at any resolution / zoom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import KP, KP_NAMES, FIGHTERS


@dataclass
class FighterSeries:
    name: str
    kp: np.ndarray          # (F,17,2) smoothed, NaN where unknown
    conf: np.ndarray        # (F,17)
    box: np.ndarray         # (F,4) xyxy, NaN where missing
    present: np.ndarray     # (F,) bool
    scale: np.ndarray       # (F,) torso height in px (rolling median, never NaN)
    cuts: np.ndarray = None  # (F,) bool — True on frames where the camera cut (shared by both fighters)

    def pt(self, name: str) -> np.ndarray:
        return self.kp[:, KP[name], :]

    def mid(self, a: str, b: str) -> np.ndarray:
        return np.nanmean(np.stack([self.pt(a), self.pt(b)]), axis=0)

    @property
    def torso_center(self) -> np.ndarray:
        return np.nanmean(np.stack([self.mid("l_shoulder", "r_shoulder"), self.mid("l_hip", "r_hip")]), axis=0)

    @property
    def head(self) -> np.ndarray:
        face = self.kp[:, [KP["nose"], KP["l_eye"], KP["r_eye"], KP["l_ear"], KP["r_ear"]], :]
        return np.nanmean(face, axis=1)

    @property
    def hip_center(self) -> np.ndarray:
        return self.mid("l_hip", "r_hip")

    @property
    def knee_center(self) -> np.ndarray:
        return self.mid("l_knee", "r_knee")


def _interp_nan(arr: np.ndarray, limit: int) -> np.ndarray:
    """Linear interpolation of NaN runs up to `limit` frames, per column."""
    out = arr.copy()
    F = arr.shape[0]
    flat = out.reshape(F, -1)
    for c in range(flat.shape[1]):
        s = pd.Series(flat[:, c])
        flat[:, c] = s.interpolate(limit=limit, limit_area="inside").to_numpy()
    return flat.reshape(arr.shape)


def _ema(arr: np.ndarray, alpha: float, reset: np.ndarray | None = None) -> np.ndarray:
    """EMA that skips NaNs (keeps them NaN) — alpha=1 disables smoothing. Resets at `reset` frames
    (camera cuts) so a jump isn't smeared into a fake multi-frame motion."""
    if alpha >= 1.0:
        return arr
    out = arr.copy()
    F = arr.shape[0]
    prev = out[0].copy()
    for i in range(1, F):
        cur = out[i]
        if reset is not None and reset[i]:
            out[i] = cur
            prev = cur.copy()
            continue
        m = ~np.isnan(cur) & ~np.isnan(prev)
        cur = np.where(m, alpha * cur + (1 - alpha) * prev, cur)
        out[i] = cur
        prev = np.where(np.isnan(cur), prev, cur)
    return out


# (child, parent, max length in torso-heights). Real-world ratios ~0.5–0.6; margin for pose error.
LIMB_LIMITS = [
    ("l_wrist", "l_elbow", 0.85), ("r_wrist", "r_elbow", 0.85),
    ("l_elbow", "l_shoulder", 0.85), ("r_elbow", "r_shoulder", 0.85),
    ("l_ankle", "l_knee", 1.0), ("r_ankle", "r_knee", 1.0),
    ("l_knee", "l_hip", 1.0), ("r_knee", "r_hip", 1.0),
]


def _drop_impossible_limbs(kp: np.ndarray, scale: np.ndarray, max_len: float = 1.0) -> np.ndarray:
    """NaN-out child joints whose segment to the parent is longer than physically possible.
    Processed parent-first so a stolen elbow also invalidates the wrist hanging off it."""
    out = kp.copy()
    for child, parent, lim in sorted(LIMB_LIMITS, key=lambda t: t[1].endswith(("shoulder", "hip")), reverse=True):
        c, p = out[:, KP[child], :], out[:, KP[parent], :]
        d = np.linalg.norm(c - p, axis=1) / scale
        bad = ~np.isnan(d) & (d > lim * max_len)
        out[bad, KP[child], :] = np.nan
    return out


def detect_cuts(df: pd.DataFrame, F: int, diff_thresh: float = 28.0, jump_torso: float = 1.2) -> np.ndarray:
    """Camera cuts: image changes a lot (frame_diff, when present) OR both fighters' hip centres jump
    together by more than `jump_torso` torso-heights in a single frame."""
    cuts = np.zeros(F, dtype=bool)
    if "frame_diff" in df.columns:
        fd = df.groupby("frame")["frame_diff"].first()
        cuts[fd.index[fd.values > diff_thresh].to_numpy()] = True
    hips = {}
    for f in FIGHTERS:
        sub = df[df["fighter"] == f].drop_duplicates("frame").set_index("frame")
        hc = np.full((F, 2), np.nan)
        if len(sub):
            hc[sub.index.to_numpy()] = np.stack([
                (sub["kp_l_hip_x"].to_numpy() + sub["kp_r_hip_x"].to_numpy()) / 2,
                (sub["kp_l_hip_y"].to_numpy() + sub["kp_r_hip_y"].to_numpy()) / 2], axis=1)
            tor = np.linalg.norm(
                np.stack([(sub["kp_l_shoulder_x"] + sub["kp_r_shoulder_x"]) / 2 - (sub["kp_l_hip_x"] + sub["kp_r_hip_x"]) / 2,
                          (sub["kp_l_shoulder_y"] + sub["kp_r_shoulder_y"]) / 2 - (sub["kp_l_hip_y"] + sub["kp_r_hip_y"]) / 2], axis=1), axis=1)
            sc = np.nanmedian(tor) if np.isfinite(np.nanmedian(tor)) else 100.0
        else:
            sc = 100.0
        jump = np.r_[0.0, np.linalg.norm(np.diff(hc, axis=0), axis=1)] / max(sc, 1e-3)
        hips[f] = jump
    both = (hips["red"] > jump_torso) & (hips["blue"] > jump_torso)
    cuts |= np.nan_to_num(both, nan=False)
    return cuts


def build_series(df: pd.DataFrame, min_conf: float = 0.3, smooth_alpha: float = 0.6,
                 gap_limit: int = 5, max_len: float = 1.0) -> tuple[dict[str, FighterSeries], int]:
    """Returns ({fighter: FighterSeries}, n_frames). max_len scales LIMB_LIMITS (1.0 = defaults)."""
    if df.empty:
        raise ValueError("No tracks — did detection find any people?")
    F = int(df["frame"].max()) + 1
    cuts = detect_cuts(df, F)
    out: dict[str, FighterSeries] = {}
    xcols = [f"kp_{n}_x" for n in KP_NAMES]
    ycols = [f"kp_{n}_y" for n in KP_NAMES]
    ccols = [f"kp_{n}_c" for n in KP_NAMES]

    for f in FIGHTERS:
        sub = df[df["fighter"] == f].drop_duplicates("frame").set_index("frame").sort_index()
        kp = np.full((F, 17, 2), np.nan)
        conf = np.zeros((F, 17))
        box = np.full((F, 4), np.nan)
        present = np.zeros(F, dtype=bool)
        if len(sub):
            idx = sub.index.to_numpy()
            kp[idx, :, 0] = sub[xcols].to_numpy()
            kp[idx, :, 1] = sub[ycols].to_numpy()
            conf[idx] = sub[ccols].to_numpy()
            box[idx] = sub[["x1", "y1", "x2", "y2"]].to_numpy()
            present[idx] = True
        # Drop low-confidence / zero keypoints (YOLO emits (0,0) for invisible joints)
        bad = (conf < min_conf) | ((kp[..., 0] == 0) & (kp[..., 1] == 0))
        kp[bad] = np.nan

        # Scale: torso height (shoulder-mid to hip-mid). Rolling median, filled.
        sh = np.nanmean(kp[:, [KP["l_shoulder"], KP["r_shoulder"]], :], axis=1)
        hp = np.nanmean(kp[:, [KP["l_hip"], KP["r_hip"]], :], axis=1)
        torso = np.linalg.norm(sh - hp, axis=1)
        s = pd.Series(torso).rolling(45, min_periods=5, center=True).median()
        s = s.bfill().ffill()
        if s.isna().all():
            # No torso visible anywhere — fall back to box height * 0.35
            bh = pd.Series(box[:, 3] - box[:, 1]).bfill().ffill()
            s = (bh * 0.35).fillna(100.0)
        scale = np.maximum(s.to_numpy(), 1e-3)

        # Anatomy check: when two bodies overlap, pose models "steal" limbs — one fighter's wrist gets
        # attached to the other's skeleton. A 2D projection can only SHORTEN a limb, so any segment
        # longer than its real length (in torso units) is impossible. Drop those joints.
        kp = _drop_impossible_limbs(kp, scale, max_len)

        kp = _interp_nan(kp, gap_limit)
        kp = _ema(kp, smooth_alpha, reset=cuts)
        out[f] = FighterSeries(f, kp, conf, box, present, scale, cuts)
    return out, F
