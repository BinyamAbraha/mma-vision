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


def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    """EMA that skips NaNs (keeps them NaN) — alpha=1 disables smoothing."""
    if alpha >= 1.0:
        return arr
    out = arr.copy()
    F = arr.shape[0]
    prev = out[0].copy()
    for i in range(1, F):
        cur = out[i]
        m = ~np.isnan(cur) & ~np.isnan(prev)
        cur = np.where(m, alpha * cur + (1 - alpha) * prev, cur)
        out[i] = cur
        prev = np.where(np.isnan(cur), prev, cur)
    return out


def build_series(df: pd.DataFrame, min_conf: float = 0.3, smooth_alpha: float = 0.6,
                 gap_limit: int = 5) -> tuple[dict[str, FighterSeries], int]:
    """Returns ({fighter: FighterSeries}, n_frames)."""
    if df.empty:
        raise ValueError("No tracks — did detection find any people?")
    F = int(df["frame"].max()) + 1
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
        kp = _interp_nan(kp, gap_limit)
        kp = _ema(kp, smooth_alpha)

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
        out[f] = FighterSeries(f, kp, conf, box, present, scale)
    return out, F
