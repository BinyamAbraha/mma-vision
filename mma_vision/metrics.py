"""Stage 3: tracks + events -> fight metrics.

Produces (a) per-frame time series used by the overlay and (b) a summary dict
(per fighter, per round) used by the stat panel, the commentary, and summary.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tracks import FighterSeries
from . import FIGHTERS


def frame_series(series: dict[str, FighterSeries], fps: float, frame_w: int, frame_h: int) -> pd.DataFrame:
    """Per-frame metrics: distance, guard height, position, movement speed."""
    F = len(next(iter(series.values())).scale)
    red, blue = series["red"], series["blue"]
    mean_scale = (red.scale + blue.scale) / 2

    dist = np.linalg.norm(red.hip_center - blue.hip_center, axis=1) / mean_scale
    out = {"frame": np.arange(F), "t": np.arange(F) / fps, "distance": dist}
    center = np.array([frame_w / 2, frame_h * 0.6])  # rough cage centre in image space. TODO: homography from the cage floor lines
    for f in FIGHTERS:
        s = series[f]
        wrists = np.nanmean(np.stack([s.pt("l_wrist"), s.pt("r_wrist")]), axis=0)
        # +ve = hands below nose level (low guard), in torso heights
        out[f"{f}_guard"] = (wrists[:, 1] - s.head[:, 1]) / s.scale
        hc = s.hip_center
        out[f"{f}_x"] = hc[:, 0]
        out[f"{f}_y"] = hc[:, 1]
        mv = np.vstack([np.zeros((1, 2)), np.diff(hc, axis=0)]) / s.scale[:, None]
        out[f"{f}_move"] = pd.Series(np.linalg.norm(mv, axis=1)).rolling(int(fps / 2) or 1, min_periods=1).mean().to_numpy()
        out[f"{f}_center_dist"] = np.linalg.norm(hc - center, axis=1) / frame_w
    df = pd.DataFrame(out)
    # Range bucket (torso heights between hips): clinch < 0.9, punching < 1.6, kicking < 2.4, out of range
    df["range"] = pd.cut(df["distance"], [-np.inf, 0.9, 1.6, 2.4, np.inf],
                         labels=["clinch", "punch", "kick", "out"]).astype(str)
    return df


def summarize(events: pd.DataFrame, fs: pd.DataFrame, fps: float, round_len: float = 300.0,
              names: dict[str, str] | None = None) -> dict:
    names = names or {"red": "RED", "blue": "BLUE"}
    F = len(fs)
    dur = F / fps
    rounds = max(1, int(np.ceil(dur / round_len)))
    summary: dict = {"duration_sec": dur, "fps": fps, "rounds": rounds, "fighters": {}}

    for f in FIGHTERS:
        ev = events[events["fighter"] == f]
        thrown, landed = len(ev), int(ev["landed"].sum())
        by_kind = {k: {"thrown": int(len(g)), "landed": int(g["landed"].sum())} for k, g in ev.groupby("kind")}
        by_target = {k: {"thrown": int(len(g)), "landed": int(g["landed"].sum())} for k, g in ev.groupby("target")}
        per_round = []
        for r in range(rounds):
            lo, hi = r * round_len, (r + 1) * round_len
            er = ev[(ev["t"] >= lo) & (ev["t"] < hi)]
            per_round.append({"round": r + 1, "thrown": int(len(er)), "landed": int(er["landed"].sum())})
        guard = fs[f"{f}_guard"]
        valid = fs[f"{f}_center_dist"].notna()
        other = [o for o in FIGHTERS if o != f][0]
        control = float(np.nanmean(fs[f"{f}_center_dist"] < fs[f"{other}_center_dist"])) if valid.any() else float("nan")
        summary["fighters"][f] = {
            "name": names[f],
            "thrown": thrown, "landed": landed,
            "accuracy": (landed / thrown) if thrown else 0.0,
            "strikes_per_min": thrown / (dur / 60) if dur else 0.0,
            "by_kind": by_kind, "by_target": by_target, "per_round": per_round,
            "avg_guard_drop": float(np.nanmean(guard)) if guard.notna().any() else float("nan"),
            "guard_drop_trend": _trend(guard.to_numpy()),
            "avg_movement": float(np.nanmean(fs[f"{f}_move"])),
            "center_control": control,  # fraction of time closer to cage centre than opponent
        }
    rng = fs["range"].value_counts(normalize=True).to_dict()
    summary["time_in_range"] = {k: float(rng.get(k, 0.0)) for k in ["clinch", "punch", "kick", "out"]}
    summary["avg_distance"] = float(np.nanmean(fs["distance"]))
    return summary


def _trend(x: np.ndarray) -> float:
    """Slope (per minute-ish, in units of the series per 1000 frames) of a noisy series."""
    m = ~np.isnan(x)
    if m.sum() < 10:
        return float("nan")
    i = np.arange(len(x))[m]
    return float(np.polyfit(i, x[m], 1)[0] * 1000)


def running_counts(events: pd.DataFrame, F: int) -> dict[str, np.ndarray]:
    """Cumulative thrown/landed per frame - what the overlay panel displays."""
    out = {}
    for f in FIGHTERS:
        ev = events[events["fighter"] == f]
        thrown = np.zeros(F, dtype=int)
        landed = np.zeros(F, dtype=int)
        np.add.at(thrown, ev["frame"].to_numpy().clip(0, F - 1), 1)
        np.add.at(landed, ev.loc[ev["landed"], "frame"].to_numpy().clip(0, F - 1), 1)
        out[f"{f}_thrown"] = np.cumsum(thrown)
        out[f"{f}_landed"] = np.cumsum(landed)
    return out
