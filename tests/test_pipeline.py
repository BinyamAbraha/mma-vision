"""End-to-end test on the synthetic fight - no ML deps needed.

    python -m pytest tests/ -q
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_synthetic_end_to_end(tmp_path):
    from scripts.make_synthetic import main as make
    from mma_vision.config import Config
    from mma_vision.detect import read_tracks
    from mma_vision.pipeline import analyze

    vid, trk = tmp_path / "syn.mp4", tmp_path / "out" / "tracks.csv"
    make(vid, trk)
    tracks = read_tracks(trk)
    cfg = Config()
    series, events, fs, summary = analyze(tracks, cfg, 30.0, 1280, 720, tmp_path / "out")
    gt = pd.read_csv(tmp_path / "out" / "ground_truth.csv")

    # every scripted strike is found within 0.3s, nothing extra
    m = pd.merge_asof(gt.sort_values("t"), events.sort_values("t"), on="t",
                      direction="nearest", tolerance=0.3, suffixes=("_gt", ""))
    assert m["frame"].notna().all(), "missed strikes"
    assert len(events) == len(gt), f"false positives: {len(events) - len(gt)}"
    assert (m["fighter_gt"] == m["fighter"]).all()
    assert (m["limb_gt"] == m["limb"]).all()
    assert (m["landed_gt"] == m["landed"]).mean() >= 0.9
    assert (m["target_gt"] == m["target"]).mean() >= 0.8
    assert (tmp_path / "out" / "summary.png").exists()
    assert summary["fighters"]["red"]["thrown"] == int((gt.fighter == "red").sum())


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "mma_vision.pipeline", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "rerender" in r.stdout


def test_keypoint_teleport_is_not_a_strike():
    """A wrist keypoint that jumps onto the opponent's glove for a few frames (pose model 'stealing'
    a limb) must not be credited as a punch - even though it's fast, toward the opponent and extended."""
    import numpy as np
    from mma_vision.config import Config
    from mma_vision.detect import read_tracks
    from mma_vision.tracks import build_series
    from mma_vision.strikes import detect_strikes

    df = read_tracks(ROOT / "outputs" / "synthetic" / "tracks.csv")
    cfg = Config()
    # inject: in frames 100-104 (a quiet stretch), red's r_wrist takes blue's r_wrist position
    bad = df.copy()
    for fr in range(100, 105):
        b = bad[(bad.frame == fr) & (bad.fighter == "blue")].iloc[0]
        m = (bad.frame == fr) & (bad.fighter == "red")
        bad.loc[m, "kp_r_wrist_x"] = b["kp_r_wrist_x"]
        bad.loc[m, "kp_r_wrist_y"] = b["kp_r_wrist_y"]
    series, _ = build_series(bad, cfg.strikes.min_conf, cfg.strikes.smooth_alpha, max_len=cfg.strikes.max_limb_len)
    ev = detect_strikes(series, 30.0, cfg.strikes)
    phantom = ev[(ev.fighter == "red") & (ev.limb == "r_wrist") & ev.frame.between(95, 115)]
    assert len(phantom) == 0, f"teleport counted as strike: {phantom[['frame','speed','ext','support']].values.tolist()}"


def test_camera_cut_is_not_a_strike():
    """Simulate a camera cut: from one frame on, every keypoint of both fighters shifts 300px.
    The smoothing used to smear that jump into a fake multi-frame punch."""
    from mma_vision.config import Config
    from mma_vision.detect import read_tracks
    from mma_vision.tracks import build_series
    from mma_vision.strikes import detect_strikes
    from mma_vision import KP_NAMES

    df = read_tracks(ROOT / "outputs" / "synthetic" / "tracks.csv")
    cfg = Config()
    cut_frame = 450  # quiet stretch (strikes at 12.7s=381 and 14.0s=420, next at 15.5s=465)
    bad = df.copy()
    m = bad.frame >= cut_frame
    for n in KP_NAMES:
        bad.loc[m, f"kp_{n}_x"] -= 300
    for c in ("x1", "x2"):
        bad.loc[m, c] -= 300
    series, _ = build_series(bad, cfg.strikes.min_conf, cfg.strikes.smooth_alpha)
    assert series["red"].cuts[cut_frame], "cut not detected"
    ev = detect_strikes(series, 30.0, cfg.strikes)
    phantom = ev[ev.frame.between(cut_frame - 12, cut_frame + 12)]
    assert len(phantom) == 0, f"cut produced strikes: {phantom[['t','fighter','limb','speed']].values.tolist()}"
    # and the real strike right after the cut (15.5s) still counts
    assert len(ev[ev.t.between(15.3, 16.0)]) == 1
