"""End-to-end test on the synthetic fight — no ML deps needed.

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
