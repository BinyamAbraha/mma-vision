"""Identity: offline repair fixes injected swaps; online assigner survives a collision."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mma_vision.config import Config  # noqa: E402
from mma_vision.detect import read_tracks  # noqa: E402
from mma_vision.identity import fix_identity, FighterAssigner, APP_COLS, similarity  # noqa: E402


def _tracks():
    p = ROOT / "outputs" / "synthetic" / "tracks.csv"
    if not p.exists():
        from scripts.make_synthetic import main as make
        make()
    return read_tracks(p)


def test_offline_repair_fixes_injected_swap():
    df = _tracks()
    truth = df[["frame", "fighter"]].copy()
    bad = df.copy()
    m = bad["frame"].between(300, 420)              # a 4-second swap, like a clinch exit gone wrong
    bad.loc[m, "fighter"] = bad.loc[m, "fighter"].map({"red": "blue", "blue": "red"})
    assert (bad["fighter"] != truth["fighter"]).sum() > 200

    fixed, n = fix_identity(bad)
    assert n > 0
    wrong = (fixed.sort_values(["frame", "fighter"]).reset_index(drop=True)["fighter"]
             != truth.sort_values(["frame", "fighter"]).reset_index(drop=True)["fighter"]).sum()
    # repaired frame-sets are identical, so the per-row fighter columns line up again
    assert wrong == 0


def test_templates_are_distinct():
    df = _tracks()
    r = df[df.fighter == "red"][APP_COLS].to_numpy().mean(0)
    b = df[df.fighter == "blue"][APP_COLS].to_numpy().mean(0)
    assert similarity(r, b) < 0.6


def test_online_assigner_survives_collision():
    """Feed the assigner the two fighters in random order every frame, then slam their
    boxes together for 20 frames (identical centroids). Position is useless there; appearance
    must carry it. After the collision the labels must still be right."""
    df = _tracks()
    cfg = Config()
    W, H = 1280, 720
    asg = FighterAssigner(cfg, W, H)
    rng = np.random.default_rng(1)
    wrong_after = 0
    frames = sorted(df.frame.unique())
    for fr in frames:
        g = df[df.frame == fr].set_index("fighter")
        boxes = np.stack([g.loc[f, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) for f in ("red", "blue")])
        feats = [g.loc[f, APP_COLS].to_numpy(dtype=np.float32) for f in ("red", "blue")]
        if 200 <= fr < 220:  # collision: both boxes become the union box
            u = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
            boxes = np.stack([u, u + rng.normal(0, 1, 4)])
        order = rng.permutation(2)
        out = asg.assign(boxes[order], [feats[i] for i in order])
        if fr >= 220 and len(out) == 2:
            # out maps fighter -> row index in the permuted arrays; truth: permuted row j is fighter order[j]
            truth = {("red", "blue")[order[j]]: j for j in range(2)}
            wrong_after += int(out != truth)
    assert wrong_after == 0, f"{wrong_after} frames mislabelled after the collision"
