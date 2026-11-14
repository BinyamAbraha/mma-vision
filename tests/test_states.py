"""Fight state / knockdown / stance on the synthetic fight with injected situations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mma_vision import KP_NAMES, KP  # noqa: E402
from mma_vision.config import Config  # noqa: E402
from mma_vision.detect import read_tracks  # noqa: E402
from mma_vision.tracks import build_series  # noqa: E402
from mma_vision import states as fst  # noqa: E402

FPS = 30.0


def _tracks():
    p = ROOT / "outputs" / "synthetic" / "tracks.csv"
    if not p.exists():
        from scripts.make_synthetic import main as make
        make()
    return read_tracks(p)


def _series(df):
    cfg = Config()
    return build_series(df, cfg.strikes.min_conf, cfg.strikes.smooth_alpha)[0]


def test_baseline_is_standing_no_knockdowns():
    s = _series(_tracks())
    st = fst.fight_state(s)
    assert (st.state == "standing").mean() > 0.95, st.state.value_counts()
    assert len(fst.knockdowns(s, FPS)) == 0


def test_clinch_detected_when_boxes_merge():
    df = _tracks()
    m = (df.frame.between(300, 360)) & (df.fighter == "blue")
    # slide blue onto red for 2 seconds
    red_x = df[(df.fighter == "red")].set_index("frame")["kp_l_hip_x"]
    blue_x = df[(df.fighter == "blue")].set_index("frame")["kp_l_hip_x"]
    shift = (red_x - blue_x + 40).reindex(df.loc[m, "frame"]).to_numpy()
    for n in KP_NAMES:
        df.loc[m, f"kp_{n}_x"] += shift
    df.loc[m, "x1"] += shift; df.loc[m, "x2"] += shift
    st = fst.fight_state(_series(df))
    in_win = st[st.frame.between(305, 355)]
    assert (in_win.state == "clinch").mean() > 0.8, in_win.state.value_counts()
    assert (st[~st.frame.between(295, 365)].state == "standing").mean() > 0.95


def test_knockdown_detected_and_ground_state():
    df = _tracks()
    s0 = _series(df)
    scale = float(np.nanmedian(s0["blue"].scale))
    # blue goes down at frame 500: over 8 frames the hips drop 1.2 torso and the torso rotates horizontal
    for k, fr in enumerate(range(500, 690)):
        m = (df.frame == fr) & (df.fighter == "blue")
        if not m.any():
            continue
        u = min(1.0, k / 8)
        hip = df.loc[m, ["kp_l_hip_x", "kp_l_hip_y"]].to_numpy()[0]
        # upper body keypoints pivot about the hip to horizontal (pointing to image-left)
        for n in ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"):
            px, py = df.loc[m, f"kp_{n}_x"].to_numpy()[0], df.loc[m, f"kp_{n}_y"].to_numpy()[0]
            dx, dy = px - hip[0], py - hip[1]
            ang = -np.pi / 2 * u
            rx = dx * np.cos(ang) - dy * np.sin(ang)
            ry = dx * np.sin(ang) + dy * np.cos(ang)
            df.loc[m, f"kp_{n}_x"] = hip[0] + rx
            df.loc[m, f"kp_{n}_y"] = hip[1] + ry
        for n in KP_NAMES:
            df.loc[m, f"kp_{n}_y"] += 1.2 * scale * u
        df.loc[m, ["y1", "y2"]] += 1.2 * scale * u
    s = _series(df)
    kds = fst.knockdowns(s, FPS)
    assert len(kds) == 1, kds
    assert kds.iloc[0].fighter == "blue" and 500 <= kds.iloc[0].frame <= 525
    st = fst.fight_state(s)
    assert (st[st.frame.between(520, 680)].state == "ground").mean() > 0.9


def test_stance_and_switch():
    df = _tracks()
    s = _series(df)
    st = fst.stance(s)
    # synthetic stance: right foot forward for both -> both southpaw by the lead-foot definition
    for f in ("red", "blue"):
        g = st[(st.fighter == f) & st.frame.between(0, 300)]
        assert (g.stance == "southpaw").mean() > 0.9, g.stance.value_counts()
    # inject a switch: swap red's ankles and knees from frame 400 on
    for a, b in (("l_ankle", "r_ankle"), ("l_knee", "r_knee")):
        m = (df.frame >= 400) & (df.fighter == "red")
        for ax in ("x", "y"):
            ca, cb = f"kp_{a}_{ax}", f"kp_{b}_{ax}"
            tmp = df.loc[m, ca].copy()
            df.loc[m, ca] = df.loc[m, cb].to_numpy()
            df.loc[m, cb] = tmp.to_numpy()
    s2 = _series(df)
    st2 = fst.stance(s2)
    sw = fst.stance_switches(st2, FPS)
    red_sw = sw[sw.fighter == "red"]
    assert len(red_sw) == 1, red_sw
    assert red_sw.iloc[0]["to"] == "orthodox" and 400 <= red_sw.iloc[0].frame <= 420
    assert len(sw[sw.fighter == "blue"]) == 0
