"""Generate a synthetic 'fight' (two stick figures) + ground-truth tracks.

Lets you exercise strikes -> metrics -> overlay with zero ML dependencies, and
gives a tiny labelled set to sanity-check the strike detector:

    python scripts/make_synthetic.py            # writes data/clips/synthetic.mp4 + outputs/synthetic/tracks.csv
    python -m mma_vision.pipeline rerender data/clips/synthetic.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mma_vision import KP, KP_NAMES, SKELETON  # noqa: E402
from mma_vision.detect import write_tracks, TRACKS_SCHEMA  # noqa: E402
from mma_vision.identity import appearance, APP_COLS  # noqa: E402

W, H, FPS, SECONDS = 1280, 720, 30, 23
TORSO = 120.0


def stance(cx: float, cy: float, facing: int) -> np.ndarray:
    """A neutral fighting stance, 17 COCO keypoints. facing=+1 faces right."""
    k = np.zeros((17, 2))
    f = facing
    k[KP["nose"]] = (cx + 8 * f, cy - TORSO * 0.75)
    k[KP["l_eye"]] = (cx + 4 * f, cy - TORSO * 0.80); k[KP["r_eye"]] = (cx + 12 * f, cy - TORSO * 0.80)
    k[KP["l_ear"]] = (cx - 8 * f, cy - TORSO * 0.78); k[KP["r_ear"]] = (cx + 18 * f, cy - TORSO * 0.78)
    k[KP["l_shoulder"]] = (cx - 20 * f, cy - TORSO * 0.5); k[KP["r_shoulder"]] = (cx + 20 * f, cy - TORSO * 0.5)
    k[KP["l_hip"]] = (cx - 16 * f, cy + TORSO * 0.5); k[KP["r_hip"]] = (cx + 16 * f, cy + TORSO * 0.5)
    # guard up: elbows down, wrists near chin
    k[KP["l_elbow"]] = (cx + 10 * f, cy - TORSO * 0.2); k[KP["r_elbow"]] = (cx + 35 * f, cy - TORSO * 0.2)
    k[KP["l_wrist"]] = (cx + 25 * f, cy - TORSO * 0.6); k[KP["r_wrist"]] = (cx + 45 * f, cy - TORSO * 0.6)
    k[KP["l_knee"]] = (cx - 30 * f, cy + TORSO * 1.1); k[KP["r_knee"]] = (cx + 40 * f, cy + TORSO * 1.1)
    k[KP["l_ankle"]] = (cx - 40 * f, cy + TORSO * 1.7); k[KP["r_ankle"]] = (cx + 50 * f, cy + TORSO * 1.7)
    return k


def bump(t: float, t0: float, dur: float) -> float:
    """0->1->0 over [t0, t0+dur] with a fast snap (strike-like)."""
    if t < t0 or t > t0 + dur:
        return 0.0
    u = (t - t0) / dur
    return float(np.sin(np.pi * u) ** 0.7)


# (time, fighter, limb, target, lands?)   -- the ground truth
SCRIPT = [
    (2.0, "red", "r_wrist", "head", True),
    (3.5, "red", "l_wrist", "head", False),
    (5.0, "blue", "r_wrist", "body", True),
    (6.5, "blue", "r_ankle", "leg", True),
    (8.0, "red", "r_ankle", "body", True),
    (9.5, "blue", "l_wrist", "head", False),
    (11.0, "red", "r_wrist", "head", True),
    (12.5, "red", "l_wrist", "body", True),
    (14.0, "blue", "r_ankle", "head", False),
    (15.5, "blue", "r_wrist", "head", True),
    (17.0, "red", "r_ankle", "leg", True),
    # 1-2-3 combo: jab, cross, lead hook — 0.2s apart, same hand reused 0.45s later. Must count as 3.
    (20.0, "red", "l_wrist", "head", True),
    (20.2, "red", "r_wrist", "head", True),
    (20.45, "red", "l_wrist", "body", True),
    # blue answers with a 1-2 where the jab misses and the cross lands
    (21.3, "blue", "l_wrist", "head", False),
    (21.5, "blue", "r_wrist", "head", True),
]


def main(out_video=Path("data/clips/synthetic.mp4"), out_tracks=Path("outputs/synthetic/tracks.csv")):
    rng = np.random.default_rng(0)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []
    F = FPS * SECONDS
    gt = []
    for fi in range(F):
        t = fi / FPS
        # fighters drift around, ~1.7 torso-heights apart
        sway = 25 * np.sin(t * 0.7)
        gap = TORSO * 1.7 + 30 * np.sin(t * 0.4)
        red_c = np.array([W / 2 - gap / 2 + sway, H * 0.5])
        blue_c = np.array([W / 2 + gap / 2 + sway, H * 0.5])
        ks = {"red": stance(*red_c, +1), "blue": stance(*blue_c, -1)}
        # Walk-in (t=18.5-19.6): red steps forward fast, feet on the floor. Must NOT register as kicks.
        if 18.5 <= t <= 19.6:
            u = (t - 18.5) / 1.1
            step = 1.0 * TORSO
            for leg, phase in (("r", 0.0), ("l", 0.5)):   # each foot lunges forward in ~4 frames
                s_ = np.clip((u - phase) * 8, 0, 1)
                ks["red"][KP[f"{leg}_ankle"]][0] += step * s_
                ks["red"][KP[f"{leg}_knee"]][0] += step * s_ * 0.7
            ks["red"][[KP["l_hip"], KP["r_hip"], KP["l_shoulder"], KP["r_shoulder"], KP["nose"],
                       KP["l_eye"], KP["r_eye"], KP["l_ear"], KP["r_ear"], KP["l_elbow"], KP["r_elbow"],
                       KP["l_wrist"], KP["r_wrist"]]] += (step * 0.6 * u, 0)
        # apply scripted strikes
        for (t0, f, limb, target, lands) in SCRIPT:
            b = bump(t, t0, 0.35 if t0 < 19 else 0.28)
            if b <= 0:
                continue
            opp = "blue" if f == "red" else "red"
            tgt = {"head": ks[opp][KP["nose"]],
                   "body": (ks[opp][KP["l_shoulder"]] + ks[opp][KP["r_hip"]]) / 2,
                   "leg": ks[opp][KP["r_knee"]]}[target]
            reach = 1.0 if lands else (0.75 if "ankle" in limb else 0.55)  # misses stop short (kicks travel further)
            tip0 = ks[f][KP[limb]]
            ks[f][KP[limb]] = tip0 + (tgt - tip0) * reach * b
            # elbow/knee follows halfway
            joint = limb.replace("wrist", "elbow").replace("ankle", "knee")
            j0 = ks[f][KP[joint]]
            ks[f][KP[joint]] = j0 + (tgt - j0) * 0.5 * reach * b
            # opponent head snaps on impact
            if lands and b > 0.9:
                ks[opp][[KP["nose"], KP["l_eye"], KP["r_eye"], KP["l_ear"], KP["r_ear"]]] += (tgt - tip0) / np.linalg.norm(tgt - tip0 + 1e-6) * 18
            if abs(t - (t0 + 0.175)) < 0.5 / FPS:
                gt.append({"t": t0 + 0.175, "fighter": f, "limb": limb, "target": target, "landed": lands})

        frame = np.full((H, W, 3), (40, 44, 52), np.uint8)
        cv2.ellipse(frame, (W // 2, int(H * 0.72)), (520, 160), 0, 0, 360, (70, 75, 85), -1)
        jit = {f: ks[f] + rng.normal(0, 1.2, ks[f].shape) for f in ks}   # detector jitter
        for f, col in (("red", (60, 60, 230)), ("blue", (230, 140, 50))):
            k = jit[f]
            for a, b_ in SKELETON:
                cv2.line(frame, tuple(k[KP[a]].astype(int)), tuple(k[KP[b_]].astype(int)), col, 6, cv2.LINE_AA)
            # "shorts": a filled quad between hips and knees so the appearance model has something to see
            quad = np.array([k[KP["l_hip"]], k[KP["r_hip"]], k[KP["r_knee"]], k[KP["l_knee"]]]).astype(int)
            cv2.fillConvexPoly(frame, quad, col, cv2.LINE_AA)
            cv2.circle(frame, tuple(k[KP["nose"]].astype(int)), 22, col, -1, cv2.LINE_AA)
        for f in ("red", "blue"):
            k = jit[f]
            x1, y1 = k.min(0) - 15; x2, y2 = k.max(0) + 15
            box = np.array([x1, y1, x2, y2])
            row = {"frame": fi, "t": t, "fighter": f, "track_id": 1 if f == "red" else 2,
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2, "box_conf": 0.9}
            for i, n in enumerate(KP_NAMES):
                row[f"kp_{n}_x"], row[f"kp_{n}_y"], row[f"kp_{n}_c"] = k[i, 0], k[i, 1], 0.95
            for c, v in zip(APP_COLS, appearance(frame, box, k, np.full(17, 0.95))):
                row[c] = float(v)
            rows.append(row)
        writer.write(frame)
    writer.release()
    df = pd.DataFrame(rows)
    df.attrs.update({"fps": FPS, "width": W, "height": H, "video": str(out_video), "schema": TRACKS_SCHEMA})
    write_tracks(df, out_tracks)
    pd.DataFrame(gt).drop_duplicates().to_csv(out_tracks.parent / "ground_truth.csv", index=False)
    print(f"wrote {out_video} ({F} frames) and {out_tracks} + ground_truth.csv ({len(SCRIPT)} strikes)")


if __name__ == "__main__":
    main()
