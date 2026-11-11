"""Stage 1: video -> per-frame fighter keypoints.

Runs Ultralytics YOLO-pose with ByteTrack, then collapses the arbitrary set of
tracked people into exactly two fighters ("red" / "blue"). Output is a long-form
DataFrame cached to Parquet/CSV so downstream stages never re-run inference.

Columns: frame, t, fighter, track_id, x1, y1, x2, y2, box_conf, kp_{name}_x, kp_{name}_y, kp_{name}_c
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import KP_NAMES
from .config import Config, pick_device
from .identity import FighterAssigner, appearance, APP_COLS, APP_DIM

log = logging.getLogger(__name__)
TRACKS_SCHEMA = 3  # bump when columns change so stale caches are rebuilt


def _load_model(cfg: Config):
    from ultralytics import YOLO  # imported lazily so the rest of the package works without torch

    try:
        return YOLO(cfg.model.weights)
    except Exception as e:  # weights not downloadable / unknown name
        log.warning("Could not load %s (%s). Falling back to %s", cfg.model.weights, e, cfg.model.fallback_weights)
        return YOLO(cfg.model.fallback_weights)


def extract_tracks(video: str | Path, cfg: Config, cache: str | Path | None = None,
                   max_frames: int | None = None, progress: bool = True) -> pd.DataFrame:
    """Run pose+track on a video. Caches to `cache` (parquet or csv) if given."""
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cfg.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if cache and (Path(cache).exists() or Path(cache).with_suffix(".csv").exists()):
        cached = read_tracks(cache)
        have = int(cached["frame"].max()) + 1 if len(cached) else 0
        want = min(max_frames, n_total) if max_frames else n_total
        same_weights = cached.attrs.get("weights", cfg.model.weights) == cfg.model.weights
        same_schema = cached.attrs.get("schema") == TRACKS_SCHEMA
        # Reuse only if the cache covers the frames we want, came from the same model, and has the current columns.
        if have >= want * 0.95 and same_weights and same_schema:
            log.info("Loading cached tracks from %s (%d frames)", cache, have)
            return cached
        why = (f"covers {have}/{want} frames" if not have >= want * 0.95 else
               f"was made with {cached.attrs.get('weights')}" if not same_weights else "predates the appearance features")
        log.info("Cache %s - re-running inference", why)

    device = pick_device(cfg.model.device)
    log.info("Device: %s | video %dx%d @ %.2f fps, %d frames", device, w, h, fps, n_total)
    model = _load_model(cfg)
    assigner = FighterAssigner(cfg, w, h)

    rows: list[dict] = []
    stream = model.track(
        source=str(video), stream=True, persist=True, verbose=False,
        tracker=cfg.model.tracker, imgsz=cfg.model.imgsz, conf=cfg.model.conf,
        device=device, half=(cfg.model.half and "cuda" in device), classes=[0],
    )
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=max_frames or n_total, unit="f", desc="pose+track")
        except ImportError:
            pass

    prev_small = None
    frame_diff = 0.0
    for fi, res in enumerate(stream):
        if max_frames and fi >= max_frames:
            break
        if bar:
            bar.update(1)
        # Scene-change score: mean abs diff of a 32x18 grayscale thumbnail vs the previous frame (0-255).
        # Cuts score 40+, normal motion < 10. Stored per row so analysis can ignore cut frames.
        if res.orig_img is not None:
            small = cv2.resize(cv2.cvtColor(res.orig_img, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
            frame_diff = float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0
            prev_small = small
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(boxes), -1)
        kxy = res.keypoints.xy.cpu().numpy()              # (N,17,2)
        kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        img = res.orig_img
        feats = [appearance(img, boxes[i], kxy[i], kc[i]) if img is not None else None for i in range(len(boxes))]

        for fighter, i in assigner.assign(boxes, feats).items():
            row = {
                "frame": fi, "t": fi / fps, "fighter": fighter, "track_id": int(ids[i]),
                "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
                "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]), "box_conf": float(confs[i]),
                "frame_diff": frame_diff,
            }
            for k, name in enumerate(KP_NAMES):
                row[f"kp_{name}_x"] = float(kxy[i, k, 0])
                row[f"kp_{name}_y"] = float(kxy[i, k, 1])
                row[f"kp_{name}_c"] = float(kc[i, k])
            fv = feats[i] if feats[i] is not None else np.zeros(APP_DIM, dtype=np.float32)
            for c, v in zip(APP_COLS, fv):
                row[c] = float(v)
            rows.append(row)
    if bar:
        bar.close()

    df = pd.DataFrame(rows)
    df.attrs.update({"fps": fps, "width": w, "height": h, "video": str(video),
                     "max_frames": max_frames, "weights": cfg.model.weights, "schema": TRACKS_SCHEMA})
    if cache:
        write_tracks(df, cache)
    return df


# ---- IO helpers (attrs don't survive parquet/csv, so store a sidecar) ----

def write_tracks(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    import json
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump({k: v for k, v in df.attrs.items()}, f)


def read_tracks(path: str | Path) -> pd.DataFrame:
    import json
    path = Path(path)
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        with open(meta) as f:
            df.attrs.update(json.load(f))
    return df
