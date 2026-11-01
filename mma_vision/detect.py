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

from . import KP_NAMES, FIGHTERS
from .config import Config, pick_device

log = logging.getLogger(__name__)


def _load_model(cfg: Config):
    from ultralytics import YOLO  # imported lazily so the rest of the package works without torch

    try:
        return YOLO(cfg.model.weights)
    except Exception as e:  # weights not downloadable / unknown name
        log.warning("Could not load %s (%s). Falling back to %s", cfg.model.weights, e, cfg.model.fallback_weights)
        return YOLO(cfg.model.fallback_weights)


class FighterAssigner:
    """Keeps red/blue identity stable across ByteTrack ID switches.

    Per frame: keep people taller than min_box_height_frac, take the two largest
    boxes, then match them to the previous red/blue centroids (swap if that
    reduces total distance). During warmup, red = leftmost fighter.
    """

    def __init__(self, cfg: Config, frame_w: int, frame_h: int):
        self.cfg = cfg.fighters
        self.w, self.h = frame_w, frame_h
        self.prev: dict[str, np.ndarray | None] = {f: None for f in FIGHTERS}
        self.locked = False
        self.n_seen = 0

    def assign(self, boxes: np.ndarray) -> dict[str, int]:
        """boxes: (N,4) xyxy. Returns {fighter: row_index} for up to 2 fighters."""
        if len(boxes) == 0:
            return {}
        heights = boxes[:, 3] - boxes[:, 1]
        keep = np.where(heights >= self.cfg.min_box_height_frac * self.h)[0]
        if len(keep) == 0:
            keep = np.argsort(-heights)[:2]
        areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
        top = keep[np.argsort(-areas)[:2]]
        cents = np.stack([(boxes[top, 0] + boxes[top, 2]) / 2, (boxes[top, 1] + boxes[top, 3]) / 2], axis=1)

        if not self.locked or any(v is None for v in self.prev.values()):
            # Warmup / cold start: red is leftmost.
            order = np.argsort(cents[:, 0])
            out = {}
            for f, i in zip(FIGHTERS, order):
                out[f] = int(top[i])
                self.prev[f] = cents[i]
            self.n_seen += 1
            if self.n_seen >= self.cfg.warmup_frames and len(top) == 2:
                self.locked = True
            return out

        # Match to previous centroids, try both pairings.
        if len(top) == 1:
            d = {f: np.linalg.norm(cents[0] - self.prev[f]) for f in FIGHTERS}
            f = min(d, key=d.get)
            self.prev[f] = cents[0]
            return {f: int(top[0])}

        d_straight = np.linalg.norm(cents[0] - self.prev["red"]) + np.linalg.norm(cents[1] - self.prev["blue"])
        d_swap = np.linalg.norm(cents[1] - self.prev["red"]) + np.linalg.norm(cents[0] - self.prev["blue"])
        pair = (0, 1) if d_straight <= d_swap else (1, 0)
        out = {}
        for f, i in zip(FIGHTERS, pair):
            jump = np.linalg.norm(cents[i] - self.prev[f])
            if jump > self.cfg.max_jump_frac * self.w:
                # Likely a camera cut; re-anchor rather than drop.
                log.debug("Large jump for %s (%.0f px) — re-anchoring", f, jump)
            self.prev[f] = cents[i]
            out[f] = int(top[i])
        return out


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
        # Reuse only if the cache covers the frames we want AND came from the same model.
        if have >= want * 0.95 and same_weights:
            log.info("Loading cached tracks from %s (%d frames)", cache, have)
            return cached
        why = f"covers {have}/{want} frames" if same_weights else f"was made with {cached.attrs.get('weights')}"
        log.info("Cache %s — re-running inference", why)

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

    for fi, res in enumerate(stream):
        if max_frames and fi >= max_frames:
            break
        if bar:
            bar.update(1)
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(boxes), -1)
        kxy = res.keypoints.xy.cpu().numpy()              # (N,17,2)
        kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])

        for fighter, i in assigner.assign(boxes).items():
            row = {
                "frame": fi, "t": fi / fps, "fighter": fighter, "track_id": int(ids[i]),
                "x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
                "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]), "box_conf": float(confs[i]),
            }
            for k, name in enumerate(KP_NAMES):
                row[f"kp_{name}_x"] = float(kxy[i, k, 0])
                row[f"kp_{name}_y"] = float(kxy[i, k, 1])
                row[f"kp_{name}_c"] = float(kc[i, k])
            rows.append(row)
    if bar:
        bar.close()

    df = pd.DataFrame(rows)
    df.attrs.update({"fps": fps, "width": w, "height": h, "video": str(video),
                     "max_frames": max_frames, "weights": cfg.model.weights})
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
