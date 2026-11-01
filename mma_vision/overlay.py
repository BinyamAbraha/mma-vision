"""Stage 4: render the broadcast-style overlay.

Skeletons + boxes per fighter, strike flashes with labels, a live stat panel
(thrown / landed / accuracy / pace), a range indicator, and a top-down-ish
minimap with position trails. Pure OpenCV so it runs anywhere.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import KP, SKELETON, FIGHTERS
from .config import Config
from .metrics import running_counts
from .strikes import label as strike_label
from .tracks import FighterSeries

log = logging.getLogger(__name__)
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (170, 170, 170)
GREEN, RED_X = (80, 220, 120), (80, 80, 255)


def _color(cfg: Config, f: str):
    return tuple(int(c) for c in (cfg.overlay.red_bgr if f == "red" else cfg.overlay.blue_bgr))


def _ok(p) -> bool:
    return p is not None and not np.any(np.isnan(p))


def _alpha_rect(img, x1, y1, x2, y2, color=BLACK, alpha=0.55):
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return
    rect = np.full_like(sub, color, dtype=np.uint8)
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, rect, alpha, 0)


def _text(img, s, org, scale=0.55, color=WHITE, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(img, s: FighterSeries, fi: int, color, thickness=2):
    kp = s.kp[fi]
    for a, b in SKELETON:
        pa, pb = kp[KP[a]], kp[KP[b]]
        if _ok(pa) and _ok(pb):
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), color, thickness, cv2.LINE_AA)
    for k in range(17):
        if _ok(kp[k]):
            cv2.circle(img, tuple(kp[k].astype(int)), 3, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(kp[k].astype(int)), 3, color, 1, cv2.LINE_AA)


def draw_box(img, s: FighterSeries, fi: int, color, name: str):
    b = s.box[fi]
    if not _ok(b):
        return
    x1, y1, x2, y2 = b.astype(int)
    # corner brackets look more "broadcast" than a full rect
    L = max(10, (x2 - x1) // 6)
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 2, cv2.LINE_AA)
    _alpha_rect(img, x1, max(0, y1 - 22), x1 + 12 * len(name) + 10, y1, color, 0.85)
    _text(img, name, (x1 + 5, max(14, y1 - 6)), 0.5, WHITE, 1, shadow=False)


def draw_panel(img, cfg: Config, counts: dict, fi: int, t: float, fs_row: pd.Series, names: dict):
    h, w = img.shape[:2]
    pw, ph = 300, 118
    x0, y0 = 16, 16
    _alpha_rect(img, x0, y0, x0 + pw, y0 + ph, BLACK, 0.6)
    _text(img, "AI FIGHT TRACKER", (x0 + 10, y0 + 20), 0.5, GREY)
    mm, ss = divmod(int(t), 60)
    _text(img, f"{mm:02d}:{ss:02d}", (x0 + pw - 60, y0 + 20), 0.5, GREY)
    _text(img, "THROWN  LANDED   ACC", (x0 + 120, y0 + 42), 0.42, GREY)
    for i, f in enumerate(FIGHTERS):
        y = y0 + 64 + i * 24
        col = _color(cfg, f)
        cv2.rectangle(img, (x0 + 10, y - 13), (x0 + 16, y + 3), col, -1)
        _text(img, names[f][:9], (x0 + 22, y), 0.5, WHITE)
        th, la = int(counts[f"{f}_thrown"][fi]), int(counts[f"{f}_landed"][fi])
        acc = f"{100 * la / th:.0f}%" if th else "--"
        _text(img, f"{th:>4}", (x0 + 130, y), 0.5, WHITE)
        _text(img, f"{la:>4}", (x0 + 192, y), 0.5, GREEN)
        _text(img, f"{acc:>5}", (x0 + 245, y), 0.5, WHITE)
    rng = str(fs_row.get("range", "")).upper()
    d = fs_row.get("distance", np.nan)
    rng_txt = f"RANGE: {rng}" + (f"  ({d:.1f})" if not np.isnan(d) else "")
    _text(img, rng_txt, (x0 + 10, y0 + ph - 8), 0.42, GREY)


def draw_minimap(img, cfg: Config, trails: dict[str, deque], frame_w: int, frame_h: int):
    h, w = img.shape[:2]
    mw, mh = 170, 110
    x0, y0 = w - mw - 16, h - mh - 16
    _alpha_rect(img, x0, y0, x0 + mw, y0 + mh, BLACK, 0.6)
    # octagon outline
    cx, cy, r = x0 + mw // 2, y0 + mh // 2, min(mw, mh) // 2 - 8
    pts = np.array([[cx + r * np.cos(a), cy + r * np.sin(a)] for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)]).astype(int)
    cv2.polylines(img, [pts], True, GREY, 1, cv2.LINE_AA)
    _text(img, "CAGE VIEW", (x0 + 8, y0 + 14), 0.38, GREY)
    for f in FIGHTERS:
        col = _color(cfg, f)
        pts_f = [p for p in trails[f] if _ok(p)]
        for j, p in enumerate(pts_f):
            # image x -> map x; image y (depth proxy) -> map y
            mx = int(cx + (p[0] / frame_w - 0.5) * (2 * r * 0.95))
            my = int(cy + (p[1] / frame_h - 0.6) * (2 * r * 1.6))
            mx, my = np.clip(mx, x0 + 4, x0 + mw - 4), np.clip(my, y0 + 4, y0 + mh - 4)
            rad = 5 if j == len(pts_f) - 1 else 2
            cv2.circle(img, (mx, my), rad, col, -1, cv2.LINE_AA)


def render(video: str | Path, series: dict[str, FighterSeries], events: pd.DataFrame, fs: pd.DataFrame,
           cfg: Config, out_path: str | Path, max_frames: int | None = None, progress: bool = True) -> Path:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cfg.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    F = len(fs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts = running_counts(events, F)
    names = {"red": cfg.red_name, "blue": cfg.blue_name}
    ev_by_frame: dict[int, list] = {}
    for _, e in events.iterrows():
        ev_by_frame.setdefault(int(e["frame"]), []).append(e)
    active: list[tuple[int, pd.Series]] = []  # (start_frame, event)
    trails = {f: deque(maxlen=int(fps * 3)) for f in FIGHTERS}

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=min(F, max_frames or F), unit="f", desc="render")
        except ImportError:
            pass

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi >= F or (max_frames and fi >= max_frames):
            break
        for f in FIGHTERS:
            s = series[f]
            col = _color(cfg, f)
            trails[f].append(s.hip_center[fi])
            if not s.present[fi] and np.all(np.isnan(s.kp[fi])):
                continue
            if cfg.overlay.draw_skeleton:
                draw_skeleton(frame, s, fi, col)
            if cfg.overlay.draw_boxes:
                draw_box(frame, s, fi, col, names[f])

        # strike flashes
        for e in ev_by_frame.get(fi, []):
            active.append((fi, e))
        active = [(st, e) for st, e in active if fi - st < cfg.overlay.flash_frames]
        for st, e in active:
            age = (fi - st) / cfg.overlay.flash_frames
            s = series[e["fighter"]]
            tip = s.pt(e["limb"])[min(int(e["frame"]), F - 1)]
            if not _ok(tip):
                continue
            col = GREEN if e["landed"] else RED_X
            rad = int(14 + 26 * age)
            cv2.circle(frame, tuple(tip.astype(int)), rad, col, 2, cv2.LINE_AA)
            if age < 0.8:
                _text(frame, strike_label(e), (int(tip[0]) + 18, int(tip[1]) - 12), 0.55, col, 2)

        if cfg.overlay.stat_panel:
            draw_panel(frame, cfg, counts, fi, fi / fps, fs.iloc[fi], names)
        if cfg.overlay.minimap:
            draw_minimap(frame, cfg, trails, w, h)

        writer.write(frame)
        fi += 1
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    cap.release()
    writer.release()

    # Re-encode to H.264 so it plays in browsers / phones and keeps the audio track
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(video),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "20", "-c:a", "aac", "-shortest", str(out_path)]
        r = subprocess.run(cmd)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            return out_path
        log.warning("ffmpeg re-encode failed; keeping mp4v output")
    tmp.rename(out_path)
    return out_path
