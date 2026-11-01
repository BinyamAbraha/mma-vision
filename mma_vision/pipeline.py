"""CLI orchestrator.

    python -m mma_vision.pipeline run data/clips/fight.mp4
    python -m mma_vision.pipeline run fight.mp4 --device mps --red "Fighter A" --blue "Fighter B"
    python -m mma_vision.pipeline run fight.mp4 --max-frames 300          # quick test
    python -m mma_vision.pipeline rerender fight.mp4 --out outputs/fight   # tweak overlay, no re-inference
    python -m mma_vision.pipeline analyze outputs/fight/tracks.parquet     # re-run strikes+metrics from cache

Outputs land in outputs/<clip-name>/:
  tracks.parquet   raw per-frame keypoints (cached — inference never re-runs)
  events.csv       strike events
  frames.csv       per-frame metrics (distance, guard, position)
  summary.json     fight summary
  summary.png      shareable stat card
  annotated.mp4    the overlay video
  commentary.json / commented.mp4 (if enabled)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from .config import Config, load_config, pick_device

log = logging.getLogger("mma_vision")


def _video_meta(video: Path, cfg: Config):
    cap = cv2.VideoCapture(str(video))
    fps = cfg.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def analyze(tracks, cfg: Config, fps: float, w: int, h: int, out: Path):
    """tracks DataFrame -> events, frame-series, summary (+ files)."""
    from .tracks import build_series
    from .strikes import detect_strikes
    from .metrics import frame_series, summarize
    from .charts import summary_card

    series, F = build_series(tracks, cfg.strikes.min_conf, cfg.strikes.smooth_alpha)
    events = detect_strikes(series, fps, cfg.strikes)
    fs = frame_series(series, fps, w, h)
    names = {"red": cfg.red_name, "blue": cfg.blue_name}
    summary = summarize(events, fs, fps, cfg.round_length_sec, names)

    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "events.csv", index=False)
    fs.to_csv(out / "frames.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    try:
        summary_card(events, fs, summary, out / "summary.png")
    except Exception as e:
        log.warning("summary card failed: %s", e)
    log.info("%d strike events (%d landed). Red %d/%d, Blue %d/%d",
             len(events), int(events["landed"].sum()),
             summary["fighters"]["red"]["landed"], summary["fighters"]["red"]["thrown"],
             summary["fighters"]["blue"]["landed"], summary["fighters"]["blue"]["thrown"])
    return series, events, fs, summary


def cmd_run(a):
    from .detect import extract_tracks
    from .overlay import render
    from . import commentary as cm

    cfg = _cfg_from_args(a)
    video = Path(a.video)
    out = Path(a.out) if a.out else Path("outputs") / video.stem
    out.mkdir(parents=True, exist_ok=True)
    fps, w, h = _video_meta(video, cfg)
    log.info("Device: %s", pick_device(cfg.model.device))

    t0 = time.time()
    tracks = extract_tracks(video, cfg, cache=out / "tracks.parquet", max_frames=a.max_frames)
    log.info("tracking done in %.1fs (%d rows)", time.time() - t0, len(tracks))

    series, events, fs, summary = analyze(tracks, cfg, fps, w, h, out)

    if not a.no_render:
        t0 = time.time()
        vid = render(video, series, events, fs, cfg, out / "annotated.mp4", max_frames=a.max_frames)
        log.info("rendered %s in %.1fs", vid, time.time() - t0)
        if cfg.commentary.enabled:
            lines = cm.generate(events, summary, cfg.commentary, out / "commentary.json")
            log.info("%d commentary lines -> %s", len(lines), out / "commentary.json")
            res = cm.mux(vid, lines, cfg.commentary, out / "tts", out / "commented.mp4")
            if res:
                log.info("commentary audio muxed -> %s", res)
    print(f"\nDone. Outputs in {out}/")


def cmd_rerender(a):
    from .detect import read_tracks
    from .overlay import render

    cfg = _cfg_from_args(a)
    video = Path(a.video)
    out = Path(a.out) if a.out else Path("outputs") / video.stem
    fps, w, h = _video_meta(video, cfg)
    tracks = read_tracks(out / "tracks.parquet")
    series, events, fs, _ = analyze(tracks, cfg, fps, w, h, out)
    render(video, series, events, fs, cfg, out / "annotated.mp4", max_frames=a.max_frames)
    print(f"\nRe-rendered -> {out / 'annotated.mp4'}")


def cmd_analyze(a):
    from .detect import read_tracks
    cfg = _cfg_from_args(a)
    tracks = read_tracks(a.tracks)
    fps = cfg.fps_override or tracks.attrs.get("fps", 30.0)
    w, h = tracks.attrs.get("width", 1920), tracks.attrs.get("height", 1080)
    out = Path(a.out) if a.out else Path(a.tracks).parent
    analyze(tracks, cfg, fps, w, h, out)
    print(f"\nAnalysis written to {out}/")


def _cfg_from_args(a) -> Config:
    cfg = load_config(getattr(a, "config", "config.yaml"))
    if getattr(a, "device", None):
        cfg.model.device = a.device
    if getattr(a, "weights", None):
        cfg.model.weights = a.weights
    if getattr(a, "red", None):
        cfg.red_name = a.red
    if getattr(a, "blue", None):
        cfg.blue_name = a.blue
    if getattr(a, "commentary", False):
        cfg.commentary.enabled = True
    if getattr(a, "tts", None):
        cfg.commentary.tts = a.tts
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="mma_vision", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--out", help="output dir (default outputs/<clip>)")
        sp.add_argument("--red", help="red corner fighter name")
        sp.add_argument("--blue", help="blue corner fighter name")
        sp.add_argument("--max-frames", type=int, help="process only the first N frames (quick test)")

    r = sub.add_parser("run", help="full pipeline: track -> analyze -> render")
    r.add_argument("video"); common(r)
    r.add_argument("--device", help="auto|cpu|mps|cuda")
    r.add_argument("--weights", help="e.g. yolo26s-pose.pt")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--commentary", action="store_true", help="generate commentary script (+ TTS if --tts)")
    r.add_argument("--tts", choices=["none", "elevenlabs", "kokoro"])
    r.set_defaults(fn=cmd_run)

    rr = sub.add_parser("rerender", help="re-run analysis + overlay from cached tracks")
    rr.add_argument("video"); common(rr)
    rr.set_defaults(fn=cmd_rerender)

    an = sub.add_parser("analyze", help="strikes + metrics from a tracks file")
    an.add_argument("tracks"); common(an)
    an.set_defaults(fn=cmd_analyze)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    a.fn(a)


if __name__ == "__main__":
    main()
