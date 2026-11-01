"""Config loading + device selection."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    # Ultralytics pose weights. yolo26n-pose is fastest; bump to s/m on GPU.
    # Falls back to yolo11n-pose.pt automatically if the yolo26 weights can't be fetched.
    weights: str = "yolo26n-pose.pt"
    fallback_weights: str = "yolo11n-pose.pt"
    imgsz: int = 640
    conf: float = 0.35
    tracker: str = "bytetrack.yaml"
    device: str = "auto"  # auto | cpu | mps | cuda | cuda:0
    half: bool = False     # fp16 on CUDA only


@dataclass
class FighterCfg:
    # Fighters = the two largest, most persistent person tracks.
    # Everyone else (ref, corners, crowd) is dropped.
    min_box_height_frac: float = 0.15   # ignore people shorter than 15% of frame height
    warmup_frames: int = 15             # frames used to lock initial red/blue assignment
    max_jump_frac: float = 0.35         # max centroid jump (frac of frame width) to keep identity


@dataclass
class StrikeCfg:
    smooth_alpha: float = 0.6         # EMA smoothing on keypoints (1 = no smoothing)
    vel_thresh: float = 0.09          # limb speed, in torso-heights per frame, to count as a strike
    toward_cos: float = 0.3           # min cosine between limb velocity and direction to opponent
    land_dist: float = 0.45           # limb tip within this many torso-heights of a target => landed
    head_react_vel: float = 0.04      # opponent head speed jump after impact that confirms a landed shot
    react_window: int = 6             # frames to look for head reaction after peak
    cooldown: int = 10                # frames before the same limb can strike again
    min_conf: float = 0.3             # keypoint confidence below this is treated as missing


@dataclass
class OverlayCfg:
    draw_skeleton: bool = True
    draw_boxes: bool = True
    stat_panel: bool = True
    minimap: bool = True
    flash_frames: int = 12
    font_scale: float = 0.55
    red_bgr: tuple = (60, 60, 230)
    blue_bgr: tuple = (230, 140, 50)


@dataclass
class CommentaryCfg:
    enabled: bool = False
    provider: str = "template"   # template | claude
    model: str = "claude-sonnet-4-5"
    tts: str = "none"            # none | elevenlabs | kokoro
    voice: str = "Rachel"
    every_sec: float = 8.0       # min seconds between commentary lines
    style: str = "energetic UFC broadcast color commentator, short punchy sentences"


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    fighters: FighterCfg = field(default_factory=FighterCfg)
    strikes: StrikeCfg = field(default_factory=StrikeCfg)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    commentary: CommentaryCfg = field(default_factory=CommentaryCfg)
    fps_override: float | None = None
    round_length_sec: float = 300.0
    red_name: str = "RED"
    blue_name: str = "BLUE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc, data: dict | None):
    if not data:
        return dc
    for k, v in data.items():
        if hasattr(dc, k):
            cur = getattr(dc, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                _merge(cur, v)
            else:
                setattr(dc, k, tuple(v) if isinstance(cur, tuple) and isinstance(v, list) else v)
    return dc


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def pick_device(requested: str = "auto") -> str:
    """cuda > mps > cpu. Honors an explicit request."""
    if requested and requested != "auto":
        return requested
    env = os.environ.get("MMA_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
