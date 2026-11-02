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
    w_app: float = 0.6                  # weight of appearance (vs position) in the identity cost
    swap_margin: float = 0.12           # cost margin required to flip red/blue vs. the position-favoured pairing
    sep_iou: float = 0.05               # boxes with IoU below this count as "separated" (safe to learn appearance)
    tmpl_rate: float = 0.03             # EMA rate for appearance templates
    repair_identity: bool = True        # offline Viterbi pass that fixes swaps using the whole clip
    repair_switch_penalty: float = 0.6  # higher = fewer, longer swap segments


@dataclass
class StrikeCfg:
    smooth_alpha: float = 0.65        # EMA smoothing on keypoints (1 = none). Too low merges fast combos.
    vel_thresh: float = 0.16          # punch: wrist speed, in torso-heights per frame, to count as a strike
    vel_thresh_kick: float = 0.2      # kick: ankle speed threshold (kicks are faster than steps)
    clinch_range: float = 0.9         # (torso, hip-to-hip) closer than this = clinch / pocket
    clinch_speed_mult: float = 1.25   # speed thresholds are multiplied by this in the clinch (hands are always 'near the target' there)
    max_head_react: float = 0.5       # opponent head 'reaction' above this (torso/frame) is a keypoint jump, ignored
    min_kick_lift: float = 0.3        # kick: kicking ankle must be this far (torso) ABOVE the standing ankle
    toward_cos: float = 0.3           # min cosine between limb velocity and direction to opponent (hooks curve)
    min_ext_punch: float = 0.5        # wrist at least this far (torso) from own shoulder at impact (body shots & hooks are bent)
    min_ext_kick: float = 0.6         # ankle at least this far (torso) from own hip at impact (low: 2D foreshortening)
    min_ext_gain: float = 0.25        # (diagnostic only now) retraction gain, kept in events.csv
    min_punch_travel: float = 0.35    # punch: wrist must have moved this far (torso) within the lookback window
    impact_window: int = 6            # frames after the speed peak to look for impact (punch)
    impact_window_kick: int = 12      # kicks unfold slower: chamber -> extension
    min_kick_travel: float = 0.5      # kick: ankle must have moved this far (torso) within the lookback window
    ext_lookback: int = 8             # frames before the speed peak used for the extension/travel baseline
    land_dist: float = 0.35           # gap (torso) between limb tip and target SURFACE to call it landed
    land_dist_kick_tight: float = 0.12  # body/leg kicks: landed on distance alone only if this close...
    body_react_vel: float = 0.05      # ...otherwise the opponent's hips must get shoved (torso/frame)
    cut_guard: int = 10               # ignore candidates within this many frames of a camera cut
    head_react_vel: float = 0.03      # opponent head speed jump after impact that confirms a landed shot
    react_window: int = 6             # frames to look for head reaction after peak
    cooldown: int = 8                 # frames before the same hand can punch again (8 @30fps ≈ a fast double jab)
    cooldown_kick: int = 15           # frames before the same leg can kick again (a kick + return takes ~0.5s)
    min_gap_any: int = 4              # min frames between ANY two strikes by one fighter (a 1-2 is ~5 frames)
    combo_window: int = 15            # frames after a strike during which the next one counts as a combo
    combo_relax: float = 0.6          # multiply extension/retraction gates by this inside a combo
    max_miss_dist: float = 0.9        # (torso) a punch miss further than this from the target is hand-fighting, not a strike
    max_miss_dist_kick: float = 1.4   # kicks that get checked/parried stop further out
    min_peak_support: int = 1         # neighbouring frames (of 4) that must carry >= support_frac of peak speed — kills keypoint teleports
    support_frac: float = 0.2         # a 3-frame real punch has neighbours at ~30%; a teleport at ~3%
    max_limb_len: float = 1.0         # multiplier on anatomical limb-length limits used to drop 'stolen' joints
    miss_speed_mult: float = 1.3      # a miss needs this x the speed threshold (landed shots don't)
    min_conf: float = 0.3             # keypoint confidence below this is treated as missing
    min_miss_conf: float = 0.0        # optional extra FP guard: drop *missed* strikes with conf below this


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
