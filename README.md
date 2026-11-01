# MMA Vision 🥊

Computer-vision fight analysis for MMA footage. Detects and tracks both fighters, estimates
their pose, detects strikes (thrown / landed / target), computes fight metrics, and renders a
broadcast-style overlay — optionally with AI commentary.

```
video ──► YOLO26-pose + ByteTrack ──► fighter keypoints ──► strike detector ──► metrics
                                                                   │
                                        annotated.mp4 ◄── overlay ◄┘──► summary.png / summary.json
                                               │
                                     (optional) Claude commentary ──► TTS ──► commented.mp4
```

**Post-processed, not real-time.** Runs on CPU, Apple Silicon (MPS) or CUDA. No GPU required.

## Quick start

```bash
git clone <this repo> && cd mma-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 30-second smoke test — no model download, no footage needed
python scripts/make_synthetic.py
python -m mma_vision.pipeline rerender data/clips/synthetic.mp4
open outputs/synthetic/annotated.mp4 outputs/synthetic/summary.png

# Real footage
scripts/download_clip.sh "https://youtube.com/watch?v=..." fight 00:10:00 00:12:00
python -m mma_vision.pipeline run data/clips/fight.mp4 --red "FIGHTER A" --blue "FIGHTER B"
```

First real run downloads the pose weights (~6 MB for `yolo26n-pose`). Device is auto-selected
(`cuda` → `mps` → `cpu`); override with `--device`.

Outputs in `outputs/<clip>/`:

| file | what |
|---|---|
| `annotated.mp4` | overlay video — skeletons, strike flashes, live stat panel, cage minimap |
| `summary.png` | shareable stat card (totals, by target, landed-over-time, range) |
| `summary.json` | per-fighter numbers: thrown/landed/accuracy/pace, by kind & target, per round, guard drop, centre control |
| `events.csv` | every strike: frame, fighter, limb, kind, target, landed, confidence |
| `frames.csv` | per-frame distance, range bucket, guard height, position, movement |
| `tracks.parquet` | raw keypoints — **cached**, so inference never re-runs |

## Iterating

Inference is the slow part and it's cached. Everything downstream re-runs in seconds:

```bash
# edit config.yaml (thresholds, colours, names), then:
python -m mma_vision.pipeline rerender data/clips/fight.mp4
# or just the numbers, no video:
python -m mma_vision.pipeline analyze outputs/fight/tracks.parquet
```

Quick test on the first 10 seconds: `--max-frames 300`.

## How strike detection works

No training data. For each fighter's wrists and ankles:

1. Limb-tip speed, normalised by the fighter's torso height (so thresholds work at any zoom).
2. A **strike** = a speed peak above `vel_thresh` whose direction points at the opponent (`toward_cos`).
3. Impact frame = max limb extension right after the peak.
4. **Landed** if the tip is within `land_dist` torso-heights of a target zone (head / body / legs)
   *or* nearly so and the opponent's head snaps in the next few frames (`head_react_vel`).
5. Wrist → punch (head / body); ankle → kick (head / body / leg). Same-limb cooldown prevents double counts.

On the synthetic test fight it finds 11/11 strikes with 0 false positives and 10/11 correct targets
(`python -m pytest tests/`). On real footage expect roughly 75–85 % agreement with official stats —
clinch work and elbows/knees are the weak spots (see Roadmap). All thresholds live in `config.yaml`;
`events.csv` has a `conf` column so you can filter.

Swap in a learned classifier later by replacing `strikes.detect_strikes` — the event schema stays the same.

## Metrics layered on top

- **Output & accuracy** — thrown / landed / % per fighter, per round, by kind and target.
- **Pace** — strikes per minute, landed-over-time curve.
- **Range control** — hip-to-hip distance in torso-heights, bucketed into clinch / punch / kick / out.
- **Cage control** — fraction of time each fighter is nearer the cage centre.
- **Guard** — average hand height vs. head, and its trend across the fight (fatigue signal).
- **Movement** — average displacement speed per fighter.

## AI commentary (optional)

```bash
# free, offline, template-based script
python -m mma_vision.pipeline run fight.mp4 --commentary
# Claude-written broadcast commentary (needs ANTHROPIC_API_KEY, < $1 per fight)
#   set commentary.provider: claude in config.yaml
# add a voice: --tts kokoro (local, free)  |  --tts elevenlabs (ELEVENLABS_API_KEY)
python -m mma_vision.pipeline run fight.mp4 --commentary --tts kokoro
```

The script is written to `commentary.json` first, so you can edit it before spending TTS credits.

## Free GPU

`scripts/kaggle_run.ipynb` — Kaggle gives ~30 GPU hours/week free. Use `--weights yolo26s-pose.pt`
or `m` there for better keypoints; stick to `n` on CPU.

## Footage tips

- Wide, static camera angle. Every hard cut resets tracking; close-ups lose a fighter.
- 720p is plenty. 1080p + `model.imgsz: 960` helps when fighters are small in frame.
- Red = the fighter on the **left** in the first second of the clip.

## Roadmap

- [ ] Elbow & knee detection (currently only wrists/ankles)
- [ ] Clinch / takedown state (boxes overlap → "clinch", tracked separately from striking)
- [ ] Small keypoint-sequence classifier (jab / cross / hook / roundhouse) trained on ~300 labelled clips
- [ ] Homography from cage lines → true top-down minimap
- [ ] Streamlit / HF Space demo

## Stack

[Ultralytics YOLO26-pose](https://docs.ultralytics.com/) · ByteTrack · OpenCV · pandas · matplotlib ·
optional Claude / ElevenLabs / Kokoro. Ultralytics is AGPL-3.0 — fine for an open-source project.
