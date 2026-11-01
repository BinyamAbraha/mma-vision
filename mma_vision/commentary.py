"""Stage 5 (optional): events -> timed commentary lines -> TTS -> muxed onto the video.

Two text providers:
  template — free, offline, deterministic. Good enough for a demo.
  claude   — sends the event log to Claude for broadcast-style colour commentary
             (needs ANTHROPIC_API_KEY; a whole fight costs well under $1).

TTS providers: none | elevenlabs (ELEVENLABS_API_KEY) | kokoro (local, free).
Commentary is written to commentary.json either way, so you can review/edit
the script before paying for voice.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import CommentaryCfg
from .strikes import label as strike_label

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- text ----

def _pick_moments(events: pd.DataFrame, every_sec: float, duration: float) -> list[dict]:
    """Choose which events get a commentary line: spaced out, biased to landed/high-conf."""
    moments, last_t = [], -1e9
    for _, e in events.sort_values("t").iterrows():
        if e["t"] - last_t < every_sec:
            continue
        if not e["landed"] and e["conf"] < 0.5 and random.random() < 0.7:
            continue  # skip most whiffs
        moments.append(e.to_dict())
        last_t = e["t"]
    return moments


def template_commentary(events: pd.DataFrame, summary: dict, cfg: CommentaryCfg) -> list[dict]:
    names = {f: summary["fighters"][f]["name"].title() for f in summary["fighters"]}
    lines = [{"t": 0.5, "text": f"{names['red']} in the red corner, {names['blue']} in blue. Here we go."}]
    land = ["{n} lands a clean {w}!", "Big {w} from {n}, that one connected.", "{n} with the {w} — right on the money.",
            "Oh! {n} finds a home for that {w}.", "{w} lands for {n}."]
    miss = ["{n} throws the {w}, just misses.", "{n} looking for the {w}, nothing there.", "{w} from {n} comes up short."]
    for e in _pick_moments(events, cfg.every_sec, summary["duration_sec"]):
        what = strike_label(e).replace(" LANDED", "").replace(" MISS", "").lower()
        tpl = random.choice(land if e["landed"] else miss)
        lines.append({"t": float(e["t"]) + 0.3, "text": tpl.format(n=names[e["fighter"]], w=what)})
    # Closing stat line
    r, b = summary["fighters"]["red"], summary["fighters"]["blue"]
    last_t = lines[-1]["t"] if lines else 0.0
    lines.append({"t": max(last_t + 2.5, summary["duration_sec"] - 5),
                  "text": f"Strikes landed so far: {names['red']} {r['landed']} of {r['thrown']}, "
                          f"{names['blue']} {b['landed']} of {b['thrown']}."})
    return sorted(lines, key=lambda l: l["t"])


def claude_commentary(events: pd.DataFrame, summary: dict, cfg: CommentaryCfg) -> list[dict]:
    import anthropic  # pip install anthropic

    names = {f: summary["fighters"][f]["name"] for f in summary["fighters"]}
    ev_text = "\n".join(
        f"t={e.t:6.1f}s  {names[e.fighter]:>8}  {strike_label(e):<18} conf={e.conf:.2f}"
        for e in events.itertuples()
    )
    prompt = f"""You are a {cfg.style}.
Below is a computer-vision event log from an MMA clip ({summary['duration_sec']:.0f}s). Red={names['red']}, Blue={names['blue']}.
Write commentary as JSON: a list of objects {{"t": seconds, "text": "..."}}.
Rules: one line per ~{cfg.every_sec:.0f}s at most; each line under 14 words so it fits before the next action;
react to landed strikes more than misses; mention pace/range shifts when they happen; add a short stat recap near the end.
Only output the JSON list.

SUMMARY: {json.dumps({k: v for k, v in summary.items() if k != 'fighters'})}
FIGHTERS: {json.dumps(summary['fighters'])}
EVENTS:
{ev_text}
"""
    client = anthropic.Anthropic()
    msg = client.messages.create(model=cfg.model, max_tokens=2000, messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    text = text[text.find("["): text.rfind("]") + 1]
    lines = json.loads(text)
    return [{"t": float(l["t"]), "text": str(l["text"])} for l in lines]


def generate(events: pd.DataFrame, summary: dict, cfg: CommentaryCfg, out_json: Path) -> list[dict]:
    if cfg.provider == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            lines = claude_commentary(events, summary, cfg)
        except Exception as e:
            log.warning("Claude commentary failed (%s); using template", e)
            lines = template_commentary(events, summary, cfg)
    else:
        if cfg.provider == "claude":
            log.warning("ANTHROPIC_API_KEY not set; using template commentary")
        lines = template_commentary(events, summary, cfg)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(lines, f, indent=2)
    return lines


# ----------------------------------------------------------------- tts ----

def tts_clip(text: str, out_wav: Path, cfg: CommentaryCfg) -> bool:
    try:
        if cfg.tts == "elevenlabs":
            from elevenlabs.client import ElevenLabs  # pip install elevenlabs
            client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
            audio = client.text_to_speech.convert(text=text, voice_id=cfg.voice, output_format="mp3_44100_128")
            mp3 = out_wav.with_suffix(".mp3")
            with open(mp3, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), str(out_wav)], check=True)
            return True
        if cfg.tts == "kokoro":
            import soundfile as sf  # pip install kokoro soundfile
            from kokoro import KPipeline
            pipe = KPipeline(lang_code="a")
            import numpy as np
            chunks = [audio for _, _, audio in pipe(text, voice=cfg.voice if cfg.voice != "Rachel" else "am_michael")]
            sf.write(str(out_wav), np.concatenate(chunks), 24000)
            return True
    except Exception as e:
        log.warning("TTS failed for '%s...': %s", text[:30], e)
    return False


def mux(video_in: Path, lines: list[dict], cfg: CommentaryCfg, workdir: Path, video_out: Path) -> Path | None:
    """Synthesise each line, place it at its timestamp, mix under the original audio."""
    if cfg.tts == "none" or not shutil.which("ffmpeg"):
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    inputs, filters, idx = ["-i", str(video_in)], [], 1
    for i, l in enumerate(lines):
        wav = workdir / f"line_{i:03d}.wav"
        if tts_clip(l["text"], wav, cfg):
            inputs += ["-i", str(wav)]
            filters.append(f"[{idx}:a]adelay={int(l['t'] * 1000)}|{int(l['t'] * 1000)}[a{idx}]")
            idx += 1
    if idx == 1:
        return None
    mix_in = "".join(f"[a{i}]" for i in range(1, idx))
    fc = ";".join(filters + [f"[0:a]volume=0.35[orig]", f"[orig]{mix_in}amix=inputs={idx}:normalize=0[aout]"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc,
           "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(video_out)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        # video may have no audio stream — retry without the original track
        fc = ";".join(filters + [f"{mix_in}amix=inputs={idx - 1}:normalize=0[aout]"])
        cmd[cmd.index("-filter_complex") + 1] = fc
        r = subprocess.run(cmd)
    return video_out if r.returncode == 0 else None
