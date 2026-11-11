"""Static summary graphic (PNG) - the thing you post alongside the video."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summary_card(events: pd.DataFrame, fs: pd.DataFrame, summary: dict, out_png: Path,
                 red_rgb=(0.90, 0.24, 0.24), blue_rgb=(0.20, 0.55, 0.90)) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444", "axes.labelcolor": "#ddd",
                         "xtick.color": "#bbb", "ytick.color": "#bbb", "text.color": "#eee"})
    bg = "#111318"
    fig = plt.figure(figsize=(12, 7), facecolor=bg)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    R, B = summary["fighters"]["red"], summary["fighters"]["blue"]
    cols = {"red": red_rgb, "blue": blue_rgb}

    fig.suptitle(f"{R['name']}  vs  {B['name']}  -  AI strike analysis", fontsize=17, fontweight="bold", y=0.98)

    # 1. totals
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(bg)
    labels = ["Thrown", "Landed"]
    x = np.arange(2)
    ax.bar(x - 0.18, [R["thrown"], R["landed"]], 0.36, color=cols["red"], label=R["name"])
    ax.bar(x + 0.18, [B["thrown"], B["landed"]], 0.36, color=cols["blue"], label=B["name"])
    ax.set_xticks(x, labels); ax.set_title("Strikes", color="#eee"); ax.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

    # 2. by target
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(bg)
    tg = ["head", "body", "leg"]
    rv = [R["by_target"].get(t, {}).get("landed", 0) for t in tg]
    bv = [B["by_target"].get(t, {}).get("landed", 0) for t in tg]
    x = np.arange(3)
    ax.bar(x - 0.18, rv, 0.36, color=cols["red"]); ax.bar(x + 0.18, bv, 0.36, color=cols["blue"])
    ax.set_xticks(x, [t.title() for t in tg]); ax.set_title("Landed by target", color="#eee")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

    # 3. accuracy + pace text
    ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(bg); ax.axis("off")
    txt = (f"{R['name']}\n  accuracy {100*R['accuracy']:.0f}%   pace {R['strikes_per_min']:.1f}/min\n"
           f"  centre control {100*(R['center_control'] or 0):.0f}%\n\n"
           f"{B['name']}\n  accuracy {100*B['accuracy']:.0f}%   pace {B['strikes_per_min']:.1f}/min\n"
           f"  centre control {100*(B['center_control'] or 0):.0f}%\n\n"
           f"Time in range\n  clinch {100*summary['time_in_range']['clinch']:.0f}%  punch {100*summary['time_in_range']['punch']:.0f}%\n"
           f"  kick {100*summary['time_in_range']['kick']:.0f}%  out {100*summary['time_in_range']['out']:.0f}%")
    ax.text(0, 1, txt, va="top", fontsize=10.5, family="monospace")

    # 4. cumulative landed over time
    ax = fig.add_subplot(gs[1, :2]); ax.set_facecolor(bg)
    for f in ["red", "blue"]:
        ev = events[(events["fighter"] == f) & events["landed"]].sort_values("t")
        ax.step(np.r_[0, ev["t"].to_numpy(), summary["duration_sec"]],
                np.r_[0, np.arange(1, len(ev) + 1), len(ev)], where="post", color=cols[f], lw=2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("landed"); ax.set_title("Landed strikes over time", color="#eee")
    ax.grid(alpha=0.15)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

    # 5. distance over time
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(bg)
    d = pd.Series(fs["distance"]).rolling(15, min_periods=1).mean()
    ax.plot(fs["t"], d, color="#ccc", lw=1.2)
    for y, lab in [(0.9, "clinch"), (1.6, "punch"), (2.4, "kick")]:
        ax.axhline(y, color="#555", lw=0.8, ls="--"); ax.text(fs["t"].iloc[-1], y, f" {lab}", fontsize=7, va="bottom", color="#888")
    ax.set_title("Range (torso-heights)", color="#eee"); ax.set_xlabel("time (s)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

    fig.text(0.99, 0.01, "YOLO-pose + ByteTrack + heuristic strike detector · post-processed, not real-time",
             ha="right", fontsize=8, color="#777")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    return out_png
