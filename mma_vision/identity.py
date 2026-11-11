"""Fighter identity: appearance features + online assignment + offline swap repair.

Position alone can't keep red/blue straight when the fighters collide - at the
moment of overlap "nearest previous centroid" is a coin flip. So we also build a
colour signature per fighter (shorts + torso HSV histograms), use it in the
per-frame assignment, and then run an offline pass over the whole clip that finds
the globally consistent labelling (a 2-state Viterbi: keep / swapped).

Feature layout (APP_DIM floats per detection): [shorts H16, shorts S8, torso H16, torso S8]
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd

from . import KP, FIGHTERS

log = logging.getLogger(__name__)

H_BINS, S_BINS = 16, 8  # TODO: try LAB instead of HSV, skin tones under arena lighting drift a lot
REGION_DIM = H_BINS + S_BINS
APP_DIM = 2 * REGION_DIM
APP_COLS = [f"app_{i}" for i in range(APP_DIM)]


# ------------------------------------------------------------------ features --

def _region_rect(kxy: np.ndarray, kc: np.ndarray, names: list[str], box: np.ndarray,
                 fallback: tuple[float, float, float, float], min_conf: float = 0.3) -> tuple[int, int, int, int]:
    pts = [kxy[KP[n]] for n in names if kc[KP[n]] >= min_conf and not (kxy[KP[n]] == 0).all()]
    x1, y1, x2, y2 = box
    if len(pts) >= 3:
        p = np.array(pts)
        rx1, ry1 = p.min(0); rx2, ry2 = p.max(0)
        pad_x, pad_y = 0.15 * (rx2 - rx1) + 4, 0.1 * (ry2 - ry1) + 4
        rx1, rx2 = rx1 - pad_x, rx2 + pad_x
        ry1, ry2 = ry1 - pad_y, ry2 + pad_y
    else:
        fx1, fy1, fx2, fy2 = fallback
        bw, bh = x2 - x1, y2 - y1
        rx1, rx2 = x1 + fx1 * bw, x1 + fx2 * bw
        ry1, ry2 = y1 + fy1 * bh, y1 + fy2 * bh
    return int(rx1), int(ry1), int(rx2), int(ry2)


def _hsv_hist(img_hsv: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = img_hsv.shape[:2]
    x1, y1, x2, y2 = rect
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    out = np.zeros(REGION_DIM, dtype=np.float32)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return out
    crop = img_hsv[y1:y2, x1:x2]
    # ignore very dark / very unsaturated pixels (shadows, mat, white canvas)
    mask = ((crop[..., 2] > 40) & (crop[..., 1] > 30)).astype(np.uint8)
    if mask.sum() < 20:
        mask = None
    hh = cv2.calcHist([crop], [0], mask, [H_BINS], [0, 180]).ravel()
    ss = cv2.calcHist([crop], [1], mask, [S_BINS], [0, 256]).ravel()
    out[:H_BINS] = hh / (hh.sum() + 1e-6)
    out[H_BINS:] = ss / (ss.sum() + 1e-6)
    return out


def appearance(img_bgr: np.ndarray, box: np.ndarray, kxy: np.ndarray, kc: np.ndarray) -> np.ndarray:
    """APP_DIM colour signature of one detection: shorts region + torso region."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    shorts = _region_rect(kxy, kc, ["l_hip", "r_hip", "l_knee", "r_knee"], box, (0.2, 0.45, 0.8, 0.75))
    torso = _region_rect(kxy, kc, ["l_shoulder", "r_shoulder", "l_hip", "r_hip"], box, (0.2, 0.18, 0.8, 0.5))
    return np.concatenate([_hsv_hist(hsv, shorts), _hsv_hist(hsv, torso)])


def shirt_score(img_bgr: np.ndarray, box: np.ndarray, kxy: np.ndarray, kc: np.ndarray) -> float:
    """Fraction of the torso crop that is NOT skin. Fighters are shirtless, so this is low for them
    (Pereira's chest tattoos push him to ~0.3); the referee in a black shirt is 0.9+.

    Skin is detected by hue, not brightness: dark skin in shadow is still red/brown (H near 0/180 in
    OpenCV's 0-180 scale) while black fabric under arena lights goes blue-grey or unsaturated.
    Tested on a backlit frame where Adesanya's torso had median V=31 and still scored 0.15."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = _region_rect(kxy, kc, ["l_shoulder", "r_shoulder", "l_hip", "r_hip"], box, (0.25, 0.2, 0.75, 0.5))
    h, w = hsv.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    c = hsv[y1:y2, x1:x2]
    H, S, V = c[..., 0], c[..., 1], c[..., 2]
    skin = ((H <= 25) | (H >= 160)) & (S >= 35) & (V >= 20)
    return float(1.0 - skin.mean())


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection averaged over the 4 sub-histograms -> [0,1]."""
    if a is None or b is None:
        return 0.5
    sims = []
    for lo, hi in ((0, H_BINS), (H_BINS, REGION_DIM), (REGION_DIM, REGION_DIM + H_BINS), (REGION_DIM + H_BINS, APP_DIM)):
        sa, sb = a[lo:hi], b[lo:hi]
        if sa.sum() > 0 and sb.sum() > 0:
            sims.append(float(np.minimum(sa, sb).sum()))
    return float(np.mean(sims)) if sims else 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------ online assign --

class FighterAssigner:
    """Picks the two fighters each frame and keeps red/blue identity stable.

    Cost of assigning detection d to fighter f =
        (1 - w_app) * min(1, |centroid_d - prev_f| / (0.5 * frame_w))  +  w_app * (1 - sim(app_d, template_f))
    The pairing with the lower total cost wins, but only beats the *previous* pairing by a
    margin (hysteresis). Appearance templates are EMA-updated only when the two boxes are
    well separated, so a clinch can't blend the two fighters' colours together.
    Red = leftmost fighter during warmup.
    """

    def __init__(self, cfg, frame_w: int, frame_h: int):
        self.cfg = cfg.fighters
        self.w, self.h = frame_w, frame_h
        self.prev: dict[str, np.ndarray | None] = {f: None for f in FIGHTERS}
        self.tmpl: dict[str, np.ndarray | None] = {f: None for f in FIGHTERS}
        self.locked = False
        self.n_seen = 0
        self.last_pair = (0, 1)

    def _pick_two(self, boxes: np.ndarray) -> np.ndarray:
        heights = boxes[:, 3] - boxes[:, 1]
        keep = np.where(heights >= self.cfg.min_box_height_frac * self.h)[0]
        if len(keep) == 0:
            keep = np.argsort(-heights)[:2]
        areas = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
        return keep[np.argsort(-areas)[:2]]

    def _update_tmpl(self, f: str, feat: np.ndarray, rate: float):
        if feat is None:
            return
        self.tmpl[f] = feat.copy() if self.tmpl[f] is None else (1 - rate) * self.tmpl[f] + rate * feat

    def _cost(self, cent: np.ndarray, feat: np.ndarray | None, f: str) -> float:
        pos = min(1.0, np.linalg.norm(cent - self.prev[f]) / (0.5 * self.w))
        app = 1.0 - similarity(feat, self.tmpl[f])
        return (1 - self.cfg.w_app) * pos + self.cfg.w_app * app

    def assign(self, boxes: np.ndarray, feats: list[np.ndarray | None] | None = None,
               shirts: list[float] | None = None) -> dict[str, int]:
        if len(boxes) == 0:
            return {}
        if shirts is not None:
            keep = np.array([i for i in range(len(boxes)) if shirts[i] < self.cfg.max_shirt], dtype=int)
            if len(keep) == 0:
                return {}
            sub = self.assign(boxes[keep], [feats[i] for i in keep] if feats is not None else None, None)
            return {f: int(keep[i]) for f, i in sub.items()}
        top = self._pick_two(boxes)
        cents = np.stack([(boxes[top, 0] + boxes[top, 2]) / 2, (boxes[top, 1] + boxes[top, 3]) / 2], axis=1)
        feats = [None] * len(boxes) if feats is None else feats
        separated = len(top) == 2 and _iou(boxes[top[0]], boxes[top[1]]) < self.cfg.sep_iou

        # ---- warmup: red = leftmost; build templates quickly
        if not self.locked or any(v is None for v in self.prev.values()):
            order = np.argsort(cents[:, 0])
            out = {}
            for f, i in zip(FIGHTERS, order):
                out[f] = int(top[i])
                self.prev[f] = cents[i]
                if separated or len(top) == 1:
                    self._update_tmpl(f, feats[top[i]], 0.3)
            self.n_seen += 1
            if self.n_seen >= self.cfg.warmup_frames and len(top) == 2:
                self.locked = True
            return out

        # ---- one visible person: whichever fighter it looks/moves like (unless it's neither)
        if len(top) == 1:
            c = {f: self._cost(cents[0], feats[top[0]], f) for f in FIGHTERS}
            f = min(c, key=c.get)
            sim = similarity(feats[top[0]], self.tmpl[f])
            jump = np.linalg.norm(cents[0] - self.prev[f]) / self.w
            if self.tmpl[f] is not None and sim < self.cfg.min_sim and jump > self.cfg.impostor_jump:
                return {}
            self.prev[f] = cents[0]
            return {f: int(top[0])}

        # ---- two people: compare both pairings with hysteresis toward the previous one
        c_straight = self._cost(cents[0], feats[top[0]], "red") + self._cost(cents[1], feats[top[1]], "blue")
        c_swap = self._cost(cents[1], feats[top[1]], "red") + self._cost(cents[0], feats[top[0]], "blue")
        prev_is_straight = self.last_pair == (0, 1)
        # top[] is sorted by area each frame, so "previous pairing" has to be judged by position:
        # which pairing keeps each fighter nearest to where it was?
        d_straight = np.linalg.norm(cents[0] - self.prev["red"]) + np.linalg.norm(cents[1] - self.prev["blue"])
        d_swap = np.linalg.norm(cents[1] - self.prev["red"]) + np.linalg.norm(cents[0] - self.prev["blue"])
        favored = (0, 1) if d_straight <= d_swap else (1, 0)
        c_fav, c_alt = (c_straight, c_swap) if favored == (0, 1) else (c_swap, c_straight)
        pair = favored if c_alt > c_fav - self.cfg.swap_margin else ((1, 0) if favored == (0, 1) else (0, 1))
        margin = abs(c_straight - c_swap)

        out = {}
        for f, i in zip(FIGHTERS, pair):
            # Impostor check: when a fighter goes down, the referee becomes the 2nd-biggest person in
            # frame. If this detection looks nothing like the fighter AND isn't where he just was,
            # leave the slot empty rather than draw his skeleton on the ref.
            sim = similarity(feats[top[i]], self.tmpl[f])
            jump = np.linalg.norm(cents[i] - self.prev[f]) / self.w
            if self.tmpl[f] is not None and sim < self.cfg.min_sim and jump > self.cfg.impostor_jump:
                continue
            self.prev[f] = cents[i]
            out[f] = int(top[i])
            if separated and margin > self.cfg.swap_margin:
                self._update_tmpl(f, feats[top[i]], self.cfg.tmpl_rate)
        self.last_pair = pair
        return out


# ----------------------------------------------------------- offline repair --

def fix_identity(df: pd.DataFrame, anchor_frames: int = 90, switch_penalty: float = 0.6,
                 sep_iou: float = 0.05, min_sim: float = 0.35, max_shirt: float = 0.7) -> tuple[pd.DataFrame, int]:
    """Globally consistent red/blue labelling via 2-state Viterbi over frames.

    State 0 = labels as stored, 1 = swapped. Emission = appearance agreement with templates
    built from the first `anchor_frames` well-separated frames (where red = left is trusted).
    Returns (fixed df, number of frames whose labels were swapped).
    """
    if df.empty or not all(c in df.columns for c in APP_COLS):
        return df, 0
    frames = np.sort(df["frame"].unique())
    by_frame = {f: g for f, g in df.groupby("frame")}

    def feats_of(g: pd.DataFrame) -> dict[str, np.ndarray | None]:
        out = {}
        for f in FIGHTERS:
            r = g[g["fighter"] == f]
            out[f] = r[APP_COLS].to_numpy()[0].astype(np.float32) if len(r) else None
        return out

    def boxes_of(g: pd.DataFrame) -> dict[str, np.ndarray | None]:
        out = {}
        for f in FIGHTERS:
            r = g[g["fighter"] == f]
            out[f] = r[["x1", "y1", "x2", "y2"]].to_numpy()[0] if len(r) else None
        return out

    # Templates from the anchor window (both visible, separated)
    tm = {f: [] for f in FIGHTERS}
    for fr in frames[:anchor_frames]:
        g = by_frame[fr]
        ft, bx = feats_of(g), boxes_of(g)
        if all(ft[f] is not None for f in FIGHTERS) and _iou(bx["red"], bx["blue"]) < sep_iou:
            for f in FIGHTERS:
                tm[f].append(ft[f])
    if min(len(v) for v in tm.values()) < 5:
        log.warning("identity: not enough separated anchor frames to build templates - skipping repair")
        return df, 0
    T = {f: np.mean(v, axis=0) for f, v in tm.items()}
    if similarity(T["red"], T["blue"]) > 0.9:
        log.warning("identity: fighters look alike (sim %.2f) - appearance repair may be unreliable",
                    similarity(T["red"], T["blue"]))

    # Emissions: score for keep vs swap
    n = len(frames)
    e = np.zeros((n, 2))
    for i, fr in enumerate(frames):
        ft = feats_of(by_frame[fr])
        keep = swap = 0.0
        for f in FIGHTERS:
            if ft[f] is None:
                continue
            other = "blue" if f == "red" else "red"
            keep += similarity(ft[f], T[f])
            swap += similarity(ft[f], T[other])
        e[i] = (keep, swap)

    # Viterbi (maximise score)
    score = np.zeros((n, 2)); back = np.zeros((n, 2), dtype=int)
    score[0] = e[0] + np.array([0.0, -switch_penalty])  # start in "keep"
    for i in range(1, n):
        for s in (0, 1):
            stay = score[i - 1, s]
            move = score[i - 1, 1 - s] - switch_penalty
            back[i, s] = s if stay >= move else 1 - s
            score[i, s] = max(stay, move) + e[i, s]
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(score[-1]))
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]

    swapped = frames[states == 1]
    df = df.copy()
    if len(swapped):
        m = df["frame"].isin(swapped)
        df.loc[m, "fighter"] = df.loc[m, "fighter"].map({"red": "blue", "blue": "red"})

    if "shirt" in df.columns:
        shirted = df["shirt"] > max_shirt
        if shirted.any():
            log.info("identity: dropped %d detections wearing a shirt (referee/corner)", int(shirted.sum()))
            df = df[~shirted]
            if df.empty:
                return df, 0
    # Impostor rows: a detection that resembles NEITHER fighter (the referee, a cornerman, a cameraman)
    # got a fighter's label. Drop it - an empty slot beats a skeleton on the ref.
    feats = df[APP_COLS].to_numpy().astype(np.float32)
    sim_own = np.array([similarity(ft, T[f]) for ft, f in zip(feats, df["fighter"])])
    sim_oth = np.array([similarity(ft, T["blue" if f == "red" else "red"]) for ft, f in zip(feats, df["fighter"])])
    df["_imp"] = ((sim_own < min_sim) & (sim_oth < min_sim)).astype(int)
    # ignore isolated single frames (one bad crop): need 2 of 3 consecutive frames, per fighter
    df = df.sort_values(["fighter", "frame"])
    run = df.groupby("fighter")["_imp"].transform(lambda x: x.rolling(3, center=True, min_periods=1).sum())
    imp_rows = (run >= 2) & (df["_imp"] == 1)
    n_imp = int(imp_rows.sum())
    if n_imp:
        log.info("identity: dropped %d detections that resemble neither fighter (referee/corner)", n_imp)
        df = df[~imp_rows]
    df = df.drop(columns="_imp").sort_values(["frame", "fighter"]).reset_index(drop=True)

    if len(swapped):
        # report the swap segments
        segs, start, prev = [], swapped[0], swapped[0]
        for fr in swapped[1:]:
            if fr != prev + 1:
                segs.append((start, prev)); start = fr
            prev = fr
        segs.append((start, prev))
        log.info("identity: repaired %d swapped frames in %d segment(s): %s", len(swapped), len(segs),
                 ", ".join(f"{a}-{b}" for a, b in segs[:8]) + (" ..." if len(segs) > 8 else ""))
    return df, int(len(swapped))
