#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moodboard Layout (No-CLI, CLIP-enabled) + Title/Palette Strips
--------------------------------------------------------------
- Adds readable panels for `title.png` and `palette.png` (if present)
- Lays out all other images within the remaining "grid area"
- Preserves your original center-weighted force layout aesthetics
- Keeps strict bounds (no overlap with strips), zoom-to-fit within grid area

Run:
    python moodboard_layout_nocli_title_palette.py
"""

import os
import io
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# -----------------------------
# User Configuration
# -----------------------------

class Config:
    IMAGES_DIR = "./images"
    OUT_DIR = "./outputs"
    CANVAS_WIDTH = 1600
    CANVAS_HEIGHT = 1000
    MAX_THUMB = 320               # max long side per thumbnail (prevents huge tiles)
    MARGIN = 12                   # outer canvas margin in px
    FILL_RATIO = 0.92             # target area coverage (0-1); raise to fill more, but risk overlap
    COMPACT_STEPS = 10            # left/up compaction sweeps
    STEPS = 2200                  # force/relax iterations (overlap resolver)
    SEED = 42
    EXCLUDE = []  # we will auto-exclude title/palette if present

    # Specials (title/palette)
    TITLE_NAME = "title.png"
    PALETTE_NAME = "palette.png"
    PLACE_TITLE = True
    PLACE_PALETTE = True
    TITLE_MAX_FRACTION = 0.22     # fraction of canvas height (top/bottom) or width (left/right)
    PALETTE_MAX_FRACTION = 0.18
    STRIP_PADDING = 10            # inner padding for title/palette inside their strips

    # Embedding mix
    USE_CLIP = True               # set False to disable CLIP
    CLIP_MODEL = "ViT-B/32"
    CLIP_DEVICE = "auto"          # "cuda", "cpu", or "auto"
    CLIP_WEIGHT = 1.0
    COLOR_WEIGHT = 0.6

    # Center-weighted aesthetic knobs
    CENTER_BIAS_STRENGTH = 0.35  # 0..0.6 Pull PCA targets toward center
    CENTER_BIAS_GAMMA = 1.2      # 1..1.6 nonlinearity for the bias
    CENTER_FORCE_K = 0.02        # tiny pull during layout (0.01..0.04)
    RADIAL_SCALE_MAX = 1.18      # size at center (1.10..1.25)
    RADIAL_SCALE_MIN = 0.88      # size at rim   (0.80..0.95)
    RADIAL_SCALE_ETA = 1.2       # falloff exponent (1..1.6)

    # System stability knobs
    ALLOW_DUPLICATE_OMP = False   # True -> sets KMP_DUPLICATE_LIB_OK=TRUE (unsafe but unblocks some Windows setups)
    OMP_NUM_THREADS = 1           # limits OpenMP threads for stability/perf predictability


# Set env vars BEFORE any possible torch import
if Config.ALLOW_DUPLICATE_OMP:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", str(Config.OMP_NUM_THREADS))

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ImgMeta:
    filename: str
    w: int
    h: int
    scale: float
    x: float = 0.0
    y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    avg_colors: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (0,0,0),(0,0,0),(0,0,0),(0,0,0)
    )
    neighbors: Optional[List[Tuple[str, float]]] = None  # list of (filename, distance)


# -----------------------------
# Utilities
# -----------------------------

def load_and_thumb(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert('RGBA')
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        if w >= h:
            scale = max_side / w
        else:
            scale = max_side / h
        img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)
    return img


def corner_means(img: Image.Image, patch_frac: float = 0.18) -> Tuple[Tuple[float, float, float], ...]:
    """Return mean RGB for four corners: tl, tr, bl, br."""
    w, h = img.size
    pw, ph = max(1, int(w*patch_frac)), max(1, int(h*patch_frac))
    rgb = img.convert('RGB')
    tl = rgb.crop((0, 0, pw, ph))
    tr = rgb.crop((w - pw, 0, w, ph))
    bl = rgb.crop((0, h - ph, pw, h))
    br = rgb.crop((w - pw, h - ph, w, h))
    def mean_rgb(im: Image.Image):
        arr = np.asarray(im).reshape(-1, 3).astype(np.float32) / 255.0
        return tuple(arr.mean(0).tolist())
    return (mean_rgb(tl), mean_rgb(tr), mean_rgb(bl), mean_rgb(br))


def standardize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def pca_project(X: np.ndarray, out_dim: int = 2) -> np.ndarray:
    """Simple PCA via SVD (on centered data)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return np.dot(Xc, Vt[:out_dim].T)


def rects_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def overlap_amount(ax, ay, aw, ah, bx, by, bw, bh) -> Tuple[float, float]:
    """Return minimal push vector (dx, dy) to separate A and B if overlapping; else (0,0)."""
    if not rects_overlap(ax, ay, aw, ah, bx, by, bw, bh):
        return 0.0, 0.0
    dx1 = (ax + aw) - bx
    dx2 = (bx + bw) - ax
    dy1 = (ay + ah) - by
    dy2 = (by + bh) - ay
    dx = dx1 if abs(dx1) < abs(dx2) else -dx2
    dy = dy1 if abs(dy1) < abs(dy2) else -dy2
    if abs(dx) < abs(dy):
        return dx, 0.0
    else:
        return 0.0, dy


def total_area(metas):
    return sum(int(m.w*m.scale)*int(m.h*m.scale) for m in metas)


# ---- Area-aware helpers (respect grid area bounds) ----

def can_place(m, x, y, metas, bounds, margin):
    x0,y0,x1,y1 = bounds
    W = x1-x0; H=y1-y0
    mw, mh = int(m.w*m.scale), int(m.h*m.scale)
    if x < x0+margin or y < y0+margin or x+mw > x1-margin or y+mh > y1-margin:
        return False
    for o in metas:
        if o is m: 
            continue
        ow, oh = int(o.w*o.scale), int(o.h*o.scale)
        if rects_overlap(x, y, mw, mh, o.x, o.y, ow, oh):
            return False
    return True


def compact_left_up(metas, bounds, margin=8, sweeps=4):
    """Repeatedly try to slide each rect left, then up, to reduce whitespace inside bounds."""
    x0,y0,x1,y1 = bounds
    metas_sorted = metas[:]
    for _ in range(sweeps):
        metas_sorted.sort(key=lambda m: (m.y, m.x))
        for m in metas_sorted:
            # left
            step = 6
            moved = True
            while moved:
                moved = False
                nx = max(x0+margin, m.x - step)
                if can_place(m, nx, m.y, metas_sorted, bounds, margin):
                    m.x = nx
                    moved = True
                else:
                    if step > 1:
                        step = max(1, step//2)
                    else:
                        break
            # up
            step = 6
            moved = True
            while moved:
                moved = False
                ny = max(y0+margin, m.y - step)
                if can_place(m, m.x, ny, metas_sorted, bounds, margin):
                    m.y = ny
                    moved = True
                else:
                    if step > 1:
                        step = max(1, step//2)
                    else:
                        break


def zoom_to_fit(metas, bounds, margin=8):
    """Scale and translate the union bbox to fill the bounds with margin."""
    x0,y0,x1,y1 = bounds
    minx = min(m.x for m in metas); miny = min(m.y for m in metas)
    maxx = max(m.x + int(m.w*m.scale) for m in metas)
    maxy = max(m.y + int(m.h*m.scale) for m in metas)
    bw = maxx - minx; bh = maxy - miny
    if bw <= 0 or bh <= 0:
        return
    W = x1-x0; H=y1-y0
    sx = (W - 2*margin) / max(1.0, bw)
    sy = (H - 2*margin) / max(1.0, bh)
    s = max(0.1, min(sx, sy)) * 0.985  # safety
    for m in metas:
        m.x = x0 + (m.x - minx) * s + margin
        m.y = y0 + (m.y - miny) * s + margin
        m.scale *= s

def bias_toward_center(x, y, bounds, strength=0.35, gamma=1.2):
    """Pull (x,y) toward *grid area* center smoothly."""
    x0,y0,x1,y1 = bounds
    W=x1-x0; H=y1-y0
    cx, cy = x0 + W * 0.5, y0 + H * 0.5
    dx, dy = x - cx, y - cy
    r = math.hypot(dx, dy) / (0.5 * math.hypot(W, H) + 1e-6)
    r = max(0.0, min(1.0, r))
    s = (1.0 - strength) + strength * (r ** gamma)
    return cx + dx * s, cy + dy * s


def scale_by_radius(metas, bounds, max_scale=1.18, min_scale=0.88, eta=1.2):
    """Scale each tile by distance to grid center: bigger near center, smaller near edge."""
    x0,y0,x1,y1 = bounds
    W=x1-x0; H=y1-y0
    cx, cy = x0 + W * 0.5, y0 + H * 0.5
    R = 0.5 * math.hypot(W, H)
    for m in metas:
        mw, mh = int(m.w * m.scale), int(m.h * m.scale)
        mx, my = m.x + mw * 0.5, m.y + mh * 0.5
        r = min(1.0, math.hypot(mx - cx, my - cy) / (R + 1e-6))
        s = max_scale - (max_scale - min_scale) * (r ** eta)
        m.scale *= float(s)


# -----------------------------
# Layout algorithm (force/anneal), area-aware
# -----------------------------

def layout_images(metas: List[ImgMeta], bounds: Tuple[int,int,int,int], steps: int = 1500, 
                  attraction: float = 0.16, repel: float = 0.85,
                  boundary_push: float = 0.6, cooling: float = 0.995,
                  jitter: float = 0.8, seed: int = 42):
    x0,y0,x1,y1 = bounds
    W=x1-x0; H=y1-y0
    rng = random.Random(seed)
    temp = 1.0
    for t in range(steps):
        # jitter
        for m in metas:
            m.x += (rng.random() - 0.5) * jitter * temp
            m.y += (rng.random() - 0.5) * jitter * temp

        # attraction to target
        for m in metas:
            m.x += (m.target_x - m.x) * attraction
            m.y += (m.target_y - m.y) * attraction

        # gentle center pull for cohesion
        center_k = getattr(Config, 'CENTER_FORCE_K', 0.02)
        cx, cy = x0 + W * 0.5, y0 + H * 0.5
        for m in metas:
            mw, mh = int(m.w * m.scale), int(m.h * m.scale)
            mx, my = m.x + mw * 0.5, m.y + mh * 0.5
            m.x += (cx - mx) * center_k
            m.y += (cy - my) * center_k

        # resolve overlaps
        for i in range(len(metas)):
            a = metas[i]
            aw, ah = int(a.w * a.scale), int(a.h * a.scale)
            for j in range(i+1, len(metas)):
                b = metas[j]
                bw, bh = int(b.w * b.scale), int(b.h * b.scale)
                dx, dy = overlap_amount(a.x, a.y, aw, ah, b.x, b.y, bw, bh)
                if dx != 0.0 or dy != 0.0:
                    a.x -= dx * 0.5 * repel
                    a.y -= dy * 0.5 * repel
                    b.x += dx * 0.5 * repel
                    b.y += dy * 0.5 * repel

        # boundaries (to grid area)
        for m in metas:
            mw, mh = int(m.w * m.scale), int(m.h * m.scale)
            if m.x < x0: m.x += (x0 - m.x) * boundary_push
            if m.y < y0: m.y += (y0 - m.y) * boundary_push
            if m.x + mw > x1: m.x -= (m.x + mw - x1) * boundary_push
            if m.y + mh > y1: m.y -= (m.y + mh - y1) * boundary_push

        temp *= cooling


# -----------------------------
# CLIP + color features
# -----------------------------

def color_feature_vector(corners: List[Tuple[Tuple[float, float, float], ...]]) -> np.ndarray:
    # 4 corners * 3 channels = 12 dims
    arr = np.array([[c for corner in cs for c in corner] for cs in corners], dtype=np.float32)
    return arr


def embed_with_clip(imgs: List[Image.Image]) -> np.ndarray:
    """Return L2-normalized CLIP image embeddings."""
    import torch  # imported here so env vars above take effect
    import clip
    device = "cuda" if (Config.CLIP_DEVICE == "auto" and torch.cuda.is_available()) else (
             "cpu" if Config.CLIP_DEVICE in ["auto", "cpu"] else "cuda")
    model, preprocess = clip.load(Config.CLIP_MODEL, device=device, jit=False)
    model.eval()
    tensors = []
    for im in imgs:
        tensors.append(preprocess(im.convert('RGB')).unsqueeze(0))
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.detach().cpu().numpy().astype(np.float32)
    return feats_np


# -----------------------------
# Reward (optional anneal keep/revert)
# -----------------------------

def compute_reward(metas: List[ImgMeta], bounds: Tuple[int,int,int,int]) -> float:
    """Higher is better: in-bounds + neighbor cohesion - overlap penalty (inside grid area)."""
    x0,y0,x1,y1 = bounds
    W=x1-x0; H=y1-y0
    in_bounds = 0.0
    for m in metas:
        mw, mh = int(m.w * m.scale), int(m.h * m.scale)
        if x0 <= m.x and y0 <= m.y and m.x + mw <= x1 and m.y + mh <= y1:
            in_bounds += 1.0
    in_bounds /= max(1, len(metas))

    overlap_pen = 0.0
    for i in range(len(metas)):
        a = metas[i]
        aw, ah = int(a.w * a.scale), int(a.h * a.scale)
        ra = (a.x, a.y, aw, ah)
        for j in range(i+1, len(metas)):
            b = metas[j]
            bw, bh = int(b.w * b.scale), int(b.h * b.scale)
            rb = (b.x, b.y, bw, bh)
            if rects_overlap(*ra, *rb):
                ix = max(0, min(a.x+aw, b.x+bw) - max(a.x, b.x))
                iy = max(0, min(a.y+ah, b.y+bh) - max(a.y, b.y))
                overlap_pen += (ix * iy)
    overlap_pen = overlap_pen / max(1.0, float(W*H))

    # neighbor cohesion in target vs final
    k = 4
    tgt = np.array([[m.target_x, m.target_y] for m in metas], dtype=np.float32)
    fin = np.array([[m.x, m.y] for m in metas], dtype=np.float32)
    dt = np.sqrt(((tgt[:,None,:]-tgt[None,:,:])**2).sum(-1))
    df = np.sqrt(((fin[:,None,:]-fin[None,:,:])**2).sum(-1))
    np.fill_diagonal(dt, np.inf)
    np.fill_diagonal(df, 0.0)
    neighbor_score = 0.0
    for i in range(len(metas)):
        nn = np.argsort(dt[i])[:k]
        neighbor_score += float(np.exp(-0.002 * df[i, nn].mean()))
    neighbor_score /= max(1, len(metas))

    return 0.7 * in_bounds + 0.3 * neighbor_score - 0.8 * overlap_pen


# -----------------------------
# Title/Palette helpers
# -----------------------------

def choose_title_side(w,h,CW,CH):
    ar = w / max(1,h)
    return "top" if ar>=1.25 else "left"

def choose_palette_side(w,h,CW,CH,avoid):
    ar = w / max(1,h)
    candidates = ["bottom","right"] if avoid in ["top","left"] else ["top","left"]
    if ar>=1.25:
        for s in ["bottom","top"]:
            if s in candidates: return s
    else:
        for s in ["right","left"]:
            if s in candidates: return s
    return candidates[0]

def reserve_strip(img: Image.Image, side: str, max_fraction: float, CW: int, CH: int, gap: int):
    w,h = img.size
    if side in ("top","bottom"):
        max_h = int(CH*max_fraction)
        sc = min(1.0, max_h/h if h>0 else 1.0)
        scaled_h = int(h*sc)
        strip_h = min(max_h, scaled_h + gap*2)
        strip = (0, 0, CW, strip_h) if side=="top" else (0, CH-strip_h, CW, CH)
        return strip, sc
    else:
        max_w = int(CW*max_fraction)
        sc = min(1.0, max_w/w if w>0 else 1.0)
        scaled_w = int(w*sc)
        strip_w = min(max_w, scaled_w + gap*2)
        strip = (0, 0, strip_w, CH) if side=="left" else (CW-strip_w, 0, CW, CH)
        return strip, sc


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    rng = random.Random(Config.SEED)
    os.makedirs(Config.OUT_DIR, exist_ok=True)

    names_all = [n for n in os.listdir(Config.IMAGES_DIR) if not n.startswith('.')]
    valid_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    files = [(n, os.path.join(Config.IMAGES_DIR, n)) for n in names_all
             if os.path.splitext(n)[1].lower() in valid_ext]
    if not files:
        raise SystemExit(f"No images found in {Config.IMAGES_DIR}")

    # Separate title/palette
    title_pair = next(((n,p) for (n,p) in files if n==Config.TITLE_NAME), None) if Config.PLACE_TITLE else None
    palette_pair = next(((n,p) for (n,p) in files if n==Config.PALETTE_NAME), None) if Config.PLACE_PALETTE else None
    main_pairs = [(n,p) for (n,p) in files if n not in {Config.TITLE_NAME, Config.PALETTE_NAME}]

    thumbs: List[Image.Image] = []
    metas: List[ImgMeta] = []
    for (n, p) in main_pairs:
        im = load_and_thumb(p, Config.MAX_THUMB)
        w, h = im.size
        metas.append(ImgMeta(filename=n, w=w, h=h, scale=1.0))
        thumbs.append(im)

    # Load specials
    CW, CH = Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT
    grid_x0 = Config.MARGIN; grid_y0 = Config.MARGIN
    grid_x1 = CW - Config.MARGIN; grid_y1 = CH - Config.MARGIN
    specials_meta = []
    t_im = p_im = None
    t_strip = p_strip = None

    if title_pair:
        tn,tp = title_pair
        t_im = load_and_thumb(tp, max(CW,CH))
        side = choose_title_side(*t_im.size, CW, CH)
        strip, t_scale = reserve_strip(t_im, side, Config.TITLE_MAX_FRACTION, CW, CH, Config.STRIP_PADDING)
        x0,y0,x1,y1 = strip
        if side=="top": grid_y0 = max(grid_y0, y1)
        elif side=="bottom": grid_y1 = min(grid_y1, y0)
        elif side=="left": grid_x0 = max(grid_x0, x1)
        else: grid_x1 = min(grid_x1, x0)
        t_strip = strip
        specials_meta.append(("title", side, strip, t_im, t_scale))

    if palette_pair:
        pn,pp = palette_pair
        p_im = load_and_thumb(pp, max(CW,CH))
        avoid = specials_meta[0][1] if specials_meta else None
        side = choose_palette_side(*p_im.size, CW, CH, avoid)
        strip, p_scale = reserve_strip(p_im, side, Config.PALETTE_MAX_FRACTION, CW, CH, Config.STRIP_PADDING)
        # avoid overlap w/ title strip
        if t_strip is not None:
            tx0,ty0,tx1,ty1 = t_strip; x0,y0,x1,y1 = strip
            if not (x1<=tx0 or tx1<=x0 or y1<=ty0 or ty1<=y0):
                # shrink palette strip slightly to avoid collide
                if side in ("top","bottom"):
                    h = y1-y0; y1 = y1 - int(h*0.12) if side=="top" else y0 + int(h*0.88)
                else:
                    w = x1-x0; x1 = x1 - int(w*0.12) if side=="left" else x0 + int(w*0.88)
                strip = (x0,y0,x1,y1)
        x0,y0,x1,y1 = strip
        if side=="top": grid_y0 = max(grid_y0, y1)
        elif side=="bottom": grid_y1 = min(grid_y1, y0)
        elif side=="left": grid_x0 = max(grid_x0, x1)
        else: grid_x1 = min(grid_x1, x0)
        p_strip = strip
        specials_meta.append(("palette", side, strip, p_im, p_scale))

    grid_bounds = (grid_x0, grid_y0, grid_x1, grid_y1)

    # Features
    clip_feats = None
    if Config.USE_CLIP and len(thumbs)>0:
        try:
            clip_feats = embed_with_clip(thumbs)
        except Exception as e:
            raise RuntimeError("CLIP embedding failed. Ensure 'clip' and 'torch' are installed and working.\n"
                               f"Original error: {e}")

    corner_cols = [corner_means(im) for im in thumbs]
    color_feats = np.array([[c for corner in cs for c in corner] for cs in corner_cols], dtype=np.float32)

    blocks = []
    if clip_feats is not None:
        blocks.append(standardize(clip_feats) * Config.CLIP_WEIGHT)
    else:
        blocks.append(np.zeros((len(thumbs), 1), dtype=np.float32))

    blocks.append(standardize(color_feats) * Config.COLOR_WEIGHT)
    X = np.concatenate(blocks, axis=1).astype(np.float32)

    # Project to 2D and map to GRID AREA (not full canvas)
    if len(thumbs)>0:
        Z = pca_project(X, out_dim=2)
        Zmin = Z.min(axis=0, keepdims=True); Zmax = Z.max(axis=0, keepdims=True)
        Z = (Z - Zmin) / (Zmax - Zmin + 1e-6)
        for i, m in enumerate(metas):
            tx = grid_bounds[0] + Config.MARGIN + Z[i, 0] * ((grid_bounds[2]-grid_bounds[0]) - 2*Config.MARGIN - m.w * m.scale)
            ty = grid_bounds[1] + Config.MARGIN + Z[i, 1] * ((grid_bounds[3]-grid_bounds[1]) - 2*Config.MARGIN - m.h * m.scale)
            tx, ty = bias_toward_center(
                tx, ty,
                grid_bounds,
                strength=getattr(Config, 'CENTER_BIAS_STRENGTH', 0.35),
                gamma=getattr(Config, 'CENTER_BIAS_GAMMA', 1.2)
            )
            m.x = m.target_x = float(tx)
            m.y = m.target_y = float(ty)
            m.avg_colors = corner_cols[i]

    # Initial layout inside grid
    layout_images(metas, grid_bounds, steps=Config.STEPS, seed=Config.SEED)

    # Center-weighted radial scaling
    scale_by_radius(
        metas,
        grid_bounds,
        max_scale=getattr(Config, 'RADIAL_SCALE_MAX', 1.18),
        min_scale=getattr(Config, 'RADIAL_SCALE_MIN', 0.88),
        eta=getattr(Config, 'RADIAL_SCALE_ETA', 1.2)
    )
    layout_images(metas, grid_bounds, steps=max(600, Config.STEPS//3), seed=Config.SEED)

    # Scale uniformly to reach target fill ratio (relative to grid area)
    area_w = grid_bounds[2]-grid_bounds[0]
    area_h = grid_bounds[3]-grid_bounds[1]
    canvas_area = area_w * area_h
    current_area = total_area(metas)
    target_area = max(1, int(Config.FILL_RATIO * canvas_area))
    if current_area > 0:
        f = (target_area / current_area) ** 0.5
        for m in metas:
            m.scale *= f
        layout_images(metas, grid_bounds, steps=max(600, Config.STEPS//3), seed=Config.SEED)

    # Compaction and zoom-to-fit within grid
    compact_left_up(metas, grid_bounds, margin=Config.MARGIN, sweeps=Config.COMPACT_STEPS)
    zoom_to_fit(metas, grid_bounds, margin=Config.MARGIN)
    layout_images(metas, grid_bounds, steps=400, seed=Config.SEED)

    # Optional light anneal keep/revert
    base_reward = compute_reward(metas, grid_bounds)
    for _ in range(120):
        if not metas: break
        i = rng.randrange(len(metas))
        m = metas[i]
        old = (m.x, m.y)
        m.x += (rng.random() - 0.5) * 20
        m.y += (rng.random() - 0.5) * 20
        new_reward = compute_reward(metas, grid_bounds)
        if new_reward < base_reward:
            m.x, m.y = old
        else:
            base_reward = new_reward

    # Neighbors (cosine distance in feature space)
    if len(thumbs)>0:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
        D = 1.0 - (Xn @ Xn.T)  # cosine distance
        for i, m in enumerate(metas):
            order = np.argsort(D[i])
            m.neighbors = [(metas[j].filename, float(D[i, j])) for j in order[1:6]]

    # Composite and save
    canvas = Image.new('RGBA', (Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT), (255, 255, 255, 255))

    # Paste main images
    for m, im in zip(metas, thumbs):
        w, h = int(m.w * m.scale), int(m.h * m.scale)
        to_paste = im.resize((w, h), Image.LANCZOS) if (w, h) != im.size else im
        canvas.alpha_composite(to_paste, (int(m.x), int(m.y)))

    # Paste title/palette centered within their strips (readable, no relevance scaling)
    specials_serial = []
    for kind, side, strip, im, sc in specials_meta:
        x0,y0,x1,y1 = strip; iw,ih = im.size
        if side in ("top","bottom"):
            max_h = (y1-y0) - Config.STRIP_PADDING*2
            s = min(sc, max_h/ih if ih>0 else 1.0)
        else:
            max_w = (x1-x0) - Config.STRIP_PADDING*2
            s = min(sc, max_w/iw if iw>0 else 1.0)
        s = max(s, 0.01)
        ww, hh = int(iw*s), int(ih*s)
        px = int(x0 + (x1-x0-ww)*0.5); py = int(y0 + (y1-y0-hh)*0.5)
        canvas.alpha_composite(im.resize((ww,hh), Image.LANCZOS), (px,py))
        specials_serial.append({
            'filename': Config.TITLE_NAME if kind=="title" else Config.PALETTE_NAME,
            'region': kind, 'x': px, 'y': py, 'scale': s, 'w': iw, 'h': ih, 'strip': {'x0':x0,'y0':y0,'x1':x1,'y1':y1}
        })

    os.makedirs(Config.OUT_DIR, exist_ok=True)
    out_png = os.path.join(Config.OUT_DIR, 'moodboard.png')
    out_json = os.path.join(Config.OUT_DIR, 'placements.json')
    canvas.save(out_png)

    serializable = [asdict(m) for m in metas]
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'canvas': {'width': Config.CANVAS_WIDTH, 'height': Config.CANVAS_HEIGHT},
            'grid_bounds': {'x0': grid_bounds[0], 'y0': grid_bounds[1], 'x1': grid_bounds[2], 'y1': grid_bounds[3]},
            'used_clip': bool(Config.USE_CLIP),
            'title_palette': specials_serial,
            'images': serializable
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_png} and {out_json}")

if __name__ == '__main__':
    main()
