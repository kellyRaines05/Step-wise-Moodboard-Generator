#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moodboard (No-CLI) — Auto Title & Auto Palette (fixed strips: no overlap)
"""

import os, io, json, math, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Config:
    IMAGES_DIR = "./images"
    OUT_DIR = "./outputs"

    CANVAS_WIDTH = 1600
    CANVAS_HEIGHT = 1000
    MARGIN = 12

    MAX_THUMB = 360
    FILL_RATIO = 0.92
    STEPS = 2500
    COMPACT_STEPS = 10
    SEED = 42

    TITLE_TEXT = "Minimal Interior"
    TITLE_MAX_FRACTION = 0.18
    PALETTE_MAX_FRACTION = 0.16
    STRIP_PADDING = 6
    PALETTE_SWATCHES = 5
    PALETTE_METHOD = "kmeans"

    USE_CLIP = True
    CLIP_MODEL = "ViT-B/32"
    CLIP_DEVICE = "auto"
    CLIP_WEIGHT = 1.0
    COLOR_WEIGHT = 0.6

    CENTER_BIAS_STRENGTH = 0.35
    CENTER_BIAS_GAMMA = 1.2
    CENTER_FORCE_K = 0.02
    RADIAL_SCALE_MAX = 1.18
    RADIAL_SCALE_MIN = 0.88
    RADIAL_SCALE_ETA = 1.2

    # Title sizing
    TITLE_FONT_MAX = 150          # cap the auto-fit
    TITLE_TEXT_SHRINK = 50        # shrink ONLY the text by this many pts (badge stays same)

    ALLOW_DUPLICATE_OMP = False
    OMP_NUM_THREADS = 1

if Config.ALLOW_DUPLICATE_OMP:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", str(Config.OMP_NUM_THREADS))

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
    neighbors: Optional[List[Tuple[str, float]]] = None

def load_and_thumb(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    w,h = im.size
    if max(w,h) > max_side:
        sc = max_side / max(w,h)
        im = im.resize((max(1,int(w*sc)), max(1,int(h*sc))), Image.LANCZOS)
    return im

def corner_means(img: Image.Image, patch_frac: float = 0.18):
    w,h = img.size
    pw, ph = max(1,int(w*patch_frac)), max(1,int(h*patch_frac))
    rgb = img.convert('RGB')
    def mean_rgb(bb):
        arr = np.asarray(bb).reshape(-1,3).astype(np.float32)/255.0
        return tuple(arr.mean(0).tolist())
    tl = rgb.crop((0,0,pw,ph)); tr = rgb.crop((w-pw,0,w,ph))
    bl = rgb.crop((0,h-ph,pw,h)); br = rgb.crop((w-pw,h-ph,w,h))
    return (mean_rgb(tl), mean_rgb(tr), mean_rgb(bl), mean_rgb(br))

def standardize(X: np.ndarray, eps=1e-6):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True)
    return (X-mu)/(sd+eps)

def pca_project(X: np.ndarray, out_dim=2):
    Xc = X - X.mean(0, keepdims=True)
    _,_,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:out_dim].T

def rects_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

def overlap_amount(ax, ay, aw, ah, bx, by, bw, bh):
    if not rects_overlap(ax, ay, aw, ah, bx, by, bw, bh):
        return 0.0, 0.0
    dx1 = (ax + aw) - bx; dx2 = (bx + bw) - ax
    dy1 = (ay + ah) - by; dy2 = (by + bh) - ay
    dx = dx1 if abs(dx1) < abs(dx2) else -dx2
    dy = dy1 if abs(dy1) < abs(dy2) else -dy2
    return (dx, 0.0) if abs(dx) < abs(dy) else (0.0, dy)

def total_area(metas): 
    return sum(int(m.w*m.scale)*int(m.h*m.scale) for m in metas)

def can_place(m, x, y, metas, bounds, margin):
    x0,y0,x1,y1=bounds
    mw, mh = int(m.w*m.scale), int(m.h*m.scale)
    if x < x0+margin or y < y0+margin or x+mw > x1-margin or y+mh > y1-margin: return False
    for o in metas:
        if o is m: continue
        ow,oh=int(o.w*o.scale),int(o.h*o.scale)
        if rects_overlap(x,y,mw,mh,o.x,o.y,ow,oh): return False
    return True

def compact_left_up(metas, bounds, margin=8, sweeps=4):
    x0,y0,x1,y1=bounds
    L=metas[:]
    for _ in range(sweeps):
        L.sort(key=lambda m:(m.y,m.x))
        for m in L:
            step=6; moved=True
            while moved:
                moved=False; nx=max(x0+margin, m.x-step)
                if can_place(m, nx, m.y, L, bounds, margin):
                    m.x=nx; moved=True
                else:
                    if step>1: step=max(1, step//2)
                    else: break
            step=6; moved=True
            while moved:
                moved=False; ny=max(y0+margin, m.y-step)
                if can_place(m, m.x, ny, L, bounds, margin):
                    m.y=ny; moved=True
                else:
                    if step>1: step=max(1, step//2)
                    else: break

def zoom_to_fit(metas, bounds, margin=8):
    x0,y0,x1,y1=bounds
    minx=min(m.x for m in metas); miny=min(m.y for m in metas)
    maxx=max(m.x+int(m.w*m.scale) for m in metas)
    maxy=max(m.y+int(m.h*m.scale) for m in metas)
    bw=maxx-minx; bh=maxy-miny
    if bw<=0 or bh<=0: return
    W=x1-x0; H=y1-y0
    sx=(W-2*margin)/max(1.0,bw); sy=(H-2*margin)/max(1.0,bh)
    s=max(0.1, min(sx,sy))*0.985
    for m in metas:
        m.x = x0 + (m.x - minx)*s + margin
        m.y = y0 + (m.y - miny)*s + margin
        m.scale *= s

def bias_toward_center(x,y,bounds,strength=0.35,gamma=1.2):
    x0,y0,x1,y1=bounds; W=x1-x0; H=y1-y0
    cx,cy=x0+W*0.5, y0+H*0.5
    dx,dy=x-cx, y-cy
    r = math.hypot(dx,dy)/(0.5*math.hypot(W,H)+1e-6)
    r=max(0.0,min(1.0,r))
    s=(1.0-strength)+strength*(r**gamma)
    return cx+dx*s, cy+dy*s

def scale_by_radius(metas,bounds,max_scale=1.18,min_scale=0.88,eta=1.2):
    x0,y0,x1,y1=bounds; W=x1-x0; H=y1-y0
    cx,cy=x0+W*0.5, y0+H*0.5; R=0.5*math.hypot(W,H)
    for m in metas:
        mw, mh = int(m.w*m.scale), int(m.h*m.scale)
        mx,my=m.x+mw*0.5, m.y+mh*0.5
        r=min(1.0, math.hypot(mx-cx,my-cy)/(R+1e-6))
        s=max_scale - (max_scale-min_scale)*(r**eta)
        m.scale*=float(s)

def layout_images(metas, bounds, steps=1500, attraction=0.16, repel=0.85,
                  boundary_push=0.6, cooling=0.995, jitter=0.8, seed=42):
    x0,y0,x1,y1=bounds; W=x1-x0; H=y1-y0
    rng=random.Random(seed); temp=1.0
    for _ in range(steps):
        for m in metas:
            m.x += (rng.random()-0.5)*jitter*temp
            m.y += (rng.random()-0.5)*jitter*temp
        for m in metas:
            m.x += (m.target_x - m.x)*attraction
            m.y += (m.target_y - m.y)*attraction
        center_k=Config.CENTER_FORCE_K; cx,cy=x0+W*0.5, y0+H*0.5
        for m in metas:
            mw, mh = int(m.w*m.scale), int(m.h*m.scale)
            mx,my=m.x+mw*0.5, m.y+mh*0.5
            m.x += (cx-mx)*center_k; m.y += (cy-my)*center_k
        for i in range(len(metas)):
            a=metas[i]; aw,ah=int(a.w*a.scale),int(a.h*a.scale)
            for j in range(i+1,len(metas)):
                b=metas[j]; bw,bh=int(b.w*b.scale),int(b.h*b.scale)
                dx,dy=overlap_amount(a.x,a.y,aw,ah,b.x,b.y,bw,bh)
                if dx or dy:
                    a.x -= dx*0.5*repel; a.y -= dy*0.5*repel
                    b.x += dx*0.5*repel; b.y += dy*0.5*repel
        for m in metas:
            mw, mh = int(m.w*m.scale), int(m.h*m.scale)
            if m.x < x0: m.x += (x0-m.x)*boundary_push
            if m.y < y0: m.y += (y0-m.y)*boundary_push
            if m.x+mw > x1: m.x -= (m.x+mw-x1)*boundary_push
            if m.y+mh > y1: m.y -= (m.y+mh-y1)*boundary_push
        temp*=cooling

def embed_with_clip(imgs: List[Image.Image]) -> np.ndarray:
    import torch, clip
    device = "cuda" if (Config.CLIP_DEVICE=="auto" and torch.cuda.is_available()) else ("cpu" if Config.CLIP_DEVICE in ["auto","cpu"] else "cuda")
    model, preprocess = clip.load(Config.CLIP_MODEL, device=device, jit=False)
    model.eval()
    batch = torch.cat([preprocess(im.convert('RGB')).unsqueeze(0) for im in imgs], dim=0).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch); feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)

def generate_palette_swatches(imgs: List[Image.Image], k: int) -> List[Tuple[int,int,int]]:
    samples = []
    for im in imgs:
        arr = np.asarray(im.convert('RGB'))
        h,w,_ = arr.shape
        idx = np.random.choice(h*w, size=min(2000, h*w), replace=False)
        sub = arr.reshape(-1,3)[idx]
        samples.append(sub)
    if not samples:
        return [(200,200,200)]*k
    X = np.vstack(samples).astype(np.float32)
    if Config.PALETTE_METHOD == "median":
        qs = np.linspace(0.1, 0.9, k)
        cols = [tuple(np.quantile(X, q, axis=0).astype(int).tolist()) for q in qs]
        return cols
    np.random.seed(0)
    cent = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(12):
        d = ((X[:,None,:]-cent[None,:,:])**2).sum(-1)
        a = d.argmin(1)
        for i in range(k):
            pts = X[a==i]
            if len(pts)>0: cent[i] = pts.mean(0)
    cols = [tuple(np.clip(c.astype(int),0,255).tolist()) for c in cent]
    cols.sort(key=lambda c: 0.2126*c[0]+0.7152*c[1]+0.0722*c[2], reverse=True)
    return cols

def render_palette_image(colors: List[Tuple[int,int,int]], size: Tuple[int,int]) -> Image.Image:
    W,H = size
    im = Image.new("RGBA", (W,H), (255,255,255,0))
    draw = ImageDraw.Draw(im)
    n = len(colors); gap = 8
    sw = W - 2*gap; sh = (H - (n+1)*gap) // n if n>0 else H-gap*2
    y = gap
    for c in colors:
        draw.rectangle([gap, y, gap+sw, y+sh], fill=(*c,255))
        y += sh + gap
    return im

def render_title_image(text: str, max_size: Tuple[int,int]) -> Image.Image:
    """Fit the largest font (including badge padding), then shrink *only* the text by Config.TITLE_TEXT_SHRINK."""
    W, H = max_size

    # Badge padding (must be used in fit and drawing so badge never overflows)
    BADGE_WPAD = 40  # total extra width (left+right)
    BADGE_HPAD = 24  # total extra height (top+bottom)

    # Base font (safe fallback)
    try:
        base_font_path = "C:\Windows\Fonts\BRADHITC.ttf"
        base_font = ImageFont.truetype(base_font_path, size=96)
    except Exception:
        base_font_path = None
        base_font = ImageFont.load_default()

    # Binary search the largest font size that fits WITH the badge padding
    lo, hi = 18, int(Config.TITLE_FONT_MAX)
    best_size = lo
    probe = ImageDraw.Draw(Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            f = ImageFont.truetype(base_font_path, size=mid) if base_font_path else ImageFont.load_default()
        except Exception:
            f = ImageFont.load_default()

        tw, th = probe.textbbox((0, 0), text, font=f)[2:]
        tw -= probe.textbbox((0, 0), text, font=f)[0]
        th -= probe.textbbox((0, 0), text, font=f)[1]

        if tw + BADGE_WPAD <= W and th + BADGE_HPAD <= H:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Font for badge sizing (largest that fits)
    try:
        badge_font = ImageFont.truetype(base_font_path, size=best_size) if base_font_path else ImageFont.load_default()
    except Exception:
        badge_font = ImageFont.load_default()
    # Badge (background) size from the fitted font
    probe2 = ImageDraw.Draw(Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    bbox_badge = probe2.textbbox((0, 0), text, font=badge_font)
    badge_tw = bbox_badge[2] - bbox_badge[0]
    badge_th = bbox_badge[3] - bbox_badge[1]
    badge_w = badge_tw + BADGE_WPAD
    badge_h = badge_th + BADGE_HPAD

    # Now shrink ONLY the text by the configured small margin
    shrunk_size = max(1, best_size - int(Config.TITLE_TEXT_SHRINK))
    try:
        text_font = ImageFont.truetype(base_font_path, size=shrunk_size) if base_font_path else ImageFont.load_default()
    except Exception:
        text_font = ImageFont.load_default()
    bbox_text = probe2.textbbox((0, 0), text, font=text_font)
    text_tw = bbox_text[2] - bbox_text[0]
    text_th = bbox_text[3] - bbox_text[1]

    # Render centered: compute common center, then draw badge and text around it
    img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    badge_x = cx - badge_w // 2
    badge_y = cy - badge_h // 2
    text_x = cx - text_tw // 2
    text_y = cy - text_th // 2

    # Draw badge
    badge = Image.new("RGBA", (badge_w, badge_h), (230, 222, 214, 255))
    img.alpha_composite(badge, (max(0, badge_x), max(0, badge_y)))

    # Draw the (slightly smaller) title text
    draw.text((text_x, text_y), text, fill=(30, 30, 30, 255), font=text_font)
    return img

def compute_reward(metas, bounds):
    x0,y0,x1,y1=bounds; W=x1-x0; H=y1-y0
    in_bounds = sum(1.0 for m in metas if (x0<=m.x and y0<=m.y and m.x+int(m.w*m.scale)<=x1 and m.y+int(m.h*m.scale)<=y1))/max(1,len(metas))
    overlap_pen=0.0
    for i in range(len(metas)):
        a=metas[i]; aw,ah=int(a.w*a.scale),int(a.h*a.scale); ra=(a.x,a.y,aw,ah)
        for j in range(i+1,len(metas)):
            b=metas[j]; bw,bh=int(b.w*b.scale),int(b.h*b.scale); rb=(b.x,b.y,bw,bh)
            if rects_overlap(*ra,*rb):
                ix=max(0, min(a.x+aw,b.x+bw)-max(a.x,b.x))
                iy=max(0, min(a.y+ah,b.y+bh)-max(a.y,b.y))
                overlap_pen += ix*iy
    overlap_pen /= max(1.0, float(W*H))
    return 0.7*in_bounds - 0.8*overlap_pen

def main():
    rng = random.Random(Config.SEED)
    names = [n for n in os.listdir(Config.IMAGES_DIR) if not n.startswith(".")]
    valid = {".png",".jpg",".jpeg",".webp",".bmp"}
    files = [(n, str(Path(Config.IMAGES_DIR)/n)) for n in names if Path(n).suffix.lower() in valid]
    if not files:
        raise SystemExit("No images found in ./images")

    thumbs=[]; metas=[]
    for n,p in files:
        im = load_and_thumb(p, Config.MAX_THUMB)
        thumbs.append(im); metas.append(ImgMeta(filename=n, w=im.size[0], h=im.size[1], scale=1.0))

    CW,CH = Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT
    grid_x0=Config.MARGIN; grid_y0=Config.MARGIN
    grid_x1=CW-Config.MARGIN; grid_y1=CH-Config.MARGIN

    # Title strip
    title_side = "top" if len(thumbs)==0 or sum(im.size[0] for im in thumbs)/max(1,len(thumbs)) >= sum(im.size[1] for im in thumbs)/max(1,len(thumbs)) else "left"
    if Config.TITLE_TEXT:
        if title_side in ("top","bottom"):
            strip_h = int(CH * Config.TITLE_MAX_FRACTION)
            title_strip = (0, 0, CW, strip_h)
            grid_y0 = max(grid_y0, strip_h)
        else:
            strip_w = int(CW * Config.TITLE_MAX_FRACTION)
            title_strip = (0, 0, strip_w, CH)
            grid_x0 = max(grid_x0, strip_w)
        title_img = render_title_image(
            Config.TITLE_TEXT,
            (title_strip[2]-title_strip[0]-Config.STRIP_PADDING*2,
             title_strip[3]-title_strip[1]-Config.STRIP_PADDING*2)
        )
    else:
        title_img=None; title_strip=None

    # Palette strip — trimmed to avoid the title strip
    palette_colors = generate_palette_swatches(thumbs, Config.PALETTE_SWATCHES)
    if title_side in ("top","bottom"):
        palette_side = "right"
        strip_w = int(CW * Config.PALETTE_MAX_FRACTION)
        palette_strip = (CW - strip_w, grid_y0, CW, grid_y1)
        grid_x1 = min(grid_x1, CW - strip_w)
    else:
        palette_side = "bottom"
        strip_h = int(CH * Config.PALETTE_MAX_FRACTION)
        palette_strip = (grid_x0, CH - strip_h, grid_x1, CH)
        grid_y1 = min(grid_y1, CH - strip_h)

    palette_img = render_palette_image(
        palette_colors,
        (max(1, palette_strip[2]-palette_strip[0]-Config.STRIP_PADDING*2),
         max(1, palette_strip[3]-palette_strip[1]-Config.STRIP_PADDING*2))
    )

    grid_bounds = (grid_x0, grid_y0, grid_x1, grid_y1)

    # Features
    clip_feats=None
    if Config.USE_CLIP:
        try:
            clip_feats = embed_with_clip(thumbs)
        except Exception as e:
            raise RuntimeError(f"CLIP embedding failed: {e}. Ensure torch+clip are installed.")

    color_feats = np.array([[c for corner in corner_means(im) for c in corner] for im in thumbs], np.float32)
    blocks=[]
    if clip_feats is not None:
        blocks.append(standardize(clip_feats)*Config.CLIP_WEIGHT)
    else:
        blocks.append(np.zeros((len(thumbs),1),np.float32))
    blocks.append(standardize(color_feats)*Config.COLOR_WEIGHT)
    X = np.concatenate(blocks,1).astype(np.float32)

    Z = pca_project(X, 2)
    Zmin=Z.min(0, keepdims=True); Zmax=Z.max(0, keepdims=True)
    Z = (Z - Zmin) / (Zmax - Zmin + 1e-6)
    for i,m in enumerate(metas):
        tx = grid_bounds[0] + Config.MARGIN + Z[i,0]*((grid_bounds[2]-grid_bounds[0])-2*Config.MARGIN - m.w*m.scale)
        ty = grid_bounds[1] + Config.MARGIN + Z[i,1]*((grid_bounds[3]-grid_bounds[1])-2*Config.MARGIN - m.h*m.scale)
        tx,ty = bias_toward_center(tx,ty,grid_bounds,Config.CENTER_BIAS_STRENGTH,Config.CENTER_BIAS_GAMMA)
        m.x = m.target_x = float(tx)
        m.y = m.target_y = float(ty)

    layout_images(metas, grid_bounds, steps=Config.STEPS, seed=Config.SEED)
    scale_by_radius(metas, grid_bounds, Config.RADIAL_SCALE_MAX, Config.RADIAL_SCALE_MIN, Config.RADIAL_SCALE_ETA)
    layout_images(metas, grid_bounds, steps=max(600,Config.STEPS//3), seed=Config.SEED)

    area_w = grid_bounds[2]-grid_bounds[0]; area_h = grid_bounds[3]-grid_bounds[1]
    target_area = int(max(1, Config.FULL_RATIO if hasattr(Config,'FULL_RATIO') else Config.FILL_RATIO)*area_w*area_h)
    cur_area = total_area(metas)
    if cur_area>0:
        f=(target_area/cur_area)**0.5
        for m in metas: m.scale*=f
        layout_images(metas, grid_bounds, steps=max(600,Config.STEPS//3), seed=Config.SEED)

    compact_left_up(metas, grid_bounds, margin=Config.MARGIN, sweeps=Config.COMPACT_STEPS)
    zoom_to_fit(metas, grid_bounds, margin=Config.MARGIN)
    layout_images(metas, grid_bounds, steps=400, seed=Config.SEED)

    canvas = Image.new("RGBA", (Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT), (255,255,255,255))

    # Title pinned near top edge
    if title_img is not None:
        tx0, ty0, tx1, ty1 = title_strip
        tw, th = title_img.size
        px = tx0 + (tx1 - tx0 - tw) // 2
        py = ty0 + 2
        canvas.alpha_composite(title_img, (px, py))

    # Palette
    px0,py0,px1,py1 = palette_strip
    pw,ph = palette_img.size
    px = px0 + (px1-px0 - pw)//2
    py = py0 + (py1-py0 - ph)//2
    canvas.alpha_composite(palette_img, (px,py))

    # Main images
    for m,im in zip(metas, thumbs):
        w,h = int(m.w*m.scale), int(m.h*m.scale)
        to_paste = im.resize((w,h), Image.LANCZOS) if (w,h)!=im.size else im
        canvas.alpha_composite(to_paste, (int(m.x), int(m.y)))

    out_dir = Path(Config.OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir/"moodboard.png"
    out_json = out_dir/"placements.json"
    canvas.save(out_png)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "canvas": {"width": Config.CANVAS_WIDTH, "height": Config.CANVAS_HEIGHT},
            "grid_bounds": {"x0": grid_bounds[0], "y0": grid_bounds[1], "x1": grid_bounds[2], "y1": grid_bounds[3]},
            "title": {"text": Config.TITLE_TEXT, "strip": title_strip if title_img is not None else None, "side": title_side},
            "palette": {"side": "right" if title_side in ("top","bottom") else "bottom", "strip": palette_strip, "colors": palette_colors},
            "images": [asdict(m) for m in metas]
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_png} and {out_json}")

if __name__ == "__main__":
    main()
