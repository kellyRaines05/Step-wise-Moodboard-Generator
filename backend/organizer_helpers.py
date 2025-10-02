import math, random
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from backend.constants import *

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
        center_k=CENTER_FORCE_K; cx,cy=x0+W*0.5, y0+H*0.5
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
    device = "cuda" if (CLIP_DEVICE=="auto" and torch.cuda.is_available()) else ("cpu" if CLIP_DEVICE in ["auto","cpu"] else "cuda")
    model, preprocess = clip.load(CLIP_MODEL, device=device, jit=False)
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
    if PALETTE_METHOD == "median":
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
        base_font_path = "C:/Windows/Fonts/BRADHITC.ttf"
        base_font = ImageFont.truetype(base_font_path, size=96)
    except Exception:
        base_font_path = None
        base_font = ImageFont.load_default()

    # Binary search the largest font size that fits WITH the badge padding
    lo, hi = 18, int(TITLE_FONT_MAX)
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
    shrunk_size = max(1, best_size - int(TITLE_TEXT_SHRINK))
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

def apply_shape_mask(img: Image.Image, shape: str, feather: int = 1) -> Image.Image:
    """
    Return a copy of `img` with an alpha mask applied for the desired shape.
    shape: "rectangle", "oval", "circle"
    feather: small integer (0..3) for softer edges (anti-alias); 0 for hard edges.
    """
    shape = (shape or "rectangle").strip().lower()
    if shape == "rectangle":
        return img  # unchanged

    w, h = img.size
    # Supersample factor for smoother edges
    ss = 4
    mw, mh = w * ss, h * ss
    from PIL import ImageDraw

    # Create an oversized mask for antialiasing
    mask_big = Image.new("L", (mw, mh), 0)
    draw = ImageDraw.Draw(mask_big)

    if shape == "oval":
        # full ellipse in the rect
        draw.ellipse([0, 0, mw - 1, mh - 1], fill=255)
    elif shape == "circle":
        # centered circle with diameter = min(w,h)
        d = min(mw, mh)
        x0 = (mw - d) // 2
        y0 = (mh - d) // 2
        draw.ellipse([x0, y0, x0 + d - 1, y0 + d - 1], fill=255)
    else:
        # fallback to rectangle if unknown
        return img

    # Optional feathering: slight blur via downscale (already anti-aliased), plus extra soften
    mask = mask_big.resize((w, h), Image.LANCZOS)
    if feather and feather > 0:
        # simple extra soften by shrinking and re-enlarging a tiny bit
        fw = max(1, w - feather)
        fh = max(1, h - feather)
        mask = mask.resize((fw, fh), Image.LANCZOS).resize((w, h), Image.LANCZOS)

    # Apply alpha: keep RGB, replace alpha with mask
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, *rest = img.split() if img.mode == "RGBA" else (*img.split(),)
    shaped = Image.merge("RGBA", (img.split()[0], img.split()[1], img.split()[2], mask))
    return shaped

def _rgb_palette_distance(im: Image.Image, palette_rgb: np.ndarray) -> float:
    """
    Mean distance of image average color to closest palette swatch (0..1 after normalization).
    """
    arr = np.asarray(im.convert('RGB')).reshape(-1, 3).astype(np.float32) / 255.0
    mean = arr.mean(axis=0, keepdims=True)  # (1,3)
    # Euclidean to each swatch
    d = np.sqrt(((mean - palette_rgb) ** 2).sum(axis=1))  # (K,)
    return float(d.min())

def _edge_density(im: Image.Image) -> float:
    """
    Simple edge density via Sobel-ish magnitude from a downsized luminance image.
    Returns ~0..1 (normalized).
    """
    small = im.convert('L').resize((96, 96), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    gx = np.zeros_like(arr); gy = np.zeros_like(arr)
    # quick 3x3 gradients
    gx[1:-1,1:-1] = (arr[1:-1,2:] - arr[1:-1,:-2]) * 0.5
    gy[1:-1,1:-1] = (arr[2:,1:-1] - arr[:-2,1:-1]) * 0.5
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.clip(mag.mean()*2.0, 0.0, 1.0))

def _aspect_pleasantness(w: int, h: int) -> float:
    """
    Scores aspect ratios near ~1.2–1.6 slightly higher; penalizes extremes.
    Normalize to 0..1.
    """
    r = max(w, h) / max(1.0, min(w, h))
    target = 1.4
    diff = abs(r - target)
    # decay; diff=0 -> 1, diff>=1.4 -> ~0
    score = np.exp(-diff * 1.2)
    return float(np.clip(score, 0.0, 1.0))

def _clip_title_similarity(clip_img_feats: np.ndarray, clip_txt_feat: np.ndarray) -> np.ndarray:
    """
    cosine similarity (0..1) between each image and the title text embedding.
    """
    # feats are L2-normalized in your embed function already; still guard:
    I = clip_img_feats / (np.linalg.norm(clip_img_feats, axis=1, keepdims=True) + 1e-6)
    t = clip_txt_feat / (np.linalg.norm(clip_txt_feat) + 1e-6)
    sim = I @ t  # [-1..1]
    sim = (sim + 1.0) * 0.5  # map to 0..1
    return sim

def _embed_title_text(text: str):
    import torch, clip
    device = "cuda" if (CLIP_DEVICE=="auto" and torch.cuda.is_available()) else ("cpu" if CLIP_DEVICE in ["auto","cpu"] else "cuda")
    model, preprocess = clip.load(CLIP_MODEL, device=device, jit=False)
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        feat = model.encode_text(tokens)[0]
        feat = feat / feat.norm()
    return feat.detach().cpu().numpy().astype(np.float32)

def assign_shapes_autonomous(thumbs, metas, palette_colors, clip_img_feats=None, title_text: str="") -> None:
    """
    Mutates each ImgMeta by setting a new attribute `shape` to 'circle'|'oval'|'rectangle'
    based on a blended aesthetic score with diversity caps.
    """
    N = len(thumbs)
    if N == 0:
        return

    # Normalize palette to 0..1
    pal = np.array(palette_colors, dtype=np.float32) / 255.0  # (K,3)

    # Collect components
    edge = np.array([_edge_density(im) for im in thumbs], dtype=np.float32)            # 0..1
    aspect = np.array([_aspect_pleasantness(m.w, m.h) for m in metas], dtype=np.float32)  # 0..1
    palclose = np.array([1.0 - _rgb_palette_distance(im, pal) for im in thumbs], dtype=np.float32)  # higher=closer -> 0..1

    # Title relevance via CLIP (optional)
    if USE_CLIP and clip_img_feats is not None and title_text:
        txt_feat = _embed_title_text(title_text)
        rel = _clip_title_similarity(clip_img_feats, txt_feat).astype(np.float32)
    else:
        rel = np.zeros(N, dtype=np.float32)

    # Blend to aesthetic score
    score = (W_CLIP_TITLE * rel
            + W_PALETTE    * palclose
            + W_ENTROPY    * edge
            + W_ASPECT     * aspect)

    # Normalize to 0..1
    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min())
    else:
        score[:] = 0.5

    # Rank and assign with caps for diversity
    order = np.argsort(-score)  # high to low
    max_circle = int(MAX_CIRCLE_FRAC * N)
    max_oval   = int(MAX_OVAL_FRAC   * N)
    want_circle = int(CIRCLE_TOP_P * N)
    want_oval   = int(OVAL_NEXT_P  * N)

    # Initial desired bins
    circle_cut = min(want_circle, max_circle)
    oval_cut   = min(want_circle + want_oval, max_oval + circle_cut)

    shapes = ['rectangle'] * N
    for i, idx in enumerate(order):
        if i < circle_cut:
            shapes[idx] = 'circle'
        elif i < oval_cut:
            shapes[idx] = 'oval'
        else:
            shapes[idx] = 'rectangle'

    # write back
    for i, m in enumerate(metas):
        setattr(m, 'shape', shapes[i])


