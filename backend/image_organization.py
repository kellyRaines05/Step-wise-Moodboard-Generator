import os, json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import uuid
from datetime import date
from backend.organizer_helpers import *
from backend.request_models import Moodboard
from backend.constants import *

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
    shape: str = "rectangle"  # added for shape assignment

if ALLOW_DUPLICATE_OMP:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", str(OMP_NUM_THREADS))

class ImageOrganization:
    def __init__(self, title: str, images_dir: str="../images", out_dir: str="../past_moodboards"):
        self.images_dir = images_dir
        self.out_dir = out_dir
        self.title = title
        self.canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (255,255,255,255))

    def organize_images(self):
        names = [n for n in os.listdir(self.images_dir) if not n.startswith(".")]
        valid = {".png",".jpg",".jpeg",".webp",".bmp"}
        files = [(n, str(Path(self.images_dir)/n)) for n in names if Path(n).suffix.lower() in valid]
        if not files:
            raise SystemExit("No images found in ./images")

        thumbs=[]; metas=[]
        for n,p in files:
            im = load_and_thumb(p, MAX_THUMB)
            thumbs.append(im)
            metas.append(ImgMeta(filename=n, w=im.size[0], h=im.size[1], scale=1.0))

        CW,CH = CANVAS_WIDTH, CANVAS_HEIGHT
        grid_x0=MARGIN; grid_y0=MARGIN
        grid_x1=CW-MARGIN; grid_y1=CH-MARGIN

        # Title strip
        title_side = "top" if len(thumbs)==0 or sum(im.size[0] for im in thumbs)/max(1,len(thumbs)) >= sum(im.size[1] for im in thumbs)/max(1,len(thumbs)) else "left"
        if self.title:
            if title_side in ("top","bottom"):
                strip_h = int(CH * TITLE_MAX_FRACTION)
                title_strip = (0, 0, CW, strip_h)
                grid_y0 = max(grid_y0, strip_h)
            else:
                strip_w = int(CW * TITLE_MAX_FRACTION)
                title_strip = (0, 0, strip_w, CH)
                grid_x0 = max(grid_x0, strip_w)
            title_img = render_title_image(
                self.title,
                (title_strip[2]-title_strip[0]-STRIP_PADDING*2,
                 title_strip[3]-title_strip[1]-STRIP_PADDING*2)
            )
        else:
            title_img=None; title_strip=None

        # Palette strip — trimmed to avoid the title strip
        palette_colors = generate_palette_swatches(thumbs, PALETTE_SWATCHES)
        if title_side in ("top","bottom"):
            palette_side = "right"
            strip_w = int(CW * PALETTE_MAX_FRACTION)
            palette_strip = (CW - strip_w, grid_y0, CW, grid_y1)
            grid_x1 = min(grid_x1, CW - strip_w)
        else:
            palette_side = "bottom"
            strip_h = int(CH * PALETTE_MAX_FRACTION)
            palette_strip = (grid_x0, CH - strip_h, grid_x1, CH)
            grid_y1 = min(grid_y1, CH - strip_h)

        palette_img = render_palette_image(
            palette_colors,
            (max(1, palette_strip[2]-palette_strip[0]-STRIP_PADDING*2),
             max(1, palette_strip[3]-palette_strip[1]-STRIP_PADDING*2))
        )

        grid_bounds = (grid_x0, grid_y0, grid_x1, grid_y1)

        # Features
        clip_feats=None
        if USE_CLIP:
            try:
                clip_feats = embed_with_clip(thumbs)
            except Exception as e:
                raise RuntimeError(f"CLIP embedding failed: {e}. Ensure torch+clip are installed.")

        color_feats = np.array([[c for corner in corner_means(im) for c in corner] for im in thumbs], np.float32)
        blocks=[]
        if clip_feats is not None:
            blocks.append(standardize(clip_feats)*CLIP_WEIGHT)
        else:
            blocks.append(np.zeros((len(thumbs),1),np.float32))
        blocks.append(standardize(color_feats)*COLOR_WEIGHT)
        X = np.concatenate(blocks,1).astype(np.float32)

        # === Shape assignment block ===
        clip_feats_np = clip_feats if clip_feats is not None else None
        if SHAPE_MODE.lower() == "auto":
            assign_shapes_autonomous(
                thumbs, metas, palette_colors,
                clip_img_feats=clip_feats_np,
                title_text=self.title or ""
            )
        else:
            for m in metas:
                m.shape = SHAPE.lower()
        # ==============================

        Z = pca_project(X, 2)
        Zmin=Z.min(0, keepdims=True); Zmax=Z.max(0, keepdims=True)
        Z = (Z - Zmin) / (Zmax - Zmin + 1e-6)
        for i,m in enumerate(metas):
            tx = grid_bounds[0] + MARGIN + Z[i,0]*((grid_bounds[2]-grid_bounds[0])-2*MARGIN - m.w*m.scale)
            ty = grid_bounds[1] + MARGIN + Z[i,1]*((grid_bounds[3]-grid_bounds[1])-2*MARGIN - m.h*m.scale)
            tx,ty = bias_toward_center(tx,ty,grid_bounds,CENTER_BIAS_STRENGTH,CENTER_BIAS_GAMMA)
            m.x = m.target_x = float(tx)
            m.y = m.target_y = float(ty)

        layout_images(metas, grid_bounds, steps=STEPS, seed=SEED)
        scale_by_radius(metas, grid_bounds, RADIAL_SCALE_MAX, RADIAL_SCALE_MIN, RADIAL_SCALE_ETA)
        layout_images(metas, grid_bounds, steps=max(600,STEPS//3), seed=SEED)

        area_w = grid_bounds[2]-grid_bounds[0]; area_h = grid_bounds[3]-grid_bounds[1]
        target_area = int(max(1, FILL_RATIO)*area_w*area_h)
        cur_area = total_area(metas)
        if cur_area>0:
            f=(target_area/cur_area)**0.5
            for m in metas: m.scale*=f
            layout_images(metas, grid_bounds, steps=max(600,STEPS//3), seed=SEED)

        compact_left_up(metas, grid_bounds, margin=MARGIN, sweeps=COMPACT_STEPS)
        zoom_to_fit(metas, grid_bounds, margin=MARGIN)
        layout_images(metas, grid_bounds, steps=400, seed=SEED)

        # Title pinned near top edge
        if title_img is not None:
            tx0, ty0, tx1, ty1 = title_strip
            tw, th = title_img.size
            px = tx0 + (tx1 - tx0 - tw) // 2
            py = ty0 + 2
            self.canvas.alpha_composite(title_img, (px, py))

        # Palette
        px0,py0,px1,py1 = palette_strip
        pw,ph = palette_img.size
        px = px0 + (px1-px0 - pw)//2
        py = py0 + (py1-py0 - ph)//2
        self.canvas.alpha_composite(palette_img, (px,py))

        # Main images
        for m, im in zip(metas, thumbs):
            w, h = int(m.w * m.scale), int(m.h * m.scale)
            to_paste = im.resize((w, h), Image.LANCZOS) if (w, h) != im.size else im

            # === apply shape mask here ===
            to_paste = apply_shape_mask(to_paste, m.shape, feather=SHAPE_EDGE_FEATHER)
            # =============================

            self.canvas.alpha_composite(to_paste, (int(m.x), int(m.y)))

        out_json = f"{self.out_dir}/{self.title}_{uuid.uuid4()}_placement.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
                "grid_bounds": {"x0": grid_bounds[0], "y0": grid_bounds[1], "x1": grid_bounds[2], "y1": grid_bounds[3]},
                "title": {"text": self.title, "strip": title_strip if title_img is not None else None, "side": title_side},
                "palette": {"side": "right" if title_side in ("top","bottom") else "bottom", "strip": palette_strip, "colors": palette_colors},
                "images": [asdict(m) for m in metas]
            }, f, ensure_ascii=False, indent=2)
            print(f"Wrote {out_json}")
    
    def save_moodboard(self, prompt: str):
        out_dir = Path(self.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        id = str(uuid.uuid4())
        out_png = f"{out_dir}/{self.title}_{id}.png"
        self.canvas.save(out_png)

        out_json = f"{out_dir}/past_moodboards.json"
        new_moodboard = Moodboard(title=self.title, prompt=prompt, image_url=out_png, date_created=str(date.today()))

        if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(new_moodboard)
       
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_png} and saved to {out_json}")
        return out_json
