# HELPER FUNCTION CONSTANTS - organizer_helpers.py
TITLE_FONT_MAX = 150
TITLE_TEXT_SHRINK = 50

CENTER_FORCE_K = 0.02

CLIP_MODEL = "ViT-B/32"
CLIP_DEVICE = "auto"
PALETTE_METHOD = "kmeans"

# IMAGE ORGANIZATION CONSTANTS - image_organization.py
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 800
MARGIN = 12

MAX_THUMB = 360
FILL_RATIO = 0.92
STEPS = 2500
COMPACT_STEPS = 10
SEED = 42

# --- Auto-shape controls ---
SHAPE_MODE = "auto"   # "auto" or "fixed"; if "fixed", use SHAPE below
SHAPE = "rectangle"   # used only when SHAPE_MODE="fixed"  ("rectangle"|"oval"|"circle")

# Aesthetic score weights (tweak to taste)
W_CLIP_TITLE = 0.5     # relevance to title text
W_PALETTE = 0.25       # closeness to palette colors
W_ENTROPY = 0.15       # texture/edge density (more = higher)
W_ASPECT = 0.10        # “pleasant” aspect; penalize extremes a bit

# Shape thresholds (after normalization to 0..1)
CIRCLE_TOP_P = 0.30    # top 30% by aesthetic score -> circle (cap below applies)
OVAL_NEXT_P  = 0.40    # next 40% -> oval; rest rectangle

# Diversity caps to avoid too many circles/ovals
MAX_CIRCLE_FRAC = 0.35
MAX_OVAL_FRAC   = 0.45

SHAPE_EDGE_FEATHER = 4

TITLE_MAX_FRACTION = 0.18
PALETTE_MAX_FRACTION = 0.16
STRIP_PADDING = 6
PALETTE_SWATCHES = 5

USE_CLIP = True
CLIP_WEIGHT = 1.0
COLOR_WEIGHT = 0.6

CENTER_BIAS_STRENGTH = 0.35
CENTER_BIAS_GAMMA = 1.2
RADIAL_SCALE_MAX = 1.18
RADIAL_SCALE_MIN = 0.88
RADIAL_SCALE_ETA = 1.2

ALLOW_DUPLICATE_OMP = False
OMP_NUM_THREADS = 1