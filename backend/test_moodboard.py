#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for the segmented moodboard codebase.

Files this expects in the same folder:
- image_organization.py
- organizer_helpers.py
- constants.py
- ./images/  (your input images)

Run:
    python test_moodboard.py
"""

import os
import uuid
from pathlib import Path

# ---- Optional: stability on Windows + OpenMP --------------------------------
# If you still see "libiomp5md.dll already initialized", uncomment:
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"

# ---- Imports from your project ----------------------------------------------
from image_organization import ImageOrganization
import constants as C

def main():
    # ---------------- Try-it knobs (comment/uncomment) ----------------
    # 1) Title to render at the top/left strip:
    title = "Test Moodboard"   # e.g., "Brutalist Interiors", "Coastal Modern", etc.

    # 2) Input/Output directories:
    images_dir = "../images"         # change if your images are elsewhere
    out_dir    = "./past_moodboards"  # where we save PNG + JSON

    # 3) Flip this to quickly disable CLIP (for fast tests) — or leave as-is:
    # C.USE_CLIP = False

    # 4) Make the title smaller or bigger quickly (only affects title rendering):
    # C.TITLE_FONT_MAX = 72
    # C.TITLE_TEXT_SHRINK = 2

    # 5) Tweak fill / steps for speed vs quality:
    # C.FILL_RATIO = 0.90
    # C.STEPS = 1200

    # ------------------------------------------------------------------

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Instantiate the organizer with a fresh canvas
    org = ImageOrganization(
        title=title,
        images_dir=images_dir,
        out_dir=out_dir,
    )

    # Generate the layout and composite onto the canvas
    org.organize_images()

    # Save the PNG and JSON (org.organize_images already wrote JSON; we write PNG here)
    org.save_moodboard()

    print("Done.")

if __name__ == "__main__":
    main()
