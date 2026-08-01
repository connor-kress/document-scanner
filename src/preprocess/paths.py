"""Folder locations and fixed numbers used everywhere else.

This file imports nothing from the project, so anything can import it safely.
"""

from __future__ import annotations

from pathlib import Path

# Worked out from this file's own location, so it doesn't matter which folder
# you run things from.
PROJECT = Path(__file__).resolve().parents[2]  # src/preprocess/ -> repo root
FRAMES = PROJECT / "data" / "raw" / "frames"
OUT = PROJECT / "data" / "processed"
FIGURES = PROJECT / "reports" / "figures"

# Photos are 1920 wide by 1080 tall -- landscape, not portrait.
FRAME_W, FRAME_H = 1920, 1080

# Size of the flattened page that `dewarp` makes. A4 shape, quarter scale.
DEWARP_SIZE = (525, 742)

#   cols
#   RAW    = original labels, corners in page order
#   CANON  = our reordered corners, in pixels
#   TARGET = the same corners as 0-1 fractions. These 8 are what a model predicts.
RAW_CORNER_COLS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]
CANON_COLS = [f"c{i}_{a}" for i in range(4) for a in ("x", "y")]
TARGET_COLS = [f"c{i}_{a}_n" for i in range(4) for a in ("x", "y")]

# rng seed
SEED = 42
