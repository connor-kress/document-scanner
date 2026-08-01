"""
Data cleaning, feature engineering and export for the SmartDoc 2015
Challenge 1 frames.

Owner: Cole Morgan.  Consumed by the EDA and modeling tracks.

Files
-----
paths      folder locations, photo size, column names
geometry   maths on the 4 corner points -- reordering, area, flattening
qc         step 1: open every photo, measure blur / brightness / colour
manifest   step 2: build the main table -- labels, measurements, splits
export     step 3: save the photos in shapes a model can use
cli        the `build_data` command

Every function is also listed here, so `from preprocess import X` works no
matter which of the files above X is really in.

    python scripts/build_data.py all

The one thing to know
---------------------
Photos are 1920x1080, landscape. The original labels name corners after the
PAGE, not the photo -- so `tl` is the page's top-left corner, which is not
always the photo's top-left corner. Every photo in background04 was taken with
the tablet sideways, so its `tl` lands somewhere else entirely. That is why
`canonicalise_quad` exists; read that function first.
"""

from .paths import (CANON_COLS, DEWARP_SIZE, FIGURES, FRAME_H, FRAME_W, FRAMES,
                    OUT, PROJECT, RAW_CORNER_COLS, SEED, TARGET_COLS)
from .geometry import (canonicalise_quad, denormalise, dewarp, is_convex,
                       load_raw_quads, normalise, polygon_area, rotation_deg,
                       side_lengths)
from .qc import run_qc
from .manifest import add_splits, build_manifest, subsample
from .export import (channel_stats, export_arrays, export_frame_tree,
                     load_frame)
from .cli import main

__all__ = [
    # paths / constants
    "PROJECT", "FRAMES", "OUT", "FIGURES", "FRAME_W", "FRAME_H", "DEWARP_SIZE",
    "RAW_CORNER_COLS", "CANON_COLS", "TARGET_COLS", "SEED",
    # geometry
    "load_raw_quads", "canonicalise_quad", "polygon_area", "side_lengths",
    "is_convex", "rotation_deg", "normalise", "denormalise", "dewarp",
    # pipeline stages
    "run_qc", "build_manifest", "add_splits", "subsample",
    "load_frame", "export_arrays", "export_frame_tree", "channel_stats",
    "main",
]
