"""Command line tool that builds the data.

    python scripts/build_data.py all
"""


from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .download import ensure_raw_data, processed_data_ready
from .export import channel_stats, export_arrays, export_frame_tree
from .manifest import build_manifest, subsample
from .paths import OUT
from .qc import run_qc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("stage", choices=["download", "qc", "manifest", "export", "stats", "all"])
    ap.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth frame of each clip in the exports")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N frames")
    ap.add_argument("--force", action="store_true", help="rebuild valid processed data")
    a = ap.parse_args(argv)

    if a.workers < 1:
        ap.error("--workers must be at least 1")
    ensure_raw_data()
    if a.stage == "download":
        return 0
    if a.stage == "all" and not a.force and processed_data_ready():
        print(f"Using existing preprocessed data: {OUT}")
        return 0
    if a.stage == "stats":
        channel_stats(a.workers)
        return 0
    if a.stage in ("qc", "all"):
        run_qc(a.workers, a.limit)
    if a.stage in ("manifest", "all"):
        build_manifest()
    if a.stage in ("export", "all"):
        df = pd.read_csv(OUT / "manifest.csv")
        sub = subsample(df, stride=a.stride)
        print(f"export subset: {len(sub)} frames (stride {a.stride}) "
              f"from {int(df.usable.sum())} usable")
        sub.to_csv(OUT / "manifest_subset.csv", index=False)
        export_arrays(sub, (64, 36), "clahe", "tab_64x36_clahe", a.workers)
        export_arrays(sub, (128, 72), "rgb", "img_128x72_rgb", a.workers)
        export_frame_tree(sub, (384, 216), a.workers)
    if a.stage == "all" and not processed_data_ready():
        raise RuntimeError(f"processed data validation failed: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
