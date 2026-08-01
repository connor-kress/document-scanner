#!/usr/bin/env python
"""Build every derived artifact from data/raw/frames.

    python scripts/build_data.py all        # qc + manifest + export (~6 min)
    python scripts/build_data.py manifest   # just re-derive features/splits
    python scripts/build_data.py export --stride 1
    python scripts/build_data.py stats      # channel mean/std
"""
import sys

from preprocess import main

if __name__ == "__main__":
    sys.exit(main())
