#!/usr/bin/env python
"""Download and build the SmartDoc data artifacts.

    python scripts/build_data.py download   # download, verify, and extract only
    python scripts/build_data.py all        # download if needed, then build
    python scripts/build_data.py manifest   # just re-derive features/splits
    python scripts/build_data.py export --stride 1
    python scripts/build_data.py stats      # channel mean/std
"""
import sys

from preprocess import main

if __name__ == "__main__":
    sys.exit(main())
