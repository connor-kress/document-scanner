#!/usr/bin/env python
"""Prepare data, train every model, and run final evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
FRAMES = RAW / "frames"
ARCHIVE = RAW / "frames.tar.gz"
PROCESSED = PROJECT / "data" / "processed"
DATA_URL = (
    "https://github.com/jchazalon/smartdoc15-ch1-dataset/"
    "releases/download/v2.0.0/frames.tar.gz"
)
DATA_SHA256 = "3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae"
BACKGROUND_DIRS = tuple(f"background{index:02d}" for index in range(1, 6))


def raw_data_ready(frames: Path = FRAMES) -> bool:
    return (frames / "metadata.csv.gz").is_file() and all(
        (frames / name).is_dir() for name in BACKGROUND_DIRS
    )


def processed_data_ready(processed: Path = PROCESSED) -> bool:
    arrays = (
        processed / "arrays" / "tab_64x36_clahe.npz",
        processed / "arrays" / "img_128x72_rgb.npz",
    )
    required = (
        processed / "qc.csv",
        processed / "manifest.csv",
        processed / "manifest_subset.csv",
        *arrays,
        processed / "frames_384" / "labels.csv",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False

    expected_columns = {"image_path", "video_id", "usable", "split_video", "split_doc"}
    try:
        with open(processed / "manifest.csv", newline="") as handle:
            reader = csv.reader(handle)
            columns = set(next(reader))
            has_row = next(reader, None) is not None
        expected_arrays = {"X.npy", "y.npy", "split_video.npy", "video_id.npy"}
        for path in arrays:
            with zipfile.ZipFile(path) as bundle:
                if not expected_arrays <= set(bundle.namelist()):
                    return False
    except (OSError, csv.Error, StopIteration, zipfile.BadZipFile):
        return False
    return has_row and expected_columns <= columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset_archive(destination: Path = ARCHIVE) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Verifying cached dataset archive: {destination}")
        if sha256(destination) == DATA_SHA256:
            return
        print("Cached archive checksum is invalid; downloading it again")

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading dataset (~1 GB) from {DATA_URL}")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response, open(partial, "wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (100 * 1024 * 1024) < len(chunk):
                    print(f"  {downloaded / total:.0%}", flush=True)
        if sha256(partial) != DATA_SHA256:
            raise RuntimeError("downloaded dataset archive failed SHA-256 verification")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in dataset archive: {member.name}")
            if not member.isfile() and not member.isdir():
                raise RuntimeError(f"unsupported entry in dataset archive: {member.name}")
        bundle.extractall(destination, filter="data")


def extract_dataset(archive: Path = ARCHIVE, frames: Path = FRAMES) -> None:
    frames.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".frames-", dir=frames.parent))
    print(f"Extracting dataset into {frames}")
    try:
        safe_extract(archive, temporary)
        source = temporary / "frames"
        if not raw_data_ready(source):
            source = temporary
        if not raw_data_ready(source):
            raise RuntimeError("dataset archive does not contain the expected frames layout")

        if frames.exists():
            backup = frames.with_name(f"frames.incomplete-{time.strftime('%Y%m%dT%H%M%S')}")
            frames.replace(backup)
            print(f"Moved incomplete existing data to {backup}")
        source.replace(frames)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def ensure_raw_data() -> None:
    if raw_data_ready():
        print(f"Using existing raw dataset: {FRAMES}")
        return
    download_dataset_archive()
    extract_dataset()
    if not raw_data_ready():
        raise RuntimeError(f"raw dataset validation failed: {FRAMES}")


def run_command(command: list[str], description: str) -> bool:
    print(f"\n{'=' * 70}\n[{description}]\nCommand: {' '.join(command)}\n{'=' * 70}")
    started = time.perf_counter()
    try:
        subprocess.run(command, check=True, cwd=PROJECT)
    except subprocess.CalledProcessError as error:
        elapsed = time.perf_counter() - started
        print(f"\nFAILED: {description} after {elapsed / 60:.1f} min")
        print(f"Exit code: {error.returncode}")
        return False
    elapsed = time.perf_counter() - started
    print(f"\nSUCCESS: {description} completed in {elapsed / 60:.1f} min")
    return True


def training_steps() -> list[tuple[list[str], str]]:
    return [
        ([sys.executable, "scripts/train_ridge.py"], "Ridge regression"),
        (
            [sys.executable, "scripts/grid_search.py", "--model", "mlp",
             "--base-config", "configs/mlp.yaml", "--train-best"],
            "MLP grid search and full-data training",
        ),
        (
            [sys.executable, "scripts/grid_search.py", "--model", "cnn",
             "--base-config", "configs/cnn_gray.yaml", "--train-best"],
            "Grayscale CNN grid search and full-data training",
        ),
        (
            [sys.executable, "scripts/grid_search.py", "--model", "cnn",
             "--base-config", "configs/cnn_rgb.yaml", "--train-best"],
            "RGB CNN grid search and full-data training",
        ),
        ([sys.executable, "scripts/final_eval.py"], "Final held-out evaluation"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) - 1)),
        help="parallel preprocessing workers (default: up to 4)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    print(f"Project root: {PROJECT}")
    ensure_raw_data()

    steps: list[tuple[list[str], str]] = []
    if processed_data_ready():
        print(f"Using existing preprocessed data: {PROCESSED}")
    else:
        steps.append(
            ([sys.executable, "scripts/build_data.py", "all", "--workers", str(args.workers)],
             "Data preprocessing")
        )
    steps.extend(training_steps())

    completed = []
    pipeline_started = time.perf_counter()
    for command, description in steps:
        if not run_command(command, description):
            print(f"\nPipeline stopped at: {description}")
            return 1
        completed.append(description)
        if description == "Data preprocessing" and not processed_data_ready():
            print(f"\nPipeline stopped: processed data validation failed in {PROCESSED}")
            return 1

    elapsed = time.perf_counter() - pipeline_started
    print("\nPIPELINE COMPLETE")
    for description in completed:
        print(f"PASS: {description}")
    print(f"Total time: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} hours)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
