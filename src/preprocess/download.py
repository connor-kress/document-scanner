"""Download, verify, extract, and validate the SmartDoc dataset."""
from __future__ import annotations

import csv
import hashlib
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from .paths import FRAMES, OUT


ARCHIVE = FRAMES.parent / "frames.tar.gz"
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


def processed_data_ready(processed: Path = OUT) -> bool:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset_archive(destination: Path = ARCHIVE) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Verifying cached dataset archive: {destination}")
        if _sha256(destination) == DATA_SHA256:
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
        if _sha256(partial) != DATA_SHA256:
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
