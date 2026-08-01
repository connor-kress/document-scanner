"""Prepocessing step to create the data formats and save them

    arrays/tab_64x36_clahe.npz   tiny grayscale, for ridge and the MLP
    arrays/img_128x72_rgb.npz    small colour, fits in memory, for the CNN
    frames_384/ + labels.csv     bigger JPEGs on disk, for a CNN that streams

    python scripts/build_data.py export
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .paths import FRAME_H, FRAME_W, FRAMES, OUT, TARGET_COLS


def load_frame(rel: str, size: tuple[int, int], mode: str = "rgb") -> np.ndarray:
    """Open one photo and shrink it to (width, height).

    We just squash it to the size asked for. No grey bars are added, because
    every photo is already the same shape, and the 0-1 labels still line up.

    mode: 'rgb'   colour, shape (h, w, 3)
          'gray'  black and white, shape (h, w)
          'clahe' black and white with the contrast evened out. Helps the
                  simple models, which cannot cope with uneven lighting.
    """
    img = cv2.imread(str(FRAMES / rel.replace("/", os.sep)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(rel)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if mode == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def _load_job(args):
    rel, size, mode = args
    return load_frame(rel, size, mode)


def export_arrays(df: pd.DataFrame, size: tuple[int, int], mode: str,
                  name: str, workers: int) -> Path:
    """Save one .npz file holding the photos, the labels, and the train/val/test
    columns together, so nobody has to match them up by hand later."""
    jobs = [(p, size, mode) for p in df.image_path]
    w, h = size
    shape = (len(jobs), h, w, 3) if mode == "rgb" else (len(jobs), h, w)
    X = np.empty(shape, np.uint8)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, arr in enumerate(pool.map(_load_job, jobs, chunksize=32)):
            X[i] = arr
            if (i + 1) % 2000 == 0:
                print(f"  {name} {i + 1}/{len(jobs)}", flush=True)

    y = df[TARGET_COLS].to_numpy(np.float32)
    dst = OUT / "arrays"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{name}.npz"
    np.savez_compressed(
        path,
        X=X,
        y=y,
        image_path=df.image_path.to_numpy(),
        video_id=df.video_id.to_numpy(),
        model_id=df.model_id.to_numpy(np.int16),
        modeltype_id=df.modeltype_id.to_numpy(np.int8),
        bg_id=df.bg_id.to_numpy(np.int8),
        split_video=df.split_video.to_numpy(),
        split_doc=df.split_doc.to_numpy(),
        fold5=df.fold5.to_numpy(np.int8),
        target_cols=np.array(TARGET_COLS),
        frame_size=np.array([FRAME_W, FRAME_H]),
    )
    print(f"wrote {path}  X{X.shape} {X.dtype}  y{y.shape}  "
          f"{path.stat().st_size / 1e6:.0f} MB")
    return path


def _resize_job(args):
    rel, size, quality, dst_root = args
    img = cv2.imread(str(FRAMES / rel.replace("/", os.sep)), cv2.IMREAD_COLOR)
    if img is None:
        return None
    out = Path(dst_root) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".jpg")),
                cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    return rel


def export_frame_tree(df: pd.DataFrame, size: tuple[int, int], workers: int,
                      quality: int = 92) -> Path:
    """Save smaller copies of the photos as JPEGs, in the same folder layout.
    Use this when the photos are too big to all fit in memory at once."""
    dst = OUT / f"frames_{size[0]}"
    jobs = [(p, size, quality, str(dst)) for p in df.image_path]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_resize_job, jobs, chunksize=32), 1):
            if i % 2000 == 0:
                print(f"  tree {i}/{len(jobs)}", flush=True)
    cols = (["image_path", "video_id", "model_id", "modeltype_id", "modeltype_name",
             "bg_id", "frame_index", "split_video", "split_doc", "fold5",
             "rot_class", "area_frac", "perspective_skew"] + TARGET_COLS)
    labels = df[cols].copy()
    labels["image_path"] = labels.image_path.str.replace(".jpeg", ".jpg", regex=False)
    labels.to_csv(dst / "labels.csv", index=False)
    print(f"wrote {dst}  ({len(df)} frames + labels.csv)")
    return dst


def _stats_job(rel: str):
    img = cv2.imread(str(FRAMES / rel.replace("/", os.sep)), cv2.IMREAD_COLOR)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64) / 255.0
    return x.sum(0), (x ** 2).sum(0), len(x)


def channel_stats(workers: int, n: int = 1500, seed: int = 0):
    """Average red, green and blue across the whole dataset.

    These are the numbers in augment.py that every photo gets adjusted by.
    Counts one photo at a time so it never has to hold them all in memory.
    """
    df = pd.read_csv(FRAMES / "metadata.csv.gz").sample(n, random_state=seed)
    s = np.zeros(3); s2 = np.zeros(3); count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for a, b, k in pool.map(_stats_job, df.image_path.tolist(), chunksize=16):
            s += a; s2 += b; count += k
    mean = s / count
    std = np.sqrt(s2 / count - mean ** 2)
    print(f"pixels: {count:,}")
    print("DATASET_MEAN =", np.round(mean, 4))
    print("DATASET_STD  =", np.round(std, 4))
    return mean, std
