"""Feeds photos and labels to a model.

    from dataset import FrameSet, TorchFrameSet
    from torch.utils.data import DataLoader

    tr = FrameSet("train", size=(224, 224), augment=True)
    dl = DataLoader(TorchFrameSet(tr), batch_size=32, shuffle=True, num_workers=4)

`FrameSet` is plain numpy so it works without PyTorch installed. Wrap it in
`TorchFrameSet` when you want a real PyTorch loader.

The labels are 8 numbers from 0 to 1 and do NOT depend on `size`, so you can
try 128 or 224 or 384 without regenerating anything.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from augment import DATASET_MEAN, DATASET_STD, augment as _augment, to_tensor_chw
from preprocess import FRAME_H, FRAME_W, OUT, TARGET_COLS, load_frame


class FrameSet:
    """One split (train, val or test). Opens photos as the model asks for them.

    split       'train', 'val' or 'test'
    scheme      'split_video' holds out whole videos (normal),
                'split_doc' holds out whole documents (harder, more honest)
    size        (width, height) to give the model
    mode        'rgb', 'gray' or 'clahe'
    augment     random tweaks. TRAINING ONLY -- this errors on val/test.
    stride      keep every Nth photo of each video (1 = all 24,889)
    decode_size how big the photo is while being tweaked. Doing it at full
                1920x1080 takes 120 ms per photo and leaves the GPU waiting;
                480x270 is 10x faster and loses nothing you'd see at 224.
    """

    def __init__(self, split: str, scheme: str = "split_video",
                 size: tuple[int, int] = (224, 224), mode: str = "rgb",
                 augment: bool = False, stride: int = 3,
                 manifest: str = "manifest.csv", seed: int = 0,
                 decode_size: tuple[int, int] = (480, 270)):
        df = pd.read_csv(OUT / manifest)
        df = df[df.usable & (df[scheme] == split)]
        df = df.sort_values(["video_id", "frame_index"])
        df = df[df.groupby("video_id").cumcount() % stride == 0].reset_index(drop=True)

        if augment and split != "train":
            raise ValueError("augment=True on a non-train split leaks jitter into eval")
        if augment and mode != "rgb":
            raise ValueError("augment is only wired up for mode='rgb'")

        self.df, self.size, self.mode, self.augment = df, size, mode, augment
        self.decode_size = decode_size
        self.paths = df.image_path.to_numpy()
        self.y = df[TARGET_COLS].to_numpy(np.float32)
        self.groups = df.video_id.to_numpy()      # pass to GroupKFold
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        y = self.y[i]
        if self.augment:
            dw, dh = self.decode_size
            img = load_frame(self.paths[i], self.decode_size, "rgb")
            # Labels are 0-1 fractions, so switching sizes is just a multiply.
            quad = y.reshape(4, 2) * [dw, dh]
            img, quad = _augment(img, quad, self.rng)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            y = (quad / [dw, dh]).ravel().astype(np.float32)
        else:
            img = load_frame(self.paths[i], self.size, self.mode)
        return to_tensor_chw(img, DATASET_MEAN, DATASET_STD), y

    def as_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Everything as one big flat table, for scikit-learn. Only do this at
        small sizes -- otherwise use the ready-made .npz files."""
        X = np.stack([self[i][0].ravel() for i in range(len(self))])
        return X, self.y


class TorchFrameSet:
    """Wraps a FrameSet so PyTorch's DataLoader can use it."""

    def __init__(self, base: FrameSet):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        import torch
        x, y = self.base[i]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def load_arrays(name: str = "img_128x72_rgb", scheme: str = "split_video"):
    """The easy way in: load one file, get train/val/test ready to use.

        parts, groups, raw = load_arrays("img_128x72_rgb")
        X_tr, y_tr = parts["train"]

    groups says which video each row came from
    """
    d = np.load(OUT / "arrays" / f"{name}.npz", allow_pickle=True)
    X, y, s, v = d["X"], d["y"], d[scheme], d["video_id"]
    parts = {p: (X[s == p], y[s == p]) for p in ("train", "val", "test")}
    groups = {p: v[s == p] for p in ("train", "val", "test")}
    return parts, groups, d
