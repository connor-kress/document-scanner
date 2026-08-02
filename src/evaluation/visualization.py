"""Prediction overlays and training curves for model comparison."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import corner_error_px, iou_batch
from preprocess import FRAME_H, FRAME_W, FRAMES, denormalise


def _load_image(rel_path: str, size: tuple[int, int] = (192, 108)) -> np.ndarray:
    image = cv2.imread(str(FRAMES / rel_path.replace("/", "\\")), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def plot_prediction_grid(
    image_paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str | Path,
    n_cols: int = 4,
) -> None:
    n = len(image_paths)
    if n == 0:
        return
    display_w, display_h = 192, 108
    scale = np.array([display_w / FRAME_W, display_h / FRAME_H])
    gt = denormalise(y_true)
    pred = denormalise(y_pred)
    iou = iou_batch(y_true, y_pred)
    err = corner_error_px(y_true, y_pred)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        ax.imshow(_load_image(str(image_paths[i]), (display_w, display_h)))
        gt_pts = gt[i] * scale
        pred_pts = pred[i] * scale
        for pts, color, linestyle in ((gt_pts, "#2a78d6", "-"), (pred_pts, "#eb6834", "--")):
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, ls=linestyle)
        ax.set_title(f"IoU {iou[i]:.2f} | err {err[i]:.0f}px", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_csv: str | Path, out_path: str | Path) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["val_iou"])
    axes[1].set_title("Validation IoU")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)