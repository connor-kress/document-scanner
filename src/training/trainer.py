"""Unified PyTorch training loop for CornerMLP and DocumentCNN.

Every epoch records train loss, val loss, val IoU, val corner error,
learning rate, and wall-clock time in results_dir/history.csv.
The best checkpoint (by val IoU) is saved to results_dir/best.pt.
Training stops when EarlyStopping triggers or max_epochs is reached.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from metrics import evaluate as _base_evaluate
from training.early_stopping import EarlyStopping
from training.checkpoint import ModelCheckpoint


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one val pass. Returns (val_loss, y_true_np, y_pred_np)."""
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        total_loss += criterion(preds, y_batch).item() * len(X_batch)
        all_true.append(y_batch.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return total_loss / len(loader.dataset), y_true, y_pred


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    result_dir: str | Path,
    max_epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    loss: str = "smooth_l1",
    device: torch.device | str | None = None,
    iou_sample: int = 500,
) -> nn.Module:
    """Train model with early stopping, LR scheduling, and checkpointing.

    Parameters
    ----------
    model : nn.Module
        Model to train (CornerMLP or DocumentCNN).
    train_loader / val_loader : DataLoader
        Yield (X, y) batches where y has shape (B, 8).
    result_dir : Path-like
        Directory for history.csv and best.pt.
    iou_sample : int
        Cap on the number of val samples used for IoU (expensive), so each
        epoch stays fast.  Set to None to compute over the full val set.

    Returns
    -------
    nn.Module
        Model with weights restored from the best-IoU checkpoint.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.SmoothL1Loss() if loss == "smooth_l1" else nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    early_stop = EarlyStopping(patience=patience, mode="max")
    ckpt = ModelCheckpoint(result_dir / "best.pt", mode="max")

    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "val_corner_err_px", "val_iou", "learning_rate", "elapsed_s",
    ]
    csv_path = result_dir / "history.csv"
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    print(f"device={device}  result_dir={result_dir}")
    t_run_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = _validate(model, val_loader, criterion, device)

        n_val = len(y_true)
        iou_limit = iou_sample if (iou_sample and iou_sample < n_val) else None
        m = _base_evaluate(y_true, y_pred, thresholds=(0.5, 0.75, 0.9), iou_limit=iou_limit)
        val_iou = m["iou"]
        val_err = m["corner_err_px"]
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        writer.writerow({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_corner_err_px": f"{val_err:.2f}",
            "val_iou": f"{val_iou:.4f}",
            "learning_rate": f"{lr_now:.2e}",
            "elapsed_s": f"{elapsed:.1f}",
        })
        csv_fh.flush()

        saved = ckpt(val_iou, model)
        scheduler.step(val_iou)

        print(
            f"[{epoch:3d}/{max_epochs}]  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"IoU {val_iou:.3f}  err {val_err:.0f}px  "
            f"lr {lr_now:.1e}"
            + ("  *" if saved else "")
        )
        if early_stop(val_iou):
            print(f"  early stop at epoch {epoch} (patience={patience})")
            break

    csv_fh.close()
    elapsed_total = time.perf_counter() - t_run_start
    print(f"done in {elapsed_total / 60:.1f} min  best val IoU={ckpt.best:.4f}")
    ckpt.load_best(model, device)
    return model
