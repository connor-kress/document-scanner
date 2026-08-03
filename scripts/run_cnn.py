#!/usr/bin/env python
"""Train the grayscale or RGB CNN using a shared architecture and split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dataset import FrameSet, TorchFrameSet
from models.cnn import DocumentCNN
from training.trainer import train
from utils.subset import grouped_subset_indices
from utils.config import CNNConfig, load_yaml, save_yaml
from utils.runs import run_directory
from utils.seed import seed_everything


def overfit_test(cfg: CNNConfig, n_samples: int = 32, max_epochs: int = 200) -> None:
    seed_everything(cfg.random_seed)
    mode = "rgb" if cfg.color_mode == "rgb" else "gray"
    in_channels = 3 if mode == "rgb" else 1
    train_set = FrameSet("train", scheme=cfg.scheme, size=tuple(cfg.image_size), mode=mode, augment=False)
    subset = Subset(TorchFrameSet(train_set), list(range(min(n_samples, len(train_set)))))
    loader = DataLoader(subset, batch_size=n_samples, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentCNN(in_channels=in_channels, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.SmoothL1Loss()
    for _ in range(max_epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    if float(loss.item()) >= 0.005:
        raise RuntimeError(f"overfit test failed; final loss={loss.item():.6f}")


def main(config_path: Path, overfit: bool, result_model_type: str | None = None) -> None:
    raw = load_yaml(config_path)
    cfg = CNNConfig(**raw)
    seed_everything(cfg.random_seed)
    mode = "rgb" if cfg.color_mode == "rgb" else "gray"
    with run_directory(PROJECT / "results", result_model_type or f"cnn_{mode}") as result_dir:
        save_yaml(cfg, result_dir / "config.yaml")
        if overfit:
            overfit_test(cfg)
        in_channels = 3 if mode == "rgb" else 1
        train_set = FrameSet("train", scheme=cfg.scheme, size=tuple(cfg.image_size), mode=mode, augment=False)
        val_set = FrameSet("val", scheme=cfg.scheme, size=tuple(cfg.image_size), mode=mode, augment=False)
        train_indices = grouped_subset_indices(train_set.groups, cfg.train_fraction, cfg.subset_seed)
        torch_train_set = TorchFrameSet(train_set)
        train_data = torch_train_set if cfg.train_fraction == 1 else Subset(torch_train_set, train_indices.tolist())
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(TorchFrameSet(val_set), batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        model = DocumentCNN(in_channels=in_channels, dropout=cfg.dropout)
        train(
            model,
            train_loader,
            val_loader,
            result_dir=result_dir,
            max_epochs=cfg.max_epochs,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            loss=cfg.loss,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overfit-test", action="store_true")
    parser.add_argument("--result-model-type", help=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args.config, args.overfit_test, args.result_model_type)
