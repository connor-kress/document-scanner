"""Dataclass configs for each model family + YAML I/O helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RidgeConfig:
    name: str = "ridge_pca256"
    model_type: str = "ridge"
    input_source: str = "tab_64x36_clahe"
    scheme: str = "split_video"
    pca_components: int = 256
    alpha: float = 1.0
    use_pca: bool = True
    sharpened: bool = False
    random_seed: int = 42


@dataclass
class MLPConfig:
    name: str = "mlp_pca256"
    model_type: str = "mlp"
    input_source: str = "tab_64x36_clahe"
    scheme: str = "split_video"
    pca_components: int = 256
    use_pca: bool = True
    sharpened: bool = False
    preprocessing: str = "none"
    hidden: list = field(default_factory=lambda: [256, 128])
    dropout: list = field(default_factory=lambda: [0.3, 0.2])
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 12
    loss: str = "smooth_l1"
    random_seed: int = 42
    train_fraction: float = 1.0
    subset_seed: int = 42


@dataclass
class CNNConfig:
    name: str = "cnn_gray"
    model_type: str = "cnn"
    color_mode: str = "gray"   # 'gray' or 'rgb'
    scheme: str = "split_video"
    image_size: list = field(default_factory=lambda: [384, 216])
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    loss: str = "smooth_l1"
    max_epochs: int = 60
    patience: int = 12
    random_seed: int = 42
    train_fraction: float = 1.0
    subset_seed: int = 42


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)
