#!/usr/bin/env python
"""Run a coarse Python-defined hyperparameter search for an MLP or CNN.

Edit the grids below for broad searches. Use the regular model runners with a
YAML config once the useful ranges have been narrowed down.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from sklearn.model_selection import ParameterGrid


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from utils.config import load_yaml, save_yaml
from utils.runs import run_directory


MLP_GRID = {
    "learning_rate": [1e-3, 3e-4],
    "batch_size": [128, 256],
    "hidden": [[256, 128], [512, 256]],
    "dropout": [[0.2, 0.2], [0.3, 0.2]],
}

CNN_GRID = {
    "learning_rate": [1e-3, 3e-4],
    "batch_size": [16, 32],
    "dropout": [0.2, 0.3],
}


def _model_type(model: str, config: dict) -> str:
    if model == "mlp":
        return "mlp"
    return "cnn_rgb" if config.get("color_mode") == "rgb" else "cnn_gray"


def _run_trial(model: str, config: dict) -> Path:
    model_type = _model_type(model, config)
    candidate_type = f"search_trial_{model_type}"
    model_root = PROJECT / "results" / candidate_type
    existing = set(model_root.iterdir()) if model_root.is_dir() else set()
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as handle:
        config_path = Path(handle.name)
    try:
        save_yaml(config, config_path)
        script = "run_mlp.py" if model == "mlp" else "run_cnn.py"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts" / script),
                "--config",
                str(config_path),
                "--result-model-type",
                candidate_type,
            ],
            cwd=PROJECT,
            check=True,
        )
    finally:
        config_path.unlink(missing_ok=True)
    for candidate in sorted(set(model_root.iterdir()) - existing, reverse=True):
        required = (candidate / "best.pt", candidate / "config.yaml", candidate / "history.csv")
        if (candidate / "COMPLETE").is_file() and all(path.is_file() for path in required):
            if load_yaml(candidate / "config.yaml")["name"] == config["name"]:
                return candidate
    raise RuntimeError(f"could not identify completed run for {config['name']}")


def _training_summary(run_dir: Path) -> dict:
    with open(run_dir / "history.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "epochs_completed": len(rows),
        "epoch_elapsed_seconds": round(sum(float(row["elapsed_s"]) for row in rows), 3),
        "best_val_iou": max(float(row["val_iou"]) for row in rows),
    }


def main(
    model: str,
    base_config_path: Path,
    train_fraction: float = 0.2,
    subset_seed: int = 42,
) -> None:
    if not 0 < train_fraction <= 1:
        raise ValueError("--train-fraction must be in the interval (0, 1]")
    base = load_yaml(base_config_path)
    grid = MLP_GRID if model == "mlp" else CNN_GRID
    model_type = _model_type(model, base)
    rows = []

    with run_directory(PROJECT / "results", f"search_{model_type}") as search_dir:
        for index, overrides in enumerate(ParameterGrid(grid), start=1):
            config = {
                **base,
                **overrides,
                "name": f"{base['name']}_{search_dir.name}_grid_{index:03d}",
                "train_fraction": train_fraction,
                "subset_seed": subset_seed,
            }
            run_dir = _run_trial(model, config)
            rows.append({
                "trial": index,
                "run_dir": str(run_dir),
                "train_fraction": train_fraction,
                "subset_seed": subset_seed,
                **overrides,
                **_training_summary(run_dir),
            })

        with open(search_dir / "experiments.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        best = max(rows, key=lambda row: float(row["best_val_iou"]))
        best_config = load_yaml(Path(str(best["run_dir"])) / "config.yaml")
        best_config.update(name=f"{base['name']}_best", train_fraction=1.0)
        save_yaml(best_config, search_dir / "best.yaml")
        print(f"selected trial {best['trial']} with IoU={best['best_val_iou']:.4f}")
        print(f"full-data config: {search_dir / 'best.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "cnn"), required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--subset-seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model, args.base_config, args.train_fraction, args.subset_seed)
