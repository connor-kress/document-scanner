#!/usr/bin/env python
"""Prepare data, train every model, and run final evaluation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


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


def pipeline_steps(workers: int) -> list[tuple[list[str], str]]:
    return [
        (
            [sys.executable, "scripts/build_data.py", "all", "--workers", str(workers)],
            "Data download and preprocessing",
        ),
        *training_steps(),
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
    completed = []
    pipeline_started = time.perf_counter()
    for command, description in pipeline_steps(args.workers):
        if not run_command(command, description):
            print(f"\nPipeline stopped at: {description}")
            return 1
        completed.append(description)

    elapsed = time.perf_counter() - pipeline_started
    print("\nPIPELINE COMPLETE")
    for description in completed:
        print(f"PASS: {description}")
    print(f"Total time: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} hours)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
