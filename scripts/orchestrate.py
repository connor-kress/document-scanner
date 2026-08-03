#!/usr/bin/env python
"""
Orchestrate the complete training pipeline.

Runs all model training scripts sequentially:
1. Ridge regression baseline (~3 min)
2. MLP grid search and full-data training of the winner
3. Grayscale CNN grid search and full-data training of the winner
4. RGB CNN grid search and full-data training of the winner
5. Final evaluation on held-out test set
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
        elapsed = time.time() - start
        print(f"\n✓ SUCCESS: {description} completed in {elapsed/60:.1f} min")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n✗ FAILED: {description} failed after {elapsed/60:.1f} min")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    """Run the full training pipeline."""
    project_root = Path(__file__).parent.parent
    # All steps
    steps = [
        (
            [sys.executable, "scripts/train_ridge.py"],
            "Ridge Regression (baseline)"
        ),
        (
            [
                sys.executable,
                "scripts/grid_search.py",
                "--model", "mlp",
                "--base-config", "configs/mlp.yaml",
                "--train-best",
            ],
            "MLP grid search and full-data training"
        ),
        (
            [
                sys.executable,
                "scripts/grid_search.py",
                "--model", "cnn",
                "--base-config", "configs/cnn_gray.yaml",
                "--train-best",
            ],
            "Grayscale CNN grid search and full-data training"
        ),
        (
            [
                sys.executable,
                "scripts/grid_search.py",
                "--model", "cnn",
                "--base-config", "configs/cnn_rgb.yaml",
                "--train-best",
            ],
            "RGB CNN grid search and full-data training"
        ),
        (
            [sys.executable, "scripts/final_eval.py"],
            "Final Evaluation (test set metrics)"
        ),
    ]
    
    print(f"\n{'='*70}")
    print(f"DOCUMENT SCANNER TRAINING ORCHESTRATION")
    print(f"{'='*70}")
    print(f"Project root: {project_root}")
    print(f"Total steps: {len(steps)}")
    print("Neural-network searches use a deterministic 20% training subset")
    
    # Run all steps
    results = {}
    pipeline_start = time.time()
    
    for cmd, description in steps:
        success = run_command(cmd, description)
        results[description] = success
        
        if not success:
            print(f"\n{'='*70}")
            print(f"Pipeline stopped at: {description}")
            print(f"{'='*70}")
            return 1
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    for desc, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {desc}")
    print(f"\nTotal time: {pipeline_elapsed/60:.1f} min ({pipeline_elapsed/3600:.2f} hours)")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
