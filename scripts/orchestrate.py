#!/usr/bin/env python
"""
Orchestrate the complete training pipeline.

Runs all model training scripts sequentially:
1. Ridge regression baseline (~3 min)
2. MLP with color and sharpening comparisons (~15-60 min)
3. CNN grayscale with overfit test (~30 min)
4. CNN RGB with overfit test (~30 min)
5. Final evaluation on held-out test set (~3 min)

Total estimated runtime: ~2-2.5 hours on GPU, ~8+ hours on CPU.
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
            [sys.executable, "scripts/run_ridge.py"],
            "Ridge Regression (baseline)"
        ),
        (
            [sys.executable, "scripts/run_mlp.py", "--compare-color", "--compare-sharpening"],
            "MLP Neural Network (with color + sharpening)"
        ),
        (
            [sys.executable, "scripts/run_cnn.py", "--config", "configs/cnn_gray.yaml", "--overfit-test"],
            "CNN Grayscale (overfit test)"
        ),
        (
            [sys.executable, "scripts/run_cnn.py", "--config", "configs/cnn_rgb.yaml", "--overfit-test"],
            "CNN RGB (overfit test)"
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
    print(f"Estimated runtime: ~2 hours (GPU) to 8+ hours (CPU)")
    
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
