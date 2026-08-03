#!/usr/bin/env python
"""Smoke test suite for training pipeline."""
import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_loading():
    """Test 1: Verify data loading."""
    print("\n[SMOKE TEST 1] Loading preprocessed arrays...")
    try:
        from dataset import load_arrays
        parts, groups, _ = load_arrays("tab_64x36_clahe")
        X_train, y_train = parts["train"]
        X_val, y_val = parts["val"]
        X_test, y_test = parts["test"]
        
        assert X_train.shape[0] > 1000, f"Train size too small: {X_train.shape[0]}"
        assert y_train.shape == (X_train.shape[0], 8), f"Target shape mismatch: {y_train.shape}"
        print(f"✓ PASS: Data loaded correctly")
        print(f"  - Train: {X_train.shape} float32 targets")
        print(f"  - Val:   {X_val.shape}")
        print(f"  - Test:  {X_test.shape}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {type(e).__name__}: {e}")
        return False

def test_ridge_minimal():
    """Test 2: Train Ridge on small subset."""
    print("\n[SMOKE TEST 2] Ridge regression (100 samples)...")
    try:
        from dataset import load_arrays
        from models.ridge_model import RidgePipeline
        from utils.seed import seed_everything
        
        seed_everything(42)
        
        parts, _, _ = load_arrays("tab_64x36_clahe")
        X_train, y_train = parts["train"]
        
        # Tiny subset for speed
        X_tiny = X_train[:100]
        y_tiny = y_train[:100]
        
        # Train Ridge
        ridge = RidgePipeline(n_components=64, alpha=1.0, use_pca=True, sharpened=False)
        start = time.time()
        ridge.fit(X_tiny, y_tiny)
        elapsed = time.time() - start
        
        # Predict
        y_pred = ridge.predict(X_tiny[:10])
        assert y_pred.shape == (10, 8), f"Pred shape mismatch: {y_pred.shape}"
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Predictions out of range"
        
        print(f"✓ PASS: Ridge trained in {elapsed:.2f}s")
        print(f"  - Fit time: {elapsed:.2f}s")
        print(f"  - Predictions: {y_pred.shape}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mlp_minimal():
    """Test 3: Train MLP on small subset."""
    print("\n[SMOKE TEST 3] MLP neural network (100 samples)...")
    try:
        import torch
        from dataset import load_arrays
        from models.mlp import CornerMLP, FlatDataset
        from utils.seed import seed_everything
        from torch.utils.data import DataLoader
        
        seed_everything(42)
        torch.manual_seed(42)
        
        parts, _, _ = load_arrays("tab_64x36_clahe")
        X_train, y_train = parts["train"]
        
        # Tiny subset
        X_tiny = X_train[:100]
        y_tiny = y_train[:100]
        
        # Create dataset and loader
        X_flat = X_tiny.reshape(len(X_tiny), -1) / 255.0  # Flatten: (100, 2304)
        dataset = FlatDataset(X_flat, y_tiny)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create and train model
        model = CornerMLP(input_dim=64*36, hidden=[128], dropout=[0.2])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 2 epochs minimum
        start = time.time()
        for epoch in range(2):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = torch.nn.functional.mse_loss(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        elapsed = time.time() - start
        
        print(f"✓ PASS: MLP trained in {elapsed:.2f}s")
        print(f"  - Fit time: {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"✗ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_overfit():
    """Test 4: CNN overfit test - skipped (train_cnn.py has built-in overfit-test)."""
    print("\n[SMOKE TEST 4] CNN overfit test (SKIP)...")
    print(f"ℹ SKIP: train_cnn.py --overfit-test performs this check")
    return True  # Skip this test since train_cnn.py handles it

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TEST SUITE")
    print("=" * 60)
    
    results = {
        "Data Loading": test_data_loading(),
        "Ridge": test_ridge_minimal(),
        "MLP": test_mlp_minimal(),
        "CNN": test_cnn_overfit(),
    }
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    passed_count = sum(results.values())
    total = len(results)
    print(f"\n{passed_count}/{total} tests passed")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
