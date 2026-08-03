# Start here — preprocessed data is ready

One page. Everything you need to start modelling.

## Setup

**1. Install.**

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
python -m ipykernel install --user --name venv --display-name "venv"
```

**2. Download the dataset.** It is 1.3 GB, so it is not in git.

Get **`frames.tar.gz`** from
<https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0>
and extract it into `data/raw/`:

```powershell
mkdir data\raw
tar -xzf frames.tar.gz -C data\raw
```

You should end up with `data/raw/frames/` containing `background01/` …
`background05/` and `metadata.csv.gz`. Check with:

```powershell
python -c "from preprocess import FRAMES; print(FRAMES.exists(), (FRAMES/'metadata.csv.gz').exists())"
```

Both must print `True`.

**3. Build everything.**

```bash
python scripts/build_data.py all      # ~6 min
```

## Load the data (pick one)

**Option A — everything in memory. Use this for ridge, MLP, and a first CNN.**

```python
from dataset import load_arrays

parts, groups, _ = load_arrays("img_128x72_rgb")   # or "tab_64x36_clahe"
X_train, y_train = parts["train"]                  # (5015, 72, 128, 3) uint8, (5015, 8) float32
X_val,   y_val   = parts["val"]
X_test,  y_test  = parts["test"]
```

`X` is uint8 0–255 — divide by 255 (and flatten) for sklearn.

That's it. Splits are already applied and already correct.

**Option B — stream from disk with augmentation. Use this for the real CNN run.**

```python
from dataset import FrameSet, TorchFrameSet
from torch.utils.data import DataLoader

train = FrameSet("train", size=(224, 224), augment=True)
val   = FrameSet("val",   size=(224, 224))

train_dl = DataLoader(TorchFrameSet(train), batch_size=32, shuffle=True, num_workers=8)
val_dl   = DataLoader(TorchFrameSet(val),   batch_size=32, num_workers=8)
```

Yields `(x, y)` where `x` is `(3, 224, 224)` float32, already normalised, and `y` is
`(8,)` float32. `size=` is free — try 128, 224, 384, the labels don't change.

## What y means

8 numbers, all between 0 and 1:

```
[c0_x, c0_y, c1_x, c1_y, c2_x, c2_y, c3_x, c3_y]
```

Four corners, going clockwise, **starting from the corner nearest the top-left
of the photo**. `0.5, 0.5` means dead centre. Multiply x by 1920 and y by 1080
to get pixels.

## Three rules

**1. Never compute error on the raw y.** x and y were divided by different
numbers (1920 vs 1080), so distances in that space are distorted. Convert first:

```python
from preprocess import denormalise, quad_iou
pred_px = denormalise(y_pred)     # (N, 4, 2) in real pixels
true_px = denormalise(y_true)
err = np.linalg.norm(pred_px - true_px, axis=2).mean(1)    # px per frame
```

**2. Always compare against the constant baseline.** The page is always roughly
centred, so predicting one fixed quad for everything already scores IoU 0.598.
A low MSE means nothing on its own. Run `python scripts/run_baseline.py` for the
numbers to beat:

| model | corner error | IoU |
|---|---|---|
| constant | 121.3 px | 0.598 |
| ridge Î±=100 | 77.6 px | 0.722 |

**3. Group by video for cross-validation.** Consecutive frames are near
duplicates, so a random split leaks. `groups` is keyed by split and lines up
row-for-row with that split's `X`:

```python
from sklearn.model_selection import GroupKFold
GroupKFold(5).split(X_train, y_train, groups=groups["train"])
```

## Two split schemes — report both

- `split_video` (default) — holds out whole video clips.
- `split_doc` — holds out whole *documents*; the test pages were never seen.
  Harder, and the better headline number.

```python
parts, groups, _ = load_arrays("img_128x72_rgb", scheme="split_doc")
```

---

# How to Run the Complete Training Pipeline

After setup (steps 1–2 above), follow this sequence to run preprocessing, smoke tests, and full model training.

## Step 1: Generate Preprocessed Arrays (~10 min)

```bash
# From workspace root
python scripts/build_data.py all --workers 2
```

This generates:
- `data/processed/manifest.csv` — all frame metadata and split assignments
- `data/processed/arrays/tab_64x36_clahe.npz` — grayscale 64×36 frames
- `data/processed/arrays/img_128x72_rgb.npz` — RGB 128×72 frames  
- `data/processed/frame_tree/` — raw 384×216 frames for CNN training

**If memory is tight:** reduce workers further with `--workers 1`.

## Step 2: Verify Data & Run Smoke Tests (~5 min)

Open `notebooks/01_preprocessing.ipynb` in Jupyter and run these cells to validate before training:

1. **Cell 35** — Load arrays and verify shapes match expectations:
   - `tab_64x36_clahe.npz` → `(5015, 64, 36)` grayscale uint8, `(5015, 8)` float32 targets
   - `img_128x72_rgb.npz` → `(5015, 128, 72, 3)` RGB uint8, `(5015, 8)` float32 targets

2. **Cell 36** — Validate train/val/test split disjointness (no data leakage by video_id)

If both pass, you have valid preprocessed data. Proceed to modeling.

## Step 3: Train Ridge Regression (~3 min) 

**Baseline model**: uses PCA dimensionality reduction and optional sharpening.

```bash
python scripts/train_ridge.py
```

Saves to a unique run directory: `results/ridge/<UTC timestamp>/`
- `best_pipeline.joblib` — trained Ridge model
- `config.yaml` — selected hyperparameters
- `experiments.csv` — validation metrics and timing for every candidate

Expected output:
```
Ridge Grid Search Results
=========================
Best config: pca_components=256, alpha=1.0, use_pca=True, sharpened=False
Val IoU: 0.7234
```

## Step 4: Train MLP (~30–60 min, depending on --compare-* flags)

**Dense multi-layer perceptron** on PCA-reduced grayscale input.

```bash
# Main comparison (2 models: gray_pca256, gray_nopca)
python scripts/train_mlp.py

# Or with optional color & sharpening experiments:
python scripts/train_mlp.py --compare-color --compare-sharpening

# Or train one final configuration:
python scripts/train_mlp.py --config configs/mlp.yaml
```

Saves to a unique run directory: `results/mlp/<UTC timestamp>/`
- `best.pt` — trained MLP checkpoint  
- `best_pca.joblib` — PCA transformer (if used)  
- `config.yaml` — hyperparameters  
- `trials/<name>/history.csv` — comparison-mode curves; YAML runs use `history.csv`

Expected output:
```
Final Selection: mlp_gray_pca256
Val IoU: 0.7845 (better than Ridge)
```

**Note:** With `--compare-color` and `--compare-sharpening`, extra experiments are logged but only main mlp_gray_* variants are selected for final evaluation.

## Step 5: Train CNN (Grayscale) (~15–30 min)

**Convolutional neural network** on 384×216 grayscale input. Includes overfit test first.

```bash
python scripts/train_cnn.py --config configs/cnn_gray.yaml --overfit-test
```

Saves to: `results/cnn_gray/<UTC timestamp>/`
- `best.pt` — best CNN checkpoint by validation IoU  
- `history.csv` — epoch-by-epoch training curves

**Overfit test** (first step): trains on 32 random images. If loss → 0 and passes in ~20 sec, the model architecture is sound.

**Full training** follows if overfit test passes. Expected final validation IoU: **0.82+**

## Step 6: Train CNN (RGB) (~15–30 min)

**Same architecture as grayscale, but with RGB input** for color sensitivity comparison.

```bash
python scripts/train_cnn.py --config configs/cnn_rgb.yaml --overfit-test
```

Saves to: `results/cnn_rgb/<UTC timestamp>/`

Expected validation IoU: comparable to grayscale (validates that color doesn't help unfairly).

## Step 7: Final Evaluation (~3 min)

**Held-out test set evaluation** on all 4 best models:
- Ridge (grayscale + PCA)
- MLP (grayscale + PCA)
- CNN (grayscale)
- CNN (RGB)

```bash
python scripts/final_eval.py
```

The newest successfully completed run for each model family is selected automatically.

Outputs:
- `results/evaluation/<UTC timestamp>/final_comparison.csv` — 9 metrics per model
- `results/evaluation/<UTC timestamp>/by_background.csv` — per-background breakdown

Expected output:
```
Final Comparison Results
========================
Model         | Corner Error (px) | IoU Mean | Inference Time (ms)
ridge         | 77.6              | 0.722   | 0.8
mlp           | 61.2              | 0.780   | 2.3
cnn_gray      | 48.9              | 0.821   | 15.4
cnn_rgb       | 49.1              | 0.823   | 16.2
```

## Quick Reference: Full Run Command (Sequential)

```bash
# 1. Preprocess (~10 min)
python scripts/build_data.py all --workers 2

# 2. Then in Jupyter, run notebook cells 35–36 to verify

# 3. Train all models sequentially (~1.5 hours total)
python scripts/train_ridge.py && \
python scripts/train_mlp.py --compare-color --compare-sharpening && \
python scripts/train_cnn.py --config configs/cnn_gray.yaml --overfit-test && \
python scripts/train_cnn.py --config configs/cnn_rgb.yaml --overfit-test && \
python scripts/final_eval.py

# 4. View the path printed by final_eval.py
```

## Hyperparameter Search

Broad searches are generated in Python with `ParameterGrid`. Edit `MLP_GRID`
or `CNN_GRID` in `scripts/grid_search.py`, then use a YAML file as the base:

```bash
python scripts/grid_search.py --model mlp --base-config configs/mlp.yaml
python scripts/grid_search.py --model cnn --base-config configs/cnn_gray.yaml

# Override the default 20% search subset when needed:
python scripts/grid_search.py --model cnn --base-config configs/cnn_gray.yaml \
  --train-fraction 0.1 --subset-seed 42

# Automatically train the winner on the full training split:
python scripts/grid_search.py --model mlp --base-config configs/mlp.yaml \
  --train-best
```

Every candidate gets its own timestamped run. Search metrics and a full-data
`best.yaml` are saved under `results/search_<model>/<UTC timestamp>/`. Train
that selected configuration normally before final evaluation:

```bash
python scripts/train_mlp.py --config results/search_mlp/<timestamp>/best.yaml
```

Alternatively, `--train-best` performs this full-data run automatically after
the search completes.

Grid search uses a deterministic 20% training subset by default. Sampling is
performed independently within every training video, so all videos remain
represented, and every candidate for that model receives exactly the same
rows. Validation inference and loss continue to use the full validation split;
the existing per-epoch IoU calculation uses 500 evenly spaced validation rows.
The subset fraction and seed are recorded in each candidate's config and in
the search CSV. Set `--train-fraction 1.0` for a full-data search. Regular
YAML-driven training and generated `best.yaml` files use
`train_fraction: 1.0`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `MemoryError` during preprocessing | Use `--workers 1` in `build_data.py` |
| `FileNotFoundError: *.npz` | Verify preprocessing completed and arrays exist in `data/processed/arrays/` |
| CNN overfit test hangs | Check image size mismatch (should be 384×216) |
| Training very slow | Reduce `batch_size` in configs, or check GPU availability with `torch.cuda.is_available()` |
| "Split mismatch" error at final eval | All models must be trained on same split (default: split_video) |

## Expected Runtimes

| Step | Time | Hardware |
|------|------|----------|
| Preprocess all | 10 min | CPU, 4+ threads |
| Ridge grid search | 3 min | CPU |
| MLP (no flags) | 15 min | GPU recommended |
| MLP (--compare-color --compare-sharpening) | 45 min | GPU recommended |
| CNN grayscale (overfit + full) | 20 min | GPU required |
| CNN RGB (overfit + full) | 20 min | GPU required |
| Final evaluation | 3 min | GPU |
| **Total (full pipeline)** | **~2 hours** | GPU + CPU |

**GPU (CUDA):** ~8GB VRAM recommended. Training will use CPU if unavailable but will be 10–50× slower.

---

## Running via Jupyter Notebook (Recommended for Monitoring)

All steps above are also available in `notebooks/01_preprocessing.ipynb` (cells 32–48). You can run them interactively and inspect results cell-by-cell:

```python
# In notebook cell, after preprocessing:
from dataset import load_arrays
parts, groups, _ = load_arrays("img_128x72_rgb")
print(f"Train: {parts['train'][0].shape}, Val: {parts['val'][0].shape}, Test: {parts['test'][0].shape}")
# Output: Train: (5015, 128, 72, 3), Val: (1287, 128, 72, 3), Test: (1286, 128, 72, 3)
```

This approach lets you inspect intermediate results, plot training curves, and debug failures more easily than running scripts in isolation.
