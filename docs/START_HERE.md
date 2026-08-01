# Start here — preprocessed data is ready

One page. Everything you need to start modelling.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
python -m ipykernel install --user --name papers-please --display-name "Papers, Please (.venv)"
python scripts/build_data.py all      # ~6 min, regenerates everything
```

**In a notebook, select the "Papers, Please (.venv)" kernel.** Otherwise Jupyter
runs a different Python and you get `ModuleNotFoundError: No module named
'preprocess'`.

## Get the data (pick one)

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
