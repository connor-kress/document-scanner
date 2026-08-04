# Preprocessing


What the data pipeline does, and the things the rest of the team has to carry
forward.

---

## First: get the data

The normal `python scripts/orchestrate.py` workflow downloads and preprocesses
the data automatically. To prepare only the data, run:

```bash
python scripts/build_data.py download  # raw data only
python scripts/build_data.py all       # raw and processed data
```

The commands verify the published checksum and reuse valid existing data.

Full steps in [START_HERE.md](START_HERE.md).

---

## What it does

The raw data is 24,889 photos from 150 short videos, plus a file listing where
each page's 4 corners are. Five steps turn that into something a model can use.

**1. Check every photo.** Open all 24,889, confirm nothing is broken, and
measure how blurry, bright and colourful each one is. Nothing was broken.

**2. Fix the corner labels.** The one real problem — see below.

**3. Drop near-duplicates.** Photos next to each other in a video are almost
identical (the page moves ~7 px out of 1920). Keeping every 3rd photo gives
8,346 real examples instead of a fake 24,889.

**4. Turn the corners into 8 numbers from 0 to 1.** Fractions across and down
the photo, so the labels work at any image size.

**5. Split by video.** Whole videos go to train, val or test — never split a
video across two piles.

---

## The problem we found

The original labels name corners after the **page**: `tl` means the page's
top-left. All 4,169 photos in background04 were filmed with the tablet turned
sideways, so their `tl` sits in a different part of the photo than everyone
else's.

Left alone, the model gets contradictory training data — the same output slot
means "left side" for most photos and "right side" for background04.

**Fixed** by renaming the corners after the **photo** instead: corner 0 is
always the one nearest the photo's top-left. The corners themselves never
moved, only their names. `rot_deg` and `doc_corner_at_slot0` still record which
way the page was turned, so nothing is lost.

---

## What you get

Everything lands in `data/processed/`.

| File | What it is |
|---|---|
| `manifest.csv` | one row per photo — labels, measurements, splits |
| `arrays/tab_64x36_clahe.npz` | tiny grayscale photos, for ridge and the MLP |
| `arrays/img_128x72_rgb.npz` | small colour photos, fits in memory, for the CNN |
| `frames_384/` | bigger JPEGs on disk, for a CNN that streams |

Load it:

```python
from dataset import load_arrays
parts, groups, _ = load_arrays("img_128x72_rgb")
X_train, y_train = parts["train"]
```

---

## ⚠️ Three things you must carry forward

### 1. Score everything through `metrics.py`

```python
from metrics import evaluate
evaluate(y_true, y_pred)
```

Do not write your own IoU or error function. If ridge and the CNN are scored
differently, the comparison table in the report is wrong and nothing will tell
you.

### 2. Beat the benchmark, and always show it

The page is nearly always near the middle of the photo, so guessing the **same
answer every time** already scores well. Report your model next to this:

| model | corner error | IoU | IoU > 0.9 |
|---|---|---|---|
| constant (learns nothing) | 121 px | 0.598 | 0% |
| ridge | 78 px | 0.722 | 7% |
| your model | ? | ? | ? |

Run it any time: `python scripts/run_baseline.py`

**MSE on the 0-1 labels is misleading.** The constant guess scores 0.004, which
looks perfect and means nothing. Report corner error in pixels and IoU.

### 3. Never split randomly

Use the `split_video` or `split_doc` columns, or `GroupKFold` with the groups
we provide:

```python
GroupKFold(5).split(X_train, y_train, groups=groups["train"])
```

A random split puts near-identical photos in both train and test. 91% of test
photos would have a twin in training, and your score would be fiction.

---

## Also worth knowing

- **Convert to pixels before measuring anything.** x was divided by 1920 and y
  by 1080, so distances on the 0-1 numbers are wrong. `metrics.py` handles this
  for you; if you go around it, call `denormalise()` first.
- **Image size is free.** Labels are fractions, so 128, 224 and 384 all work
  without regenerating anything.
- **Augmentation is training only.** `FrameSet(..., augment=True)` errors on
  val and test on purpose.
- **background04 is in all three splits** on purpose — it's the only source of
  sideways pages.

Charts backing all of this are in `reports/figures/`.

Questions → Cole.
