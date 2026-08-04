# Papers, Please

Locating the four corners of a document in smartphone camera frames, as a
multi-output regression problem. ML term project.

**Connor Kress** · **Joshua Rivera** · **Cole Morgan** · **Samy Katibi** · **Vinh Nguyen**

---

## Setup

From a Python 3.11+ environment, install the project and run the complete
pipeline:

```bash
python -m pip install -e .
python scripts/orchestrate.py
```

The orchestrator downloads and verifies the public SmartDoc dataset when it is
missing, preprocesses it, searches model configurations, trains each selected
model on the full training split, and runs final evaluation. Existing valid
data is reused. Use `python scripts/orchestrate.py --workers 1` only if
preprocessing runs out of memory.

To prepare the data without training models, run:

```bash
python scripts/build_data.py download  # raw data only
python scripts/build_data.py all       # raw and processed data
```

Both commands verify and reuse valid existing data. Pass `--force` to `all`
only when the processed artifacts need to be rebuilt.

## Using the data

```python
from dataset import load_arrays

parts, groups, _ = load_arrays("img_128x72_rgb")
X_train, y_train = parts["train"]     # splits already applied
```

**Read [`docs/START_HERE.md`](docs/START_HERE.md) before modelling** — one page,
and it covers the three mistakes that will otherwise cost you a rerun.

## Layout

```
src/              the shared library — import from here, don't copy code out of it
  preprocess/       the data pipeline, one module per job
    paths.py          where things live, frame dimensions, column names
    geometry.py       corner canonicalisation, area, convexity, dewarp
    qc.py             stage 1 — decode every frame, blur/exposure/colour
    manifest.py       stage 2 — features, flags, splits, subsample
    export.py         stage 3 — arrays, resized frame tree, channel stats
    cli.py            the build_data command
  augment.py        jitter that moves the labels with the image
  dataset.py        loaders (in-memory arrays, or streaming with augmentation)
  metrics.py        corner error + IoU  ← every model scores through this
  viz.py            the figure pack
  baselines.py      constant and ridge references
notebooks/        one per person, numbered; don't edit someone else's
scripts/          things you run from the terminal
docs/             the proposal, plus the hand-off write-ups
reports/figures/  committed plots for the presentation
data/             git-ignored: raw frames in, processed artifacts out
```
