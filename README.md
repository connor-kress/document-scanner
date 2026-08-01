# Papers, Please

Locating the four corners of a document in smartphone camera frames, as a
multi-output regression problem. ML term project.

**Connor Kress** · **Joshua Rivera** · **Cole Morgan** · **Samy Katibi** · **Vinh Nguyen**

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -e .                # makes `src/` importable everywhere -- no path hacks
python -m ipykernel install --user --name papers-please --display-name "Papers, Please (.venv)"
```

That last line registers this venv as a notebook kernel. **In VS Code or Jupyter,
pick "Papers, Please (.venv)" in the kernel selector** (top right of the
notebook). If you get `ModuleNotFoundError: No module named 'preprocess'`, the
notebook is running some other Python — that's the fix.

The dataset is not in git (1.2 GB). Put the SmartDoc 2015 Challenge 1 frames at
`data/raw/frames/`, then:

```bash
python scripts/build_data.py all      # ~6 min on 10 cores
```

That writes `data/processed/` — the manifest, the model-ready arrays, and a
resized frame tree. Everything under `data/` is regenerable and git-ignored.

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

## Who owns what

| Area | Owner |
|---|---|
| Preprocessing, augmentation, data hand-off | Cole |
| EDA and visualisation | Joshua, Samy |
| Model implementation, CV, tuning | Vinh |
| Cloud GPU, results interpretation | Connor |

## Conventions

- **`src/` is shared, notebooks are personal.** If you write something another
  person will call, it belongs in `src/`. Numbered notebooks have one owner —
  two people editing one `.ipynb` produces a merge conflict you cannot resolve
  by hand.
- **Score every model through `metrics.evaluate()`.** If ridge and the CNN
  use different IoU implementations the comparison table in the report is wrong
  and nothing will make that obvious.
- **Never split randomly.** Consecutive video frames are near-duplicates. Use
  the `split_video` / `split_doc` columns, or `GroupKFold` with
  `groups=groups["train"]`.
- **Regenerate, don't commit.** Nothing in `data/` goes in git.
