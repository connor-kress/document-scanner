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
python -m ipykernel install --user --name venv --display-name "venv"
```

That last line registers this venv as a notebook kernel. **In VS Code or Jupyter,
pick "venv" in the kernel selector** (top right of the
notebook). If you get `ModuleNotFoundError: No module named 'preprocess'`, the
notebook is running some other Python — that's the fix.

## Get the data

The dataset is 1.3 GB, so it is not in git. Download it once:

1. Go to the SmartDoc 2015 Challenge 1 release:
   <https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0>
2. Download **`frames.tar.gz`** (~1 GB).
3. Extract it into `data/raw/` so you end up with a `frames` folder:

   ```powershell
   mkdir data\raw
   tar -xzf frames.tar.gz -C data\raw
   ```

   `tar` ships with Windows 10+, macOS and Linux. If the archive extracts as
   loose files instead of a `frames` folder, move them into `data/raw/frames/`
   yourself.

You should end up with exactly this:

```
data/raw/frames/
├── background01/ ... background05/     the 24,889 photos
├── metadata.csv.gz                     the corner labels
├── README.md, LICENCE, VERSION
└── original_datasets_files.txt
```

Check it worked:

```powershell
python -c "from preprocess import FRAMES; print(FRAMES.exists(), (FRAMES/'metadata.csv.gz').exists())"
```

Both should print `True`. Then build everything:

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
