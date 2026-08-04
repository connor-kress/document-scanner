# Papers, Please

Locating the four corners of a document in smartphone camera frames, as a
multi-output regression problem. Machine Learning term project.

**Connor Kress** · **Joshua Rivera** · **Cole Morgan** · **Samy Katibi** · **Vinh Nguyen**

---

## Quick Start

```bash
git clone https://github.com/connor-kress/document-scanner.git
cd document-scanner
python -m pip install -e .
python scripts/infer.py samples/sample_1.jpg
```

These commands work in macOS/Linux terminals and Windows PowerShell with
Python 3.11+ available as `python`. If your system uses `python3` instead,
replace `python` with `python3` in both commands.

The first inference run automatically downloads and verifies the pretrained
models, then displays the predicted document corners.

## Training and Data

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

To download the pretrained model release without running the training pipeline:

```bash
python scripts/download_models.py
```

The script verifies the release checksum and installs the Ridge, MLP, grayscale
CNN, RGB CNN, and evaluation artifacts under the git-ignored `models/`
directory. Existing complete downloads are reused; pass `--force` to replace
one.

The default is pinned for reproducibility. Select another model release or
explicitly opt into the newest release with:

```bash
python scripts/download_models.py --version 1.0.0
python scripts/download_models.py --latest
```

Run inference on an image with the released RGB CNN (the highest-performing
model and the default):

```bash
python scripts/infer.py path/to/document.jpg
```

Select another released model, or use the newest completed training run under
`results/`, with:

```bash
python scripts/infer.py path/to/document.jpg --model-type ridge
python scripts/infer.py path/to/document.jpg --model-type cnn-rgb --latest-trained
```

Custom neural weights use their adjacent `config.yaml` by default. The config
can also be selected explicitly; an MLP config that enables PCA requires its
matching PCA artifact:

```bash
python scripts/infer.py path/to/document.jpg --model-type cnn-rgb \
  --weights path/to/model.pt --config path/to/config.yaml
python scripts/infer.py path/to/document.jpg --model-type mlp \
  --weights path/to/model.pt --pca path/to/pca.joblib --config path/to/config.yaml
```

Custom artifact paths and `--latest-trained` require an explicit model type.
Predicted corners are printed in pixels relative to the original image.
The overlay is displayed by default; save it, suppress the window, or do both
with:

```bash
python scripts/infer.py path/to/document.jpg --output prediction.png
python scripts/infer.py path/to/document.jpg --no-show
python scripts/infer.py path/to/document.jpg --no-show --output prediction.png
```

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
models/           git-ignored: downloaded pretrained model artifacts
```
