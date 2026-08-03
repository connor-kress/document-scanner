# Training and evaluation results guide

This document travels with the exported result bundle. It explains what was
run, how to interpret the files and metrics, and what conclusions are supported
by this experiment.

## Executive summary

The complete pipeline succeeded on 2026-08-03 using one NVIDIA B200 on
HiPerGator. Preprocessing, hyperparameter search, full-data retraining, and
held-out evaluation all completed with exit code 0.

The RGB CNN is the strongest model on the held-out `split_video` test set:

| Model | Corner error | Mean IoU | Median IoU | IoU >= 0.90 |
|---|---:|---:|---:|---:|
| Ridge | 77.62 px | 0.7247 | 0.7550 | 6.54% |
| MLP | 57.05 px | 0.7951 | 0.8329 | 20.62% |
| Grayscale CNN | 26.36 px | 0.8918 | 0.9111 | 58.76% |
| **RGB CNN** | **18.94 px** | **0.9202** | **0.9291** | **77.78%** |

The headline conclusion is that the learned CNNs substantially outperform the
linear and dense baselines, and RGB improves over grayscale in this run. These
numbers apply to a video-grouped split, not unseen-document generalization.

## Experiment design

The processed dataset contains 8,346 frames sampled with stride 3 from 24,889
usable SmartDoc frames across 150 videos and 30 document models.

| Split | Frames | Purpose |
|---|---:|---|
| Train | 5,015 | Fit search candidates and final selected models |
| Validation | 1,648 | Select hyperparameters and checkpoints |
| Test | 1,683 | One final comparison after selection |

All reported headline results use `split_video`. Entire videos are assigned to
one split, preventing adjacent near-duplicate frames from leaking across
train/validation/test.

The neural-model searches used a deterministic, video-stratified 20% subset of
the training split. Validation still used the complete validation split. The
best search configuration was then retrained with `train_fraction: 1.0` before
the test set was evaluated.

Search size:

- MLP: 16 candidates, then one full-data winner.
- Grayscale CNN: 8 candidates, then one full-data winner.
- RGB CNN: 8 candidates, then one full-data winner.
- Ridge: the repository's full Ridge/PCA/alpha comparison.

## Selected configurations

| Model | Selected configuration |
|---|---|
| Ridge | grayscale CLAHE, no PCA, alpha 100 |
| MLP | PCA 256; hidden `[512, 256]`; dropout `[0.3, 0.2]`; batch 128; LR 0.001 |
| Gray CNN | 384x216 gray; batch 16; dropout 0.3; LR 0.0003 |
| RGB CNN | 384x216 RGB; batch 32; dropout 0.2; LR 0.001 |

The search CSVs are the authority for candidate-level validation results. The
final model `config.yaml` files are the authority for what was retrained.

## How to read the metrics

### Corner error in pixels

`corner_err_px` is the mean Euclidean distance between predicted and true page
corners after coordinates are converted back to the original 1920x1080 frame.
It averages four corner errors per frame, then all frames. Lower is better.

This is the easiest geometric metric to explain: the RGB CNN's predicted
corners are about 18.94 pixels from the annotations on average.

### Normalized corner error

`corner_err_norm` divides mean corner error by the original frame diagonal.
Lower is better. It supports comparisons across different source resolutions,
although all frames in this experiment share one resolution.

### Intersection over Union

IoU measures overlap between the predicted and true document quadrilaterals.
It ranges from 0 to 1; higher is better:

- `iou_mean`: average over all test frames.
- `iou_median`: the middle test frame, less sensitive to a few severe failures.
- `iou>=0.50`, `iou>=0.75`, `iou>=0.90`: fraction of frames meeting each
  quality threshold.

The RGB CNN reaches IoU >= 0.90 on 77.78% of held-out frames and IoU >= 0.75 on
99.47% of them.

### Inference time

`inference_time_ms` is model-only, batched inference measured on the B200. For
PyTorch models, batches were already loaded and transferred to the GPU before
timing. It excludes JPEG decode, resize, normalization, filesystem access,
model loading, and host-to-device transfer. Do not present it as camera-to-result
latency or compare it directly with a mobile deployment.

For deployment latency, benchmark the complete preprocessing + model pipeline
on the target device with batch size 1.

## Per-background interpretation

`by_background.csv` uses this ID mapping:

| ID | Dataset background |
|---:|---|
| 0 | background01 |
| 1 | background02 |
| 2 | background03 |
| 3 | background04 |
| 4 | background05 |

The RGB CNN is best in every background. Its mean IoU ranges from 0.8892 on
background05 to 0.9331 on background04. Background05 is therefore the clearest
remaining condition to inspect qualitatively. Per-background rows are useful
for diagnosing domain difficulty, but they are not separate independently
selected test sets.

## Bundle layout

```text
document-scanner-results-20260803/
├── README_RESULTS.md              this guide
├── TRAINING_RUNBOOK.md            cluster workflow and measured timing
├── results/                       every search/final/evaluation artifact
├── logs/                          complete Slurm stdout/stderr
├── configs/                       base YAML configurations
├── slurm/                         submitted batch scripts
├── scripts/                       orchestration/training/evaluation entrypoints
├── metadata/manifest_subset.csv   labels, groups, and split assignments
└── provenance/                    commit, environment, file manifest/checksums
```

Important result directories contain a `COMPLETE` marker. Incomplete or failed
runs must not be selected as final models.

The authoritative final evaluation is:

```text
results/evaluation/20260803T082352.568466Z/
├── COMPLETE
├── final_comparison.csv
└── by_background.csv
```

Final trained models:

```text
results/ridge/20260803T023507.175011Z/
results/mlp/20260803T023813.462718Z/
results/cnn_gray/20260803T051641.941832Z/
results/cnn_rgb/20260803T080348.581259Z/
```

`best.pt` is a PyTorch state dictionary, not a self-contained application.
Reconstruct the architecture and preprocessing from the adjacent `config.yaml`
and repository code. Ridge is serialized as `best_pipeline.joblib`; only load
joblib/pickle artifacts from trusted sources.

## Quick local analysis

After extracting the archive:

```bash
cd document-scanner-results-20260803
python - <<'PY'
import pandas as pd

root = "results/evaluation/20260803T082352.568466Z"
comparison = pd.read_csv(f"{root}/final_comparison.csv")
by_background = pd.read_csv(f"{root}/by_background.csv")
print(comparison.sort_values("iou_mean", ascending=False).to_string(index=False))
print(by_background.pivot(index="background", columns="model", values="iou_mean"))
PY
```

Training curves are ordinary CSV files. For example:

```python
import pandas as pd

history = pd.read_csv(
    "results/cnn_rgb/20260803T080348.581259Z/history.csv"
)
history.plot(x="epoch", y=["train_loss", "val_loss"])
history.plot(x="epoch", y="val_iou")
```

## What is and is not included

Included:

- all 32 neural search trials and their histories/checkpoints/configs;
- Ridge experiments and final serialized pipeline;
- all selected full-data model checkpoints;
- final aggregate and per-background evaluation;
- logs, configs, scripts, split manifest, and provenance.

Excluded:

- the 1 GB release archive;
- 24,889 raw frames;
- resized/processed image arrays and frame trees;
- the Python virtual environment.

Those excluded inputs are reproducible from the official dataset and code and
are not training/evaluation results.

## Limits and responsible claims

- Results are for `split_video`. They do not establish performance on document
  models absent from training. Run and report `split_doc` separately for that.
- Hyperparameters and checkpoints were selected using validation IoU. The test
  set was used for the final table, not search selection.
- Only one random seed (`42`) is represented. The table does not quantify
  run-to-run variance.
- SmartDoc is a fixed benchmark. Performance on modern phones, new page types,
  motion blur, occlusion, or other capture conditions requires new evaluation.
- Background-level differences describe this dataset and should not be treated
  as universal causal effects.

## Transfer and integrity verification

From a local terminal, copy both the archive and checksum sidecar:

```bash
rsync -avP \
  <hipergator-user>@<hipergator-login-host>:/blue/thai/vinhnguyen1/document-scanner/exports/document-scanner-results-20260803.tar.gz* \
  .
```

Then verify and extract locally:

```bash
sha256sum --check document-scanner-results-20260803.tar.gz.sha256
tar -xzf document-scanner-results-20260803.tar.gz
```

On macOS, use `shasum -a 256` and compare its output with the digest in the
`.sha256` file if `sha256sum` is unavailable.
