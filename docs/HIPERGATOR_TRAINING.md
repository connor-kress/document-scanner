# HiPerGator training runbook and status

Last updated: 2026-08-03 (America/New_York)

## Current status

- The repository has been reviewed from raw-data ingestion through final evaluation.
- A workspace-local `.venv` uses HiPerGator's `pytorch/2.7` module (PyTorch
  2.7.1, CUDA 12.8) plus this project's declared Python dependencies.
- The dataset, preprocessing products, Slurm logs, checkpoints, and final CSVs
  are excluded from Git via `data/`, `logs/`, and `results/`.
- Preprocessing, all model searches, full-data retraining, and final held-out
  evaluation completed successfully. Both Slurm jobs exited with code `0:0`.
- Submission IDs are also recorded in the ignored file
  `logs/latest_pipeline.env`.

| Stage | Job ID | Final state | Elapsed | Dependency |
|---|---:|---|---:|---|
| Download + preprocess | `38569072` | `COMPLETED (0:0)` | 3m 27s | none |
| B200 training + evaluation | `38569073` | `COMPLETED (0:0)` | 5h 50m 01s | `afterok:38569072` |

The full pipeline finished at 2026-08-03 04:24:45 EDT. Evaluation is already
complete; it was the final orchestrator stage, not a separate next step.

## What ran

The pipeline was split into two jobs so downloading and CPU preprocessing did
not consume billed B200 time.

1. `slurm/preprocess.sbatch` runs on `hpg-default` with 16 CPUs and 48 GB RAM.
   It calls `scripts/download_smartdoc.sh`, then
   `python scripts/build_data.py all --workers 16`, and finally validates every
   expected export.
2. `slurm/train_b200.sbatch` requested one B200, 8 CPUs, and 64 GB RAM. It ran
   the smoke suite and then `python scripts/orchestrate.py --workers 8`.
3. `orchestrate.py` ran Ridge, the 16-trial MLP search plus full retraining, the
   8-trial grayscale CNN search plus full retraining, the 8-trial RGB CNN search
   plus full retraining, and finally held-out evaluation.

The jobs are connected with Slurm's `afterok` dependency. Training becomes
eligible only if dataset download, extraction, preprocessing, and artifact
validation all exit successfully.

## Dataset handling

The official [SmartDoc v2.0.0 release](https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0)
contains two large archives. This project needs only `frames.tar.gz`, because it
contains the 24,889 video frames and `metadata.csv.gz` ground truth consumed by
`src/preprocess/`. The optional `models.tar.gz` contains clean document-model
images and is not referenced anywhere in this codebase.

Download details verified from the official release API and checksum asset:

| Property | Value |
|---|---|
| Asset | `frames.tar.gz` |
| Exact size | 1,019,404,933 bytes (about 972 MiB) |
| SHA-256 | `3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae` |
| Local archive | `data/downloads/frames-v2.0.0.tar.gz` |
| Extracted root | `data/raw/frames/` |

The download is resumable and idempotent. Before extraction, the script checks
the official SHA-256 and tests the gzip/tar structure. After extraction it
requires `metadata.csv.gz` and exactly 24,889 JPEG frames. A rerun skips a valid,
completed dataset and rebuilds the derived data safely.

Preprocessing performs full image QC, canonicalizes the four corner labels,
creates grouped video/document splits, keeps every third frame within each
video, and exports 8,346 examples as:

- `data/processed/arrays/tab_64x36_clahe.npz` for Ridge and MLP;
- `data/processed/arrays/img_128x72_rgb.npz` for color comparisons;
- `data/processed/frames_384/` and manifests for streaming experiments.

## Submit and monitor

The environment has already been created for the current checkout. From the
repository root, the reproducible setup is:

```bash
module purge
module load pytorch/2.7
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

Submit both jobs with:

```bash
bash slurm/submit_pipeline.sh
```

Monitor without waiting for completion:

```bash
source logs/latest_pipeline.env
squeue -j "${PREPROCESS_JOB_ID},${TRAIN_JOB_ID}"
tail -f "logs/docscan_prep_${PREPROCESS_JOB_ID}.out"
tail -f "logs/docscan_train_${TRAIN_JOB_ID}.out"
```

After either job leaves the queue, inspect accounting and the failure code with:

```bash
sacct -j "${PREPROCESS_JOB_ID},${TRAIN_JOB_ID}" \
  --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES
```

If preprocessing fails, the training job remains blocked by `DependencyNeverSatisfied`.
Fix the cause, then rerun `bash slurm/submit_pipeline.sh`; the dataset download
resumes or skips as appropriate. Do not remove the existing results unless an
intentional clean run is required.

## Measured timing

These are the measured times from jobs `38569072` and `38569073`. Queue time is
not included.

| Stage | Actual elapsed time | Notes |
|---|---:|---|
| Download validation + preprocessing | 3m 27s | cached archive; 24,889-frame QC and 8,346 exports |
| Ridge search | 0.4 min | selected no-PCA Ridge, alpha 100 |
| MLP grid + full retraining | 2.9 min | 16 subset trials; trial 7 selected |
| Grayscale CNN grid + retraining | 178.1 min | trial 4 selected |
| RGB CNN grid + retraining | 167.3 min | trial 5 selected |
| Final held-out evaluation | 1.0 min | all four completed model families |
| **B200 training/evaluation job** | **5h 50m 01s** | orchestrator reported 349.7 min |
| **Compute time end to end** | **5h 53m 28s** | preprocessing plus B200 job |

The preprocessing job has a 2-hour limit and the training job has a 12-hour
limit. The longer GPU allocation accommodates the pulled orchestrator's model
grid searches, full-data retraining, and final evaluation.

## B200 performance expectations

HiPerGator documents `hpg-b200` as its NVIDIA DGX B200 partition and accepts
`--gpus=1` or equivalent GRES syntax for a single GPU. A B200 has 180 GB HBM3e;
this repository's CNN has only 2,890,600 parameters, so capacity is not a
constraint. See the [HiPerGator GPU guide](https://docs.rc.ufl.edu/scheduler/gpu_access/)
and [NVIDIA DGX B200 system specifications](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html).

The B200 will not deliver anything close to its headline large-model throughput
on this workload. The current code trains in float32 without automatic mixed
precision, uses `num_workers=0`, decodes/resizes original JPEGs synchronously,
and calculates validation IoU on the CPU each epoch. Ridge and PCA are also CPU
work. Consequently:

- GPU memory use should be only a small fraction of 180 GB;
- GPU utilization may be bursty or low while the input pipeline and metrics run;
- a B200 should keep CNN compute short, but end-to-end speed may be similar to a
  less powerful GPU unless data loading, PCA reuse, and mixed precision are
  optimized later.

No performance refactor is included in this first run, preserving the exact
experiment definitions in `orchestrate.py`. The Slurm output records the actual
GPU model, software versions, epoch times, early-stopping points, and total
times needed to replace these estimates with measurements.

## Final held-out test results

Evaluation completed successfully using the latest completed run from each
model family:

| Model | Corner error | Mean IoU | Median IoU | IoU >= 0.90 |
|---|---:|---:|---:|---:|
| Ridge | 77.62 px | 0.7247 | 0.7550 | 6.54% |
| MLP | 57.05 px | 0.7951 | 0.8329 | 20.62% |
| Grayscale CNN | 26.36 px | 0.8918 | 0.9111 | 58.76% |
| **RGB CNN** | **18.94 px** | **0.9202** | **0.9291** | **77.78%** |

The RGB CNN is the best model on every headline held-out metric. Its selected
configuration uses batch size 32, learning rate 0.001, dropout 0.2, and the
384x216 RGB input.

Authoritative artifacts:

- `results/evaluation/20260803T082352.568466Z/final_comparison.csv`
- `results/evaluation/20260803T082352.568466Z/by_background.csv`
- `results/ridge/20260803T023507.175011Z/`
- `results/mlp/20260803T023813.462718Z/`
- `results/cnn_gray/20260803T051641.941832Z/`
- `results/cnn_rgb/20260803T080348.581259Z/`

The evaluation run contains `COMPLETE`, so rerunning `final_eval.py` is not
needed unless model artifacts or the held-out split change. The next step is to
review the aggregate/per-background tables, produce report figures, and decide
whether a separate `split_doc` experiment is required for the harder headline
comparison.

These outputs, all checkpoints, histories, logs, the 1 GB archive, and all
processed data remain local to HiPerGator and cannot be added to Git under the
repository's ignore rules. The scripts and this runbook remain tracked so the
run can be reproduced.
