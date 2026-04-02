# PPG Heart Rate Estimation

This repository reproduces heart-rate estimation on the `GalaxyPPG` dataset with:

- raw input: Galaxy Watch `PPG.csv`
- reference labels: Polar H10 `HR.csv` or `IBI.csv`
- classical baselines: peak detection and spectral HR
- foundation-model features: PulsePPG and PaPaGei
- downstream regressors: linear, ridge, random forest, gradient boosting

The repository is intended to be runnable without manual code edits. The only external inputs are:

1. the raw `GalaxyPPG` dataset
2. the official pretrained model checkpoints

The minimal model-definition code required to load the checkpoints is already vendored inside this repository under `src/vendor/`.

## Reproducibility status

The repository now supports both:

- end-to-end reproduction from raw `GalaxyPPG` data
- shortest-path reproduction from the saved processed windows and saved feature caches

The train/validation/test split is fixed by `configs/galaxyppg_submission_split.json`, and the training code now uses the saved validation-fold assignments instead of recomputing them ad hoc.

## Environment setup

Tested environment:

- Python `3.13.9`
- Windows
- CPU execution

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU execution, replace the `torch` wheel with the one that matches your CUDA environment.

## What is included in the repository

Key repository contents:

```text
ppg_project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- galaxyppg_submission_split.json
|   `-- submission_protocol.json
|-- src/
|   |-- baseline/
|   |-- data/
|   |-- models/
|   |-- regression/
|   |-- utils/
|   `-- vendor/
|       |-- UPSTREAM.md
|       |-- pulseppg_resnet1d.py
|       |-- papagei_resnet.py
|       `-- resnet1d_shared.py
|-- data/
|   |-- raw/
|   |   `-- GalaxyPPG/
|   |-- processed/
|   `-- summary/
|-- external/
|   `-- .gitkeep
`-- experiments/
```

`external/` is intentionally kept almost empty in version control. It is the local checkpoint root where users place downloaded model weights at the documented paths.

## 1. How to obtain raw data

The raw dataset is not bundled in this repository.

Use the official GalaxyPPG sources:

- official dataset archive: `https://doi.org/10.5281/zenodo.14635823`
- supplementary code repository: `https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code`
- Galaxy Watch logger repository: `https://github.com/Kaist-ICLab/GalaxyPPG-Logger`

After download, extract the dataset so that the repository-local path is:

```text
data/raw/GalaxyPPG/
```

The preprocessing code expects `Meta.csv` directly inside `data/raw/GalaxyPPG/`.

## 2. Exact expected GalaxyPPG raw-data folder structure

This project only needs the files below for preprocessing and label generation:

```text
data/raw/GalaxyPPG/
|-- Meta.csv
|-- P02/
|   |-- Event.csv
|   |-- GalaxyWatch/
|   |   `-- PPG.csv
|   `-- PolarH10/
|       |-- HR.csv
|       `-- IBI.csv
|-- P03/
|   |-- Event.csv
|   |-- GalaxyWatch/
|   |   `-- PPG.csv
|   `-- PolarH10/
|       |-- HR.csv
|       `-- IBI.csv
`-- ...
```

Notes:

- `GalaxyWatch/PPG.csv` is required as model input.
- `PolarH10/HR.csv` is the default reference label source.
- `PolarH10/IBI.csv` is also supported.
- `Event.csv` is used for `ENTER/EXIT` session boundaries.
- The official dataset contains additional files such as `ACC.csv`, `ECG.csv`, `E4/*`, and `GalaxyWatch/HR.csv`; this repository does not need them for the main reported pipeline.
- `P01` is excluded from the fixed split because the official raw release does not provide usable Galaxy Watch PPG plus event annotations for this project.

## 3. How to obtain model weights

Only the pretrained checkpoint files are required. No extra upstream model-code checkout is needed at runtime.

### PulsePPG checkpoint

- source: official PulsePPG Zenodo record `https://doi.org/10.5281/zenodo.17270930`
- expected filename: `checkpoint_best.pkl`
- recommended placement after creating the directory locally:

```text
external/pulseppg/checkpoint_best.pkl
```

Backward-compatible legacy placement is also supported:

```text
external/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl
```

### PaPaGei checkpoint

- source: official PaPaGei Zenodo record `https://zenodo.org/records/13983110`
- expected filename: `papagei_s.pt`
- expected placement after creating the directory locally:

```text
external/papagei-foundation-model/weights/papagei_s.pt
```

## 4. How to obtain required external model code

No extra external model repository checkout is required for feature extraction.

This repository uses the "vendor the minimal required source files" option from the reproducibility checklist. The minimal checkpoint-compatible runtime code is vendored in:

- `src/vendor/pulseppg_resnet1d.py`
- `src/vendor/papagei_resnet.py`
- `src/vendor/resnet1d_shared.py`

Upstream provenance is pinned explicitly in:

- `src/vendor/UPSTREAM.md`

Vendored source provenance:

- PulsePPG upstream repository: `https://github.com/maxxu05/pulseppg`
- PulsePPG upstream commit: `716eaf9cf966e8f76436f2263872ef38b1f90166`
- PulsePPG upstream source file used for vendoring: `pulseppg/nets/ResNet1D/ResNet1D_Net.py`
- PaPaGei upstream repository: `https://github.com/Nokia-Bell-Labs/papagei-foundation-model`
- PaPaGei upstream commit: `0c537dad4d2850e15b724260de820dd68d77f0b0`
- PaPaGei upstream source file used for vendoring: `models/resnet.py`

Because the minimal runtime source is already in `src/vendor/`, another user can clone this repository and run feature extraction without checking out any additional model repository. The only external requirement is downloading the official checkpoint files to the documented paths.

## Fixed split and validation folds

The repository ships a fully fixed split definition in:

- `configs/galaxyppg_submission_split.json`

It defines:

- exact train participants
- exact test participants
- exact validation-fold participant assignments
- random seed `42`

The training code now uses the saved validation folds from this file instead of recomputing them dynamically.

## Preprocessing details

Implemented in:

- `src/data/loader.py`
- `src/data/preprocessing.py`
- `src/data/windowing.py`
- `src/data/export_processed.py`

Important preprocessing rules:

- Polar `phoneTimestamp` is shifted by `-9 hours` (`32400000 ms`) to align with Galaxy Watch timestamps.
- Sessions are built from `Event.csv` `ENTER/EXIT` pairs.
- Window length is `10 s`.
- Stride is `2 s`.
- Window label is `median(HR values inside the window)`.
- Only valid PPG status values `{0, 500}` are kept.
- Minimum PPG coverage per window is `0.8`.
- Metrics drop `NaN` pairs instead of extrapolating edge values.

Failure behavior:

- preprocessing fails if the dataset root does not exist
- preprocessing fails if `Meta.csv` is missing
- preprocessing fails if no participant folders such as `P02/` are found
- preprocessing fails if the split config references participant folders that are not present
- preprocessing fails if zero windows are produced

## 5. Preprocessing command

Exact preprocessing command:

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json
```

Expected outputs:

- `data/processed/galaxyppg_hr_w10_s2_median_manifest.json`
- `data/processed/windows/galaxyppg_hr_w10_s2_median_windows.jsonl.gz`
- `data/processed/labels/galaxyppg_hr_w10_s2_median_labels.csv`

The processed manifest also stores:

- the fixed train participants
- the fixed test participants
- the fixed validation-fold assignments
- the split-config path used to build the cache

## 6. Reproducible baseline commands

### Baseline from raw data

```bash
python -m src.baseline.run_baseline --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json --method peak --output-dir experiments/reproduced_submission/baseline_peak_from_raw
```

### Baseline from processed data

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_hr_w10_s2_median_manifest.json --method peak --output-dir experiments/reproduced_submission/baseline_peak
```

The processed-data command is the shortest reproducible baseline path because it reuses the saved split labels in the manifest and does not depend on re-running raw preprocessing.

## 7. Embedding extraction command for PulsePPG

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/pulseppg_results/2026-03-18/full --batch-size 128 --device cpu
```

Expected outputs:

- `experiments/pulseppg_results/2026-03-18/full/pulseppg_features.npy`
- `experiments/pulseppg_results/2026-03-18/full/pulseppg_metadata.csv`
- `experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json`

## 8. Embedding extraction command for PaPaGei

```bash
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/papagei_results/2026-03-18/full --batch-size 128 --device cpu
```

Expected outputs:

- `experiments/papagei_results/2026-03-18/full/papagei_features.npy`
- `experiments/papagei_results/2026-03-18/full/papagei_metadata.csv`
- `experiments/papagei_results/2026-03-18/full/papagei_manifest.json`

## 9. Training and evaluation commands

### Train a regressor

Example: PulsePPG + random forest

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-03-23/regression_random_forest
```

Example: PaPaGei + ridge

```bash
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-03-18/regression_ridge
```

Supported regressors:

- `linear`
- `ridge`
- `random_forest`
- `gradient_boosting`

### Generate evaluation plots for a saved regression run

```bash
python -m src.regression.plot_regression_results --result-dir experiments/pulseppg_results/2026-03-23/regression_random_forest
```

### Generate evaluation plots for a saved baseline run

```bash
python -m src.baseline.plot_baseline_results --result-dir experiments/baseline_results/2026-03-11
```

## 10. Exact command that reproduces the final reported number

The best reported result is `PulsePPG + Random Forest`.

Shortest exact reproduction command from saved feature caches:

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/reproduced_submission/pulseppg_random_forest
```

Expected metrics:

- `MAE = 7.286407769380653`
- `RMSE = 11.30392115001437`

Expected outputs:

- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_metrics.json`
- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_predictions.csv`
- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_run_log.json`

## Full end-to-end command order

1. Install the environment.

```bash
pip install -r requirements.txt
```

2. Download and extract `GalaxyPPG` to `data/raw/GalaxyPPG/`.

3. Create the local checkpoint folders and place the official checkpoints:

```text
external/pulseppg/checkpoint_best.pkl
external/papagei-foundation-model/weights/papagei_s.pt
```

4. Preprocess the raw dataset.

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json
```

5. Extract PulsePPG embeddings.

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/pulseppg_results/2026-03-18/full --batch-size 128 --device cpu
```

6. Extract PaPaGei embeddings.

```bash
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/papagei_results/2026-03-18/full --batch-size 128 --device cpu
```

7. Train downstream regressors.

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor linear --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-03-18/regression_linear
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-03-18/regression_ridge
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor gradient_boosting --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-03-23/regression_gradient_boosting
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-03-23/regression_random_forest
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor linear --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-03-18/regression_linear
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-03-18/regression_ridge
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor gradient_boosting --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-03-23/regression_gradient_boosting
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-03-23/regression_random_forest
```

8. Optionally regenerate plots.

```bash
python -m src.baseline.plot_baseline_results --result-dir experiments/baseline_results/2026-03-11
python -m src.regression.plot_regression_results --result-dir experiments/pulseppg_results/2026-03-23/regression_random_forest
python -m src.regression.plot_regression_results --result-dir experiments/papagei_results/2026-03-23/regression_random_forest
```

## Saved artifacts included in the repository

The repository includes:

- processed manifest, window cache, and label CSV
- full PulsePPG feature bundle
- full PaPaGei feature bundle
- baseline metrics and predictions
- regression metrics and predictions for all reported experiments
- run logs for all reported experiments
- estimator artifacts and plot directories for the reported experiments
- the vendored minimal model source needed for PulsePPG and PaPaGei feature extraction
- summary indexes:
  - `experiments/reported_results_summary.md`
  - `experiments/reported_results_summary.json`
  - `experiments/reproduced_submission/submission_run_summary.md`
  - `experiments/reproduced_submission/submission_run_summary.json`

## Reported experiment summary

For the bundled experiment inventory and metric table, see:

- `experiments/reported_results_summary.md`
- `experiments/reported_results_summary.json`

## Requirements for reproducible reruns

You should not need to edit code if:

- the raw dataset is placed under `data/raw/GalaxyPPG/`
- the official checkpoints are placed at the documented paths
- you use the commands above

If you change dataset paths or checkpoint paths, pass them through the documented CLI arguments instead of editing source files.
