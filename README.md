# PPG Heart Rate Estimation

This repository targets heart-rate estimation on the `GalaxyPPG` dataset using:

- Galaxy Watch PPG as input
- Polar H10 HR / IBI as reference labels
- classical baselines
- two foundation models: `PulsePPG` and `PaPaGei`

The repository is organized to be reproducible without manual code edits.  
Only dataset placement and model-weight placement are external requirements, and both are documented below.

## What Must Be In The Repository

```text
ppg_project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- galaxyppg_submission_split.json
|   `-- submission_protocol.json
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- cache.py
|   |   |-- export_processed.py
|   |   |-- loader.py
|   |   |-- preprocessing.py
|   |   `-- windowing.py
|   |-- baseline/
|   |   |-- __init__.py
|   |   |-- peak_detection.py
|   |   |-- plot_baseline_results.py
|   |   |-- run_baseline.py
|   |   `-- spectral_hr.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- common.py
|   |   |-- papagei_feature.py
|   |   `-- pulseppg_feature.py
|   |-- regression/
|   |   |-- __init__.py
|   |   |-- evaluate.py
|   |   |-- plot_regression_results.py
|   |   `-- train_regressor.py
|   `-- utils/
|       |-- __init__.py
|       |-- generate_progress_report.py
|       `-- metrics.py
|-- data/
|   |-- raw/
|   |   |-- README.md
|   |   `-- GalaxyPPG/
|   |       `-- README.md
|   |-- processed/
|   |   |-- README.md
|   |   |-- galaxyppg_hr_w10_s2_median_manifest.json
|   |   |-- windows/
|   |   |   `-- galaxyppg_hr_w10_s2_median_windows.jsonl.gz
|   |   `-- labels/
|   |       `-- galaxyppg_hr_w10_s2_median_labels.csv
|   `-- summary/
|       |-- README.md
|       `-- galaxyppg_schema.md
|-- external/
|   |-- pulseppg/
|   |   |-- LICENSE
|   |   |-- README.md
|   |   `-- pulseppg/
|   |       |-- nets/
|   |       |   `-- ResNet1D/
|   |       |       `-- ResNet1D_Net.py
|   |       `-- experiments/
|   |           `-- out/
|   |               `-- pulseppg/
|   |                   `-- .gitkeep
|   `-- papagei-foundation-model/
|       |-- LICENSE
|       |-- README.md
|       |-- models/
|       |   |-- __init__.py
|       |   `-- resnet.py
|       `-- weights/
|           `-- .gitkeep
|-- experiments/
|   |-- baseline_results/
|   |   `-- 2026-03-11/
|   |       |-- peak_metrics.json
|   |       |-- peak_predictions.csv
|   |       |-- spectral_metrics.json
|   |       `-- spectral_predictions.csv
|   |-- pulseppg_results/
|   |   |-- 2026-03-18/
|   |   |   `-- full/
|   |   |       |-- pulseppg_features.npy
|   |   |       |-- pulseppg_manifest.json
|   |   |       `-- pulseppg_metadata.csv
|   |   `-- 2026-03-23/
|   |       `-- regression_random_forest/
|   |           |-- pulseppg_random_forest_metrics.json
|   |           `-- pulseppg_random_forest_predictions.csv
|   |-- papagei_results/
|   |   |-- 2026-03-18/
|   |   |   `-- full/
|   |   |       |-- papagei_features.npy
|   |   |       |-- papagei_manifest.json
|   |   |       `-- papagei_metadata.csv
|   |   `-- 2026-03-23/
|   |       `-- regression_random_forest/
|   |           |-- papagei_random_forest_metrics.json
|   |           `-- papagei_random_forest_predictions.csv
|   `-- reproduced_submission/
|       |-- submission_run_summary.json
|       |-- submission_run_summary.md
|       |-- baseline_peak/
|       |   |-- peak_metrics.json
|       |   |-- peak_predictions.csv
|       |   `-- peak_run_log.json
|       `-- pulseppg_random_forest/
|           |-- pulseppg_random_forest_metrics.json
|           |-- pulseppg_random_forest_predictions.csv
|           `-- pulseppg_random_forest_run_log.json
`-- notebooks/
    `-- exploration.ipynb
```

## Environment Setup

Exact package versions are stored in `requirements.txt`.

Recommended Python version:

- `Python 3.13.x`

Install:

```bash
pip install -r requirements.txt
```

If you need GPU acceleration, replace the `torch` package with the wheel that matches your CUDA environment.

## Raw Dataset Placement

If you want to rebuild preprocessing from raw data, place the raw `GalaxyPPG` files under:

```text
data/raw/GalaxyPPG/
```

The repository already includes a reusable processed cache under `data/processed/`, so raw data is not required to reproduce the final downstream numbers.

## Model Weights

The repository includes the exact expected weight directories, but not the large official checkpoint files themselves.

Download locations:

- `PulsePPG`: `https://doi.org/10.5281/zenodo.17270930`
- `PaPaGei`: `https://zenodo.org/records/13983110`

Place the files here:

- `PulsePPG` checkpoint:
  `external/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl`
- `PaPaGei-S` checkpoint:
  `external/papagei-foundation-model/weights/papagei_s.pt`

No code changes are needed if you use these default paths.

## Fixed Split And Reproducibility Metadata

The repository includes two reproducibility config files:

- `configs/galaxyppg_submission_split.json`
- `configs/submission_protocol.json`

They define:

- exact train participants
- exact test participants
- deterministic validation folds used by `GroupKFold`
- random seed `42`
- preprocessing rules
- timestamp alignment details
- NaN handling rules
- exclusion rules
- exact saved artifact paths used in the submission

Important note on validation:

- there is no single fixed held-out validation set
- model selection uses deterministic participant-level `GroupKFold(n_splits=5)` over the fixed training participants
- the exact validation-participant IDs for all five folds are stored in `configs/galaxyppg_submission_split.json`

## Preprocessing Details

The preprocessing pipeline is implemented by:

- `src/data/loader.py`
- `src/data/preprocessing.py`
- `src/data/windowing.py`
- `src/data/export_processed.py`

Covered behavior:

- timestamp alignment:
  Polar `phoneTimestamp` is shifted by `-9 hours` (`32400000 ms`)
- session extraction:
  `Event.csv` `ENTER/EXIT` pairs
- windowing:
  `10 s` windows with `2 s` stride
- label generation:
  median HR inside each window
- NaN handling:
  no edge extrapolation for interpolated reference values; metrics drop NaN pairs
- resampling:
  model input resampling is done in `src/models/common.py`
- normalization:
  optional z-score and band-pass preprocessing for foundation-model input
- exclusion rules:
  keep only valid PPG status values, require minimum PPG coverage and minimum reference samples
- split:
  participant-level split fixed by `configs/galaxyppg_submission_split.json`

Exact preprocessing command:

```bash
python -m src.data.export_processed --split-config configs/galaxyppg_submission_split.json
```

## Embedding Extraction Scripts

Separate scripts are provided for each model:

- `python -m src.models.pulseppg_feature`
- `python -m src.models.papagei_feature`

Exact commands used for the saved full feature caches:

```bash
python -m src.models.pulseppg_feature --output-dir experiments/pulseppg_results/2026-03-18/full --batch-size 128 --device cpu
python -m src.models.papagei_feature --output-dir experiments/papagei_results/2026-03-18/full --batch-size 128 --device cpu
```

## Training And Evaluation Script

The regressor training and evaluation entry point is:

```bash
python -m src.regression.train_regressor
```

Plot generation is handled by:

```bash
python -m src.regression.plot_regression_results
```

## Command Order

For a full rerun from raw data and official model weights:

1. Install the environment:

```bash
pip install -r requirements.txt
```

2. Put raw `GalaxyPPG` under `data/raw/GalaxyPPG/`

3. Download and place the model weights in the documented default paths

4. Rebuild processed data:

```bash
python -m src.data.export_processed --split-config configs/galaxyppg_submission_split.json
```

5. Extract embeddings:

```bash
python -m src.models.pulseppg_feature --output-dir experiments/pulseppg_results/2026-03-18/full --batch-size 128 --device cpu
python -m src.models.papagei_feature --output-dir experiments/papagei_results/2026-03-18/full --batch-size 128 --device cpu
```

6. Train and evaluate regressors:

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --output-dir experiments/reproduced_submission/pulseppg_random_forest
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-03-18/full/papagei_manifest.json --regressor random_forest --random-state 42 --output-dir experiments/reproduced_submission/papagei_random_forest
```

7. Optionally generate plots:

```bash
python -m src.regression.plot_regression_results --result-dir experiments/reproduced_submission/pulseppg_random_forest
python -m src.regression.plot_regression_results --result-dir experiments/reproduced_submission/papagei_random_forest
```

## Exact Command That Reproduces The Final Best Number

The best submission number is `PulsePPG + Random Forest`.

Because the repository already includes the exact full PulsePPG embedding cache, the shortest exact reproduction command is:

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --output-dir experiments/reproduced_submission/pulseppg_random_forest
```

Expected reproduced metrics:

- `MAE = 7.286407769380653`
- `RMSE = 11.30392115001437`

The reproduced outputs are saved to:

- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_metrics.json`
- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_predictions.csv`
- `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_run_log.json`

## Saved Logs, Metrics, And Predictions

The repository includes:

- saved evaluation metrics in JSON
- saved predictions in CSV
- run logs in JSON for reproduced submission runs
- a compact command summary in:
  `experiments/reproduced_submission/submission_run_summary.md`

Key files:

- baseline reproduced log:
  `experiments/reproduced_submission/baseline_peak/peak_run_log.json`
- best-model reproduced log:
  `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_run_log.json`
- baseline reproduced predictions:
  `experiments/reproduced_submission/baseline_peak/peak_predictions.csv`
- best-model reproduced predictions:
  `experiments/reproduced_submission/pulseppg_random_forest/pulseppg_random_forest_predictions.csv`
