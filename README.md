# PPG Heart Rate Estimation

This repository reproduces heart-rate estimation on the `GalaxyPPG` dataset with a corrected, auditable processing pipeline.

The current default flow uses:

- raw input: Galaxy Watch `PPG.csv`
- mandatory loader-level Galaxy Watch PPG inversion
- canonical internal signal schema: `canonical_ppg_v1`
- reference labels: Polar H10 IBI by default, with ECG-derived beat intervals also supported
- fixed 10-second windows and 2-second stride
- target: median beat-interval instantaneous HR inside each window
- classical baselines: peak detection and spectral HR
- foundation-model features: PulsePPG and PaPaGei
- downstream regressors: linear, ridge, random forest, gradient boosting
- selectable experiment modes: `harmonized` and `model_faithful`

The repository is intended to be runnable without manual source edits. The only external inputs are:

1. the raw `GalaxyPPG` dataset
2. the official pretrained model checkpoints

The minimal model-definition code required to load the checkpoints is already vendored under `src/vendor/`.

## Corrected Flow

The corrected GalaxyPPG flow is anchored by:

- canonical schema: `src/data/canonical.py`
- label generation: `src/data/labels.py`
- experiment modes: `configs/experiment_modes.json`
- protocol record: `configs/submission_protocol.json`
- corrected processed manifest: `data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json`
- raw/inverted PPG visual check: `notebooks/galaxyppg_ppg_inversion_check.ipynb`
- week deliverable summary: `weekplan/week1.md`

`reports/`, root-level `week1.md`, `Current.md`, and local `.docx` planning files are personal review artifacts and are not required for the reproducible pipeline.

## Repository Layout

```text
ppg_project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- experiment_modes.json
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
|-- notebooks/
|-- weekplan/
`-- experiments/
```

`external/` is intentionally kept almost empty in version control. It is the local checkpoint root where users place downloaded model weights at the documented paths.

## Environment

Tested environment:

- Python `3.13.9`
- Windows
- CPU execution

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, replace the `torch` wheel with the one that matches your CUDA environment.

## Raw Data

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

Expected participant structure:

```text
data/raw/GalaxyPPG/
|-- Meta.csv
|-- P02/
|   |-- Event.csv
|   |-- GalaxyWatch/
|   |   |-- PPG.csv
|   |   `-- ACC.csv
|   `-- PolarH10/
|       |-- ECG.csv
|       |-- HR.csv
|       `-- IBI.csv
|-- P03/
|   |-- Event.csv
|   |-- GalaxyWatch/
|   |   |-- PPG.csv
|   |   `-- ACC.csv
|   `-- PolarH10/
|       |-- ECG.csv
|       |-- HR.csv
|       `-- IBI.csv
`-- ...
```

Notes:

- `GalaxyWatch/PPG.csv` is required as model input.
- `GalaxyWatch/ACC.csv` is loaded into the canonical accelerometer table when available.
- `PolarH10/IBI.csv` is the default corrected reference source.
- `PolarH10/ECG.csv` is supported by detecting R peaks inside each window and deriving RR intervals.
- `PolarH10/HR.csv` remains available for legacy `provided_hr_samples` labels.
- `Event.csv` is used for `ENTER/EXIT` session boundaries.
- `P01` is excluded from the fixed split because the official raw release does not provide usable Galaxy Watch PPG plus event annotations for this project.

## Model Weights

Only the pretrained checkpoint files are required. No extra upstream model-code checkout is needed at runtime.

### PulsePPG

- source: official PulsePPG Zenodo record `https://doi.org/10.5281/zenodo.17270930`
- expected filename: `checkpoint_best.pkl`
- recommended placement:

```text
external/pulseppg/checkpoint_best.pkl
```

Backward-compatible legacy placement is also supported:

```text
external/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl
```

### PaPaGei

- source: official PaPaGei Zenodo record `https://zenodo.org/records/13983110`
- expected filename: `papagei_s.pt`
- expected placement:

```text
external/papagei-foundation-model/weights/papagei_s.pt
```

## Vendored Runtime Code

No extra external model repository checkout is required for feature extraction.

The minimal checkpoint-compatible runtime code is vendored in:

- `src/vendor/pulseppg_resnet1d.py`
- `src/vendor/papagei_resnet.py`
- `src/vendor/resnet1d_shared.py`

Upstream provenance is pinned in `src/vendor/UPSTREAM.md`.

## Fixed Split

The fixed participant split is defined in:

```text
configs/galaxyppg_submission_split.json
```

It records:

- train participants
- held-out test participants
- deterministic participant-level validation folds
- random seed `42`

Training uses the saved validation-fold assignments instead of recomputing them ad hoc.

## Canonical Schema

The canonical schema is defined in `src/data/canonical.py` as `canonical_ppg_v1`.

Canonical PPG fields include:

```text
participant_id, timestamp_ms, ppg, ppg_raw, ppg_inverted,
ppg_canonical_source, session_id, session_name, activity_label,
dataset, sensor
```

Canonical accelerometer fields include:

```text
participant_id, timestamp_ms, acc_x, acc_y, acc_z,
session_id, session_name, activity_label, dataset, sensor
```

Canonical reference fields include:

```text
participant_id, timestamp_ms, ecg_uv, rr_interval_ms, hr_bpm,
reference_source, session_id, session_name, activity_label,
dataset, sensor
```

Dataset-specific corrections happen at the loader boundary. Downstream windowing, feature extraction, and regression consume canonical fields so that GalaxyPPG, PPG-DaLiA, and WildPPG can later share the same contracts.

## Loader-Level PPG Inversion

Galaxy Watch PPG inversion is explicit and mandatory in `src/data/loader.py`.

The raw value from `GalaxyWatch/PPG.csv` is preserved as:

```text
ppg_raw
```

The canonical downstream signal is:

```text
ppg = -ppg_raw
```

Each row records:

```text
ppg_inverted = True
ppg_canonical_source = "GalaxyWatch/PPG.csv:ppg_raw_inverted"
```

The inversion is intentionally not hidden in model-specific preprocessing.

## Label Generation

The shared target logic is implemented in `src/data/labels.py`.

Corrected default:

```text
window_seconds = 10
stride_seconds = 2
label_method = beat_interval_instant_hr
label_aggregation = median
min_valid_beats = 2
```

The target rule is:

```text
instant_hr_bpm = 60000 / rr_interval_ms
label_hr_bpm = median(instant_hr_bpm inside the 10-second window)
```

For IBI references, `rr_interval_ms` comes from `PolarH10/IBI.csv`.

For ECG references, likely R peaks are detected inside each 10-second window, adjacent R-R intervals are converted to instantaneous HR, and the same median target rule is used.

Windows are dropped when they fail either condition:

- PPG coverage is below `0.8`
- fewer than `2` valid beat intervals are available

Legacy HR-sample labels are still available through `provided_hr_samples`, but the corrected GalaxyPPG export uses `beat_interval_instant_hr`.

## Preprocessing Modes

Experiment modes are defined in:

```text
configs/experiment_modes.json
```

### Harmonized

Use `harmonized` for controlled comparisons.

It keeps windows, labels, split, resampling policy, band-pass filtering, and per-window z-score normalization consistent across feature extractors.

### Model Faithful

Use `model_faithful` for model-specific assumptions while keeping the corrected windows and labels fixed.

PulsePPG normalization options:

- `none`
- `per_window_zscore`
- `person_specific_zscore`
- `causal_running_zscore`

PaPaGei is evaluated as frozen embeddings with downstream linear or ridge probes.

Feature manifests record:

- `experiment_mode`
- preprocessing mode
- target sampling rate
- band-pass setting
- normalization strategy
- operation order

Regression metrics propagate this preprocessing metadata.

## Processed Data Export

Build the corrected processed cache:

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json
```

Expected outputs:

```text
data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json
data/processed/windows/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_windows.jsonl.gz
data/processed/labels/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_labels.csv
```

The corrected manifest records:

- canonical schema version
- dataset root
- reference source
- PPG inversion status and canonical source
- window length and stride
- label method, aggregation, formula, ECG/IBI construction, and drop rule
- train/test participants
- validation-fold assignments
- processed artifact paths

Current corrected GalaxyPPG export:

```text
artifact_name = galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median
num_windows = 35135
num_train_windows = 27504
num_test_windows = 7631
num_participants = 23
```

## Baseline Commands

Run a baseline from the corrected processed cache:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --output-dir experiments/reproduced_corrected_2026-04-24/baseline_peak
```

Run from raw data:

```bash
python -m src.baseline.run_baseline --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json --method peak --output-dir experiments/reproduced_corrected_2026-04-24/baseline_peak_from_raw
```

Supported baseline methods:

- `peak`
- `spectral`

## Feature Extraction

### PulsePPG Harmonized

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/pulseppg_results/2026-04-24/full_harmonized --batch-size 128 --device cpu
```

### PulsePPG Model Faithful

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode model_faithful --normalization causal_running_zscore --output-dir experiments/pulseppg_results/2026-04-24/full_model_faithful_causal --batch-size 128 --device cpu
```

### PaPaGei Harmonized

```bash
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/papagei_results/2026-04-24/full_harmonized --batch-size 128 --device cpu
```

### PaPaGei Model Faithful

```bash
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode model_faithful --output-dir experiments/papagei_results/2026-04-24/full_model_faithful --batch-size 128 --device cpu
```

Expected feature outputs:

```text
<output-dir>/<model>_features.npy
<output-dir>/<model>_metadata.csv
<output-dir>/<model>_manifest.json
```

## Regression

Example: PulsePPG + random forest

```bash
python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-04-24/full_harmonized/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/pulseppg_results/2026-04-24/regression_random_forest
```

Example: PaPaGei + ridge

```bash
python -m src.regression.train_regressor --feature-manifest experiments/papagei_results/2026-04-24/full_harmonized/papagei_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/papagei_results/2026-04-24/regression_ridge
```

Supported regressors:

- `linear`
- `ridge`
- `random_forest`
- `gradient_boosting`

Regression metrics include feature preprocessing metadata when it is available in the feature manifest.

## Plotting

Generate evaluation plots for a saved regression run:

```bash
python -m src.regression.plot_regression_results --result-dir experiments/pulseppg_results/2026-04-24/regression_random_forest
```

Generate evaluation plots for a saved baseline run:

```bash
python -m src.baseline.plot_baseline_results --result-dir experiments/reproduced_corrected_2026-04-24/baseline_peak
```

## Week 2 Corrected GalaxyPPG Benchmark

Week 2 outputs are stored under:

```text
experiments/week2_galaxyppg_corrected_2026-05-01/
```

The Week 2 configs are:

```text
configs/week2_galaxyppg_harmonized.json
configs/week2_galaxyppg_model_faithful.json
configs/week2_galaxyppg_inversion_ablation.json
```

Run the corrected GalaxyPPG harmonized classical baselines:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --ppg-source canonical --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/baseline_peak
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method spectral --ppg-source canonical --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/baseline_spectral
```

Run the inversion ablation for classical baselines:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --ppg-source raw --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/baseline_peak_raw
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method spectral --ppg-source raw --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/baseline_spectral_raw
```

Extract frozen embeddings for harmonized foundation-model runs:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features --batch-size 256 --device cpu
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/papagei_features --batch-size 128 --device cpu
```

Train probes and practical upper-bound regressors from a saved feature manifest:

```bash
python -m src.regression.train_regressor --feature-manifest experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features/pulseppg_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_ridge
python -m src.regression.train_regressor --feature-manifest experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_random_forest
```

For model-faithful PulsePPG, rerun feature extraction with each normalization variant:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode model_faithful --normalization causal_running_zscore --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/model_faithful/pulseppg_features_causal_running_zscore --batch-size 256 --device cpu
```

For foundation-model inversion ablations, extract raw non-inverted features with `--ppg-source raw` and train the same probe:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --ppg-source raw --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/pulseppg_features_raw --batch-size 256 --device cpu
python -m src.regression.train_regressor --feature-manifest experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/pulseppg_features_raw/pulseppg_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/pulseppg_ridge_raw
```

After all selected runs are present, generate standardized prediction files, metrics, tables, figures, and the Week 2 memo:

```bash
python -m src.utils.build_week2_artifacts --search-root experiments/week2_galaxyppg_corrected_2026-05-01/runs --output-root experiments/week2_galaxyppg_corrected_2026-05-01 --tag-name week2-galaxyppg-corrected-2026-05-01
```

Expected Week 2 summary artifacts:

```text
experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv
experiments/week2_galaxyppg_corrected_2026-05-01/metrics/overall_metrics.csv
experiments/week2_galaxyppg_corrected_2026-05-01/metrics/participant_level_metrics.csv
experiments/week2_galaxyppg_corrected_2026-05-01/metrics/activity_level_metrics.csv
experiments/week2_galaxyppg_corrected_2026-05-01/tables/main_benchmark_table.csv
experiments/week2_galaxyppg_corrected_2026-05-01/tables/inversion_ablation_table.csv
experiments/week2_galaxyppg_corrected_2026-05-01/week2_memo.md
```

## Week 3 Regime Analysis and Oracle Routing

Week 3 uses the corrected GalaxyPPG Week 2 predictions to test whether estimator dominance is regime-dependent before training any deployable router. The analysis pairs a selected classical expert with a selected foundation-model expert on shared valid windows, computes per-window error gaps, adds activity/participant/HR-range/motion/PPG-quality regime features, and reports an oracle router upper bound.

Build the default Week 3 artifact set:

```bash
python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root experiments/week3_galaxyppg_regime_oracle_2026-05-13
```

The default selection chooses the lowest-MAE inverted classical run and the lowest-MAE inverted foundation-model run from the Week 2 prediction table. You can override the expert choice, for example:

```bash
python -m src.utils.build_week3_artifacts --foundation-model pulseppg --foundation-regressor ridge --foundation-preprocessing-mode harmonized
```

Expected Week 3 artifacts:

```text
experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_window_regime_expert_errors.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_oracle_router_predictions.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/selected_expert_oracle_summary.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/oracle_all_pairs_table.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_activity.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_participant.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_hr_range.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_intensity.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_ppg_quality.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_and_quality.csv
experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/oracle_vs_selected_experts.png
experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/winner_rate_by_activity.png
experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/motion_quality_error_gap_heatmap.png
experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/window_error_gap_distribution.png
experiments/week3_galaxyppg_regime_oracle_2026-05-13/week3_regime_analysis.md
```

## Week 4 Lightweight Routing

Week 4 trains the first lightweight motion- and quality-aware router on GalaxyPPG. It uses the Week 3 paired expert predictions and inference-time regime features, then evaluates participant-level out-of-fold gates with leave-one-participant-out routing predictions.

The routed system intentionally remains simple and interpretable:

- `hard_gate`: logistic gate chooses either the classical expert or foundation-model expert.
- `soft_gate`: logistic gate probability is used as the foundation-model weight, and the two expert predictions are averaged.
- feature ablations: `motion_only`, `quality_only`, and `motion_quality`.

Build the Week 4 artifact set:

```bash
python -m src.utils.build_week4_artifacts --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --output-root experiments/week4_galaxyppg_lightweight_router_2026-05-13
```

Routing feature groups:

```text
motion_only:
  acc_norm_mean, acc_norm_std, acc_dominant_frequency_hz,
  acc_cadence_band_power_fraction

quality_only:
  ppg_amplitude_range, ppg_clipping_rate, ppg_flatline_rate,
  ppg_autocorr_peak_strength, ppg_spectral_entropy,
  ppg_spectral_peak_sharpness, beat_count_consistency,
  peak_spectral_disagreement_bpm

motion_quality:
  all motion and quality features
```

Expected Week 4 artifacts:

```text
experiments/week4_galaxyppg_lightweight_router_2026-05-13/predictions/week4_routed_predictions.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_fold_summary.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_feature_coefficients.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/participant_level_routing_metrics.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/activity_level_routing_metrics.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/hard_soft_router_mae.png
experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/best_router_error_cdf.png
experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/combined_gate_feature_coefficients.png
experiments/week4_galaxyppg_lightweight_router_2026-05-13/week4_lightweight_router.md
```

## Full Reproduction Order

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Download and extract `GalaxyPPG` to `data/raw/GalaxyPPG/`.

3. Place official checkpoints:

```text
external/pulseppg/checkpoint_best.pkl
external/papagei-foundation-model/weights/papagei_s.pt
```

4. Export corrected processed windows and labels.

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json
```

5. Inspect raw versus inverted PPG.

```text
notebooks/galaxyppg_ppg_inversion_check.ipynb
```

6. Run a baseline or extract model features with a selected experiment mode.

7. Train downstream regressors using the saved feature manifest.

## Verification Performed

The corrected flow was smoke-tested with:

```text
python -m compileall src
```

Additional checks:

- JSON parse check for `configs/submission_protocol.json`
- JSON parse check for `configs/experiment_modes.json`
- JSON parse check for the corrected processed manifest
- notebook JSON parse check
- default manifest load check: `35135` windows
- first P02 raw-loader check: `ppg = -ppg_raw`
- P02 IBI smoke test: `1507` windows, first window `valid_beat_count=14`
- P02 ECG smoke test: `1507` windows, first window `valid_beat_count=16`

## Legacy Results

Older experiment folders and some historical feature caches may still correspond to the former `galaxyppg_hr_w10_s2_median` workflow. Those results should be treated as legacy HR-sample-label results and not mixed with corrected `galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median` metrics.

For corrected reruns, use the manifest path:

```text
data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json
```

## Requirements for Reproducible Reruns

You should not need to edit source code if:

- the raw dataset is placed under `data/raw/GalaxyPPG/`
- the official checkpoints are placed at the documented paths
- you use the documented CLI arguments for paths and experiment modes

If you change dataset paths or checkpoint paths, pass them through CLI arguments instead of editing source files.
