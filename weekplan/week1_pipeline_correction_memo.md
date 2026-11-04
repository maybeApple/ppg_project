# Week 1 Memo: Corrected GalaxyPPG Pipeline

Date: 2026-04-24

## Purpose

This memo documents the Week 1 correction to the GalaxyPPG heart-rate estimation pipeline. The goal was to turn the earlier dataset-specific workflow into an auditable cross-dataset workflow that can later support GalaxyPPG, PPG-DaLiA, and WildPPG under one consistent processing contract.

The correction has four main parts:

1. Galaxy Watch PPG inversion is now explicit and mandatory at load time.
2. A canonical internal schema now represents PPG, accelerometer, participant/session/activity metadata, and ECG/IBI references.
3. Window labels now use one shared ECG/IBI beat-interval target definition.
4. Preprocessing is now selected through experiment configuration as either `harmonized` or `model_faithful`.

These changes matter because heart-rate estimation results are sensitive to signal polarity, target definition, window filtering, and preprocessing assumptions. If those choices are hidden in model-specific scripts, experiments can appear comparable while actually using different signal conventions or different physiological targets.

## 1. Loader-Level Galaxy Watch PPG Inversion

The GalaxyPPG loader now reads the original Galaxy Watch PPG value into:

```text
ppg_raw
```

It then creates the canonical downstream model input as:

```text
ppg = -ppg_raw
```

Every loaded PPG row records:

```text
ppg_inverted = True
ppg_canonical_source = "GalaxyWatch/PPG.csv:ppg_raw_inverted"
```

This correction is implemented in `src/data/loader.py`. The raw waveform remains available for inspection and visualization, but all downstream processing consumes the canonical `ppg` field.

Scientifically, this matters because PPG polarity changes what algorithms see as a peak, a trough, or a pulse upstroke. Peak-detection baselines, learned feature extractors, and quality-control plots can all behave differently when polarity is inconsistent. Applying inversion inside individual model scripts would create a high risk of accidental double inversion, missing inversion, or inconsistent treatment across experiments. Moving inversion to the loader makes the signal convention a dataset-specific correction applied exactly once.

The notebook `notebooks/galaxyppg_ppg_inversion_check.ipynb` visualizes `ppg_raw` versus canonical `ppg` across multiple GalaxyPPG sessions and includes a scatter check of the invariant:

```text
canonical ppg = -ppg_raw
```

## 2. Canonical Internal Schema

The new canonical schema is defined in `src/data/canonical.py` as:

```text
canonical_ppg_v1
```

The schema separates the data into reusable signal groups.

Canonical PPG fields:

```text
participant_id, timestamp_ms, ppg, ppg_raw, ppg_inverted,
ppg_canonical_source, session_id, session_name, activity_label,
dataset, sensor
```

Canonical accelerometer fields:

```text
participant_id, timestamp_ms, acc_x, acc_y, acc_z,
session_id, session_name, activity_label, dataset, sensor
```

Canonical reference fields:

```text
participant_id, timestamp_ms, ecg_uv, rr_interval_ms, hr_bpm,
reference_source, session_id, session_name, activity_label,
dataset, sensor
```

For GalaxyPPG, the loader now reads Galaxy Watch PPG, Galaxy Watch accelerometer, Polar IBI/HR reference, Polar ECG, and event sessions into this contract.

Scientifically, the canonical schema separates dataset ingestion from experimental modeling. Dataset-specific work, such as timestamp alignment, sensor naming, activity labels, and Galaxy Watch PPG inversion, happens at the boundary of the dataset loader. After that point, downstream code operates on common fields. This reduces the chance that later PPG-DaLiA or WildPPG integration will introduce hidden branches in model code. It also makes the processed manifest auditable because it records which canonical schema version was used.

## 3. Unified ECG/IBI Beat-Interval Target Definition

The previous window-label logic aggregated HR samples directly inside each window. That behavior was replaced with `src/data/labels.py`, which centralizes target generation.

The corrected default rule is:

```text
window_seconds = 10
stride_seconds = 2
instant_hr_bpm = 60000 / rr_interval_ms
label_hr_bpm = median instantaneous HR inside the 10-second window
drop window if valid beat intervals < 2
```

For IBI references, `rr_interval_ms` comes directly from `PolarH10/IBI.csv`.

For ECG references, the code detects likely R peaks inside each window, computes adjacent R-R intervals, filters those intervals to a physiologically plausible range, converts each interval to instantaneous HR, and then applies the same median target rule.

The corrected GalaxyPPG processed manifest is:

```text
data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json
```

It records:

```text
reference_source = ibi
label_method = beat_interval_instant_hr
label_aggregation = median
window_seconds = 10.0
stride_seconds = 2.0
min_valid_beats = 2
```

Scientifically, this matters because a device-provided HR stream may be smoothed, delayed, or filtered by proprietary algorithms. Beat intervals are closer to the physiological event timing that defines heart rate. Using instantaneous HR from beat intervals gives a clearer target: the model predicts the central heart-rate state represented by cardiac intervals within the same 10-second PPG window. The median reduces sensitivity to isolated interval errors while preserving a robust central estimate.

The window discard rule is also important. A window with too few valid beats cannot support a reliable beat-interval HR target. Dropping such windows prevents the model from training or being evaluated against weak labels.

## 4. Configuration-Driven Experiment Modes

Preprocessing is now selected from:

```text
configs/experiment_modes.json
```

Two modes are defined.

`harmonized` mode is for controlled comparisons. It uses the same corrected windows, labels, split, resampling policy, band-pass filter, and per-window z-score normalization for all feature extractors.

`model_faithful` mode keeps the corrected windows and labels fixed but allows model-specific signal assumptions to be selected explicitly. PulsePPG supports:

```text
none
per_window_zscore
person_specific_zscore
causal_running_zscore
```

PaPaGei remains a frozen-embedding evaluation, with linear or ridge regression used downstream.

The feature extraction scripts write preprocessing metadata into their feature manifests, including:

```text
experiment_mode
target_sampling_hz
apply_bandpass
normalization
operation_order
```

Regression metrics then propagate this metadata.

Scientifically, this matters because there are two legitimate but different evaluation questions. The harmonized setting asks which method performs better under the same controlled preprocessing. The model-faithful setting asks how each method behaves under assumptions closer to its intended use. Keeping both modes configuration-driven makes this distinction visible instead of embedding it as scattered source-code branches.

## 5. Reproducibility State

The corrected processed export currently contains:

```text
num_windows = 35135
num_train_windows = 27504
num_test_windows = 7631
num_participants = 23
```

The verification checks completed for this correction were:

```text
python -m compileall src
```

Additional smoke checks confirmed:

- the corrected manifest loads successfully
- the default manifest contains `label_method=beat_interval_instant_hr`
- PPG inversion is recorded as `ppg_inverted=True`
- the first loaded P02 sample satisfies `ppg = -ppg_raw`
- P02 IBI windowing produces 1507 windows
- P02 ECG windowing produces 1507 windows

## Conclusion

The corrected pipeline separates three concerns that were previously too easy to mix together:

1. dataset-specific signal correction
2. physiological target generation
3. model-specific preprocessing assumptions

This separation improves auditability and scientific comparability. GalaxyPPG-specific PPG polarity is handled once in the loader. The heart-rate target is generated from ECG/IBI beat intervals using one reusable rule. Preprocessing assumptions are recorded in configuration and manifests. These changes make later GalaxyPPG, PPG-DaLiA, and WildPPG comparisons less likely to be affected by hidden preprocessing differences or inconsistent label definitions.
