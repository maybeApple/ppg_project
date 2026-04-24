# Week 1 Corrected GalaxyPPG Pipeline Deliverable

Date: 2026-04-24

## Scope

This deliverable implements the corrected GalaxyPPG processing direction:

- Galaxy Watch PPG inversion is explicit, mandatory, and applied exactly once in the loader.
- A reusable canonical schema is defined for future GalaxyPPG, PPG-DaLiA, and WildPPG ingestion.
- Window labels use one shared ECG/IBI beat-interval target definition with 10-second windows and 2-second stride.
- Preprocessing is selectable through configuration as either `harmonized` or `model_faithful`.
- Reproducible artifacts are available through an updated README, processed manifest, and raw-versus-inverted PPG notebook.

## Implemented Changes

### Loader-level PPG inversion

`src/data/loader.py` now reads Galaxy Watch `PPG.csv` into `ppg_raw` and exposes canonical downstream `ppg` as:

```text
ppg = -ppg_raw
```

Each row records:

```text
ppg_inverted = True
ppg_canonical_source = "GalaxyWatch/PPG.csv:ppg_raw_inverted"
```

This prevents model scripts from applying hidden or inconsistent polarity corrections.

### Canonical internal schema

`src/data/canonical.py` defines `canonical_ppg_v1`.

The schema includes:

- timestamped canonical PPG, raw PPG, inversion metadata, participant ID, session name, activity label, dataset, and sensor
- timestamped accelerometer x/y/z with participant/session/activity metadata
- timestamped ECG or IBI reference fields with `ecg_uv`, `rr_interval_ms`, `hr_bpm`, and `reference_source`

GalaxyPPG now loads PPG, accelerometer, IBI/HR reference, ECG, and event sessions into this internal contract.

### Unified label generation

`src/data/labels.py` centralizes target creation.

The corrected default rule is:

```text
window_seconds = 10
stride_seconds = 2
instant_hr_bpm = 60000 / rr_interval_ms
target = median instantaneous HR inside the window
drop window if valid beat intervals < 2
```

For IBI references, `rr_interval_ms` comes from `PolarH10/IBI.csv`. For ECG references, R peaks are detected within the window, adjacent R-R intervals are converted to instantaneous HR, and the same median target rule is used.

### Experiment modes

`configs/experiment_modes.json` defines two selectable modes:

- `harmonized`: controlled comparison with shared corrected windows, labels, split, resampling, band-pass filtering, and per-window z-score normalization
- `model_faithful`: same corrected windows and labels, but with model-specific assumptions selected explicitly; PulsePPG supports `none`, `per_window_zscore`, `person_specific_zscore`, and `causal_running_zscore`

PulsePPG and PaPaGei feature manifests now record the selected preprocessing block, mode, normalization, band-pass setting, and operation order. Regression metrics propagate this metadata.

## Reproducible Artifacts

Corrected processed manifest:

```text
data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json
```

Corrected processed windows and labels:

```text
data/processed/windows/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_windows.jsonl.gz
data/processed/labels/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_labels.csv
```

Visualization notebook:

```text
notebooks/galaxyppg_ppg_inversion_check.ipynb
```

Primary documentation:

```text
README.md
data/processed/README.md
configs/submission_protocol.json
```

Personal review files and existing reports are intentionally not part of the commit.

## Reproduction Commands

Build the corrected processed cache:

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json
```

Run the corrected peak baseline from processed data:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --output-dir experiments/reproduced_corrected_2026-04-24/baseline_peak
```

Run PulsePPG in harmonized mode:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --batch-size 128 --device cpu
```

Run PulsePPG in model-faithful mode with causal running normalization:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode model_faithful --normalization causal_running_zscore --batch-size 128 --device cpu
```

## Verification

Completed checks:

- `python -m compileall src`
- JSON parse check for `configs/submission_protocol.json`, `configs/experiment_modes.json`, the corrected processed manifest, and the inversion notebook
- default manifest load check: `35135` windows, `label_method=beat_interval_instant_hr`, `ppg_inverted=True`
- raw-loader check: first P02 sample has `ppg = -ppg_raw`
- IBI smoke test for P02: `1507` windows, first window `valid_beat_count=14`
- ECG smoke test for P02: `1507` windows, first window `valid_beat_count=16`

## Git Boundary

Do not commit:

- existing `reports/` outputs
- root-level `week1.md`
- `new_plan.docx`
- `Current.md`
- generated experiment-result records unless explicitly requested

Commit:

- corrected source code
- corrected configuration files
- corrected processed manifest/windows/labels
- inversion visualization notebook
- `weekplan/week1.md`
