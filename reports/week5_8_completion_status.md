# Week 5-8 Completion Status

## Executive Summary

Week 5-8 engineering infrastructure is largely complete, but the full scientific objectives remain partially pending until real PPG-DaLiA and WildPPG wrist external-validation runs are completed.

The repository now supports external dataset ingestion, manifest-based processed exports, reusable baseline and embedding commands, participant-level statistics, reproducibility packaging, and frozen-result generation. It does not yet contain real PPG-DaLiA or WildPPG wrist metrics, and no claim should be made that external validation is complete.

## Week 5 PPG-DaLiA Status

Complete:

- `src/data/ppgdalia_loader.py` maps PPG-DaLiA wrist BVP, wrist ACC, and chest ECG/R-peaks into the canonical schema.
- `src/data/export_ppgdalia.py` writes 10-second, 2-second stride processed windows with median beat-interval instantaneous HR labels.
- `data/raw/PPG-DaLiA/README.md` documents raw-data placement.
- `scripts/smoke_ppgdalia_export.py` verifies export plus baseline reuse with temporary synthetic data.

Pending:

- Real PPG-DaLiA within-dataset baselines.
- Real PulsePPG / PaPaGei embedding extraction and linear/Ridge probes.
- Cross-dataset router transfer from GalaxyPPG to PPG-DaLiA.
- Written numerical analysis of whether complementarity survives dataset/device shift.

## Week 6 WildPPG Wrist Status

Complete:

- `src/data/wildppg_loader.py` implements manifest-driven file and column mapping for wrist PPG, wrist ACC, ECG, or RR references.
- `src/data/export_wildppg_wrist.py` writes processed windows using the same 10-second, 2-second stride, median beat-interval label rule.
- `data/raw/WildPPG-wrist/README.md` documents the required manifest.
- `scripts/smoke_wildppg_wrist_export.py` verifies manifest parsing, export, and baseline reuse with temporary synthetic data.

Pending:

- Real WildPPG wrist within-dataset baselines.
- Real PulsePPG / PaPaGei embedding extraction and linear/Ridge probes.
- Cross-dataset router transfer from GalaxyPPG to WildPPG wrist.
- Full external-validation table across GalaxyPPG, PPG-DaLiA, and WildPPG wrist.

## Week 7 Statistics and Ablations Status

Complete:

- `src/utils/build_week7_statistics.py` aggregates by participant, compares the routed system against each participant's best single expert, computes bootstrap confidence intervals, paired t-test, and Wilcoxon signed-rank test, and writes CSV/JSON/Markdown outputs.
- Week 2-4 generated artifacts locally cover inversion, preprocessing, normalization, expert, and router ablations when regenerated.

Pending:

- Final paper-ready ablation tables must be regenerated from the final selected experiment set.
- External-dataset ablations require real PPG-DaLiA and WildPPG wrist runs.

## Week 8 Frozen Package and Manuscript Status

Complete:

- `src/utils/freeze_final_results.py` copies selected Week 2/3/4/7 artifacts and writes a `final_frozen_manifest.json` plus README.
- Missing artifact groups are now explicitly recorded with `copied: false` and `reason: source_missing`.
- `paper/first_draft_skeleton.md` provides a manuscript scaffold without claiming unavailable results.
- `reports/week8_figure_status.md` tracks Figure 1-8 readiness.

Pending:

- Final frozen result directory should be regenerated after all final experiments complete.
- Paper results text and figures remain draft/pending where external-validation runs are unavailable.

## Repository Files That Implement Each Part

| Part | Files |
| --- | --- |
| PPG-DaLiA ingestion | `src/data/ppgdalia_loader.py`, `src/data/export_ppgdalia.py`, `data/raw/PPG-DaLiA/README.md` |
| WildPPG wrist ingestion | `src/data/wildppg_loader.py`, `src/data/export_wildppg_wrist.py`, `data/raw/WildPPG-wrist/README.md` |
| External smoke tests | `scripts/smoke_ppgdalia_export.py`, `scripts/smoke_wildppg_wrist_export.py` |
| External status/result table | `src/utils/build_external_validation_status.py`, `reports/week5_6_external_validation_status.md` |
| Statistics | `src/utils/build_week7_statistics.py`, `reports/week7_final_ablations_status.md` |
| Freeze package | `src/utils/freeze_final_results.py` |
| Manuscript/figures | `paper/first_draft_skeleton.md`, `reports/week8_figure_status.md` |

## What Is Complete

- External dataset loaders and exporters are implemented.
- External processed manifests can be consumed by baseline scripts without forcing GalaxyPPG fixed splits.
- Feature extraction and regression commands can reuse external processed manifests.
- Synthetic smoke tests verify the export and baseline path without committing generated artifacts.
- Participant-level Week 7 statistics tooling is implemented.
- Freeze tooling is implemented and records missing source groups honestly.
- Documentation identifies pending real-data work.

## What Remains Pending

- Real PPG-DaLiA raw data must be placed locally and processed.
- Real WildPPG wrist raw data plus manifest must be placed locally and processed.
- Week 5/6 baselines, embeddings, probes, and router transfer experiments must be run on real data.
- Final paper figures and tables must be regenerated from the final experiment set.
- The manuscript skeleton must be filled after real results are available.

## Exact Next Commands

Run synthetic smoke tests:

```bash
python scripts/smoke_ppgdalia_export.py
python scripts/smoke_wildppg_wrist_export.py
```

Export real external processed manifests after placing raw data:

```bash
python -m src.data.export_ppgdalia --dataset-root data/raw/PPG-DaLiA --output-root data/processed
python -m src.data.export_wildppg_wrist --dataset-root data/raw/WildPPG-wrist --output-root data/processed
```

Run real external baselines:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/ppg_dalia_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --method peak --output-dir experiments/week5_ppgdalia_external_validation/runs/baseline_peak
python -m src.baseline.run_baseline --processed-manifest data/processed/ppg_dalia_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --method spectral --output-dir experiments/week5_ppgdalia_external_validation/runs/baseline_spectral
python -m src.baseline.run_baseline --processed-manifest data/processed/wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --method peak --output-dir experiments/week6_wildppg_wrist_external_validation/runs/baseline_peak
python -m src.baseline.run_baseline --processed-manifest data/processed/wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --method spectral --output-dir experiments/week6_wildppg_wrist_external_validation/runs/baseline_spectral
```

Build external status/result tables:

```bash
python -m src.utils.build_external_validation_status --experiments-root experiments --output-csv reports/external_validation_result_table.csv --output-md reports/external_validation_result_table.md
```

Run final statistics and freeze after final Week 4/7 artifacts exist:

```bash
python -m src.utils.build_week7_statistics --week4-root experiments/week4_galaxyppg_lightweight_router_2026-05-13 --output-root experiments/week7_final_statistics --router-feature-set motion_quality --router-type hard_gate
python -m src.utils.freeze_final_results --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --week4-root experiments/week4_galaxyppg_lightweight_router_2026-05-13 --week7-root experiments/week7_final_statistics
```

