# Week 2-4 Reproducibility Notes

This file lists only Week 2-4 commands, configuration files, result paths, and figure paths. Week 1 preprocessing is referenced only where it is a required dependency for the Week 2-4 experiments.

## Required Existing Inputs

- Corrected processed manifest: `data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json`
- Corrected processed windows: `data/processed/windows/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_windows.jsonl.gz`
- Fixed split config: `configs/galaxyppg_submission_split.json`
- Experiment mode config: `configs/experiment_modes.json`
- Week 2 configs:
  - `configs/week2_galaxyppg_harmonized.json`
  - `configs/week2_galaxyppg_model_faithful.json`
  - `configs/week2_galaxyppg_inversion_ablation.json`

## Week 2 Corrected GalaxyPPG Benchmark

Output root:

```text
experiments/week2_galaxyppg_corrected_2026-05-01/
```

Classical baseline commands recorded in `README.md`:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --ppg-source canonical --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/baseline_peak
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method spectral --ppg-source canonical --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/baseline_spectral
```

Inversion ablation baseline commands recorded in `README.md`:

```bash
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method peak --ppg-source raw --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/baseline_peak_raw
python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --split-config configs/galaxyppg_submission_split.json --method spectral --ppg-source raw --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/inversion_ablation/baseline_spectral_raw
```

Foundation feature extraction commands recorded in `README.md`:

```bash
python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features --batch-size 256 --device cpu
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json --experiment-config configs/experiment_modes.json --experiment-mode harmonized --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/papagei_features --batch-size 128 --device cpu
```

Example downstream regressor commands recorded in `README.md`:

```bash
python -m src.regression.train_regressor --feature-manifest experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features/pulseppg_manifest.json --regressor ridge --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_ridge
python -m src.regression.train_regressor --feature-manifest experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_features/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/week2_galaxyppg_corrected_2026-05-01/runs/harmonized/pulseppg_random_forest
```

Week 2 artifact build command:

```bash
python -m src.utils.build_week2_artifacts --search-root experiments/week2_galaxyppg_corrected_2026-05-01/runs --output-root experiments/week2_galaxyppg_corrected_2026-05-01 --tag-name week2-galaxyppg-corrected-2026-05-01
```

Week 2 result files:

- `experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/overall_metrics.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/overall_metrics.json`
- `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/participant_level_metrics.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/activity_level_metrics.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/main_benchmark_table.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/main_benchmark_table.md`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/harmonized_preprocessing_table.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/harmonized_preprocessing_table.md`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/model_faithful_preprocessing_table.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/model_faithful_preprocessing_table.md`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/inversion_ablation_table.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/tables/inversion_ablation_table.md`
- `experiments/week2_galaxyppg_corrected_2026-05-01/run_index.csv`
- `experiments/week2_galaxyppg_corrected_2026-05-01/week2_memo.md`

Week 2 figures:

- `experiments/week2_galaxyppg_corrected_2026-05-01/figures/activity_level_mae_by_model.png`
- `experiments/week2_galaxyppg_corrected_2026-05-01/figures/error_distribution_by_model.png`
- `experiments/week2_galaxyppg_corrected_2026-05-01/figures/inversion_ablation_summary.png`
- `experiments/week2_galaxyppg_corrected_2026-05-01/figures/participant_level_mae_by_model.png`

Native watch HR output:

- Missing / not found in the processed Week 2 cache; `week2_memo.md` states no native-HR result was fabricated.

## Week 3 Regime Analysis and Oracle Routing

Output root:

```text
experiments/week3_galaxyppg_regime_oracle_2026-05-13/
```

Command:

```bash
python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root experiments/week3_galaxyppg_regime_oracle_2026-05-13
```

Week 3 result files:

- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_window_regime_expert_errors.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_oracle_router_predictions.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/metrics/selected_expert_oracle_summary.json`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/selected_expert_oracle_summary.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/selected_expert_oracle_summary.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/oracle_all_pairs_table.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/oracle_all_pairs_table.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_activity.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_activity.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_participant.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_participant.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_hr_range.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_hr_range.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_intensity.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_intensity.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_ppg_quality.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_ppg_quality.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_and_quality.csv`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_and_quality.md`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/week3_regime_analysis.md`

Week 3 figures:

- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/oracle_vs_selected_experts.png`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/winner_rate_by_activity.png`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/motion_quality_error_gap_heatmap.png`
- `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/window_error_gap_distribution.png`

## Week 4 Learned Lightweight Routing

Output root:

```text
experiments/week4_galaxyppg_lightweight_router_2026-05-13/
```

Command:

```bash
python -m src.utils.build_week4_artifacts --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --output-root experiments/week4_galaxyppg_lightweight_router_2026-05-13
```

Week 4 result files:

- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/predictions/week4_routed_predictions.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/metrics/routing_summary.json`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.md`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_fold_summary.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_fold_summary.md`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_feature_coefficients.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_feature_coefficients.md`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/participant_level_routing_metrics.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/participant_level_routing_metrics.md`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/activity_level_routing_metrics.csv`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/activity_level_routing_metrics.md`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/week4_lightweight_router.md`

Week 4 figures:

- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/hard_soft_router_mae.png`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/best_router_error_cdf.png`
- `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/combined_gate_feature_coefficients.png`

## Verification Commands Run for These Report Files

These commands were used to inspect the current repository state and generated Week 2-4 report-only commit:

```bash
git status --short
git diff -- reports/week2_4_summary.md reports/week2_4_reproducibility.md README.md
git status --short
git diff --cached
```

No long-running training rerun was performed for this report.
