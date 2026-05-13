# Week 2-4 Deliverable Report

This report covers only Week 2, Week 3, and Week 4 deliverables from `new_plan.docx`. It does not summarize Week 1 or plan Week 5+ work.

## Scope

The project direction for Weeks 2-4 is to move from a corrected GalaxyPPG benchmark to evidence for regime-dependent complementarity, then to a first lightweight motion- and quality-aware router.

Primary output roots:

- Week 2: `experiments/week2_galaxyppg_corrected_2026-05-01/`
- Week 3: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/`
- Week 4: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/`
- PI-facing summary: `reports/week2_4_summary.md`
- Reproducibility notes: `reports/week2_4_reproducibility.md`

## Week 2: Corrected GalaxyPPG Benchmark

### Required deliverables from plan

- Subject-independent GalaxyPPG experiments.
- Corrected inversion-aware benchmark.
- Harmonized preprocessing and model-faithful preprocessing.
- Peak-based baseline, spectral baseline, PulsePPG, PaPaGei, and native watch HR if usable.
- Prediction, metrics, participant-level, and activity-level outputs.
- Metrics: MAE, RMSE, median absolute error, 95th percentile absolute error, catastrophic error rate above 20 bpm.
- Written conclusions addressing inversion, preprocessing mode, classical competitiveness, and PaPaGei usefulness.

### Delivered artifacts

- Standardized predictions: `experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv`
- Overall metrics: `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/overall_metrics.csv`
- Participant-level metrics: `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/participant_level_metrics.csv`
- Activity-level metrics: `experiments/week2_galaxyppg_corrected_2026-05-01/metrics/activity_level_metrics.csv`
- Main benchmark table: `experiments/week2_galaxyppg_corrected_2026-05-01/tables/main_benchmark_table.md`
- Harmonized table: `experiments/week2_galaxyppg_corrected_2026-05-01/tables/harmonized_preprocessing_table.md`
- Model-faithful table: `experiments/week2_galaxyppg_corrected_2026-05-01/tables/model_faithful_preprocessing_table.md`
- Inversion ablation table: `experiments/week2_galaxyppg_corrected_2026-05-01/tables/inversion_ablation_table.md`
- Week 2 memo: `experiments/week2_galaxyppg_corrected_2026-05-01/week2_memo.md`
- Figures:
  - `experiments/week2_galaxyppg_corrected_2026-05-01/figures/activity_level_mae_by_model.png`
  - `experiments/week2_galaxyppg_corrected_2026-05-01/figures/participant_level_mae_by_model.png`
  - `experiments/week2_galaxyppg_corrected_2026-05-01/figures/error_distribution_by_model.png`
  - `experiments/week2_galaxyppg_corrected_2026-05-01/figures/inversion_ablation_summary.png`

### Key results

From `tables/main_benchmark_table.md`, the best overall completed run was PulsePPG harmonized random forest:

- MAE: 6.8176 bpm
- RMSE: 11.3431 bpm
- Median AE: 3.6610 bpm
- P95 AE: 25.7977 bpm
- Catastrophic error rate: 0.0799

Other relevant rows:

- PulsePPG model-faithful causal-running-zscore random forest: MAE 6.8606, P95 AE 25.5460.
- Peak-based classical: MAE 7.8658, median AE 2.8210, P95 AE 37.1319.
- PaPaGei ridge: MAE 11.4621, P95 AE 35.8505.
- Spectral classical: MAE 18.6033, P95 AE 63.0116.

### Week 2 conclusions

- Inversion was included as a formal ablation. The current measured effect is mixed rather than uniformly positive. The correction remains methodologically required, but the table should not be described as a universal performance improvement.
- Harmonized versus model-faithful preprocessing did not substantially change the main ranking among completed runs. PulsePPG remained strongest, peak-based classical remained competitive, PaPaGei trailed PulsePPG and peak-based classical, and spectral was weakest.
- The peak-based classical baseline remains competitive in lower-motion and cleaner regimes, especially by median absolute error.
- PaPaGei remains useful as a corrected benchmark comparator, but current GalaxyPPG results do not support making it the main routed expert.
- Native watch HR output was missing / not available in the processed cache; no native-HR result was fabricated.

## Week 3: Regime Analysis and Oracle Routing

### Required deliverables from plan

- Compute per-window absolute error for classical expert and foundation-model expert.
- Compute the error gap between them.
- Analyze winners by activity, participant, HR range, accelerometer-derived motion intensity, and PPG signal-quality indicators.
- Compute an oracle router that chooses the expert with lower true error for each window.
- Decide whether routing is worth pursuing.
- Produce a regime analysis document, oracle-routing result table, and at least one draft figure.

### Delivered artifacts

- Window-level paired expert errors and regime features: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_window_regime_expert_errors.csv`
- Oracle predictions: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_oracle_router_predictions.csv`
- Oracle summary: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/selected_expert_oracle_summary.md`
- All classical/foundation pair oracle audit table: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/oracle_all_pairs_table.md`
- Regime tables:
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_activity.md`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_participant.md`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_hr_range.md`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_intensity.md`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_ppg_quality.md`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/tables/regime_by_motion_and_quality.md`
- Regime memo: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/week3_regime_analysis.md`
- Draft figures:
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/oracle_vs_selected_experts.png`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/winner_rate_by_activity.png`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/motion_quality_error_gap_heatmap.png`
  - `experiments/week3_galaxyppg_regime_oracle_2026-05-13/figures/window_error_gap_distribution.png`

### Selected experts

- Classical expert: `peak_based/harmonized/NA/classical_peak`
- Foundation-model expert: `pulseppg/harmonized/per_window_zscore/random_forest`
- Paired valid windows: 7629

### Oracle result

From `tables/selected_expert_oracle_summary.md`:

- Classical expert MAE: 7.8658; P95 AE: 37.1319; catastrophic error rate: 0.1060.
- Foundation expert MAE: 6.8182; P95 AE: 25.7984; catastrophic error rate: 0.0800.
- Oracle router MAE: 4.4969; P95 AE: 19.1051; catastrophic error rate: 0.0482.

The oracle improves over the best single expert by 2.3213 bpm MAE, 6.6933 bpm P95 AE, and 0.0317 absolute catastrophic-error-rate reduction. This is large enough to justify proceeding to learned routing.

### Regime findings

- Foundation dominated high-motion activities by mean error, including jogging and running.
- Classical remained better in several lower-motion or cleaner activities, including keyboard-typing, ssst-prep, rest-5, mobile-typing, rest-2, screen-reading, standing, and baseline.
- Motion-quality bins showed the largest foundation advantage under high_motion/low_quality, while lower-motion bins often favored the classical method.
- This supports the central claim that estimator dominance is regime-dependent rather than globally fixed.

## Week 4: Lightweight Learned Routing

### Required deliverables from plan

- Implement a lightweight hard gate that chooses one expert.
- Implement a lightweight soft gate that combines predictions with a learned weight.
- Avoid large end-to-end neural mixture-of-experts.
- Use inference-time motion and quality features.
- Train motion-only, quality-only, and motion-plus-quality combined gates.
- Compare hard versus soft routing.
- Decide which experts should remain in the final paper.

### Delivered artifacts

- Routed predictions: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/predictions/week4_routed_predictions.csv`
- Main routing summary: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.md`
- Fold summary: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_fold_summary.md`
- Gate coefficients: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/gate_feature_coefficients.md`
- Participant-level routing metrics: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/participant_level_routing_metrics.md`
- Activity-level routing metrics: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/activity_level_routing_metrics.md`
- Week 4 memo: `experiments/week4_galaxyppg_lightweight_router_2026-05-13/week4_lightweight_router.md`
- Figures:
  - `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/hard_soft_router_mae.png`
  - `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/best_router_error_cdf.png`
  - `experiments/week4_galaxyppg_lightweight_router_2026-05-13/figures/combined_gate_feature_coefficients.png`

### Routing features

Motion features:

- `acc_norm_mean`
- `acc_norm_std`
- `acc_dominant_frequency_hz`
- `acc_cadence_band_power_fraction`

Quality features:

- `ppg_amplitude_range`
- `ppg_clipping_rate`
- `ppg_flatline_rate`
- `ppg_autocorr_peak_strength`
- `ppg_spectral_entropy`
- `ppg_spectral_peak_sharpness`
- `beat_count_consistency`
- `peak_spectral_disagreement_bpm`

### Learned routing result

From `tables/routing_summary.md`:

- Best single foundation expert: MAE 6.8182, P95 AE 25.7984, catastrophic error rate 0.0800.
- Motion-plus-quality soft gate: MAE 6.4001, P95 AE 26.3065, catastrophic error rate 0.0783.
- Motion-plus-quality hard gate: MAE 6.4426, P95 AE 25.5632, catastrophic error rate 0.0750.
- Quality-only soft gate: MAE 6.4898.
- Motion-only soft gate: MAE 6.5120.
- Oracle router remains the upper bound: MAE 4.4969, P95 AE 19.1051, catastrophic error rate 0.0482.

### Week 4 conclusions

- The first learned router improves MAE over the best single expert.
- The motion-plus-quality hard gate gives the best learned tail robustness among the routed variants.
- Combined motion-plus-quality features outperform motion-only and quality-only by MAE, supporting the planned title framing.
- The recommended main routed system is peak-based classical plus PulsePPG.
- PaPaGei should remain in the paper as a benchmark unless later external validation changes its role.

## Missing or Requires Rerun

- Native watch HR output is missing from the current processed Week 2 cache.
- Week 2 used a fixed subject-independent held-out split, not full LOSO or repeated grouped cross-validation.
- Participant-level confidence intervals and paired significance tests are not yet generated for Week 2-4.
- Week 4 routing is currently GalaxyPPG-only; external validation remains future work.
- Large prediction CSVs and model/embedding caches are intentionally not part of the minimal git submission.

## Minimal Reproduction Files Submitted

The git submission should include:

- Week 2-4 report files under `reports/`.
- This weekplan report.
- Week 2 configs under `configs/`.
- README Week 2-4 command sections.
- Week 2-4 artifact builders under `src/utils/`.
- Source changes required for raw/canonical inversion ablations and expanded robustness metrics.
- Small result summaries, metrics, tables, memos, and figures for Week 2-4.

The git submission should exclude:

- Raw data.
- Processed data.
- Model weights.
- Feature `.npy` caches.
- Estimator `.joblib` files.
- Large prediction CSVs.
- Full `runs/` directories.
