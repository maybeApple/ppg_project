# Week 3 Memo: GalaxyPPG Regime Analysis and Oracle Routing

## Objective

Week 3 tests whether estimator dominance is regime-dependent on the corrected GalaxyPPG benchmark. It compares one selected classical expert with one selected foundation-model expert on the same windows, computes per-window error gaps, summarizes winners by activity, participant, HR range, motion intensity, and PPG quality, and reports an oracle router upper bound.

## Inputs

- Standardized Week 2 predictions: `experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv`
- Processed windows with PPG waveforms and labels: `data/processed/windows/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_windows.jsonl.gz`
- Output root: `experiments/week3_galaxyppg_regime_oracle_2026-05-13`

## Selected Experts

- Classical expert: `peak_based/harmonized/NA/classical_peak`
- Foundation expert: `pulseppg/harmonized/per_window_zscore/random_forest`

The default selection chooses the lowest-MAE inverted classical run and the lowest-MAE inverted foundation-model run. The builder also writes `tables/oracle_all_pairs_table.csv` so this choice can be audited against every completed classical/foundation pair.

## Oracle Result

The selected-pair oracle chooses, for each window, the expert with the smaller true absolute error. It is not deployable; it estimates routing headroom.

| expert | run | MAE | RMSE | median_absolute_error | p95_absolute_error | catastrophic_error_rate_20bpm | n_windows | gain_vs_best_single_MAE | gain_vs_best_single_p95 | gain_vs_best_single_catastrophic_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical | peak_based/harmonized/NA/classical_peak | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 | 7629 | -1.0475 | -11.3335 | -0.0261 |
| foundation | pulseppg/harmonized/per_window_zscore/random_forest | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 | 7629 | 0.0000 | 0.0000 | 0.0000 |
| oracle_router | per-window lower true error | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 | 7629 | 2.3213 | 6.6933 | 0.0317 |

Best single-expert MAE was `6.818` bpm. The oracle MAE was `4.497` bpm, for a headroom of `2.321` bpm. Best single-expert P95 AE was `25.798` bpm; oracle P95 AE was `19.105` bpm.

## Regime Findings

Positive error gap means the foundation expert had lower absolute error than the classical expert. Negative error gap means the classical expert won.

Activity summary:

| activity | n_windows | classical_MAE | foundation_MAE | oracle_MAE | mean_error_gap_classical_minus_foundation | median_error_gap_classical_minus_foundation | classical_win_rate | foundation_win_rate | tie_rate | oracle_gain_vs_best_single_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jogging | 280 | 29.1948 | 14.3685 | 12.3562 | 14.8263 | 14.0846 | 0.2464 | 0.7536 | 0.0000 | 2.0123 |
| running | 277 | 40.4288 | 27.5799 | 21.5881 | 12.8489 | 11.7550 | 0.4116 | 0.5884 | 0.0000 | 5.9918 |
| tsst-speech | 435 | 8.3926 | 5.1951 | 3.7368 | 3.1975 | 0.8156 | 0.4115 | 0.5885 | 0.0000 | 1.4583 |
| rest-4 | 280 | 11.2074 | 8.1527 | 5.5024 | 3.0548 | -0.3977 | 0.5357 | 0.4643 | 0.0000 | 2.6502 |
| walking | 278 | 18.5971 | 16.4837 | 10.0309 | 2.1134 | -2.2357 | 0.5288 | 0.4712 | 0.0000 | 6.4528 |
| ssst-sing | 59 | 11.3962 | 9.9331 | 7.4914 | 1.4631 | 0.1157 | 0.4915 | 0.5085 | 0.0000 | 2.4417 |
| tsst-prep | 430 | 4.9239 | 4.4270 | 2.8575 | 0.4970 | -0.2110 | 0.5186 | 0.4814 | 0.0000 | 1.5695 |
| rest-3 | 283 | 3.9994 | 3.7414 | 2.8445 | 0.2579 | 0.0928 | 0.4770 | 0.5230 | 0.0000 | 0.8969 |

Motion-quality summary:

| motion_intensity_bin | ppg_autocorr_quality_bin | n_windows | classical_MAE | foundation_MAE | oracle_MAE | mean_error_gap_classical_minus_foundation | median_error_gap_classical_minus_foundation | classical_win_rate | foundation_win_rate | tie_rate | oracle_gain_vs_best_single_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_motion | low_quality | 1007 | 20.0766 | 15.4680 | 10.5645 | 4.6086 | 0.0755 | 0.4995 | 0.5005 | 0.0000 | 4.9035 |
| high_motion | mid_quality | 770 | 16.5361 | 13.1860 | 9.1216 | 3.3501 | -0.0807 | 0.5039 | 0.4961 | 0.0000 | 4.0643 |
| high_motion | high_quality | 766 | 12.9113 | 9.6113 | 6.5032 | 3.3000 | 0.5455 | 0.4634 | 0.5366 | 0.0000 | 3.1081 |
| low_motion | high_quality | 897 | 3.1062 | 2.9501 | 2.0562 | 0.1560 | 0.1984 | 0.4560 | 0.5440 | 0.0000 | 0.8940 |
| mid_motion | high_quality | 880 | 4.3122 | 4.6381 | 2.8805 | -0.3259 | -0.2270 | 0.5307 | 0.4693 | 0.0000 | 1.4317 |
| low_motion | low_quality | 685 | 2.8212 | 3.2036 | 2.1073 | -0.3824 | -0.2436 | 0.5518 | 0.4482 | 0.0000 | 0.7139 |
| mid_motion | mid_quality | 812 | 3.6523 | 4.1116 | 2.4011 | -0.4593 | -0.3138 | 0.5530 | 0.4470 | 0.0000 | 1.2511 |
| mid_motion | low_quality | 851 | 4.0900 | 4.5650 | 2.6701 | -0.4750 | -0.4464 | 0.5699 | 0.4301 | 0.0000 | 1.4199 |

## Artifacts

- Window-level paired errors and regime features: `predictions/week3_window_regime_expert_errors.csv`
- Oracle predictions: `predictions/week3_oracle_router_predictions.csv`
- Selected-pair oracle table: `tables/selected_expert_oracle_summary.csv`
- All-pair oracle audit table: `tables/oracle_all_pairs_table.csv`
- Regime tables: `tables/regime_by_activity.csv`, `tables/regime_by_participant.csv`, `tables/regime_by_hr_range.csv`, `tables/regime_by_motion_intensity.csv`, `tables/regime_by_ppg_quality.csv`, `tables/regime_by_motion_and_quality.csv`
- Draft figures: `figures/oracle_vs_selected_experts.png`, `figures/winner_rate_by_activity.png`, `figures/motion_quality_error_gap_heatmap.png`, `figures/window_error_gap_distribution.png`

## Feature Notes

- No missing feature inputs detected.

Motion bins are tertiles of accelerometer-norm standard deviation. PPG quality bins are tertiles of autocorrelation peak strength in plausible HR-period lags. These are descriptive Week 3 regime features, not a trained router.

## Conclusion

The oracle gain quantifies whether routing is worth pursuing before Week 4. A meaningful oracle improvement, especially in P95 AE or catastrophic error rate, supports the paper direction; a negligible oracle improvement would weaken it. The generated all-pair table should be used to choose which experts remain in the Week 4 routed system.

## Reproducibility

- branch: `main`
- commit hash: `298f37dd5b4000158394a175817f55f5ce4f729e`
- command: `python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root experiments/week3_galaxyppg_regime_oracle_2026-05-13`
