# Week 4 Memo: Lightweight Motion- and Quality-Aware Routing

## Objective

Week 4 implements the first deployable-style lightweight router between the selected classical expert and PulsePPG-style foundation expert from Week 3. The gate uses only inference-time motion and PPG-quality features, and is evaluated with participant-level leave-one-participant-out routing predictions across the available corrected GalaxyPPG held-out participants.

## Inputs

- Week 3 paired window features: `experiments/week3_galaxyppg_regime_oracle_2026-05-13/predictions/week3_window_regime_expert_errors.csv`
- Week 2 standardized predictions for peak/spectral auxiliary quality features: `experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv`
- Output root: `experiments/week4_galaxyppg_lightweight_router_2026-05-13`

## Gate Variants

- `motion_only`: accelerometer norm mean/std, dominant accelerometer frequency, cadence-band power fraction
- `quality_only`: PPG amplitude range, clipping rate, flatline rate, autocorrelation peak strength, spectral entropy, spectral peak sharpness, beat-count consistency, peak-vs-spectral HR disagreement
- `motion_quality`: all motion and quality features combined

For each feature set, the builder trains:

- `hard_gate`: logistic gate chooses either the classical or foundation expert.
- `soft_gate`: logistic probability is used as the foundation weight, and predictions are averaged.

## Main Result

Best learned router by MAE:

| method | feature_set | routing_type | n_windows | gate_accuracy | mean_foundation_weight | MAE | RMSE | median_absolute_error | p95_absolute_error | catastrophic_error_rate_20bpm | gain_vs_best_single_MAE | gain_vs_best_single_p95 | gain_vs_best_single_catastrophic_rate | oracle_gain_recovered_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learned_router | motion_quality | soft_gate | 7629 |  | 0.5040 | 6.4001 | 11.8270 | 2.7968 | 26.3065 | 0.0783 | 0.4181 | -0.5081 | 0.0017 | 0.1801 |

Best hard gate:

| method | feature_set | routing_type | n_windows | gate_accuracy | mean_foundation_weight | MAE | RMSE | median_absolute_error | p95_absolute_error | catastrophic_error_rate_20bpm | gain_vs_best_single_MAE | gain_vs_best_single_p95 | gain_vs_best_single_catastrophic_rate | oracle_gain_recovered_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learned_router | motion_quality | hard_gate | 7629 | 0.5318 | 0.4419 | 6.4426 | 11.2408 | 3.1821 | 25.5632 | 0.0750 | 0.3756 | 0.2352 | 0.0050 | 0.1618 |

Best soft gate:

| method | feature_set | routing_type | n_windows | gate_accuracy | mean_foundation_weight | MAE | RMSE | median_absolute_error | p95_absolute_error | catastrophic_error_rate_20bpm | gain_vs_best_single_MAE | gain_vs_best_single_p95 | gain_vs_best_single_catastrophic_rate | oracle_gain_recovered_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learned_router | motion_quality | soft_gate | 7629 |  | 0.5040 | 6.4001 | 11.8270 | 2.7968 | 26.3065 | 0.0783 | 0.4181 | -0.5081 | 0.0017 | 0.1801 |

Full routing summary:

| method | feature_set | routing_type | n_windows | gate_accuracy | mean_foundation_weight | MAE | RMSE | median_absolute_error | p95_absolute_error | catastrophic_error_rate_20bpm | gain_vs_best_single_MAE | gain_vs_best_single_p95 | gain_vs_best_single_catastrophic_rate | oracle_gain_recovered_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical_expert | NA | NA | 7629 |  |  | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 | -1.0475 | -11.3335 | -0.0261 | -0.4513 |
| foundation_expert | NA | NA | 7629 |  |  | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| learned_router | motion_quality | soft_gate | 7629 |  | 0.5040 | 6.4001 | 11.8270 | 2.7968 | 26.3065 | 0.0783 | 0.4181 | -0.5081 | 0.0017 | 0.1801 |
| learned_router | motion_quality | hard_gate | 7629 | 0.5318 | 0.4419 | 6.4426 | 11.2408 | 3.1821 | 25.5632 | 0.0750 | 0.3756 | 0.2352 | 0.0050 | 0.1618 |
| learned_router | quality_only | soft_gate | 7629 |  | 0.5084 | 6.4898 | 12.1463 | 2.8044 | 27.0068 | 0.0814 | 0.3284 | -1.2084 | -0.0014 | 0.1415 |
| learned_router | motion_only | soft_gate | 7629 |  | 0.4980 | 6.5120 | 12.0637 | 2.8014 | 27.6397 | 0.0835 | 0.3062 | -1.8413 | -0.0035 | 0.1319 |
| learned_router | motion_only | hard_gate | 7629 | 0.5461 | 0.1707 | 6.6802 | 11.9203 | 2.9121 | 27.8470 | 0.0907 | 0.1380 | -2.0486 | -0.0107 | 0.0595 |
| learned_router | quality_only | hard_gate | 7629 | 0.4833 | 0.4830 | 6.8085 | 11.8323 | 3.3752 | 26.4798 | 0.0798 | 0.0098 | -0.6814 | 0.0001 | 0.0042 |
| oracle_router | NA | NA | 7629 |  |  | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 | 2.3213 | 6.6933 | 0.0317 | 1.0000 |

## Gate Interpretation

Largest mean absolute standardized coefficients for the combined gate:

| feature | abs_coefficient |
| --- | --- |
| ppg_amplitude_range | 0.2606 |
| acc_norm_std | 0.2127 |
| peak_spectral_disagreement_bpm | 0.1977 |
| ppg_spectral_entropy | 0.1754 |
| acc_norm_mean | 0.1005 |
| ppg_autocorr_peak_strength | 0.0890 |
| beat_count_consistency | 0.0673 |
| ppg_flatline_rate | 0.0641 |

Positive coefficients push the hard gate toward the foundation expert. Negative coefficients push it toward the classical expert.

## Expert Decision

The Week 3 oracle table showed the main routing pair should remain `peak_based` plus `PulsePPG` for the first routed system. PaPaGei remains useful as a benchmark, but the current routed implementation keeps the paper's main system focused on one classical expert and one PulsePPG-based expert.

## Artifacts

- Routed predictions: `predictions/week4_routed_predictions.csv`
- Main comparison table: `tables/routing_summary.csv`
- Gate fold summary: `tables/gate_fold_summary.csv`
- Gate coefficients: `tables/gate_feature_coefficients.csv`
- Participant-level routing metrics: `tables/participant_level_routing_metrics.csv`
- Activity-level routing metrics: `tables/activity_level_routing_metrics.csv`
- Figures: `figures/hard_soft_router_mae.png`, `figures/best_router_error_cdf.png`, `figures/combined_gate_feature_coefficients.png`

## Limitations

This is a first GalaxyPPG router trained and evaluated only across the Week 2 held-out participants using participant-level out-of-fold gates. It is suitable for deciding whether hard/soft lightweight routing is promising, but final claims still require the later external-validation weeks.

## Reproducibility

- branch: `main`
- commit hash: `298f37dd5b4000158394a175817f55f5ce4f729e`
- command: `python -m src.utils.build_week4_artifacts --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --output-root experiments/week4_galaxyppg_lightweight_router_2026-05-13`
