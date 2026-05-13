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
