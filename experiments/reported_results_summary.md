# Reported Results Summary

- Fixed split config: `configs/galaxyppg_submission_split.json`
- Processed manifest: `data/processed/galaxyppg_hr_w10_s2_median_manifest.json`

| Name | Kind | MAE | RMSE | Result Dir |
| --- | --- | ---: | ---: | --- |
| peak | baseline | 7.641879 | 15.847827 | `experiments/baseline_results/2026-03-11` |
| spectral | baseline | 18.733783 | 29.436620 | `experiments/baseline_results/2026-03-11` |
| peak | baseline | 7.641879 | 15.847827 | `experiments/reproduced_submission/baseline_peak` |
| pulseppg_linear | regression | 7.948564 | 12.143205 | `experiments/pulseppg_results/2026-03-18/regression_linear` |
| pulseppg_ridge | regression | 7.871293 | 12.080186 | `experiments/pulseppg_results/2026-03-18/regression_ridge` |
| pulseppg_gradient_boosting | regression | 7.492755 | 11.809258 | `experiments/pulseppg_results/2026-03-23/regression_gradient_boosting` |
| pulseppg_random_forest | regression | 7.286408 | 11.303921 | `experiments/pulseppg_results/2026-03-23/regression_random_forest` |
| pulseppg_random_forest | regression | 7.286408 | 11.303921 | `experiments/reproduced_submission/pulseppg_random_forest` |
| papagei_linear | regression | 11.241105 | 17.039752 | `experiments/papagei_results/2026-03-18/regression_linear` |
| papagei_ridge | regression | 11.057308 | 16.529641 | `experiments/papagei_results/2026-03-18/regression_ridge` |
| papagei_gradient_boosting | regression | 11.909232 | 16.137550 | `experiments/papagei_results/2026-03-23/regression_gradient_boosting` |
| papagei_random_forest | regression | 11.532532 | 15.724710 | `experiments/papagei_results/2026-03-23/regression_random_forest` |
