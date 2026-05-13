# Week 2 Memo: GalaxyPPG Corrected Benchmark

## 1. Objective

Week 2 evaluates GalaxyPPG only, using subject-independent participant splits, corrected loader-level PPG inversion, ECG/IBI-derived beat-interval labels, and the configured harmonized/model-faithful preprocessing modes. No PPG-DaLiA or WildPPG experiments are included.

## 2. Repository and pipeline status

Week 1 prerequisites were present before this benchmark: GalaxyPPG inversion is explicit in the loader, the `canonical_ppg_v1` schema exists, labels use 10-second windows with 2-second stride and median instantaneous HR from beat intervals, and `configs/experiment_modes.json` defines `harmonized` and `model_faithful` modes.

Outputs are organized under `experiments/week2_galaxyppg_corrected_2026-05-01`. Standardized prediction CSVs are stored in `predictions/`, metrics in `metrics/`, benchmark tables in `tables/`, and figures in `figures/`.

## 3. Experimental design

- Dataset: GalaxyPPG
- Evaluation: subject-independent fixed held-out participant split with participant-level validation folds
- Window: 10 seconds, 2 seconds stride
- Label: median instantaneous HR from IBI/ECG beat intervals
- Metrics: MAE, RMSE, median absolute error, 95th percentile absolute error, catastrophic error rate above 20 bpm
- Prediction export: `experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv`

## 4. Methods compared

Completed runs discovered in the result tree:

| model_name | preprocessing_mode | normalization_strategy | probe_type | regressor_type | inversion | n_windows |
| --- | --- | --- | --- | --- | --- | --- |
| pulseppg | harmonized | per_window_zscore | NA | random_forest | True | 7631 |
| pulseppg | model_faithful | causal_running_zscore | NA | random_forest | True | 7631 |
| peak_based | harmonized | NA | NA | classical_peak | False | 7631 |
| peak_based | harmonized | NA | NA | classical_peak | True | 7629 |
| pulseppg | harmonized | per_window_zscore | ridge | ridge | True | 7631 |
| pulseppg | model_faithful | per_window_zscore | ridge | ridge | True | 7631 |
| pulseppg | model_faithful | none | ridge | ridge | True | 7631 |
| pulseppg | model_faithful | person_specific_zscore | ridge | ridge | True | 7631 |
| pulseppg | model_faithful | causal_running_zscore | ridge | ridge | True | 7631 |
| pulseppg | harmonized | per_window_zscore | ridge | ridge | False | 7631 |
| pulseppg | harmonized | per_window_zscore | linear | linear | True | 7631 |
| pulseppg | model_faithful | per_window_zscore | linear | linear | True | 7631 |
| pulseppg | model_faithful | none | linear | linear | True | 7631 |
| pulseppg | model_faithful | person_specific_zscore | linear | linear | True | 7631 |
| pulseppg | model_faithful | causal_running_zscore | linear | linear | True | 7631 |
| papagei | harmonized | per_window_zscore | ridge | ridge | False | 7631 |
| papagei | harmonized | per_window_zscore | ridge | ridge | True | 7631 |
| papagei | model_faithful | per_window_zscore | ridge | ridge | True | 7631 |
| papagei | harmonized | per_window_zscore | NA | random_forest | True | 7631 |
| papagei | model_faithful | per_window_zscore | NA | random_forest | True | 7631 |
| papagei | harmonized | per_window_zscore | linear | linear | True | 7631 |
| papagei | model_faithful | per_window_zscore | linear | linear | True | 7631 |
| spectral | harmonized | NA | NA | classical_spectral | False | 7631 |
| spectral | harmonized | NA | NA | classical_spectral | True | 7631 |

Native watch HR was unavailable in the processed GalaxyPPG cache used here; no native-HR result was fabricated.

## 5. Main benchmark results

Best by MAE: pulseppg / harmonized / per_window_zscore / random_forest (MAE=6.818)

Best by P95 AE: pulseppg / model_faithful / causal_running_zscore / random_forest (p95_absolute_error=25.546)

Best by catastrophic error rate: pulseppg / model_faithful / causal_running_zscore / random_forest (catastrophic_error_rate_20bpm=0.080)

The paper-facing tables are:

- `tables/main_benchmark_table.csv`
- `tables/harmonized_preprocessing_table.csv`
- `tables/model_faithful_preprocessing_table.csv`

## 6. Inversion ablation

The inversion ablation table is `tables/inversion_ablation_table.csv`. Positive delta values mean the non-inverted raw signal was worse than the corrected inverted signal.

| model_name | preprocessing_mode | normalization_strategy | probe_type | regressor_type | MAE_with_inversion | MAE_without_inversion | delta_MAE | RMSE_with_inversion | RMSE_without_inversion | delta_RMSE | median_absolute_error_with_inversion | median_absolute_error_without_inversion | delta_median_absolute_error | p95_absolute_error_with_inversion | p95_absolute_error_without_inversion | delta_p95_absolute_error | catastrophic_error_rate_20bpm_with_inversion | catastrophic_error_rate_20bpm_without_inversion | delta_catastrophic_error_rate_20bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pulseppg | harmonized | per_window_zscore | ridge | ridge | 8.0508 | 8.0647 | 0.0139 | 12.4977 | 12.3567 | -0.1410 | 4.8660 | 4.9205 | 0.0544 | 27.9149 | 26.9895 | -0.9254 | 0.0921 | 0.0946 | 0.0025 |
| spectral | harmonized | NA | NA | classical_spectral | 18.6033 | 18.6033 | 0.0000 | 29.5835 | 29.5835 | 0.0000 | 6.6629 | 6.6629 | 0.0000 | 63.0116 | 63.0116 | 0.0000 | 0.3285 | 0.3285 | 0.0000 |
| papagei | harmonized | per_window_zscore | ridge | ridge | 11.4621 | 11.4027 | -0.0594 | 16.5681 | 16.9615 | 0.3934 | 8.0845 | 7.9342 | -0.1503 | 35.8505 | 35.5326 | -0.3180 | 0.1407 | 0.1303 | -0.0105 |
| peak_based | harmonized | NA | NA | classical_peak | 7.8658 | 7.4705 | -0.3953 | 16.1052 | 16.0299 | -0.0753 | 2.8210 | 2.4523 | -0.3687 | 37.1319 | 38.0321 | 0.9002 | 0.1060 | 0.1008 | -0.0053 |

## 7. Preprocessing-mode comparison

The harmonized and model-faithful tables are stored separately. Interpret ranking changes only among completed runs with the same model/checkpoint availability.

## 8. Classical baseline regime analysis

Week 2 reports activity-level summaries only; deeper regime maps and oracle routing are intentionally left for Week 3. Activity-level metrics are stored in `metrics/activity_level_metrics.csv`.

## 9. PaPaGei assessment

PaPaGei should be retained as a benchmark when completed runs are available, but Week 2 treats it as a frozen-embedding comparator rather than a routing expert decision. Routing decisions are outside the Week 2 boundary.

## 10. Participant-level observations

Participant-level metrics are stored in `metrics/participant_level_metrics.csv`. These participant aggregates are the correct unit for later confidence intervals and paired significance tests.

## 11. Limitations

- Native watch HR was unavailable in the processed GalaxyPPG cache used here; no native-HR result was fabricated.
- No completed foundation-model run was missing from the discovered result set.
- Stronger regressors are reported only as practical upper bounds and should not be framed as the core novelty.
- The fixed split is subject-independent, but it is not leave-one-subject-out.

## 12. Conclusion

1. Did inversion materially change performance? Use `tables/inversion_ablation_table.csv`; positive deltas show the corrected inversion improved the metric.
2. Did model-faithful preprocessing change the ranking? Use the harmonized and model-faithful tables; ranking should be interpreted among matched completed runs.
3. Does the classical baseline remain competitive in any regimes? Use `metrics/activity_level_metrics.csv`; detailed regime analysis is Week 3.
4. Is PaPaGei still useful as a benchmark after the comparison is corrected? Yes, if completed frozen-embedding linear/Ridge runs are available; it should remain a benchmark comparator, not the main routing claim.

## 13. Reproducibility

- branch: `main`
- commit hash: `298f37dd5b4000158394a175817f55f5ce4f729e`
- tag name: `week2-galaxyppg-corrected-2026-05-01`
- configs used: `configs/galaxyppg_submission_split.json`, `configs/experiment_modes.json`
- main output location: `experiments/week2_galaxyppg_corrected_2026-05-01`
- working tree note: Week 2 code changes and generated artifacts are present in the working tree; the git tag points to the current HEAD because no commit was requested during this run.
