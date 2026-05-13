# Week 2-4 Project Summary

## Executive Summary

Weeks 2-4 moved the project from a corrected GalaxyPPG benchmark into evidence for a motion- and quality-aware routing paper direction. Week 2 established the corrected GalaxyPPG subject-independent benchmark and saved paper-facing metrics. Week 3 tested whether estimator dominance is regime-dependent and found substantial oracle-routing headroom. Week 4 implemented the first lightweight learned router using inference-time motion and PPG quality features.

The current evidence supports continuing with a routed system built around one classical expert plus one PulsePPG-based expert. The strongest single expert in the Week 2 benchmark was PulsePPG harmonized random forest (MAE 6.8176 bpm, P95 AE 25.7977 bpm, catastrophic error rate 0.0799). The Week 3 oracle router reduced MAE to 4.4969 bpm and P95 AE to 19.1051 bpm on the paired expert windows, showing that the routing idea has meaningful headroom. The Week 4 learned combined motion-plus-quality router improved MAE over the best single expert, with the soft gate reaching MAE 6.4001 bpm and the hard gate improving tail robustness to P95 AE 25.5632 bpm and catastrophic error rate 0.0750.

## Week 2 Corrected Benchmark

Week 2 evaluated corrected GalaxyPPG with subject-independent held-out participants. The benchmark outputs are under `experiments/week2_galaxyppg_corrected_2026-05-01/`, with standardized predictions, overall metrics, participant-level metrics, activity-level metrics, benchmark tables, figures, and a memo.

Main corrected benchmark results from `tables/main_benchmark_table.md`:

| Method | Mode | Regressor | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PulsePPG | harmonized | random_forest | 6.8176 | 11.3431 | 3.6610 | 25.7977 | 0.0799 |
| PulsePPG | model_faithful causal_running_zscore | random_forest | 6.8606 | 11.3287 | 3.7344 | 25.5460 | 0.0798 |
| Peak-based classical | harmonized | classical_peak | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 |
| PulsePPG | harmonized | ridge | 8.0508 | 12.4977 | 4.8660 | 27.9149 | 0.0921 |
| PaPaGei | harmonized | ridge | 11.4621 | 16.5681 | 8.0845 | 35.8505 | 0.1407 |
| Spectral classical | harmonized | classical_spectral | 18.6033 | 29.5835 | 6.6629 | 63.0116 | 0.3285 |

The Week 2 artifact builder also produced participant-level results in `metrics/participant_level_metrics.csv` and activity-level results in `metrics/activity_level_metrics.csv`. These are the correct files for later participant-based confidence intervals and paired tests.

Inversion ablation was present but did not produce a simple uniform improvement across methods. In `tables/inversion_ablation_table.md`, non-inverted minus inverted MAE deltas were: PulsePPG ridge +0.0139, spectral 0.0000, PaPaGei ridge -0.0594, and peak-based -0.3953. Therefore, based on the current Week 2 table, inversion is methodologically required by the corrected pipeline but did not materially improve every benchmark metric in this held-out result set. The peak-based MAE was lower without inversion, while its P95 AE was worse without inversion by 0.9002 bpm.

Harmonized versus model-faithful preprocessing did not substantially change the broad ranking among completed runs. PulsePPG stayed strongest, peak-based classical remained competitive, PaPaGei stayed behind PulsePPG and peak-based classical, and spectral was weakest. Within PulsePPG ridge/linear probes, the normalization variants were nearly tied. The model-faithful random forest PulsePPG run slightly improved P95 AE versus harmonized random forest (25.5460 versus 25.7977) but had slightly worse MAE (6.8606 versus 6.8176).

Native watch HR output was not available in the processed GalaxyPPG cache used for Week 2. The Week 2 memo explicitly states that no native-HR result was fabricated. This remains missing / requires rerun only if a usable native watch HR signal is later added to the processed cache.

Week 2 conclusions:

- Inversion: methodologically corrected and included as an ablation; current measured performance impact is mixed rather than uniformly positive.
- Model-faithful preprocessing: did not change the main ranking among completed runs; PulsePPG remained strongest.
- Classical baseline: peak-based classical remained competitive, especially by median AE and in several low-motion or cleaner regimes.
- PaPaGei: still useful as a corrected benchmark comparator, but current GalaxyPPG results do not support making it the main routed expert.

## Week 3 Regime Analysis and Oracle Routing

Week 3 used `experiments/week3_galaxyppg_regime_oracle_2026-05-13/` to analyze complementarity between the selected classical expert and foundation-model expert:

- Classical expert: `peak_based/harmonized/NA/classical_peak`
- Foundation expert: `pulseppg/harmonized/per_window_zscore/random_forest`
- Paired valid windows: 7629

For every paired window, the Week 3 table records classical absolute error, foundation absolute error, the error gap `classical_abs_error - foundation_abs_error`, the winning expert, and regime features. The main per-window file is `predictions/week3_window_regime_expert_errors.csv`.

Oracle-routing result from `tables/selected_expert_oracle_summary.md`:

| Expert | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 |
| Foundation | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 |
| Oracle router | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 |

The oracle recovered a gain of 2.3213 bpm MAE versus the best single expert, 6.6933 bpm in P95 AE, and 0.0317 absolute catastrophic-error-rate reduction. This is large enough to justify implementing a learned router in Week 4.

Regime analysis showed that estimator dominance is not uniform. Activity summaries from `tables/regime_by_activity.md` show foundation dominance in high-motion activities such as jogging (foundation MAE 14.3685 versus classical MAE 29.1948) and running (27.5799 versus 40.4288). Classical was better in several lower-motion or cleaner regimes, including keyboard-typing, ssst-prep, rest-5, mobile-typing, rest-2, screen-reading, standing, and baseline. Walking was mixed: foundation had lower mean MAE, but classical won a slightly larger window share.

Motion-quality summaries from `tables/regime_by_motion_and_quality.md` also support complementarity. High-motion bins favored the foundation expert by mean error gap, especially high_motion/low_quality (classical MAE 20.0766, foundation MAE 15.4680, oracle MAE 10.5645). Lower-motion and mid-motion bins often favored the classical expert by mean gap.

Draft Week 3 figures were generated:

- `figures/oracle_vs_selected_experts.png`
- `figures/winner_rate_by_activity.png`
- `figures/motion_quality_error_gap_heatmap.png`
- `figures/window_error_gap_distribution.png`

## Week 4 Learned Routing

Week 4 implemented a lightweight router in `experiments/week4_galaxyppg_lightweight_router_2026-05-13/`. The gate is intentionally interpretable and uses participant-level leave-one-participant-out routing predictions over the Week 3 paired windows.

Routing feature sets:

- Motion-only: accelerometer norm mean, accelerometer norm standard deviation, dominant accelerometer frequency, cadence-band power fraction.
- Quality-only: PPG amplitude range, clipping rate, flatline rate, autocorrelation peak strength, spectral entropy, spectral peak sharpness, beat-count consistency, and peak-vs-spectral HR disagreement.
- Motion-plus-quality combined: all motion and quality features.

Routing types:

- Hard gate: logistic gate selects either classical or foundation.
- Soft gate: logistic probability is used as the foundation weight and combines both expert predictions.

Main learned routing results from `tables/routing_summary.md`:

| Method | Feature set | Routing type | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Foundation expert | NA | NA | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 |
| Motion+quality router | combined | soft_gate | 6.4001 | 11.8270 | 2.7968 | 26.3065 | 0.0783 |
| Motion+quality router | combined | hard_gate | 6.4426 | 11.2408 | 3.1821 | 25.5632 | 0.0750 |
| Quality-only router | quality_only | soft_gate | 6.4898 | 12.1463 | 2.8044 | 27.0068 | 0.0814 |
| Motion-only router | motion_only | soft_gate | 6.5120 | 12.0637 | 2.8014 | 27.6397 | 0.0835 |
| Oracle router | NA | NA | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 |

The best learned MAE came from the motion-plus-quality soft gate. The best tail robustness among learned routers came from the motion-plus-quality hard gate, which improved P95 AE and catastrophic error rate relative to the foundation expert. Motion-plus-quality combined outperformed motion-only and quality-only by MAE, supporting the title framing that both motion and quality cues matter.

The combined gate coefficient summary in `week4_lightweight_router.md` identified the largest mean absolute standardized coefficients as PPG amplitude range, accelerometer norm standard deviation, peak-spectral disagreement, PPG spectral entropy, accelerometer norm mean, PPG autocorrelation peak strength, beat-count consistency, and PPG flatline rate.

Final expert selection recommendation: keep the main routed system focused on peak-based classical plus PulsePPG. Keep PaPaGei in the paper as a benchmark comparator unless later external validation changes its role.

## Key Findings

- The corrected GalaxyPPG benchmark favors PulsePPG random forest overall, but the peak-based classical method remains competitive in cleaner and lower-motion regimes.
- Oracle routing substantially improves both average and tail metrics, so the core routing idea is supported by the Week 3 evidence.
- Learned lightweight routing already improves MAE over the best single expert; the combined hard gate is better for robustness metrics than the best single expert.
- Motion-plus-quality features are more useful than either motion-only or quality-only in the first learned router.
- PaPaGei remains useful as a corrected benchmark, but current results do not justify using it as the main routed foundation expert.

## Current Risks / Missing Items

- Native watch HR output is missing / not available in the processed Week 2 cache.
- Inversion ablation is mixed; it supports methodological correction but does not currently show a uniform performance improvement.
- Week 2 uses a fixed subject-independent held-out split, not leave-one-subject-out or repeated grouped cross-validation.
- Learned Week 4 routing was evaluated only on GalaxyPPG held-out participants, not on external datasets.
- Participant-level confidence intervals and paired significance tests for Week 2-4 are not yet present in the current artifacts.
- Large prediction CSVs and generated experimental artifacts exist in the working tree but should not be committed as part of this report-only change.

## Recommended Next Steps Before External Validation

- Freeze the Week 2-4 result directories used in this summary and avoid mixing them with later reruns.
- Add participant-level confidence intervals and paired tests for best single expert versus learned router and oracle, especially for MAE, P95 AE, and catastrophic error rate.
- Decide whether to report soft-gate MAE or hard-gate robustness as the primary Week 4 result; the current evidence favors describing both.
- If native watch HR becomes available, rerun the Week 2 benchmark with that comparator.
- Before moving to external validation, keep the main routed pair as peak-based classical plus PulsePPG and treat PaPaGei as a benchmark unless new evidence changes this.
