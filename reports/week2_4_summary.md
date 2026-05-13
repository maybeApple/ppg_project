# Week 2-4 Project Summary

## Executive summary

Weeks 2-4 moved the project from a corrected GalaxyPPG benchmark into evidence for a motion- and quality-aware routing paper direction. Week 2 established the corrected GalaxyPPG subject-independent benchmark. Week 3 tested whether estimator dominance is regime-dependent and quantified the oracle-routing upper bound. Week 4 implemented the first lightweight learned router using inference-time motion and PPG quality features.

The current evidence supports continuing with a routed system built around one classical expert plus one PulsePPG-based expert. The strongest single expert in the Week 2 benchmark was **PulsePPG harmonized random forest**, with **MAE = 6.8176 bpm**, **P95 absolute error = 25.7977 bpm**, and **catastrophic error rate = 0.0799**. The Week 3 oracle router reduced MAE to **4.4969 bpm** and P95 absolute error to **19.1051 bpm** on paired expert windows, showing meaningful routing headroom. The Week 4 learned combined motion-plus-quality router improved MAE over the best single expert, with the soft gate reaching **MAE = 6.4001 bpm** and the hard gate improving tail robustness to **P95 absolute error = 25.5632 bpm** and **catastrophic error rate = 0.0750**.

Important repository note: the latest `main` branch documents the Week 2-4 commands and expected output paths, but generated result files were pruned from the committed package. The numerical results below are summarized from the earlier Week 2-4 report commit and the pruned artifact diff in the repository history, not from currently committed CSV artifacts on `main`.

## Week 2 corrected GalaxyPPG benchmark

Week 2 evaluated corrected GalaxyPPG with subject-independent held-out participants. The benchmark was subject-independent, but it used a fixed held-out split rather than LOSO or repeated grouped cross-validation.

### Main benchmark results

| Method | Mode | Regressor | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PulsePPG | harmonized | random_forest | 6.8176 | 11.3431 | 3.6610 | 25.7977 | 0.0799 |
| PulsePPG | model_faithful causal_running_zscore | random_forest | 6.8606 | 11.3287 | 3.7344 | 25.5460 | 0.0798 |
| Peak-based classical | harmonized | classical_peak | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 |
| PulsePPG | harmonized | ridge | 8.0508 | 12.4977 | 4.8660 | 27.9149 | 0.0921 |
| PaPaGei | harmonized | ridge | 11.4621 | 16.5681 | 8.0845 | 35.8505 | 0.1407 |
| Spectral classical | harmonized | classical_spectral | 18.6033 | 29.5835 | 6.6629 | 63.0116 | 0.3285 |

### Week 2 interpretation

- **Best single expert:** PulsePPG with harmonized preprocessing and Random Forest downstream regressor.
- **Inversion ablation:** mixed rather than uniformly positive. Inversion remains methodologically required for GalaxyPPG, but the held-out metrics do not show uniform improvement for every estimator.
- **Harmonized vs model-faithful preprocessing:** the broad ranking did not substantially change among completed runs. PulsePPG remained strongest; peak-based classical remained competitive; PaPaGei stayed behind PulsePPG and peak-based classical.
- **Classical baseline:** peak-based classical remains scientifically important because it performs well in several cleaner / lower-motion regimes and has a strong median AE.
- **PaPaGei role:** PaPaGei is useful as a corrected benchmark comparator, but current GalaxyPPG results do not support using it as the main routed expert.
- **Native watch HR output:** missing / not found. No native-HR result should be fabricated.

## Week 3 regime analysis and oracle routing

Week 3 used paired expert predictions to analyze whether estimator dominance is regime-dependent.

Selected experts:

- Classical expert: `peak_based / harmonized / classical_peak`
- Foundation expert: `PulsePPG / harmonized / per_window_zscore / random_forest`
- Paired valid windows: 7629

### Oracle-routing result

| Expert | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical | 7.8658 | 16.1052 | 2.8210 | 37.1319 | 0.1060 |
| Foundation | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 |
| Oracle router | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 |

The oracle router improved over the best single expert by **2.3213 bpm MAE**, **6.6933 bpm P95 AE**, and **0.0317 absolute catastrophic-error-rate reduction**. This is strong evidence that expert complementarity exists and that routing is worth pursuing.

### Regime-level interpretation

- Foundation-model estimation is favored in high-motion regimes, especially jogging and running.
- Peak-based classical estimation remains competitive or better in several lower-motion or cleaner regimes, including keyboard typing, rest periods, screen reading, standing, and baseline.
- Motion-quality regimes show meaningful complementarity: high-motion / low-quality windows favor the foundation expert, while lower-motion and cleaner windows often favor the classical expert.
- The Week 3 evidence supports the paper framing: no single estimator is uniformly optimal across wrist PPG conditions.

## Week 4 learned lightweight routing

Week 4 implemented a simple and interpretable learned router on GalaxyPPG.

Routing variants:

- `hard_gate`: logistic gate chooses either the classical expert or the foundation expert.
- `soft_gate`: logistic probability is used as the foundation-model weight and combines predictions.
- Feature ablations: `motion_only`, `quality_only`, and `motion_quality`.

Feature groups:

- Motion features: accelerometer norm mean/std, dominant accelerometer frequency, cadence-band power fraction.
- Quality features: PPG amplitude range, clipping rate, flatline rate, autocorrelation peak strength, spectral entropy, spectral peak sharpness, beat-count consistency, peak-vs-spectral HR disagreement.

### Learned-routing results

| Method | Feature set | Routing type | MAE | RMSE | Median AE | P95 AE | Catastrophic >20 bpm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Foundation expert | NA | NA | 6.8182 | 11.3442 | 3.6610 | 25.7984 | 0.0800 |
| Motion+quality router | combined | soft_gate | 6.4001 | 11.8270 | 2.7968 | 26.3065 | 0.0783 |
| Motion+quality router | combined | hard_gate | 6.4426 | 11.2408 | 3.1821 | 25.5632 | 0.0750 |
| Quality-only router | quality_only | soft_gate | 6.4898 | 12.1463 | 2.8044 | 27.0068 | 0.0814 |
| Motion-only router | motion_only | soft_gate | 6.5120 | 12.0637 | 2.8014 | 27.6397 | 0.0835 |
| Oracle router | NA | NA | 4.4969 | 8.9193 | 1.8548 | 19.1051 | 0.0482 |

### Week 4 interpretation

- The **motion+quality soft gate** achieved the best learned-router MAE.
- The **motion+quality hard gate** achieved better tail robustness than the foundation expert, including lower P95 AE and lower catastrophic error rate.
- Combined motion+quality features outperformed motion-only and quality-only by MAE, supporting the title claim that both motion and signal quality matter.
- The recommended main routed system should use **peak-based classical + PulsePPG**.
- PaPaGei should remain in the paper as a benchmark rather than the main routed expert unless external validation changes this conclusion.

## Key findings for PI update

1. The corrected GalaxyPPG benchmark currently favors PulsePPG random forest as the best single expert.
2. Inversion is methodologically required, but the current held-out ablation shows mixed metric effects rather than a uniform performance improvement.
3. The oracle router gives a large improvement over the best single expert, especially on P95 AE and catastrophic error rate, validating the routing direction.
4. The first learned lightweight router already improves MAE over the best single expert; the combined hard gate improves robustness metrics.
5. Motion+quality routing is stronger than motion-only or quality-only, supporting the intended scientific framing.

## Current risks / missing items

- Native watch HR output is missing / not found and is not included as a Week 2 baseline.
- Week 2 uses a fixed subject-independent held-out split rather than LOSO or repeated grouped CV.
- Participant-level confidence intervals and paired significance tests are not yet available for Week 2-4.
- Learned routing has only been validated on GalaxyPPG; PPG-DaLiA and WildPPG wrist external validation remains Week 5-7 work.
- Current `main` does not include the generated Week 2-4 CSV/figure artifacts after the pruning commit; paths are documented for reproducibility, but artifact files need to be regenerated or restored from local experiment outputs if full audit is required.

## Recommended next steps

- Regenerate or restore the frozen Week 2-4 artifact directories locally before manuscript writing.
- Add participant-level confidence intervals and paired tests for best single expert vs learned router and oracle router.
- Decide whether the paper should emphasize the soft gate for MAE or the hard gate for robustness; current evidence supports reporting both.
- If native watch HR becomes available in aligned form, rerun Week 2 with that additional baseline.
- Proceed to Week 5-7 external validation with the main expert pair set to peak-based classical + PulsePPG.
