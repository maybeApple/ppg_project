# Quality- and Motion-Aware Routing Between Classical and Foundation Models for Heart Rate Estimation from Wearable Wrist PPG

## Abstract

Placeholder. Summarize the problem, corrected GalaxyPPG preprocessing, classical/foundation expert comparison, quality- and motion-aware routing, participant-level statistical evaluation, and external-validation status. Do not finalize claims until all frozen artifacts are regenerated.

## Introduction

- Motivate wrist PPG heart-rate estimation under motion and signal-quality variation.
- Explain why a single estimator may not dominate across all physiological and activity regimes.
- State the planned contribution: a lightweight quality- and motion-aware router between a classical expert and a foundation-model expert.

## Related Work

- Classical PPG HR estimation using peak detection and spectral methods.
- Foundation models for physiological waveform representation, including PulsePPG and PaPaGei.
- Mixture/routing systems and uncertainty or quality-aware model selection.
- Dataset shift and external validation in wearable sensing.

## Datasets and Preprocessing

- GalaxyPPG corrected pipeline: loader-level Galaxy Watch PPG inversion, canonical schema, 10-second windows, 2-second stride, median beat-interval instantaneous HR label.
- PPG-DaLiA integration: wrist BVP, wrist ACC, chest ECG/R-peaks mapped to the same schema.
- WildPPG wrist integration: manifest-driven wrist PPG/ACC and ECG/RR mapping.
- External validation results: pending real PPG-DaLiA and WildPPG wrist runs.

## Methods

- Classical experts: peak detection and spectral HR.
- Foundation-model experts: PulsePPG and PaPaGei embeddings with downstream probes.
- Regime features: motion features, PPG quality features, and combined motion-plus-quality features.
- Router design: lightweight hard gate and soft gate, trained with participant-independent folds.

## Experimental Design

- Subject-independent evaluation.
- Week 2 corrected GalaxyPPG benchmark.
- Week 3 oracle complementarity analysis.
- Week 4 learned router evaluation.
- Week 5/6 external validation plan for PPG-DaLiA and WildPPG wrist.
- Week 7 participant-level confidence intervals and paired tests.

## Results

- GalaxyPPG corrected benchmark: fill from frozen Week 2 artifacts.
- Regime-dependent complementarity: fill from frozen Week 3 artifacts.
- Learned routing: fill from frozen Week 4 artifacts.
- Participant-level statistical evidence: fill from frozen Week 7 artifacts.
- External validation results: pending real PPG-DaLiA and WildPPG wrist runs.

## Discussion

- Interpret when classical and foundation-model experts are complementary.
- Discuss why motion and signal-quality features are plausible routing signals.
- Separate window-level overall improvements from participant-level statistical evidence.
- Discuss expected dataset-shift risks for external validation.

## Limitations

- External PPG-DaLiA and WildPPG wrist metrics are not yet available in the repository.
- Generated result folders are intentionally gitignored and must be regenerated or distributed separately.
- Router transfer to external datasets requires real data and a final protocol decision.

## Conclusion

Placeholder. Summarize only claims supported by final frozen artifacts.

## Reproducibility Statement

The repository provides code for corrected GalaxyPPG preprocessing, external data export, baseline evaluation, foundation-model feature extraction, downstream regression, router artifact construction, participant-level statistics, and final artifact freezing. Raw datasets, checkpoints, generated prediction CSVs, embedding arrays, trained model files, figures, and frozen result directories are not committed.

