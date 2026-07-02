# Week 7 Participant-Level Statistics

- Week 4 root: `experiments/week4_galaxyppg_lightweight_router_2026-05-13`
- Router: `motion_quality/hard_gate`
- Participants: `5`

Positive deltas mean the routed system improved over the participant's best single expert.

| metric                        |   n_participants |   mean_delta_best_single_minus_router |   median_delta_best_single_minus_router |   bootstrap_ci95_low |   bootstrap_ci95_high |   paired_ttest_p_value |   wilcoxon_signed_rank_p_value | interpretation                                                           |
|:------------------------------|-----------------:|--------------------------------------:|----------------------------------------:|---------------------:|----------------------:|-----------------------:|-------------------------------:|:-------------------------------------------------------------------------|
| MAE                           |                5 |                          -0.0903316   |                              0.0230085  |           -0.490994  |            0.31033    |               0.727709 |                         1      | positive means router improved over the participant's best single expert |
| p95_absolute_error            |                5 |                          -0.673988    |                              0.483815   |           -2.8397    |            1.33644    |               0.596193 |                         0.8125 | positive means router improved over the participant's best single expert |
| catastrophic_error_rate_20bpm |                5 |                          -0.000299586 |                             -0.00063857 |           -0.0101563 |            0.00777498 |               0.958009 |                         1      | positive means router improved over the participant's best single expert |
