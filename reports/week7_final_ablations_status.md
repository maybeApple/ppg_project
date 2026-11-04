# Week 7 Final Ablations Status

Week 7 statistics tooling exists and uses participant-level aggregation. Several planned ablations are already represented by Week 2-4 generated artifacts, but those generated experiment directories are intentionally gitignored and must be regenerated locally or distributed as release artifacts.

| Ablation | Required by plan | Current source file/path | Current status | Action needed |
| --- | --- | --- | --- | --- |
| inversion vs no inversion | yes | `experiments/week2_galaxyppg_corrected_2026-05-01/tables/inversion_ablation_table.*`; config `configs/week2_galaxyppg_inversion_ablation.json` | available from Week 2 generated artifacts; not committed as generated result package | regenerate Week 2 artifacts locally if absent |
| harmonized vs model-faithful preprocessing | yes | `experiments/week2_galaxyppg_corrected_2026-05-01/tables/harmonized_preprocessing_table.*`; `model_faithful_preprocessing_table.*` | available from Week 2 generated artifacts; not committed as generated result package | regenerate Week 2 artifacts locally if absent |
| PulsePPG normalization variants | yes | `experiments/week2_galaxyppg_corrected_2026-05-01/runs/model_faithful/pulseppg_features_*`; `embedding_manifest.csv` | available locally when Week 2 runs are present; generated caches are gitignored | rerun PulsePPG feature variants if absent |
| motion-only vs quality-only vs combined routing | yes | `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.*` | implemented in `src/utils/build_week4_artifacts.py`; generated table is gitignored | rerun Week 4 artifact builder if absent |
| hard vs soft routing | yes | `experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/routing_summary.*` | implemented in `src/utils/build_week4_artifacts.py`; generated table is gitignored | rerun Week 4 artifact builder if absent |
| linear/Ridge probe vs Random Forest or Gradient Boosting upper bound | yes | Week 2 run directories and `tables/main_benchmark_table.*` | available from Week 2 generated artifacts for completed runs | rerun missing probes/upper-bound regressors before final paper claims |
| participant-level CI/tests | yes | `src/utils/build_week7_statistics.py`; `experiments/week7_final_statistics/*` | tooling complete; generated statistics are gitignored | rerun Week 7 statistics after final Week 4 outputs exist |

## Notes

- The current statistical comparison is routed system versus each participant's best single expert, using participant-level deltas.
- Metrics covered by the Week 7 statistics script are MAE, 95th percentile absolute error, and catastrophic error rate above 20 bpm.
- External-dataset ablations remain pending until real PPG-DaLiA and WildPPG wrist runs are completed.

