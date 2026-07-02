# Final Frozen Results

This directory contains copied, immutable experiment artifacts for paper drafting and reproducibility review.

## Sources

- week2_root: `experiments/week2_galaxyppg_corrected_2026-05-01`
- week3_root: `experiments/week3_galaxyppg_regime_oracle_2026-05-13`
- week4_root: `experiments/week4_galaxyppg_lightweight_router_2026-05-13`
- week7_root: `experiments/week7_final_statistics`

## Reproduction Commands

```bash
python -m src.utils.build_week2_artifacts --search-root experiments/week2_galaxyppg_corrected_2026-05-01/runs --output-root experiments/week2_galaxyppg_corrected_2026-05-01 --tag-name week2-galaxyppg-corrected-2026-05-01
```

```bash
python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root experiments/week3_galaxyppg_regime_oracle_2026-05-13
```

```bash
python -m src.utils.build_week4_artifacts --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --output-root experiments/week4_galaxyppg_lightweight_router_2026-05-13
```

```bash
python -m src.utils.freeze_final_results --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --week4-root experiments/week4_galaxyppg_lightweight_router_2026-05-13 --week7-root experiments/week7_final_statistics
```

## Large Artifacts

Large prediction CSVs, embedding `.npy` arrays, trained downstream estimators, and per-fold router `.joblib` files are intentionally not committed in this acceptance package.

They remain available in the local source result directories listed above. If external transfer is required, publish those large files as a release artifact or archive instead of committing them to git.

## Artifact Groups

- configs: `experiments/final_frozen_results_2026-06-29/configs` copied=True
- week2_tables: `experiments/final_frozen_results_2026-06-29/week2/tables` copied=True
- week2_metrics: `experiments/final_frozen_results_2026-06-29/week2/metrics` copied=True
- week2_figures: `experiments/final_frozen_results_2026-06-29/week2/figures` copied=True
- week3_tables: `experiments/final_frozen_results_2026-06-29/week3/tables` copied=True
- week3_metrics: `experiments/final_frozen_results_2026-06-29/week3/metrics` copied=True
- week3_figures: `experiments/final_frozen_results_2026-06-29/week3/figures` copied=True
- week4_tables: `experiments/final_frozen_results_2026-06-29/week4/tables` copied=True
- week4_metrics: `experiments/final_frozen_results_2026-06-29/week4/metrics` copied=True
- week4_features: `experiments/final_frozen_results_2026-06-29/week4/features` copied=True
- week4_figures: `experiments/final_frozen_results_2026-06-29/week4/figures` copied=True

## Artifact Files

- reproducibility_manifest_json: `experiments/final_frozen_results_2026-06-29/reproducibility_manifest.json` copied=True
- reproducibility_manifest_md: `experiments/final_frozen_results_2026-06-29/reproducibility_manifest.md` copied=True
- week2_run_index: `experiments/final_frozen_results_2026-06-29/week2/run_index.csv` copied=True
- week2_run_manifest: `experiments/final_frozen_results_2026-06-29/week2/run_manifest.csv` copied=True
- week2_embedding_manifest: `experiments/final_frozen_results_2026-06-29/week2/embedding_manifest.csv` copied=True
- week2_memo: `experiments/final_frozen_results_2026-06-29/week2/week2_memo.md` copied=True
- week3_memo: `experiments/final_frozen_results_2026-06-29/week3/week3_regime_analysis.md` copied=True
- week3_metrics_json: `experiments/final_frozen_results_2026-06-29/week3/metrics.json` copied=True
- week3_run_config: `experiments/final_frozen_results_2026-06-29/week3/run_config.json` copied=True
- week3_run_log: `experiments/final_frozen_results_2026-06-29/week3/run_log.json` copied=True
- week4_memo: `experiments/final_frozen_results_2026-06-29/week4/week4_lightweight_router.md` copied=True
- week4_metrics_json: `experiments/final_frozen_results_2026-06-29/week4/metrics.json` copied=True
- week4_run_config: `experiments/final_frozen_results_2026-06-29/week4/run_config.json` copied=True
- week4_run_log: `experiments/final_frozen_results_2026-06-29/week4/run_log.json` copied=True
- week4_router_model_manifest: `experiments/final_frozen_results_2026-06-29/week4/models/router_model_manifest.json` copied=True
- week2_run_manifest_frozen_paths: `experiments/final_frozen_results_2026-06-29/week2/run_manifest_frozen_paths.csv` copied=True
- week2_embedding_manifest_frozen_paths: `experiments/final_frozen_results_2026-06-29/week2/embedding_manifest_frozen_paths.csv` copied=True
