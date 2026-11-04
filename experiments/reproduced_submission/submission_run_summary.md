# Submission Reproduction Summary

Processed-data command:

python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json

Embedding commands:

python -m src.models.pulseppg_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/pulseppg_results/2026-03-18/full --batch-size 128 --device cpu
python -m src.models.papagei_feature --manifest-path data/processed/galaxyppg_hr_w10_s2_median_manifest.json --output-dir experiments/papagei_results/2026-03-18/full --batch-size 128 --device cpu

Exact command that reproduces the final best number reported in the submission:

python -m src.regression.train_regressor --feature-manifest experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json --regressor random_forest --random-state 42 --split-config configs/galaxyppg_submission_split.json --output-dir experiments/reproduced_submission/pulseppg_random_forest

Expected reproduced metrics for the final best model:
- MAE: 7.286407769380653
- RMSE: 11.30392115001437

Reference baseline command:

python -m src.baseline.run_baseline --processed-manifest data/processed/galaxyppg_hr_w10_s2_median_manifest.json --method peak --output-dir experiments/reproduced_submission/baseline_peak

Reference baseline metrics:
- MAE: 7.641879292148965
- RMSE: 15.84782749791325

All reported experiment bundles, including logs and prediction CSVs, are indexed in:
- experiments/reported_results_summary.md
- experiments/reported_results_summary.json
