# PPG Heart Rate Estimation

This repository reproduces wrist-PPG heart-rate estimation with classical estimators, frozen PulsePPG / PaPaGei features, and motion- and signal-quality-aware routing.

The reproducible flow uses:

- Galaxy Watch `PPG.csv`, inverted exactly once at loading into `canonical_ppg_v1`
- Polar H10 IBI reference labels for GalaxyPPG; ECG-derived intervals for external datasets
- 10-second windows, 2-second stride, and median instantaneous HR from at least two valid RR intervals (35–220 bpm)
- peak and spectral estimators, frozen encoder features, and participant-grouped probe/gate fitting
- explicit input hashes, frozen predictions before final reference evaluation, and separate development and evaluation stages

The experiment core contains the data/feature pipeline, expert and router training, inference, evaluation, configurations, and runtime dependencies. Tests, audit tools, reports, notebooks, raw recordings, checkpoints, and generated results remain local. Minimal checkpoint-compatible model definitions are vendored under `src/vendor/`; their upstream revisions are recorded in [UPSTREAM.md](src/vendor/UPSTREAM.md).

## Current Results

Completed evaluations below use the frozen **v2 preprocessing recipe**. Errors are pooled MAE in bpm on common finite windows.

| Evaluation | Test participants / paired windows | Training-selected single | Original soft router | Peak |
| --- | --- | --- | --- | --- |
| GalaxyPPG fixed holdout | 5 / 7,629 | 6.792 | 6.393 | 7.866 |
| Galaxy → PPG-DaLiA | 15 / 63,737 | 12.523 | 12.135 | 8.654 |
| Galaxy → WildPPG wrist | 16 / 375,881 | 23.648 | 18.117 | 15.808 |
| PPG-DaLiA within-dataset | 3 / 12,308 | 9.544 | 8.621 | 9.544 |
| WildPPG wrist within-dataset | 4 / 94,565 | 12.292 | 12.185 | 15.511 |

The original soft router is `motion_quality_soft`. Transfer uses the Galaxy-selected RF expert and frozen gates without target fitting. Within-dataset experiments fit separate linear/Ridge probes and gates using target-training participants only.

Nested router development used only the original 18 Galaxy training participants. Across 27,497 common validation windows, Direct MQ achieved MAE **6.223**, compared with **6.655** for legacy soft routing; P95 improved from **29.612** to **27.283** bpm. Selection used equal-participant MAE with participant-level tail guards. Scores used for this development choice are not an independent test.

After fitting and freezing the selected `direct_mq_soft` system, reassessment on the same physical windows gave:

| Dataset | Preserved original router MAE | Direct MQ MAE | Original → Direct MQ P95 |
| --- | --- | --- | --- |
| GalaxyPPG | 6.393 | **6.076** | 26.729 → 25.346 |
| PPG-DaLiA | **12.135** | 12.155 | 32.784 → 32.921 |
| WildPPG wrist | 18.117 | **15.025** | 48.825 → 49.646 |

The improvement is dataset-dependent: DaLiA still favors Peak, and WildPPG's lower MAE accompanies a worse P95. These historically exposed cohorts are reassessments, not new confirmatory tests. The reported scores remain pre-v3: **no full v3 HR-model retraining or accuracy evaluation has been completed**, and the experimental v3 tools are outside this core.

## Experiment Core

```text
ppg_project/
|-- README.md
|-- requirements.txt
|-- requirements-external.txt
|-- configs/
`-- src/
    |-- baseline/
    |-- data/
    |-- models/
    |-- routing/
    |-- utils/
    `-- vendor/
```

The commands create local `data/processed/` and `experiments/` directories. Data and weights must be supplied separately.

## Environment

Tested with Python **3.13.9** on Windows using CPU execution. Dependencies are pinned; native WildPPG ECG reference extraction requires **WFDB 4.3.0**.

```bash
python -m pip install -r requirements-external.txt
```

`requirements-external.txt` includes the core dependencies and WFDB. Use `requirements.txt` for Galaxy-only execution. For GPU execution, install the PyTorch wheel matching the CUDA environment.

Git attributes preserve exact file bytes across platforms because provenance checks hash source and configuration files.

## Raw Data

Download [GalaxyPPG](https://doi.org/10.5281/zenodo.14635823), [PPG-DaLiA](https://archive.ics.uci.edu/static/public/495/ppg%2Bdalia.zip), and the native recordings from [WildPPG](https://github.com/eth-siplab/WildPPG). Preserve original filenames and place them as follows:

```text
data/raw/
|-- GalaxyPPG/
|   |-- Meta.csv
|   `-- P02/
|       |-- Event.csv
|       |-- GalaxyWatch/
|       |   |-- PPG.csv
|       |   `-- ACC.csv
|       `-- PolarH10/
|           |-- IBI.csv
|           `-- ECG.csv
|-- PPG-DaLiA/
|   |-- S1/S1.pkl
|   `-- ...
`-- WildPPG-wrist/
    |-- WildPPG_Part_an0.mat
    `-- ...
```

Galaxy requires all participants in `configs/galaxyppg_submission_split.json`; `P01` is excluded. DaLiA also accepts `S1.pkl` directly under its root and uses corrected released `rpeaks`. WildPPG native MAT input needs no CSV manifest. Only wrist green PPG and wrist ACC enter WildPPG inference; sternum ECG supplies its reference. External PPG retains native polarity.

Extract the supplied combined archive into these raw-data layouts before running the export commands.

## Model Weights

Download the official [PulsePPG checkpoint](https://doi.org/10.5281/zenodo.17270930) and [PaPaGei checkpoint](https://zenodo.org/records/13983110). The frozen configuration uses:

```text
external/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl
external/papagei-foundation-model/weights/papagei_s.pt
```

These paths are declared in `configs/conference_submission.json`. Preserve them for the complete default reproduction chain; alternative placements require explicit configuration and a separate run.

## Frozen v2 Reproduction

Run commands from the repository root. The following chain assumes a fresh clone and empty output directories. Existing local research runs must be preserved: use a separate checkout with the default layout, or consistently change all source/output paths for a new run. An incomplete cache must not be spliced into a frozen run.

The fixed split has 18 training and 5 test participants. Expert selection minimizes participant-balanced training OOF MAE. Probe tuning, scaling, imputation, and gate training stay inside participant-grouped training folds. Final predictions are frozen before evaluation joins references.

```bash
python -m src.data.export_processed --dataset-root data/raw/GalaxyPPG --split-config configs/galaxyppg_submission_split.json --reference-source ibi --min-hr-bpm 35 --max-hr-bpm 220 --output-root data/processed/corrected_v2
python -m src.utils.run_submission --config configs/conference_submission.json --stage all
python -m src.utils.run_submission --config configs/conference_submission.json --stage verify
python -m src.utils.submission_statistics --results-root experiments/conference_submission_2026_11_01 --bootstrap-iterations 10000 --random-state 42
```

The runner also supports `--stage prepare`, `train`, and `evaluate` in sequence. Reuse verifies input/artifact hashes; a changed protocol requires a new output directory. Outputs include OOF and final predictions, metrics, fitted models, selection records, and provenance. Statistics resample participants rather than overlapping windows. The historical `external_validation: requires_real_data` field in the frozen configuration predates the completed external studies; it is retained for source-protocol reproduction.

## Real External Validation

First complete the default Galaxy source run above. Export the native datasets into continuous participant arrays plus scalar window indices:

```bash
python -m src.data.export_ppgdalia --dataset-root data/raw/PPG-DaLiA --output-root data/processed/external_2026_11_01/PPG-DaLiA_v2 --compact
python -m src.data.export_wildppg_wrist --dataset-root data/raw/WildPPG-wrist --output-root data/processed/external_2026_11_01/WildPPG-wrist_v2
python -m src.utils.run_external_research --manifest data/processed/external_2026_11_01/PPG-DaLiA_v2/ppg_dalia_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --units configs/ppgdalia_acc_units.json --output-root experiments/external_validation_2026_11_01/PPG-DaLiA --workers 2 --stage all
python -m src.utils.run_external_research --manifest data/processed/external_2026_11_01/WildPPG-wrist_v2/wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json --units configs/wildppg_acc_units.json --output-root experiments/external_validation_2026_11_01/WildPPG-wrist --workers 2 --stage all
```

Stages `prepare`, `transfer`, and `within` can run separately. The runner uses `experiments/conference_submission_2026_11_01` as its frozen source. Receipts verify manifests, archives, units, source models, implementation, and prepared artifacts before reuse.

Transfer evaluates all 15 DaLiA and 16 WildPPG participants without target fitting. Separate within-dataset splits use `random.Random(42)` on sorted IDs: DaLiA holds out `S2,S3,S8`; WildPPG holds out `l38,n31,ngh,p9p`. Training uses outer 5 / inner 3 participant folds and linear/Ridge core probes. These studies are participant holdouts, not full-system LOSO.

## Adaptive Router Development and Inference

Complete the default Galaxy run and both external runs before the full reassessment:

```bash
python -m src.utils.run_router_development --config configs/router_development.json --stage all
python -m src.utils.evaluate_adaptive_candidate
python -m src.utils.compare_adaptive_candidate
python -m src.utils.predict_adaptive_hr --help
```

Development rebuilds expert selection and gate-training OOF inside each of five outer training folds, with three meta folds and nested Ridge tuning. It uses float64 downstream fitting and fixed RF experts for both encoders. The final five Galaxy participants and all external references are excluded from development. `--stage cv` and `--stage fit` separate development validation from final candidate serialization.

The evaluation freezes predictions for every dataset before loading references; the comparison requires equal complete physical-window key sets. The inference CLI consumes the fitted `adaptive_hr_system.joblib`, signal-only features, explicitly keyed embedding metadata, and two embedding matrices. It rejects target/error columns and recognized v3 caches. It does not fit or compute performance metrics.

## Scientific Limits

Current limitations remain material:

- Galaxy IBI timestamp ownership is unresolved. Historical logger/exporter evidence is required before shifting interval timestamps or claiming corrected new labels.
- WildPPG MAT acceleration is assumed to be in g, with a predeclared factor of 9.80665. Motion-dependent results are conditional on this unverified scale; PPG-only experts and quality-only gates are independent of ACC scaling.
- WildPPG references use pinned, untuned XQRS ECG detection and retain automatic-reference uncertainty. DaLiA uses corrected released R peaks; its synchronized PKL acceleration is verified in g, distinct from raw E4 CSV units of 1/64 g.
- DaLiA is external to this project's downstream fitting but appeared in upstream encoder evaluation/model selection. Small holdouts and historically inspected cohorts limit claims of independent generalization.
- The reported accuracy remains pre-v3. Independent confirmation, v3 retraining, and strict model-faithful comparisons are unfinished.
