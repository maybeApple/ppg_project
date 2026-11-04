"""Rebuild, cross-fit, freeze and evaluate the corrected research experiment.

Training never opens evaluation_reference.csv. evaluate() only loads frozen
predictions and references; it cannot call a fitting function. No legacy cache
is accepted. Each stage checks the hash of its inputs before resuming.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

from src.utils.submission_protocol import (
    BASELINES, FEATURES, MOTION, QUALITY, assert_fit_scope, fit_expert, fit_gate,
    gate_probability, grouped_splits, metrics, route, select_experts, sha256,
    validate_oof, validate_split, window_keys, write_json, apply_regime_bins,
)

LOG_PATH = None

def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_fingerprints():
    files = [p for root in ["src", "configs", "tests"] for p in Path(root).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json"}]
    files += [Path("requirements.txt")]
    return {p.as_posix(): sha256(p) for p in sorted(files)}


def initialize(config_path):
    cfg = read_json(config_path)
    required_values = {"schema_version": "submission_protocol_v2", "primary_router": "motion_quality_soft",
        "evaluation": "common_finite_windows_all_seven_baselines",
        "expert_selection": "participant_balanced_training_OOF_MAE_common_valid_windows"}
    for key, expected in required_values.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Unsupported {key} override; protocol and implementation must agree")
    if cfg["random_forest"] != {"n_estimators": 100, "max_depth": 16, "max_features": "sqrt", "min_samples_leaf_grid": [1,5]}:
        raise ValueError("Unsupported random_forest override")
    if cfg["preprocessing"]["pulseppg_sampling_hz"] != 50 or cfg["preprocessing"]["papagei_sampling_hz"] != 125:
        raise ValueError("Model sampling frequencies must match the encoder input contract")
    if cfg["outer_folds"] < 2 or cfg["inner_folds"] < 2:
        raise ValueError("Grouped nested validation requires at least two folds")
    if cfg["ridge_alphas"] != [0.1, 1.0, 10.0, 100.0] or cfg["gate"] != {
        "classifier": "LogisticRegression", "C": 1.0, "threshold": 0.5, "class_weight": None,
        "calibration": "raw_probability_with_grouped_training_diagnostics"}:
        raise ValueError("Unsupported hyperparameter override; update implementation and protocol together")
    if cfg["preprocessing"]["normalization"] == "person_specific_zscore":
        raise ValueError("Submission deployment protocol excludes offline future-person normalization")
    root = Path(cfg["output_root"])
    for directory in ["configs", "predictions", "metrics", "tables", "figures", "statistics",
                      "manifests", "features", "models", "evaluation", "logs"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    frozen = root / "configs/frozen_protocol.json"
    if frozen.exists() and read_json(frozen) != cfg:
        raise ValueError("Protocol already frozen; use a different output_root for a new protocol")
    if not frozen.exists():
        write_json(frozen, cfg)
        write_json(root / "configs/participant_split.json", read_json(cfg["split_config"]))
    elif read_json(root / "configs/participant_split.json") != read_json(cfg["split_config"]):
        raise ValueError("Participant split changed after freezing protocol")
    return cfg, root


def check_prepared(cfg, root):
    record = read_json(root / "manifests/preparation.json")
    if record["protocol_sha256"] != sha256(root / "configs/frozen_protocol.json"):
        raise ValueError("Prepared data protocol hash mismatch")
    for name, expected in record["artifacts"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    for name, expected in record["input_hashes"].items():
        if sha256(name) != expected:
            raise ValueError(f"Preparation source input changed: {name}")
    return record


def prepare(cfg, root):
    if (root / "manifests/preparation.json").exists():
        check_prepared(cfg, root)
        log("Preparation already verified; unchanged artifacts reused")
        return
    from src.data import load_processed_manifest, load_processed_windows
    from src.baseline.peak_detection import estimate_hr_from_peaks
    from src.baseline.spectral_hr import estimate_hr_from_spectrum
    from src.models.common import SignalPreprocessingConfig, build_window_signal_matrix
    from src.models.pulseppg_feature import load_pulseppg_encoder
    from src.models.papagei_feature import load_papagei_encoder
    from src.utils.build_week3_artifacts import compute_ppg_quality_features, compute_accelerometer_features
    import torch

    manifest_path = Path(cfg["dataset_manifest"])
    if not manifest_path.exists():
        write_json(root / "manifests/data_status.json", {"galaxyppg": "blocked_missing_data",
            "required_manifest": str(manifest_path), "ppgdalia": "requires_real_data", "wildppg_wrist": "requires_real_data"})
        raise FileNotFoundError(f"Run corrected raw export first: {manifest_path}")
    manifest = load_processed_manifest(manifest_path)
    label_cfg = manifest["label_generation"]
    if not (label_cfg.get("min_hr_bpm") == 35 and label_cfg.get("max_hr_bpm") == 220):
        raise ValueError("Submission requires corrected label generation with explicit 35–220 bpm RR bounds")
    for key, expected in [("window_seconds", 10), ("stride_seconds", 2),
                          ("label_aggregation", "median"), ("reference_source", "ibi")]:
        if manifest[key] != expected:
            raise ValueError(f"Unexpected main experiment {key}: {manifest[key]}")
    log("Reading freshly corrected raw-signal windows")
    windows = load_processed_windows(manifest["windows_path"])
    windows["window_key"] = window_keys(windows)
    split = read_json(cfg["split_config"])
    validate_split(windows, split["train_participants"], split["test_participants"])
    if not windows.ppg_inverted.all():
        raise ValueError("GalaxyPPG canonical polarity missing")
    for row in windows.itertuples(index=False):
        if not np.allclose(np.asarray(row.ppg_values), -np.asarray(row.ppg_raw_values), rtol=0, atol=0):
            raise ValueError("GalaxyPPG must be inverted exactly once")
    metadata = windows[["window_key", "participant_id", "session_id", "session_name", "split",
                        "window_start_ms", "window_end_ms"]].copy()
    metadata["activity"] = metadata.session_name
    metadata.to_csv(root / "features/metadata.csv", index=False)
    train_mask = windows.split.eq("train")
    targets = windows.loc[train_mask, ["window_key", "participant_id", "label_hr_bpm"]].rename(columns={"label_hr_bpm": "y_true"})
    targets.to_csv(root / "features/train_targets.csv", index=False)
    windows.loc[~train_mask, ["window_key", "label_hr_bpm"]].rename(columns={"label_hr_bpm": "y_true"}).to_csv(
        root / "evaluation/evaluation_reference.csv", index=False)
    log(f"Preparing signal-only features for {len(windows)} windows; training labels {len(targets)}")
    classical = metadata[["window_key"]].copy()
    for name, estimator in [("peak_based", estimate_hr_from_peaks), ("spectral", estimate_hr_from_spectrum)]:
        classical[name] = [estimator(row.ppg_values, row.ppg_timestamps_ms,
                           sampling_rate_hz=row.ppg_sampling_hz).predicted_hr_bpm
                           for row in windows.itertuples(index=False)]
        log(f"Computed {name}")
    classical.to_csv(root / "features/classical_predictions.csv", index=False)
    quality = pd.DataFrame([compute_ppg_quality_features(row) for _, row in windows.iterrows()])
    quality["peak_spectral_disagreement_bpm"] = (classical.peak_based - classical.spectral).abs()
    quality["window_key"] = windows.window_key
    keys = metadata.rename(columns={"window_start_ms": "window_start_time", "window_end_ms": "window_end_time"})
    acceleration, note = compute_accelerometer_features(Path(cfg["dataset_root"]), keys)
    features = quality.merge(acceleration, on="window_key", how="left", validate="one_to_one")
    features = features[["window_key"] + MOTION + QUALITY]
    features.to_csv(root / "features/router_features.csv", index=False)
    if note:
        log(note)
    torch.set_num_threads(int(cfg["torch_threads"]))
    checkpoint_records = {}
    for model_name, hz, loader in [("pulseppg", 50, load_pulseppg_encoder), ("papagei", 125, load_papagei_encoder)]:
        checkpoint = Path(cfg["checkpoints"][model_name])
        if not checkpoint.is_file():
            write_json(root / "manifests/data_status.json", {model_name: "blocked_missing_checkpoint"})
            raise FileNotFoundError(checkpoint)
        pc = cfg["preprocessing"]
        preprocess = SignalPreprocessingConfig(mode=pc["mode"], target_sampling_hz=hz,
            normalization=pc["normalization"], apply_bandpass=pc["apply_bandpass"],
            low_hz=pc["low_hz"], high_hz=pc["high_hz"], filter_order=pc["filter_order"])
        preprocess.validate_deployment()
        signal_digest = sha256(manifest["windows_path"])
        cache_record = {"source_windows_sha256": signal_digest, "checkpoint_sha256": sha256(checkpoint),
                        "model_code_sha256": sha256(Path("src/vendor") / ("pulseppg_resnet1d.py" if model_name == "pulseppg" else "papagei_resnet.py")),
                        "shared_code_sha256": sha256("src/vendor/resnet1d_shared.py"),
                        "preprocessing_code_sha256": sha256("src/models/common.py"),
                        "preprocessing": preprocess.to_dict(), "n_windows": len(windows)}
        out = root / f"features/{model_name}_embeddings.npy"
        receipt = root / f"manifests/{model_name}_embedding.json"
        if out.exists() and receipt.exists() and read_json(receipt).get("inputs") == cache_record and read_json(receipt).get("sha256") == sha256(out):
            log(f"Verified current-run {model_name} embedding cache")
            checkpoint_records[model_name] = read_json(receipt)
            continue
        log(f"Fresh {model_name} signal preprocessing and frozen encoder inference on CPU")
        matrix = build_window_signal_matrix(windows, hz, preprocessing_config=preprocess)
        encoder, device, _ = loader(checkpoint_path=checkpoint, device="cpu")
        result = []
        batch_size = int(cfg["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                output = encoder(torch.from_numpy(matrix[start:start+batch_size]).unsqueeze(1).to(device))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                result.append(output.detach().cpu().numpy())
                if start % (batch_size * 20) == 0:
                    log(f"{model_name}: {min(start+batch_size,len(matrix))}/{len(matrix)} windows")
        embeddings = np.vstack(result)
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Nonfinite {model_name} embeddings")
        np.save(out, embeddings)
        checkpoint_records[model_name] = {"inputs": cache_record, "sha256": sha256(out), "shape": list(embeddings.shape),
            "checkpoint_path": str(checkpoint), "encoder_frozen": True, "legacy_cache_used": False,
            "normalization_fit": "within_each_window_no_label_or_cross_window_statistics"}
        write_json(receipt, checkpoint_records[model_name])
        del encoder, matrix, result, embeddings
        log(f"Completed fresh {model_name} embeddings")
    artifacts = {p.relative_to(root).as_posix(): sha256(p) for p in (root / "features").iterdir() if p.is_file()}
    artifacts["evaluation/evaluation_reference.csv"] = sha256(root / "evaluation/evaluation_reference.csv")
    inputs = [manifest_path, Path(manifest["windows_path"]), Path(cfg["split_config"]),
              Path("src/models/common.py"), Path("src/vendor/pulseppg_resnet1d.py"),
              Path("src/vendor/papagei_resnet.py"), Path("src/vendor/resnet1d_shared.py"),
              Path("src/baseline/peak_detection.py"), Path("src/baseline/spectral_hr.py"),
              Path("src/utils/build_week3_artifacts.py"), Path("src/data/labels.py"),
              Path("src/data/loader.py"), Path("src/data/windowing.py"),
              *[Path(p) for p in cfg["checkpoints"].values()]]
    write_json(root / "manifests/preparation.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / "configs/frozen_protocol.json"), "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path), "windows_sha256": sha256(manifest["windows_path"]),
        "n_windows": len(windows), "n_train_windows": int(train_mask.sum()), "n_test_windows": int((~train_mask).sum()),
        "label_generation": label_cfg, "artifacts": artifacts, "checkpoints": checkpoint_records,
        "input_hashes": {str(p): sha256(p) for p in inputs}, "accelerometer_note": note})
    write_json(root / "manifests/data_status.json", {"galaxyppg": "real_data_prepared", "pulseppg": "real_checkpoint",
        "papagei": "real_checkpoint", "ppgdalia": "requires_real_data", "wildppg_wrist": "requires_real_data",
        "external_metrics_in_manuscript": False})


def train(cfg, root):
    check_prepared(cfg, root)
    if (root / "manifests/frozen_fit.json").exists():
        verify_frozen(root)
        log("Frozen training already complete; fitting skipped")
        return
    metadata = pd.read_csv(root / "features/metadata.csv")
    split = read_json(root / "configs/participant_split.json")
    allowed, test_ids = split["train_participants"], split["test_participants"]
    validate_split(metadata, allowed, test_ids)
    train_mask = metadata.split.eq("train").to_numpy()
    tr = metadata.loc[train_mask].reset_index(drop=True)
    te = metadata.loc[~train_mask].reset_index(drop=True)
    targets = pd.read_csv(root / "features/train_targets.csv")
    if set(targets.participant_id) != set(allowed) or targets.window_key.duplicated().any():
        raise ValueError("Training targets contain test participants or duplicate windows")
    if set(targets.window_key) != set(tr.window_key):
        raise ValueError("Training targets must match every training window exactly")
    tr = tr.merge(targets[["window_key", "y_true"]], validate="one_to_one", on="window_key", how="left")
    if tr.y_true.isna().any():
        raise ValueError("Missing training targets")
    # The fit phase never opens the separately stored test reference file.
    y = tr.y_true.to_numpy(float)
    groups = tr.participant_id.to_numpy()
    features = pd.read_csv(root / "features/router_features.csv")
    tr = tr.merge(features, on="window_key", validate="one_to_one")
    te = te.merge(features, on="window_key", validate="one_to_one")
    classical = pd.read_csv(root / "features/classical_predictions.csv")
    oof = tr.merge(classical, on="window_key", validate="one_to_one")
    final = te.merge(classical, on="window_key", validate="one_to_one")
    oof["fold"] = -1
    for name in BASELINES[2:]:
        oof[name] = np.nan
    outer = grouped_splits(groups, cfg["outer_folds"])
    records = []
    # Read only X for held-out participants; labels are physically absent here.
    embeddings = {model: np.load(root / f"features/{model}_embeddings.npy", mmap_mode="r")
                  for model in ["pulseppg", "papagei"]}
    for fold, (fit_idx, val_idx) in enumerate(outer, 1):
        fit_ids, val_ids = sorted(set(groups[fit_idx])), sorted(set(groups[val_idx]))
        record = {"fold": fold, "train_participants": fit_ids, "validation_participants": val_ids, "experts": {}}
        oof.loc[val_idx, "fold"] = fold
        for name in BASELINES[2:]:
            model_name, kind = name.split("_", 1)
            x_train = np.asarray(embeddings[model_name][train_mask])
            log(f"OOF fold {fold}/{len(outer)}: {name}, holding out {val_ids}")
            estimator, fit_record = fit_expert(x_train[fit_idx], y[fit_idx], groups[fit_idx], kind,
                cfg["seed"], allowed, val_ids, cfg["inner_folds"])
            oof.loc[val_idx, name] = estimator.predict(x_train[val_idx])
            record["experts"][name] = fit_record
        records.append(record)
        oof.to_csv(root / "predictions/training_oof_predictions.csv", index=False)
        write_json(root / "manifests/oof_provenance.json", records)
    validate_oof(oof, records, allowed, test_ids)
    selection = select_experts(oof, allowed)
    write_json(root / "manifests/selection.json", selection)
    log(f"Training OOF preselection: {selection['best_classical']} / {selection['best_foundation']}; best single {selection['best_single']}")
    # Refit all seven predeclared baselines for a paired comparison, with no test labels.
    final_records = {}
    for name in BASELINES[2:]:
        model_name, kind = name.split("_", 1)
        log(f"Final full-training fit: {name}")
        estimator, fit_record = fit_expert(np.asarray(embeddings[model_name][train_mask]), y, groups, kind,
            cfg["seed"], allowed, test_ids, cfg["outer_folds"])
        joblib.dump(estimator, root / f"models/{name}.joblib")
        final[name] = estimator.predict(np.asarray(embeddings[model_name][~train_mask]))
        final_records[name] = fit_record
    c, f = selection["best_classical"], selection["best_foundation"]
    valid = np.isfinite(oof[c]) & np.isfinite(oof[f]) & np.isfinite(oof.y_true)
    gate_train = oof.loc[valid].reset_index(drop=True)
    target = ((gate_train[f]-gate_train.y_true).abs() < (gate_train[c]-gate_train.y_true).abs()).astype(int).to_numpy()
    gate_train["foundation_wins"] = target
    gate_train.to_csv(root / "predictions/router_training_data.csv", index=False)
    write_json(root / "configs/router_training.json", {"features": FEATURES, "forbidden": ["reference", "valid_beat_count", "label", "expert_error"],
        "target": "foundation_absolute_error_strictly_less_than_classical_OOF_error;ties_classical",
        "fit_participants": allowed, "test_participants_excluded": test_ids, "gate": cfg["gate"],
        "selection": selection, "training_rows": len(gate_train)})
    coefficients, calibration_rows, calibration_predictions = [], [], []
    for feature_set, columns in FEATURES.items():
        log(f"Training final {feature_set} logistic router on training OOF data")
        gate = fit_gate(gate_train, target, columns, allowed, cfg["seed"])
        joblib.dump(gate, root / f"models/{feature_set}_gate.joblib")
        probability = gate_probability(gate, final, columns)
        final[f"{feature_set}_probability"] = probability
        for soft in [False, True]:
            name = feature_set + ("_soft" if soft else "_hard")
            final[name] = route(final[c], final[f], probability, soft)
        classifier = gate.steps[-1][1]
        if hasattr(classifier, "coef_"):
            for feature, value in zip(columns, classifier.coef_[0]):
                coefficients.append({"feature_set": feature_set, "feature": feature, "coefficient": float(value)})
        # Diagnostic grouped gate CV uses training subjects only. It is not a new
        # generalization estimate: the expert OOF targets have cross-fold dependence.
        p_cv = np.zeros(len(gate_train))
        for a, b in grouped_splits(gate_train.participant_id, cfg["outer_folds"]):
            assert_fit_scope(gate_train.iloc[a].participant_id, allowed, gate_train.iloc[b].participant_id)
            inner_gate = fit_gate(gate_train.iloc[a], target[a], columns, allowed, cfg["seed"])
            p_cv[b] = gate_probability(inner_gate, gate_train.iloc[b], columns)
        prior = np.zeros(len(gate_train))
        for a, b in grouped_splits(gate_train.participant_id, cfg["outer_folds"]):
            prior[b] = np.mean(target[a])
        calibration_rows.append({"feature_set": feature_set, "scope": "training_grouped_gate_diagnostic_only",
            "Brier": float(brier_score_loss(target, p_cv)), "prior_Brier": float(brier_score_loss(target, prior)),
            "log_loss": float(log_loss(target, p_cv, labels=[0,1])), "n_windows": len(target),
            "calibration_selected_on_test": False, "probability_calibration": "uncalibrated_logistic",
            "limitation": "expert OOF targets share training participants across gate folds; descriptive diagnostic"})
        calibration_predictions.append(pd.DataFrame({"window_key": gate_train.window_key, "participant_id": gate_train.participant_id,
            "feature_set": feature_set, "target": target, "probability": p_cv}))
    pd.DataFrame(coefficients).to_csv(root / "tables/gate_coefficients.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(root / "metrics/training_calibration_diagnostics.csv", index=False)
    pd.concat(calibration_predictions).to_csv(root / "predictions/training_gate_diagnostics.csv", index=False)
    # Bin edges are trained before the evaluation reference file is opened.
    edges = {"motion": tr.acc_norm_std.quantile(cfg["regime_bin_quantiles"]).tolist(),
             "quality": tr.ppg_autocorr_peak_strength.quantile(cfg["regime_bin_quantiles"]).tolist()}
    write_json(root / "configs/regime_bin_edges.json", edges)
    for key, column in [("motion", "acc_norm_std"), ("quality", "ppg_autocorr_peak_strength")]:
        final["motion_intensity_bin" if key == "motion" else "ppg_quality_bin"] = apply_regime_bins(final[column], edges[key])
    final.to_csv(root / "predictions/frozen_test_predictions_without_labels.csv", index=False)
    write_json(root / "manifests/final_expert_fits.json", final_records)
    model_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in (root/"models").glob("*.joblib")}
    frozen_paths = ["predictions/frozen_test_predictions_without_labels.csv", "manifests/selection.json",
                    "configs/frozen_protocol.json", "configs/router_training.json", "configs/regime_bin_edges.json",
                    "manifests/oof_provenance.json", "manifests/final_expert_fits.json",
                    "configs/participant_split.json", "manifests/preparation.json",
                    "predictions/training_oof_predictions.csv", "predictions/router_training_data.csv"]
    model_hashes.update({path: sha256(root / path) for path in frozen_paths})
    write_json(root / "manifests/frozen_fit.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": model_hashes, "test_labels_read_during_fit": False,
        "test_prediction_rows": len(final), "fit_participants": allowed, "test_participants": test_ids,
        "training_source_hashes": source_fingerprints()})
    write_json(root / "manifests/leakage_checks.json", {"status": "passed", "train_test_disjoint": True,
        "OOF_provenance_checked": True, "inner_preprocessing_in_pipeline": True,
        "test_reference_excluded_from_training_API": True, "gate_test_fit": False,
        "oracle_used_for_training": False, "reference_features_excluded": True,
        "selection_source": selection["selection_source"], "regime_thresholds_source": "training_only",
        "historical_test_exposure_remains": True})
    log("Expert/gate models and unlabeled test predictions frozen; test reference not read")


def verify_frozen(root):
    record = read_json(root / "manifests/frozen_fit.json")
    for relative, digest in record["artifacts"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen artifact changed: {relative}")
    return record


def evaluate(cfg, root):
    check_prepared(cfg, root)
    frozen = verify_frozen(root)
    predictions = pd.read_csv(root / "predictions/frozen_test_predictions_without_labels.csv")
    references = pd.read_csv(root / "evaluation/evaluation_reference.csv")
    frame = predictions.merge(references, on="window_key", validate="one_to_one")
    if len(frame) != len(predictions):
        raise ValueError("Evaluation reference keys missing")
    names = BASELINES + [f"{fs}_{kind}" for fs in FEATURES for kind in ["hard", "soft"]]
    common = np.isfinite(frame[names + ["y_true"]].to_numpy(float)).all(axis=1)
    coverage = {"total_test_windows": len(frame), "common_valid_windows": int(common.sum()),
                "excluded_common_windows": int((~common).sum()),
                "invalid_by_model": {name: int((~np.isfinite(frame[name])).sum()) for name in names},
                "mask_rule": "all_seven_predeclared_baselines_and_all_routers_finite;no_error_threshold_filtering"}
    write_json(root / "metrics/coverage.json", coverage)
    frame = frame.loc[common].copy()
    selection = read_json(root / "manifests/selection.json")
    for alias in ["best_classical", "best_foundation", "best_single"]:
        frame[alias] = frame[selection[alias]]
    # Oracle is computed only here, after freezing predictions. It is an analysis
    # of the selected pair and is never serialized as a deployable estimator.
    c, f = selection["best_classical"], selection["best_foundation"]
    frame["oracle"] = np.where((frame[c]-frame.y_true).abs() <= (frame[f]-frame.y_true).abs(), frame[c], frame[f])
    names += ["best_classical", "best_foundation", "best_single", "oracle"]
    identity = ["window_key", "participant_id", "session_id", "activity", "window_start_ms", "window_end_ms",
                "motion_intensity_bin", "ppg_quality_bin", "y_true"]
    long = frame.melt(id_vars=identity, value_vars=names, var_name="model_name", value_name="y_pred")
    long["absolute_error"] = (long.y_pred-long.y_true).abs()
    long.to_csv(root / "predictions/final_test_predictions.csv", index=False)
    summary = []
    for name, rows in long.groupby("model_name", sort=False):
        row = {"model_name": name, **metrics(rows.y_true, rows.y_pred), "n_participants": rows.participant_id.nunique(),
               "participant_mean_MAE": float(rows.groupby("participant_id").absolute_error.mean().mean()),
               "participant_std_MAE": float(rows.groupby("participant_id").absolute_error.mean().std(ddof=1)),
               "preprocessing_mode": cfg["preprocessing"]["mode"], "normalization": cfg["preprocessing"]["normalization"]}
        summary.append(row)
    pd.DataFrame(summary).to_csv(root / "metrics/summary.csv", index=False)
    write_json(root / "metrics/summary.json", summary)
    for group in ["participant_id", "activity", "motion_intensity_bin", "ppg_quality_bin"]:
        rows = []
        for (name, value), subset in long.groupby(["model_name", group], observed=True, dropna=False):
            rows.append({"model_name": name, group: value, **metrics(subset.y_true, subset.y_pred),
                         "n_participants": subset.participant_id.nunique()})
        pd.DataFrame(rows).to_csv(root / f"metrics/by_{group}.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", "unavailable"
    manifest = {"status": "corrected_real_data_evaluation_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "git_dirty": bool(dirty), "source_sha256": source_fingerprints(),
        "seed": cfg["seed"], "python": sys.version, "platform": platform.platform(),
        "dataset_manifest": cfg["dataset_manifest"], "dataset_schema_version": "canonical_ppg_v1_with_rr_validation_v2",
        "preparation": read_json(root / "manifests/preparation.json"), "selection": selection,
        "frozen_fit_sha256": sha256(root / "manifests/frozen_fit.json"),
        "test_reference_sha256": sha256(root / "evaluation/evaluation_reference.csv"),
        "final_predictions_sha256": sha256(root / "predictions/final_test_predictions.csv"),
        "frozen_fit": frozen, "coverage": coverage,
        "reproduction_commands": ["python -m src.data.export_processed --reference-source ibi --output-root data/processed/corrected_v2",
            "python -m src.utils.run_submission --config configs/conference_submission.json --stage all",
            "python -m src.utils.submission_statistics --results-root " + str(root),
            "python -m src.utils.build_submission_paper --results-root " + str(root)],
        "limitations": ["Fixed test cohort was historically exposed; computationally corrected rerun is not a new confirmatory cohort",
            "Only five independent test participants", "No real external dataset available", "No test-driven configuration changes permitted"]}
    write_json(root / "manifests/run_manifest.json", manifest)
    log("Frozen independent evaluation complete: " + str(root / "metrics/summary.csv"))
    print(pd.DataFrame(summary)[["model_name", "MAE", "RMSE", "p95_absolute_error", "n_windows"]].to_string(index=False), flush=True)


def main():
    global LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/conference_submission.json"))
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "verify", "all"], default="all")
    args = parser.parse_args()
    cfg, root = initialize(args.config)
    LOG_PATH = root / "logs/experiment.log"
    log("Command: " + subprocess.list2cmdline([sys.executable, "-m", "src.utils.run_submission", *sys.argv[1:]]))
    # Bound BLAS thread count to keep grouped fits reproducible and avoid oversubscription.
    with threadpool_limits(limits=4):
        if args.stage in ["prepare", "all"]:
            prepare(cfg, root)
        if args.stage in ["train", "all"]:
            train(cfg, root)
        if args.stage in ["evaluate", "all"]:
            evaluate(cfg, root)
        if args.stage == "verify":
            check_prepared(cfg, root)
            verify_frozen(root)
            oof = pd.read_csv(root / "predictions/training_oof_predictions.csv")
            split = read_json(root / "configs/participant_split.json")
            validate_oof(oof, read_json(root / "manifests/oof_provenance.json"), split["train_participants"], split["test_participants"])
            log("Input hashes, frozen artifacts and OOF provenance verified")


if __name__ == "__main__":
    main()
