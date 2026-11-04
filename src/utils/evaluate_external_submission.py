"""Evaluate an unchanged Galaxy-trained system on separately supplied wrist data.

No training, parameter search, calibration or threshold selection is performed.
An explicit acceleration-unit scale is mandatory; no device-unit conversion is
guessed. Without a documented source/external unit contract, generated metrics
are diagnostic and ineligible for a cross-dataset scientific claim.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data import load_processed_manifest, load_processed_windows
from src.models.common import SignalPreprocessingConfig
from src.utils.build_week3_artifacts import compute_ppg_quality_features, compute_single_acc_window
from src.utils.submission_protocol import (
    BASELINES, FEATURES, MOTION, QUALITY, apply_regime_bins, gate_probability,
    metrics, route, sha256, window_keys, write_json,
)

DATASETS = {"PPG-DaLiA", "WildPPG-wrist"}
MODEL_NAMES = BASELINES + [f"{features}_{kind}" for features in FEATURES for kind in ("hard", "soft")]
REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FrozenSource:
    root: Path
    config: dict
    selection: dict
    estimators: dict
    gates: dict
    checkpoints: dict
    preparation: dict
    frozen: dict
    regime_edges: dict


def _checkpoint_path(stored: str, source_root: Path) -> Path:
    candidate = Path(stored)
    for path in (candidate, REPO_ROOT / candidate, source_root / candidate):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing source checkpoint: {stored}")


def load_frozen_source(source_root: Path) -> FrozenSource:
    """Verify the original completed run before deserializing its frozen models."""
    source_root = Path(source_root).resolve()
    frozen = read_json(source_root / "manifests/frozen_fit.json")
    run = read_json(source_root / "manifests/run_manifest.json")
    if run.get("status") != "corrected_real_data_evaluation_complete":
        raise ValueError("Source must be a completed corrected real-data run.")
    if run.get("frozen_fit_sha256") != sha256(source_root / "manifests/frozen_fit.json"):
        raise ValueError("Source frozen-fit manifest hash mismatch.")
    if frozen.get("test_labels_read_during_fit") is not False:
        raise ValueError("Source lacks evidence of independent model fitting.")
    for relative, expected in frozen["artifacts"].items():
        if sha256(source_root / relative) != expected:
            raise ValueError(f"Source frozen artifact changed: {relative}")
    config = read_json(source_root / "configs/frozen_protocol.json")
    selection = read_json(source_root / "manifests/selection.json")
    if selection.get("selection_source") != "training_participants_oof_only" or run.get("selection") != selection:
        raise ValueError("Source expert selection must be the unchanged training-OOF selection.")
    preparation = read_json(source_root / "manifests/preparation.json")
    checkpoints = {}
    for name in ("pulseppg", "papagei"):
        checkpoint = _checkpoint_path(config["checkpoints"][name], source_root)
        expected = preparation["checkpoints"][name]["inputs"]["checkpoint_sha256"]
        if sha256(checkpoint) != expected:
            raise ValueError(f"Source checkpoint hash mismatch: {name}")
        checkpoints[name] = checkpoint
    # A source-trained model must receive the same signal/feature implementation.
    relevant = ["src/models/common.py", "src/models/pulseppg_feature.py", "src/models/papagei_feature.py",
                "src/vendor/pulseppg_resnet1d.py", "src/vendor/papagei_resnet.py", "src/vendor/resnet1d_shared.py",
                "src/baseline/peak_detection.py", "src/baseline/spectral_hr.py",
                "src/utils/build_week3_artifacts.py", "src/utils/submission_protocol.py"]
    source_hashes = frozen.get("training_source_hashes", {})
    for relative in relevant:
        if source_hashes.get(relative) != sha256(REPO_ROOT / relative):
            raise ValueError(f"Source inference implementation changed: {relative}")
    return FrozenSource(source_root, config, selection,
        {name: joblib.load(source_root / f"models/{name}.joblib") for name in BASELINES[2:]},
        {name: joblib.load(source_root / f"models/{name}_gate.joblib") for name in FEATURES},
        checkpoints, preparation, frozen, read_json(source_root / "configs/regime_bin_edges.json"))


def validate_external_manifest(manifest: dict, windows: pd.DataFrame) -> str:
    dataset = manifest.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"Only explicitly identified external wrist datasets are supported: {sorted(DATASETS)}")
    expected = {"window_seconds": 10, "stride_seconds": 2,
                "label_method": "beat_interval_instant_hr", "label_aggregation": "median",
                "reference_source": "ecg"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"External label/window contract mismatch: {key}")
    label = manifest.get("label_generation", {})
    if (label.get("label_schema_version") != "beat_interval_v2" or label.get("min_hr_bpm") != 35
            or label.get("max_hr_bpm") != 220 or label.get("min_valid_beats", 0) < 2):
        raise ValueError("External labels require beat_interval_v2, 35–220 bpm and at least two valid intervals.")
    required = {"participant_id", "session_id", "session_name", "window_start_ms", "window_end_ms",
                "window_length_s", "stride_s", "ppg_values", "ppg_raw_values", "ppg_timestamps_ms",
                "ppg_sampling_hz", "label_hr_bpm", "valid_beat_count", "ppg_inverted"}
    if required - set(windows) or windows.empty:
        raise ValueError(f"Missing or empty external windows: {sorted(required - set(windows))}")
    if windows[["participant_id", "session_id"]].isna().any().any():
        raise ValueError("External windows need participant/session identifiers.")
    if not windows.window_length_s.eq(10).all() or not windows.stride_s.eq(2).all():
        raise ValueError("External window rows disagree with 10 s / 2 s manifest.")
    if not np.isfinite(windows.label_hr_bpm).all() or not windows.label_hr_bpm.between(35, 220).all():
        raise ValueError("External target values violate the unified physiological contract.")
    if not np.isfinite(windows.valid_beat_count).all() or not windows.valid_beat_count.ge(2).all():
        raise ValueError("External windows contain insufficient valid reference intervals.")
    if windows.ppg_inverted.isna().any() or windows.ppg_inverted.astype(bool).any() or manifest.get("ppg_inverted") is not False:
        raise ValueError("These external loader contracts preserve native wrist polarity; Galaxy inversion is forbidden.")
    for row in windows.itertuples(index=False):
        signal, raw = np.asarray(row.ppg_values), np.asarray(row.ppg_raw_values)
        if signal.shape != raw.shape or not np.array_equal(signal, raw):
            raise ValueError("External canonical PPG must equal its recorded native raw PPG.")
        timestamps = np.asarray(row.ppg_timestamps_ms, dtype=float)
        if (signal.ndim != 1 or len(signal) < 2 or signal.shape != timestamps.shape
                or not np.isfinite(signal).all() or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) <= 0) or row.window_end_ms-row.window_start_ms != 10000
                or timestamps.min() < row.window_start_ms or timestamps.max() >= row.window_end_ms):
            raise ValueError("External signal samples and timestamps must align inside each half-open 10 s window.")
    if window_keys(windows).duplicated().any():
        raise ValueError("External window identities must be unique.")
    return dataset


def validate_unit_contract(scale: float, dataset: str, contract: dict | None = None) -> dict:
    """A positive conversion is required even when its physical evidence is absent."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("An explicit finite positive --acc-scale-to-mps2 is required.")
    contract = dict(contract or {})
    verified = False
    if contract:
        if contract.get("dataset") != dataset or contract.get("acc_scale_to_mps2") != scale:
            raise ValueError("Acceleration unit contract dataset/scale mismatch.")
        verified = (contract.get("source_native_is_mps2") is True and contract.get("external_units_verified") is True
                    and bool(contract.get("source_evidence")) and bool(contract.get("external_evidence")))
    return {"acc_scale_to_mps2": float(scale), "target_unit": "m/s^2",
            "source_feature_unit": "unchanged_source_Galaxy_native_values",
            "unit_contract_verified": verified, "contract": contract,
            "interpretation": "eligible_if_real_data" if verified else "unverified_units_diagnostic_only",
            "warning": None if verified else "Verify source Galaxy native values are m/s^2 and external conversion before scientific interpretation."}


def _participant_acceleration(dataset: str, root: Path, participant: str) -> pd.DataFrame:
    if dataset == "PPG-DaLiA":
        from src.data.ppgdalia_loader import load_ppgdalia_participant_data
        data = load_ppgdalia_participant_data(participant, root)
    else:
        from src.data.wildppg_loader import load_wildppg_participant_data
        data = load_wildppg_participant_data(participant, root)
    return data.canonical_accelerometer


def external_acceleration_features(windows: pd.DataFrame, dataset: str, root: Path,
                                   scale: float, loader=None) -> pd.DataFrame:
    """Use the exporter loader's native clock; never add Galaxy phone offsets."""
    validate_unit_contract(scale, dataset)
    loader = loader or _participant_acceleration
    rows = []
    for participant, group in windows.groupby("participant_id", sort=False):
        acc = loader(dataset, root, str(participant)).copy()
        columns = ["timestamp_ms", "acc_x", "acc_y", "acc_z"]
        if not set(columns) <= set(acc):
            raise ValueError(f"Missing canonical acceleration stream for {participant}")
        acc[columns] = acc[columns].apply(pd.to_numeric, errors="coerce")
        acc = acc.loc[np.isfinite(acc[columns]).all(axis=1)].sort_values("timestamp_ms")
        if acc.empty or acc.timestamp_ms.duplicated().any():
            raise ValueError(f"Empty or duplicate-timestamp canonical acceleration for {participant}")
        timestamps = acc.timestamp_ms.to_numpy(float)
        norm = np.linalg.norm(acc[["acc_x", "acc_y", "acc_z"]].to_numpy(float) * scale, axis=1)
        for row in group.itertuples(index=False):
            left, right = np.searchsorted(timestamps, [row.window_start_ms, row.window_end_ms], side="left")
            rows.append(compute_single_acc_window(row.window_key, timestamps[left:right], norm[left:right]))
    return pd.DataFrame(rows)


def external_signal_features(windows, dataset, dataset_root, scale, acceleration_loader=None):
    """Reference labels are excluded from the signal-only inference interface."""
    from src.baseline.peak_detection import estimate_hr_from_peaks
    from src.baseline.spectral_hr import estimate_hr_from_spectrum
    signals = windows[["window_key", "participant_id", "session_id", "session_name", "window_start_ms", "window_end_ms",
                       "window_length_s", "stride_s", "ppg_values", "ppg_raw_values", "ppg_timestamps_ms", "ppg_sampling_hz"]].copy()
    classical = pd.DataFrame({"window_key": signals.window_key})
    for name, estimator in [("peak_based", estimate_hr_from_peaks), ("spectral", estimate_hr_from_spectrum)]:
        classical[name] = [estimator(row.ppg_values, row.ppg_timestamps_ms,
                           sampling_rate_hz=row.ppg_sampling_hz).predicted_hr_bpm for row in signals.itertuples(index=False)]
    quality = pd.DataFrame([compute_ppg_quality_features(row) for _, row in signals.iterrows()])
    quality["window_key"] = signals.window_key.to_numpy()
    quality["peak_spectral_disagreement_bpm"] = (classical.peak_based - classical.spectral).abs()
    acceleration = external_acceleration_features(signals, dataset, dataset_root, scale, acceleration_loader)
    features = quality.merge(acceleration, on="window_key", validate="one_to_one", how="left")
    return signals, classical, features[["window_key", "acc_sample_count"] + MOTION + QUALITY]


def frozen_predictions(signals, classical, features, embeddings, source: FrozenSource, dataset: str):
    """Apply serialized probes/gates unchanged; this function cannot access targets."""
    if any(c.startswith(("label", "reference", "valid_beat")) for c in signals):
        raise ValueError("Inference must receive signal-only windows without external reference columns.")
    identity = ["window_key", "participant_id", "session_id", "session_name", "window_start_ms", "window_end_ms"]
    result = signals[identity].copy()
    if not set(result.window_key) == set(features.window_key) == set(classical.window_key):
        raise ValueError("External inference feature and window identities do not align.")
    result = result.merge(classical, on="window_key", validate="one_to_one", how="left")
    result = result.merge(features, on="window_key", validate="one_to_one", how="left")
    for name in BASELINES[2:]:
        model = name.split("_", 1)[0]
        x = np.asarray(embeddings[model])
        if x.ndim != 2 or len(x) != len(result) or not np.isfinite(x).all():
            raise ValueError("External embedding rows must be finite and aligned with signal windows.")
        result[name] = source.estimators[name].predict(x)
    c, f = source.selection["best_classical"], source.selection["best_foundation"]
    for feature_set, columns in FEATURES.items():
        probability = gate_probability(source.gates[feature_set], result, columns)
        result[f"{feature_set}_probability"] = probability
        for soft in (False, True):
            result[feature_set + ("_soft" if soft else "_hard")] = route(result[c], result[f], probability, soft)
    for alias in ("best_classical", "best_foundation", "best_single"):
        result[alias] = result[source.selection[alias]]
    for kind, column, output in [("motion", "acc_norm_std", "motion_intensity_bin"),
                                 ("quality", "ppg_autocorr_peak_strength", "ppg_quality_bin")]:
        result[output] = apply_regime_bins(result[column], source.regime_edges[kind])
    result["original_participant_id"] = result.participant_id
    result["participant_id"] = dataset + "::" + result.participant_id.astype(str)
    result["window_key"] = dataset + "::" + result.window_key.astype(str)
    result["dataset"] = dataset
    result["activity"] = result.session_name
    return result


def _acceleration_source_files(dataset: str, root: Path, participants) -> list[Path]:
    if dataset == "PPG-DaLiA":
        from src.data.ppgdalia_loader import discover_ppgdalia_participant_paths
        return [item.pickle_path for item in discover_ppgdalia_participant_paths(root) if item.participant_id in participants]
    from src.data.wildppg_loader import load_wildppg_manifest
    records = load_wildppg_manifest(root)
    return [root / "wildppg_wrist_manifest.csv"] + sorted({r.acc_path for r in records if r.participant_id in participants and r.acc_path})


def run_external_evaluation(processed_manifest: Path, source_results_root: Path, output_root: Path,
                            acc_scale_to_mps2: float, unit_contract: dict | None = None,
                            allow_synthetic_test: bool = False) -> dict:
    source_results_root, output_root = Path(source_results_root).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(source_results_root) or source_results_root.is_relative_to(output_root):
        raise ValueError("External output must be separate from the frozen source results.")
    if (output_root / "transfer_manifest.json").exists():
        raise FileExistsError("An external transfer is already frozen in this output directory; do not overwrite it.")
    if not Path(processed_manifest).is_file():
        raise FileNotFoundError(f"Real external processed manifest missing: {processed_manifest}")
    manifest = load_processed_manifest(processed_manifest)
    windows = load_processed_windows(manifest["windows_path"]).reset_index(drop=True)
    dataset = validate_external_manifest(manifest, windows)
    fixture = any(word in str(manifest.get("dataset_root", "")).lower() for word in ("_verification", "synthetic", "fixture"))
    if fixture and not allow_synthetic_test:
        raise ValueError("Synthetic fixture is not real external validation; --allow-synthetic-test is required.")
    units = validate_unit_contract(acc_scale_to_mps2, dataset, unit_contract)
    source = load_frozen_source(source_results_root)
    dataset_root = Path(manifest["dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Real external raw acceleration dataset missing: {dataset_root}")
    if output_root.is_relative_to(dataset_root) or dataset_root.is_relative_to(output_root):
        raise ValueError("External results must be separate from external raw data.")
    output_root.mkdir(parents=True, exist_ok=True)
    windows["window_key"] = window_keys(windows)
    contract = {"dataset": dataset, "processed_manifest_sha256": sha256(processed_manifest),
        "windows_sha256": sha256(manifest["windows_path"]), "source_frozen_fit_sha256": sha256(source.root / "manifests/frozen_fit.json"),
        "all_exporter_splits_evaluated": True, "frozen_external_units": units,
        "external_participants": sorted(windows.participant_id.unique().tolist()),
        "source_fit_participants": ["GalaxyPPG::" + p for p in source.frozen["fit_participants"]],
        "checkpoint_sha256": {name: sha256(path) for name, path in source.checkpoints.items()},
        "source_selection": source.selection, "source_preprocessing": source.config["preprocessing"],
        "external_loader_acceleration_sha256": {str(p): sha256(p) for p in _acceleration_source_files(dataset, dataset_root, set(windows.participant_id))},
        "implementation_sha256": sha256(Path(__file__)), "synthetic_test": bool(allow_synthetic_test or fixture)}
    frozen_contract = output_root / "frozen_transfer_contract.json"
    if frozen_contract.exists() and read_json(frozen_contract) != contract:
        raise ValueError("External transfer configuration changed; choose a new output directory.")
    write_json(frozen_contract, contract)
    signals, classical, features = external_signal_features(windows, dataset, dataset_root, acc_scale_to_mps2)
    from src.models.pulseppg_feature import extract_pulseppg_embeddings
    from src.models.papagei_feature import extract_papagei_embeddings
    embeddings = {}
    import torch
    torch.set_num_threads(int(source.config["torch_threads"]))
    for name, hz, extract in [("pulseppg", 50, extract_pulseppg_embeddings), ("papagei", 125, extract_papagei_embeddings)]:
        pc = source.config["preprocessing"]
        preprocess = SignalPreprocessingConfig(mode=pc["mode"], target_sampling_hz=hz,
            normalization=pc["normalization"], apply_bandpass=pc["apply_bandpass"], low_hz=pc["low_hz"],
            high_hz=pc["high_hz"], filter_order=pc["filter_order"])
        preprocess.validate_deployment()
        embeddings[name], _ = extract(signals, checkpoint_path=source.checkpoints[name], device="cpu",
                                     batch_size=source.config["batch_size"], preprocessing_config=preprocess)
    predictions = frozen_predictions(signals, classical, features, embeddings, source, dataset)
    prediction_path = output_root / "frozen_predictions_without_labels.csv"
    predictions.to_csv(prediction_path, index=False)
    write_json(output_root / "inference_freeze.json", {"predictions_sha256": sha256(prediction_path),
        "contract_sha256": sha256(frozen_contract), "no_external_model_fitting": True,
        "labels_excluded_from_prediction_interface": True})
    names = MODEL_NAMES + ["best_classical", "best_foundation", "best_single"]
    references = windows[["window_key", "label_hr_bpm"]].copy()
    references.window_key = dataset + "::" + references.window_key.astype(str)
    frame = predictions.merge(references, on="window_key", validate="one_to_one", how="left")
    common = np.isfinite(frame[names + ["label_hr_bpm"]].to_numpy(float)).all(axis=1)
    coverage = {"all_external_windows": len(frame), "common_windows": int(common.sum()),
        "excluded_windows": int((~common).sum()), "all_participants": int(frame.participant_id.nunique()),
        "evaluated_participants": int(frame.loc[common].participant_id.nunique()),
        "invalid_by_model": {name: int((~np.isfinite(frame[name])).sum()) for name in names},
        "acc_windows_without_samples": int(features.acc_sample_count.eq(0).sum()),
        "mask_rule": "one_common_finite_set_all_predeclared_experts_and_routers;no_error_filter"}
    write_json(output_root / "coverage.json", coverage)
    frame = frame.loc[common].copy()
    if frame.empty:
        raise ValueError("No common finite external windows; no research metrics generated.")
    frame = frame.rename(columns={"label_hr_bpm": "y_true"})
    identity = ["window_key", "participant_id", "original_participant_id", "dataset", "session_id", "activity",
                "window_start_ms", "window_end_ms", "motion_intensity_bin", "ppg_quality_bin", "y_true"]
    long = frame.melt(id_vars=identity, value_vars=names, var_name="model_name", value_name="y_pred")
    long.to_csv(output_root / "external_predictions.csv", index=False)
    summaries = []
    for name, group in long.groupby("model_name", sort=False):
        participant_mae = (group.y_pred-group.y_true).abs().groupby(group.participant_id).mean()
        summaries.append({"model_name": name, **metrics(group.y_true, group.y_pred),
            "n_participants": group.participant_id.nunique(), "participant_mean_MAE": participant_mae.mean(),
            "participant_std_MAE": participant_mae.std(ddof=1), "unit_contract_verified": units["unit_contract_verified"]})
    pd.DataFrame(summaries).to_csv(output_root / "summary.csv", index=False)
    for column in ["participant_id", "activity", "motion_intensity_bin", "ppg_quality_bin"]:
        rows = [{"model_name": name, column: key, **metrics(group.y_true, group.y_pred),
                 "n_participants": group.participant_id.nunique()}
                for (name, key), group in long.groupby(["model_name", column], dropna=False)]
        pd.DataFrame(rows).to_csv(output_root / f"by_{column}.csv", index=False)
    synthetic = bool(allow_synthetic_test or fixture)
    status = "synthetic_implementation_test_only" if synthetic else (
        "frozen_external_evaluation_complete" if units["unit_contract_verified"] else "requires_verified_unit_contract")
    receipt = {"status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_result_eligible": units["unit_contract_verified"] and not synthetic,
        "no_external_fit": True, "no_external_selection_or_calibration": True,
        "source_models_unchanged": True, "pretrained_weights_unchanged": True,
        "exporter_split_ignored": True, "identity_namespace": dataset + "::", "coverage": coverage,
        "contract": contract, "artifacts": {p.name: sha256(p) for p in output_root.iterdir() if p.is_file() and p.name != "status.json"},
        "limitations": ["Unit evidence must support both source and target scales before interpretation.",
                        "No model, gate, preprocessing or threshold was tuned on external labels.",
                        "Any external fine-tuning requires a separate explicitly adapted protocol."]}
    write_json(output_root / "transfer_manifest.json", receipt)
    write_json(output_root / "status.json", {"status": status, "research_result_eligible": receipt["research_result_eligible"]})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--acc-scale-to-mps2", type=float, required=True)
    parser.add_argument("--acc-unit-contract", type=Path)
    parser.add_argument("--allow-synthetic-test", action="store_true")
    args = parser.parse_args()
    try:
        contract = read_json(args.acc_unit_contract) if args.acc_unit_contract else None
        with threadpool_limits(limits=4):
            receipt = run_external_evaluation(args.processed_manifest, args.source_results_root,
                args.output_root, args.acc_scale_to_mps2, contract, args.allow_synthetic_test)
        print(json.dumps({"status": receipt["status"], "research_result_eligible": receipt["research_result_eligible"]}))
    except (FileNotFoundError, ValueError, FileExistsError) as error:
        # Never write a status into the source tree when path validation rejects an invocation.
        output, source = args.output_root.resolve(), args.source_results_root.resolve()
        safe_status = not (output.is_relative_to(source) or source.is_relative_to(output))
        if args.processed_manifest.is_file():
            try:
                raw = Path(load_processed_manifest(args.processed_manifest)["dataset_root"]).resolve()
                safe_status = safe_status and not (output.is_relative_to(raw) or raw.is_relative_to(output))
            except (KeyError, ValueError, OSError):
                safe_status = False
        if safe_status:
            status = "blocked_missing_checkpoint" if isinstance(error, FileNotFoundError) and "checkpoint" in str(error).lower() else (
                "blocked_missing_data" if isinstance(error, FileNotFoundError) else "failed_validation")
            if not (output / "transfer_manifest.json").exists():
                write_json(output / "status.json", {"status": status, "external_validation": "requires_real_data",
                    "research_result_eligible": False, "error": str(error)})
        parser.exit(2, f"External evaluation blocked: {error}\n")


if __name__ == "__main__":
    main()
