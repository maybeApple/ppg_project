"""Shared helpers for foundation-model feature extraction."""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from src.data import load_processed_manifest, load_processed_windows
from src.data.preprocessing import estimate_sampling_rate_hz


DEFAULT_PROCESSED_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json"
)
LEGACY_PROCESSED_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "galaxyppg_hr_w10_s2_median_manifest.json"
)


@dataclass(slots=True)
class EmbeddingExportSummary:
    """Summary of one exported embedding table."""

    model_name: str
    created_at_local: str
    num_windows: int
    embedding_dim: int
    target_sampling_hz: int
    features_path: str
    metadata_path: str
    manifest_path: str
    source_windows_path: str
    checkpoint_path: str | None
    external_repo_root: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class SignalPreprocessingConfig:
    """Configuration for traceable signal preprocessing before embedding export."""

    mode: str
    target_sampling_hz: int
    apply_bandpass: bool = True
    normalization: str = "per_window_zscore"
    low_hz: float = 0.5
    high_hz: float = 12.0
    filter_order: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary with an explicit operation order."""

        payload = asdict(self)
        payload["operation_order"] = self.operation_order()
        offline = self.normalization == "person_specific_zscore"
        payload["deployment_compatible"] = not offline
        payload["causality"] = "noncausal_full_participant_offline" if offline else "causal_at_window_end"
        payload["prediction_available_at"] = "window_end_ms"
        payload["normalization_scope"] = (
            "all_available_participant_windows_including_future" if offline else
            "completed_windows_of_same_participant_and_session_including_current"
            if self.normalization == "causal_running_zscore" else "current_window_only"
        )
        payload["normalization_version"] = "window_end_causal_v2"
        payload["running_history_policy"] = "reset_each_session; overlapping samples count once per completed window"
        payload["filter_causality"] = "filtfilt_within_completed_window" if self.apply_bandpass else "no_filter"
        return payload

    def validate_deployment(self, deployment_style: bool = True) -> None:
        """Exclude offline full-person statistics from deployment-style experiments."""

        if deployment_style and self.normalization == "person_specific_zscore":
            raise ValueError(
                "person_specific_zscore uses future participant windows and is a noncausal "
                "offline ablation; use causal_running_zscore for deployment-style evaluation."
            )

    def operation_order(self) -> list[str]:
        """Return the exact preprocessing order applied to the signal."""

        operations = ["timestamp_resample"]
        if self.normalization == "per_window_zscore":
            operations.append("per_window_zscore")
        if self.apply_bandpass:
            operations.append("chebyshev2_bandpass")
        if self.normalization in {"person_specific_zscore", "causal_running_zscore"}:
            operations.append(self.normalization)
        return operations


def build_signal_preprocessing_config(
    model_name: str,
    target_sampling_hz: int,
    mode: str = "harmonized",
    normalization: str | None = None,
    apply_bandpass: bool | None = None,
) -> SignalPreprocessingConfig:
    """Build a model preprocessing config without scattering model branches."""

    if mode not in {"harmonized", "model_faithful"}:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    if normalization is None:
        normalization = "per_window_zscore"
    if apply_bandpass is None:
        apply_bandpass = True

    if normalization not in {"none", "per_window_zscore", "person_specific_zscore", "causal_running_zscore"}:
        raise ValueError(f"Unsupported normalization strategy: {normalization}")

    return SignalPreprocessingConfig(
        mode=mode,
        target_sampling_hz=target_sampling_hz,
        apply_bandpass=bool(apply_bandpass),
        normalization=normalization,
    )


def load_signal_preprocessing_config(
    config_path: str | Path,
    model_name: str,
    mode_name: str | None,
    target_sampling_hz: int,
    normalization_override: str | None = None,
    apply_bandpass_override: bool | None = None,
) -> SignalPreprocessingConfig:
    """Load a signal preprocessing config from an experiment-mode JSON file."""

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    resolved_mode_name = mode_name or str(payload.get("default_mode", "harmonized"))
    modes = payload.get("modes", {})
    if resolved_mode_name not in modes:
        raise KeyError(f"Experiment mode `{resolved_mode_name}` is not defined in {config_path}.")

    mode_payload = modes[resolved_mode_name]
    preprocessing_payload = mode_payload.get("preprocessing")
    if preprocessing_payload is None:
        preprocessing_payload = mode_payload.get("models", {}).get(model_name, {}).get("preprocessing")
    if preprocessing_payload is None:
        raise KeyError(
            f"Experiment mode `{resolved_mode_name}` does not define preprocessing for model `{model_name}`."
        )

    normalization = normalization_override or preprocessing_payload.get("normalization")
    apply_bandpass = (
        apply_bandpass_override
        if apply_bandpass_override is not None
        else preprocessing_payload.get("apply_bandpass", True)
    )
    config = build_signal_preprocessing_config(
        model_name=model_name,
        target_sampling_hz=target_sampling_hz,
        mode=str(preprocessing_payload.get("mode", resolved_mode_name)),
        normalization=None if normalization is None else str(normalization),
        apply_bandpass=bool(apply_bandpass),
    )
    config.low_hz = float(preprocessing_payload.get("low_hz", config.low_hz))
    config.high_hz = float(preprocessing_payload.get("high_hz", config.high_hz))
    config.filter_order = int(preprocessing_payload.get("filter_order", config.filter_order))
    return config


def default_external_root() -> Path:
    """Return the repository-local folder that stores downloaded external checkpoints."""

    return Path(__file__).resolve().parents[2] / "external"


def load_windows_from_manifest(manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load persisted processed windows from a manifest JSON file."""

    if manifest_path is not None:
        resolved_manifest = Path(manifest_path)
    else:
        resolved_manifest = DEFAULT_PROCESSED_MANIFEST if DEFAULT_PROCESSED_MANIFEST.exists() else LEGACY_PROCESSED_MANIFEST
    manifest = load_processed_manifest(resolved_manifest)
    windows = load_processed_windows(manifest["windows_path"])
    return windows, manifest


def select_window_subset(windows: pd.DataFrame, max_windows: int | None = None) -> pd.DataFrame:
    """Optionally limit the number of processed windows for smoke testing."""

    if max_windows is None or max_windows >= len(windows):
        return windows.reset_index(drop=True)
    return windows.iloc[:max_windows].reset_index(drop=True)


def build_window_signal_matrix(
    windows: pd.DataFrame,
    target_sampling_hz: int,
    apply_zscore: bool = True,
    apply_bandpass: bool = True,
    low_hz: float = 0.5,
    high_hz: float = 12.0,
    filter_order: int = 4,
    preprocessing_config: SignalPreprocessingConfig | None = None,
) -> np.ndarray:
    """Convert window rows into a fixed-length matrix ready for model input."""

    config = preprocessing_config or SignalPreprocessingConfig(
        mode="harmonized",
        target_sampling_hz=target_sampling_hz,
        apply_bandpass=apply_bandpass,
        normalization="per_window_zscore" if apply_zscore else "none",
        low_hz=low_hz,
        high_hz=high_hz,
        filter_order=filter_order,
    )
    prepared = [
        prepare_window_signal(
            row=row,
            target_sampling_hz=config.target_sampling_hz,
            apply_zscore=config.normalization == "per_window_zscore",
            apply_bandpass=config.apply_bandpass,
            low_hz=config.low_hz,
            high_hz=config.high_hz,
            filter_order=config.filter_order,
        )
        for row in windows.itertuples(index=False)
    ]
    matrix = np.stack(prepared).astype(np.float32, copy=False)
    if config.normalization == "person_specific_zscore":
        return normalize_matrix_by_participant(matrix, windows)
    if config.normalization == "causal_running_zscore":
        return normalize_matrix_by_causal_participant_history(matrix, windows)
    return matrix


def prepare_window_signal(
    row: Any,
    target_sampling_hz: int,
    apply_zscore: bool = True,
    apply_bandpass: bool = True,
    low_hz: float = 0.5,
    high_hz: float = 12.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Resample one PPG window to a fixed sampling rate and apply light preprocessing."""

    values = np.asarray(row.ppg_values, dtype=float)
    timestamps_ms = np.asarray(row.ppg_timestamps_ms, dtype=float)
    if values.ndim != 1 or timestamps_ms.ndim != 1 or len(values) != len(timestamps_ms):
        raise ValueError("Window signal values and timestamps must be aligned 1D arrays.")
    if len(values) < 2:
        raise ValueError("At least two PPG samples are required to build a model input window.")

    window_seconds = float(row.window_length_s)
    signal_resampled = resample_signal_by_timestamp(
        values=values,
        timestamps_ms=timestamps_ms,
        window_start_ms=int(row.window_start_ms),
        window_seconds=window_seconds,
        target_sampling_hz=target_sampling_hz,
    )

    if apply_zscore:
        signal_resampled = zscore_signal(signal_resampled)
    if apply_bandpass:
        signal_resampled = bandpass_filter_ppg(
            signal_values=signal_resampled,
            sampling_hz=float(target_sampling_hz),
            low_hz=low_hz,
            high_hz=high_hz,
            filter_order=filter_order,
        )
    return signal_resampled.astype(np.float32, copy=False)


def resample_signal_by_timestamp(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    window_start_ms: int,
    window_seconds: float,
    target_sampling_hz: int,
) -> np.ndarray:
    """Resample an irregularly sampled window onto a fixed target grid."""

    sorted_order = np.argsort(timestamps_ms)
    values = values[sorted_order]
    timestamps_ms = timestamps_ms[sorted_order]

    unique_timestamps_ms, unique_indices = np.unique(timestamps_ms, return_index=True)
    values = values[unique_indices]
    timestamps_ms = unique_timestamps_ms

    relative_time_s = (timestamps_ms - float(window_start_ms)) / 1000.0
    target_length = int(round(window_seconds * target_sampling_hz))
    target_time_s = np.arange(target_length, dtype=float) / float(target_sampling_hz)

    return np.interp(
        target_time_s,
        relative_time_s,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )


def zscore_signal(signal_values: np.ndarray) -> np.ndarray:
    """Apply per-window z-score normalization with flat-signal protection."""

    mean_value = float(np.mean(signal_values))
    std_value = float(np.std(signal_values))
    if not np.isfinite(std_value) or std_value <= 0:
        return signal_values - mean_value
    return (signal_values - mean_value) / std_value


def normalize_matrix_by_participant(matrix: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """Offline NONCAUSAL full-person normalization; unsuitable for deployment claims."""

    normalized = matrix.copy()
    if "participant_id" not in windows.columns:
        raise KeyError("person_specific_zscore requires a participant_id column.")
    participants = windows["participant_id"].astype(str).to_numpy()
    for participant_id in sorted(set(participants)):
        mask = participants == participant_id
        values = normalized[mask]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if np.isfinite(std_value) and std_value > 0:
            normalized[mask] = (values - mean_value) / std_value
        else:
            normalized[mask] = values - mean_value
    return normalized.astype(np.float32, copy=False)


def normalize_matrix_by_causal_participant_history(matrix: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """Normalize at each window end using only completed same-session history.

    Running pooled moments avoid the former quadratic history concatenation. The
    current complete window is available at decision time; later-ending windows
    are excluded even when their start timestamps precede the current start.
    Equal-ending windows use their own samples plus strictly earlier history,
    so ties and arbitrary DataFrame indices cannot change the result.
    """

    required_columns = {"participant_id", "window_start_ms", "window_end_ms"}
    missing_columns = sorted(required_columns - set(windows.columns))
    if missing_columns:
        raise KeyError(f"causal_running_zscore requires columns: {missing_columns}")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(windows) or not np.isfinite(matrix).all():
        raise ValueError("Causal normalization requires finite 2D signals aligned with window metadata.")
    group_columns = ["participant_id"] + (["session_id"] if "session_id" in windows else [])
    if windows[group_columns].isna().any().any():
        raise ValueError("Causal normalization requires nonmissing participant/session identities.")
    starts = pd.to_numeric(windows["window_start_ms"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(windows["window_end_ms"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all() and np.all(ends > starts)):
        raise ValueError("Causal normalization requires finite timestamps and window_end_ms > window_start_ms.")
    normalized = matrix.copy()
    ordered = windows.reset_index(drop=True).copy()
    ordered["_row_position"] = np.arange(len(windows))
    ordered["window_end_ms"] = ends
    ordered = ordered.sort_values([*group_columns, "window_end_ms", "_row_position"])
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, group in ordered.groupby(grouper, sort=False):
        count, mean, m2 = 0, 0.0, 0.0
        for _, simultaneous in group.groupby("window_end_ms", sort=False):
            moments = []
            for position in simultaneous["_row_position"]:
                values = matrix[int(position)]
                sample_count = len(values)
                sample_mean = float(values.mean())
                sample_m2 = float(np.square(values - sample_mean).sum())
                combined_count = count + sample_count
                difference = sample_mean - mean
                combined_mean = mean + difference * sample_count / combined_count
                combined_m2 = m2 + sample_m2 + difference**2 * count * sample_count / combined_count
                std = float(np.sqrt(max(0.0, combined_m2 / combined_count)))
                normalized[int(position)] = (values - combined_mean) / std if std > 0 else values - combined_mean
                moments.append((sample_count, sample_mean, sample_m2))
            for sample_count, sample_mean, sample_m2 in moments:
                combined_count = count + sample_count
                difference = sample_mean - mean
                m2 += sample_m2 + difference**2 * count * sample_count / combined_count
                mean += difference * sample_count / combined_count
                count = combined_count
    return normalized.astype(np.float32, copy=False)


def bandpass_filter_ppg(
    signal_values: np.ndarray,
    sampling_hz: float,
    low_hz: float,
    high_hz: float,
    filter_order: int = 4,
) -> np.ndarray:
    """Apply a stable Chebyshev-II band-pass filter to a 1D PPG signal."""

    nyquist_hz = sampling_hz / 2.0
    bounded_low_hz = max(low_hz, 1e-3)
    bounded_high_hz = min(high_hz, nyquist_hz * 0.99)
    if bounded_low_hz >= bounded_high_hz:
        return signal_values

    b_coeff, a_coeff = signal.cheby2(
        filter_order,
        20,
        [bounded_low_hz, bounded_high_hz],
        btype="bandpass",
        fs=sampling_hz,
    )
    padlen = min(len(signal_values) - 1, 3 * max(len(a_coeff), len(b_coeff)))
    if padlen <= 0:
        return signal_values
    return signal.filtfilt(b_coeff, a_coeff, signal_values, padlen=padlen)


def infer_window_sampling_hz(row: Any) -> float | None:
    """Infer a window sampling rate from its timestamps when metadata is missing."""

    sampling_hz = getattr(row, "ppg_sampling_hz", None)
    if sampling_hz is not None and not pd.isna(sampling_hz):
        return float(sampling_hz)
    timestamps_ms = np.asarray(row.ppg_timestamps_ms, dtype=float)
    return estimate_sampling_rate_hz(timestamps_ms)


def save_embedding_artifacts(
    model_name: str,
    embeddings: np.ndarray,
    windows: pd.DataFrame,
    target_sampling_hz: int,
    output_dir: str | Path,
    source_windows_path: str,
    checkpoint_path: str | None,
    external_repo_root: str | Path,
    extra_manifest_fields: dict[str, Any] | None = None,
) -> EmbeddingExportSummary:
    """Persist extracted embeddings plus per-window metadata."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    features_path = target_dir / f"{model_name}_features.npy"
    metadata_path = target_dir / f"{model_name}_metadata.csv"
    manifest_path = target_dir / f"{model_name}_manifest.json"
    manifest_dir = manifest_path.parent

    np.save(features_path, embeddings)
    metadata_columns = [
        "window_uid",
        "split",
        "participant_id",
        "session_id",
        "session_name",
        "window_index",
        "window_start_ms",
        "window_end_ms",
        "window_length_s",
        "stride_s",
        "ppg_sampling_hz",
        "ppg_sample_count",
        "reference_sample_count",
        "valid_beat_count",
        "label_hr_bpm",
        "label_method",
        "label_aggregation",
        "reference_source",
        "ppg_inverted",
        "ppg_canonical_source",
    ]
    metadata = windows.loc[:, [column for column in metadata_columns if column in windows.columns]].copy()
    metadata.to_csv(metadata_path, index=False)

    summary = EmbeddingExportSummary(
        model_name=model_name,
        created_at_local=datetime.now().isoformat(),
        num_windows=int(len(windows)),
        embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        target_sampling_hz=int(target_sampling_hz),
        features_path=_portable_path_for_manifest(features_path, manifest_dir),
        metadata_path=_portable_path_for_manifest(metadata_path, manifest_dir),
        manifest_path=_portable_path_for_manifest(manifest_path, manifest_dir),
        source_windows_path=_portable_path_for_manifest(source_windows_path, manifest_dir),
        checkpoint_path=None if checkpoint_path is None else _portable_path_for_manifest(checkpoint_path, manifest_dir),
        external_repo_root=_portable_path_for_manifest(external_repo_root, manifest_dir),
    )
    manifest_dict = summary.to_dict()
    if extra_manifest_fields:
        manifest_dict.update(extra_manifest_fields)
    manifest_dict["schema_version"] = "embedding_artifact_v2"
    source_path = Path(source_windows_path)
    manifest_dict["source_windows_sha256"] = _file_sha256(source_path) if source_path.is_file() else None
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    manifest_dict["checkpoint_sha256"] = _file_sha256(checkpoint) if checkpoint is not None and checkpoint.is_file() else None
    manifest_dict["features_sha256"] = _file_sha256(features_path)
    manifest_dict["metadata_sha256"] = _file_sha256(metadata_path)
    manifest_dict["preprocessing_code_sha256"] = _file_sha256(Path(__file__))
    preprocessing = manifest_dict.get("preprocessing")
    if isinstance(preprocessing, dict):
        manifest_dict["deployment_compatible"] = preprocessing.get(
            "deployment_compatible", preprocessing.get("normalization") != "person_specific_zscore"
        )
        manifest_dict["causality"] = preprocessing.get("causality", "unspecified_legacy_preprocessing")
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _file_sha256(path: Path) -> str:
    """Stream artifact hashes without materializing a checkpoint or signal cache."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path: Path, stored_path: str | Path) -> Path:
    """Resolve an artifact path stored inside a manifest."""

    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _portable_path_for_manifest(target_path: str | Path, manifest_dir: Path) -> str:
    """Store a path relative to the manifest directory for cross-machine reuse."""

    resolved_target = Path(target_path).resolve()
    return Path(os.path.relpath(resolved_target, start=manifest_dir.resolve())).as_posix()
