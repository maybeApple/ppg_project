"""Shared helpers for foundation-model feature extraction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from src.data import load_processed_windows
from src.data.preprocessing import estimate_sampling_rate_hz


DEFAULT_PROCESSED_MANIFEST = (
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


def default_external_root() -> Path:
    """Return the repository-local folder that stores cloned third-party code."""

    return Path(__file__).resolve().parents[2] / "external"


def load_windows_from_manifest(manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load persisted processed windows from a manifest JSON file."""

    resolved_manifest = Path(manifest_path) if manifest_path is not None else DEFAULT_PROCESSED_MANIFEST
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    manifest["dataset_root"] = str(_resolve_manifest_path(resolved_manifest, manifest["dataset_root"]))
    manifest["labels_path"] = str(_resolve_manifest_path(resolved_manifest, manifest["labels_path"]))
    manifest["windows_path"] = str(_resolve_manifest_path(resolved_manifest, manifest["windows_path"]))
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
) -> np.ndarray:
    """Convert window rows into a fixed-length matrix ready for model input."""

    prepared = [
        prepare_window_signal(
            row=row,
            target_sampling_hz=target_sampling_hz,
            apply_zscore=apply_zscore,
            apply_bandpass=apply_bandpass,
            low_hz=low_hz,
            high_hz=high_hz,
            filter_order=filter_order,
        )
        for row in windows.itertuples(index=False)
    ]
    return np.stack(prepared).astype(np.float32, copy=False)


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
    metadata = windows.loc[
        :,
        [
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
            "label_hr_bpm",
            "label_aggregation",
            "reference_source",
        ],
    ].copy()
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
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


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
