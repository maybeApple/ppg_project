"""Persist processed window and label artifacts for reuse."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

WINDOW_SEQUENCE_COLUMNS = [
    "ppg_timestamps_ms",
    "ppg_values",
    "reference_timestamps_ms",
    "reference_hr_bpm_values",
    "reference_rr_interval_ms_values",
]

LABEL_COLUMNS = [
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
]


@dataclass(slots=True)
class ProcessedDatasetManifest:
    """Metadata for one persisted processed dataset snapshot."""

    artifact_name: str
    created_at_utc: str
    dataset_root: str
    reference_source: str
    window_seconds: float
    stride_seconds: float
    label_aggregation: str
    test_size: float
    random_state: int
    num_windows: int
    num_train_windows: int
    num_test_windows: int
    num_participants: int
    train_participants: list[str]
    test_participants: list[str]
    split_config_path: str | None
    split_name: str | None
    validation_strategy: dict[str, object] | None
    validation_folds: list[dict[str, object]] | None
    windows_path: str
    labels_path: str

    def to_dict(self) -> dict[str, object]:
        """Convert the manifest into a JSON-serializable dictionary."""

        return asdict(self)


def default_processed_root() -> Path:
    """Return the default processed-data root in this repository."""

    return Path(__file__).resolve().parents[2] / "data" / "processed"


def load_processed_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a processed-data manifest and resolve stored relative paths."""

    resolved_manifest_path = Path(manifest_path).resolve()
    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    for key in ("dataset_root", "windows_path", "labels_path", "split_config_path"):
        stored_path = payload.get(key)
        if not stored_path:
            continue
        payload[key] = str(resolve_manifest_path(resolved_manifest_path, stored_path))
    payload["manifest_path"] = str(resolved_manifest_path)
    return payload


def build_artifact_name(
    reference_source: str,
    window_seconds: float,
    stride_seconds: float,
    label_aggregation: str,
) -> str:
    """Build a compact, filesystem-safe artifact name from dataset settings."""

    return (
        f"galaxyppg_{reference_source}_"
        f"w{_format_float_token(window_seconds)}_"
        f"s{_format_float_token(stride_seconds)}_"
        f"{label_aggregation}"
    )


def annotate_window_dataset(
    windows: pd.DataFrame,
    train_participants: list[str],
    test_participants: list[str],
) -> pd.DataFrame:
    """Add reproducible window IDs and split labels to a window table."""

    if windows.empty:
        return windows.copy()

    split_by_participant = {participant_id: "train" for participant_id in train_participants}
    split_by_participant.update({participant_id: "test" for participant_id in test_participants})

    annotated = windows.copy()
    annotated.insert(
        0,
        "window_uid",
        (
            annotated["session_id"].astype(str)
            + ":w"
            + annotated["window_index"].astype("int64").astype(str)
        ),
    )
    annotated.insert(1, "split", annotated["participant_id"].map(split_by_participant).fillna("unassigned"))
    return annotated


def build_label_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Extract the scalar label table from a processed window dataset."""

    if windows.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return windows.loc[:, LABEL_COLUMNS].copy()


def save_processed_dataset(
    windows: pd.DataFrame,
    dataset_root: str | Path,
    reference_source: str,
    window_seconds: float,
    stride_seconds: float,
    label_aggregation: str,
    test_size: float,
    random_state: int,
    train_participants: list[str],
    test_participants: list[str],
    split_config_path: str | Path | None = None,
    split_name: str | None = None,
    validation_strategy: dict[str, object] | None = None,
    validation_folds: list[dict[str, object]] | None = None,
    output_root: str | Path | None = None,
) -> ProcessedDatasetManifest:
    """Save processed windows, labels, and a manifest to disk."""

    processed_root = Path(output_root) if output_root is not None else default_processed_root()
    processed_root.mkdir(parents=True, exist_ok=True)
    windows_dir = processed_root / "windows"
    labels_dir = processed_root / "labels"
    windows_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = build_artifact_name(
        reference_source=reference_source,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        label_aggregation=label_aggregation,
    )
    windows_path = windows_dir / f"{artifact_name}_windows.jsonl.gz"
    labels_path = labels_dir / f"{artifact_name}_labels.csv"
    manifest_path = processed_root / f"{artifact_name}_manifest.json"
    manifest_dir = manifest_path.parent

    normalized_windows = _normalize_sequence_columns(windows)
    labels = build_label_table(normalized_windows)

    normalized_windows.to_json(
        windows_path,
        orient="records",
        lines=True,
        compression="infer",
        force_ascii=False,
    )
    labels.to_csv(labels_path, index=False)

    split_counts = normalized_windows["split"].value_counts().to_dict() if not normalized_windows.empty else {}
    manifest = ProcessedDatasetManifest(
        artifact_name=artifact_name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_root=_portable_path_for_manifest(Path(dataset_root), manifest_dir),
        reference_source=reference_source,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        label_aggregation=label_aggregation,
        test_size=test_size,
        random_state=random_state,
        num_windows=int(len(normalized_windows)),
        num_train_windows=int(split_counts.get("train", 0)),
        num_test_windows=int(split_counts.get("test", 0)),
        num_participants=int(normalized_windows["participant_id"].nunique()) if not normalized_windows.empty else 0,
        train_participants=list(train_participants),
        test_participants=list(test_participants),
        split_config_path=(
            None
            if split_config_path is None
            else _portable_path_for_manifest(Path(split_config_path), manifest_dir)
        ),
        split_name=split_name,
        validation_strategy=validation_strategy,
        validation_folds=validation_folds,
        windows_path=_portable_path_for_manifest(windows_path, manifest_dir),
        labels_path=_portable_path_for_manifest(labels_path, manifest_dir),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_processed_windows(windows_path: str | Path) -> pd.DataFrame:
    """Load a persisted window dataset from JSON Lines."""

    return pd.read_json(Path(windows_path), orient="records", lines=True, compression="infer")


def load_processed_labels(labels_path: str | Path) -> pd.DataFrame:
    """Load a persisted label table from CSV."""

    return pd.read_csv(Path(labels_path))


def resolve_manifest_path(manifest_path: str | Path, stored_path: str | Path) -> Path:
    """Resolve an artifact path stored inside a manifest JSON file."""

    resolved_manifest_path = Path(manifest_path).resolve()
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (resolved_manifest_path.parent / candidate).resolve()


def _normalize_sequence_columns(windows: pd.DataFrame) -> pd.DataFrame:
    """Convert sequence-valued columns into JSON-safe Python lists."""

    normalized = windows.copy()
    for column in WINDOW_SEQUENCE_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].apply(_normalize_sequence_value)
    return normalized


def _normalize_sequence_value(value: object) -> list[object]:
    """Normalize a single list-like cell before JSON export."""

    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if pd.isna(value):
        return []
    return [_normalize_scalar(value)]


def _normalize_scalar(value: object) -> object:
    """Convert pandas / NumPy scalars and missing values into JSON-safe values."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _format_float_token(value: float) -> str:
    """Render a float for stable artifact names."""

    return f"{value:g}".replace(".", "p")


def _portable_path_for_manifest(target_path: str | Path, manifest_dir: Path) -> str:
    """Store a path relative to the manifest directory for cross-machine reuse."""

    resolved_target = Path(target_path).resolve()
    return Path(os.path.relpath(resolved_target, start=manifest_dir.resolve())).as_posix()
