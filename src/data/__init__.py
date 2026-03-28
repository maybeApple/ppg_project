"""Data access and preprocessing utilities."""

from .cache import (
    ProcessedDatasetManifest,
    annotate_window_dataset,
    build_artifact_name,
    build_label_table,
    default_processed_root,
    load_processed_labels,
    load_processed_windows,
    save_processed_dataset,
)
from .loader import (
    POLAR_PHONE_TIMESTAMP_OFFSET_MS,
    ParticipantData,
    load_metadata,
    load_participant_data,
    resolve_dataset_root,
)
from .preprocessing import AlignedSession, SessionInterval, align_participant_sessions
from .windowing import (
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    build_window_dataset,
    default_stride_rationale,
    split_by_participant,
)

__all__ = [
    "AlignedSession",
    "DEFAULT_STRIDE_SECONDS",
    "DEFAULT_WINDOW_SECONDS",
    "POLAR_PHONE_TIMESTAMP_OFFSET_MS",
    "ProcessedDatasetManifest",
    "ParticipantData",
    "SessionInterval",
    "annotate_window_dataset",
    "align_participant_sessions",
    "build_artifact_name",
    "build_label_table",
    "build_window_dataset",
    "default_processed_root",
    "default_stride_rationale",
    "load_processed_labels",
    "load_processed_windows",
    "load_metadata",
    "load_participant_data",
    "resolve_dataset_root",
    "save_processed_dataset",
    "split_by_participant",
]
