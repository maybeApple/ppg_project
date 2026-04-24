"""Window segmentation and label generation utilities."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Literal

import pandas as pd

from .loader import load_participant_data, list_participants
from .labels import LabelAggregation, LabelGenerationConfig, LabelMethod, compute_window_label
from .preprocessing import AlignedSession, align_participant_sessions

#Each sample window is 10 seconds long.The window sliding step is 2 seconds.
#Therefore, it is an overlapping sliding window, with adjacent windows overlapping for 8 seconds.
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_STRIDE_SECONDS = 2.0
DEFAULT_MIN_VALID_BEATS = 2


def build_window_dataset(
    dataset_root: str | Path | None = None,
    participant_ids: list[str] | None = None,
    reference_source: Literal["auto", "hr", "ibi", "ecg"] = "auto",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    label_aggregation: LabelAggregation = "median",
    label_method: LabelMethod = "beat_interval_instant_hr",
    min_duration_seconds: float = 10.0,
    min_reference_samples: int = 1,
    min_valid_beats: int = DEFAULT_MIN_VALID_BEATS,
    min_ppg_coverage: float = 0.8,
) -> pd.DataFrame:
    """Load, align, and window the whole dataset into a single table."""

    selected_participants = participant_ids or list_participants(dataset_root)
    all_windows: list[pd.DataFrame] = []
    label_config = LabelGenerationConfig(
        method=label_method,
        aggregation=label_aggregation,
        min_valid_beats=min_valid_beats,
        min_reference_samples=min_reference_samples,
    )

    for participant_id in selected_participants:
        participant = load_participant_data(
            participant_id=participant_id,
            dataset_root=dataset_root,
            reference_source=reference_source,
        )
        aligned_sessions = align_participant_sessions(
            participant=participant,
            min_duration_seconds=min_duration_seconds,
        )
        session_windows = generate_windows_from_sessions(
            aligned_sessions=aligned_sessions,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            label_config=label_config,
            min_ppg_coverage=min_ppg_coverage,
        )
        if not session_windows.empty:
            all_windows.append(session_windows)

    if not all_windows:
        return pd.DataFrame(
            columns=[
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
                "ppg_timestamps_ms",
                "ppg_values",
                "ppg_raw_values",
                "reference_timestamps_ms",
                "reference_hr_bpm_values",
                "reference_rr_interval_ms_values",
            ]
        )

    return pd.concat(all_windows, ignore_index=True)


def generate_windows_from_sessions(
    aligned_sessions: list[AlignedSession],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    label_config: LabelGenerationConfig | None = None,
    min_ppg_coverage: float = 0.8,
) -> pd.DataFrame:
    """Window multiple aligned sessions and concatenate the results."""

    resolved_label_config = label_config or LabelGenerationConfig()
    windows = [
        generate_session_windows(
            aligned_session=session,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            label_config=resolved_label_config,
            min_ppg_coverage=min_ppg_coverage,
        )
        for session in aligned_sessions
    ]
    windows = [frame for frame in windows if not frame.empty]
    if not windows:
        return pd.DataFrame()
    return pd.concat(windows, ignore_index=True)


def generate_session_windows(
    aligned_session: AlignedSession,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    label_config: LabelGenerationConfig | None = None,
    min_ppg_coverage: float = 0.8,
) -> pd.DataFrame:
    """Generate fixed windows and one HR label per window from a single session."""

    resolved_label_config = label_config or LabelGenerationConfig()
    if resolved_label_config.aggregation not in {"mean", "median"}:
        raise ValueError(f"Unsupported label aggregation: {resolved_label_config.aggregation}")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")

    window_ms = int(window_seconds * 1000)
    stride_ms = int(stride_seconds * 1000)
    session_duration_ms = aligned_session.end_ms - aligned_session.start_ms
    if session_duration_ms < window_ms:
        return pd.DataFrame()

    expected_sampling_hz = aligned_session.ppg_sampling_hz or 0.0
    min_ppg_samples = 1
    if expected_sampling_hz > 0:
        min_ppg_samples = max(1, math.ceil(window_seconds * expected_sampling_hz * min_ppg_coverage))

    window_rows: list[dict] = []
    for window_index, window_start_ms in enumerate(
        range(aligned_session.start_ms, aligned_session.end_ms - window_ms + 1, stride_ms)
    ):
        window_end_ms = window_start_ms + window_ms
        ppg_window = aligned_session.ppg.loc[
            (aligned_session.ppg["timestamp_ms"] >= window_start_ms)
            & (aligned_session.ppg["timestamp_ms"] < window_end_ms)
        ]
        reference_window = aligned_session.reference.loc[
            (aligned_session.reference["timestamp_ms"] >= window_start_ms)
            & (aligned_session.reference["timestamp_ms"] < window_end_ms)
        ]

        if len(ppg_window) < min_ppg_samples:
            continue

        label = compute_window_label(reference_window=reference_window, config=resolved_label_config)
        if label is None:
            continue

        ppg_inverted = bool(ppg_window["ppg_inverted"].all()) if "ppg_inverted" in ppg_window.columns else False
        ppg_canonical_source = (
            str(ppg_window["ppg_canonical_source"].iloc[0])
            if "ppg_canonical_source" in ppg_window.columns and not ppg_window.empty
            else ""
        )

        window_rows.append(
            {
                "participant_id": aligned_session.participant_id,
                "session_id": aligned_session.session_id,
                "session_name": aligned_session.session_name,
                "window_index": window_index,
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "window_length_s": window_seconds,
                "stride_s": stride_seconds,
                "ppg_sampling_hz": aligned_session.ppg_sampling_hz,
                "ppg_sample_count": int(len(ppg_window)),
                "reference_sample_count": label.reference_sample_count,
                "valid_beat_count": label.valid_beat_count,
                "label_hr_bpm": label.label_hr_bpm,
                "label_method": label.label_method,
                "label_aggregation": label.label_aggregation,
                "reference_source": aligned_session.reference_source,
                "ppg_inverted": ppg_inverted,
                "ppg_canonical_source": ppg_canonical_source,
                "ppg_timestamps_ms": ppg_window["timestamp_ms"].tolist(),
                "ppg_values": ppg_window["ppg"].tolist(),
                "ppg_raw_values": ppg_window["ppg_raw"].tolist() if "ppg_raw" in ppg_window.columns else [],
                "reference_timestamps_ms": label.reference_timestamps_ms,
                "reference_hr_bpm_values": label.reference_hr_bpm_values,
                "reference_rr_interval_ms_values": label.reference_rr_interval_ms_values,
            }
        )

    return pd.DataFrame(window_rows)


def split_by_participant(
    windows: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    test_participants: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split windows into train/test partitions strictly by participant ID."""

    if windows.empty:
        return windows.copy(), windows.copy(), [], []

    participants = sorted(windows["participant_id"].dropna().unique().tolist())
    if len(participants) < 2:
        raise ValueError("At least two participants are required for a train/test split.")

    if test_participants is None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in the open interval (0, 1).")
        shuffled = participants[:]
        random.Random(random_state).shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * test_size)))
        chosen_test_participants = sorted(shuffled[:test_count])
    else:
        unknown = sorted(set(test_participants) - set(participants))
        if unknown:
            raise ValueError(f"Unknown test participants: {unknown}")
        chosen_test_participants = sorted(set(test_participants))

    chosen_train_participants = sorted(set(participants) - set(chosen_test_participants))
    train_windows = windows.loc[windows["participant_id"].isin(chosen_train_participants)].reset_index(drop=True)
    test_windows = windows.loc[windows["participant_id"].isin(chosen_test_participants)].reset_index(drop=True)
    return train_windows, test_windows, chosen_train_participants, chosen_test_participants


def default_stride_rationale() -> str:
    """Explain the default stride choice used by this project."""

    return (
        "A 2-second stride preserves more temporal detail than 5 seconds while avoiding the "
        "extreme redundancy of a 1-second stride. With 10-second windows and participant-level "
        "splits, it is a practical default for early experiments."
    )
