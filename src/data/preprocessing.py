"""Signal preprocessing utilities for PPG and HR alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import ParticipantData


@dataclass(slots=True)
class SessionInterval:
    """A labeled time interval derived from Event.csv."""

    participant_id: str
    session_id: str
    session_name: str
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class AlignedSession:
    """PPG and reference samples clipped to one session and one shared time axis."""

    participant_id: str
    session_id: str
    session_name: str
    start_ms: int
    end_ms: int
    ppg: pd.DataFrame
    reference: pd.DataFrame
    reference_on_ppg: pd.DataFrame
    reference_source: str
    ppg_sampling_hz: float | None


def build_session_intervals(
    participant_id: str,
    events: pd.DataFrame,
    fallback_start_ms: int | None = None,
    fallback_end_ms: int | None = None,
) -> list[SessionInterval]:
    """Pair ENTER/EXIT events into session intervals.

    When Event.csv is missing, a synthetic `full_recording#1` interval is created
    from the provided fallback bounds.
    """

    if events.empty:
        if fallback_start_ms is None or fallback_end_ms is None or fallback_end_ms <= fallback_start_ms:
            return []
        return [
            SessionInterval(
                participant_id=participant_id,
                session_id=f"{participant_id}:full_recording#1",
                session_name="full_recording",
                start_ms=int(fallback_start_ms),
                end_ms=int(fallback_end_ms),
            )
        ]

    starts: dict[str, int] = {}
    counts: dict[str, int] = {}
    intervals: list[SessionInterval] = []

    for row in events.sort_values("timestamp_ms").itertuples(index=False):
        timestamp_ms = int(row.timestamp_ms)
        session_name = str(row.session)
        status = str(row.status).upper()

        if status == "ENTER":
            starts[session_name] = timestamp_ms
        elif status == "EXIT" and session_name in starts:
            start_ms = starts.pop(session_name)
            if timestamp_ms <= start_ms:
                continue
            counts[session_name] = counts.get(session_name, 0) + 1
            intervals.append(
                SessionInterval(
                    participant_id=participant_id,
                    session_id=f"{participant_id}:{session_name}#{counts[session_name]}",
                    session_name=session_name,
                    start_ms=start_ms,
                    end_ms=timestamp_ms,
                )
            )

    return intervals


def align_participant_sessions(
    participant: ParticipantData,
    min_duration_seconds: float = 10.0,
    use_valid_ppg_only: bool = True,
) -> list[AlignedSession]:
    """Clip participant signals to event sessions and place both streams on one time axis."""

    ppg = participant.ppg.copy()
    reference = participant.reference.copy()

    if use_valid_ppg_only and "is_valid_ppg" in ppg.columns:
        ppg = ppg[ppg["is_valid_ppg"]].copy()

    ppg = _sort_and_deduplicate(ppg, "timestamp_ms")
    reference = _sort_and_deduplicate(reference, "timestamp_ms")

    if ppg.empty or reference.empty:
        return []

    overlap_start_ms, overlap_end_ms = compute_overlap_bounds(ppg, reference)

    if overlap_end_ms - overlap_start_ms < min_duration_seconds * 1000:
        return []

    intervals = build_session_intervals(
        participant_id=participant.participant_id,
        events=participant.events,
        fallback_start_ms=overlap_start_ms,
        fallback_end_ms=overlap_end_ms,
    )

    aligned_sessions: list[AlignedSession] = []
    for interval in intervals:
        session_start_ms = max(interval.start_ms, overlap_start_ms)
        session_end_ms = min(interval.end_ms, overlap_end_ms)
        if session_end_ms - session_start_ms < min_duration_seconds * 1000:
            continue

        session_ppg = slice_time_range(ppg, session_start_ms, session_end_ms)
        session_reference = slice_time_range(reference, session_start_ms, session_end_ms)
        if session_ppg.empty or session_reference.empty:
            continue

        session_ppg = session_ppg.copy()
        session_reference = session_reference.copy()
        session_ppg["time_s"] = (session_ppg["timestamp_ms"] - session_start_ms) / 1000.0
        session_reference["time_s"] = (session_reference["timestamp_ms"] - session_start_ms) / 1000.0

        interpolated_reference = pd.DataFrame(
            {
                "timestamp_ms": session_ppg["timestamp_ms"].to_numpy(copy=True),
                "time_s": session_ppg["time_s"].to_numpy(copy=True),
                "reference_hr_bpm": interpolate_reference_to_target(
                    source_timestamps_ms=session_reference["timestamp_ms"].to_numpy(),
                    source_values=session_reference["hr_bpm"].to_numpy(),
                    target_timestamps_ms=session_ppg["timestamp_ms"].to_numpy(),
                ),
            }
        )

        aligned_sessions.append(
            AlignedSession(
                participant_id=participant.participant_id,
                session_id=interval.session_id,
                session_name=interval.session_name,
                start_ms=session_start_ms,
                end_ms=session_end_ms,
                ppg=session_ppg.reset_index(drop=True),
                reference=session_reference.reset_index(drop=True),
                reference_on_ppg=interpolated_reference,
                reference_source=participant.reference_source,
                ppg_sampling_hz=estimate_sampling_rate_hz(session_ppg["timestamp_ms"].to_numpy()),
            )
        )

    return aligned_sessions


def compute_overlap_bounds(ppg: pd.DataFrame, reference: pd.DataFrame) -> tuple[int, int]:
    """Return the shared time range between PPG and the reference stream."""

    overlap_start_ms = int(max(ppg["timestamp_ms"].min(), reference["timestamp_ms"].min()))
    overlap_end_ms = int(min(ppg["timestamp_ms"].max(), reference["timestamp_ms"].max()))
    return overlap_start_ms, overlap_end_ms


def slice_time_range(frame: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Slice a frame with half-open bounds `[start_ms, end_ms)`."""

    mask = (frame["timestamp_ms"] >= start_ms) & (frame["timestamp_ms"] < end_ms)
    return frame.loc[mask]


def interpolate_reference_to_target(
    source_timestamps_ms: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate reference HR values onto the PPG timestamps."""

    source_timestamps_ms = pd.to_numeric(pd.Series(source_timestamps_ms), errors="coerce").to_numpy(dtype=float)
    source_values = pd.to_numeric(pd.Series(source_values), errors="coerce").to_numpy(dtype=float)
    target_timestamps_ms = pd.to_numeric(pd.Series(target_timestamps_ms), errors="coerce").to_numpy(dtype=float)
    valid_mask = ~(np.isnan(source_timestamps_ms) | np.isnan(source_values))
    source_timestamps_ms = source_timestamps_ms[valid_mask]
    source_values = source_values[valid_mask]

    if len(source_timestamps_ms) == 0:
        return np.full(len(target_timestamps_ms), np.nan, dtype=float)
    if len(source_timestamps_ms) == 1:
        return np.full(len(target_timestamps_ms), float(source_values[0]), dtype=float)

    unique_indices = np.concatenate(([True], np.diff(source_timestamps_ms) != 0))
    source_timestamps_ms = source_timestamps_ms[unique_indices]
    source_values = source_values[unique_indices]
    return np.interp(
        target_timestamps_ms,
        source_timestamps_ms,
        source_values,
        left=np.nan,
        right=np.nan,
    )


def estimate_sampling_rate_hz(timestamps_ms: np.ndarray) -> float | None:
    """Estimate the sampling rate from timestamp deltas."""

    if len(timestamps_ms) < 2:
        return None
    diffs = np.diff(timestamps_ms.astype(float))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median_dt_ms = float(np.median(diffs))
    if median_dt_ms <= 0:
        return None
    return 1000.0 / median_dt_ms


def _sort_and_deduplicate(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Sort a frame and drop duplicate timestamps."""

    if frame.empty:
        return frame.copy()
    return frame.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last").reset_index(drop=True)
