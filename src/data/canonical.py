"""Canonical internal schema shared by wearable PPG datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

CANONICAL_SCHEMA_VERSION = "canonical_ppg_v1"

CANONICAL_PPG_COLUMNS = [
    "participant_id",
    "timestamp_ms",
    "ppg",
    "ppg_raw",
    "ppg_inverted",
    "ppg_canonical_source",
    "session_id",
    "session_name",
    "activity_label",
    "dataset",
    "sensor",
]

CANONICAL_ACCELEROMETER_COLUMNS = [
    "participant_id",
    "timestamp_ms",
    "acc_x",
    "acc_y",
    "acc_z",
    "session_id",
    "session_name",
    "activity_label",
    "dataset",
    "sensor",
]

CANONICAL_REFERENCE_COLUMNS = [
    "participant_id",
    "timestamp_ms",
    "ecg_uv",
    "rr_interval_ms",
    "hr_bpm",
    "reference_source",
    "session_id",
    "session_name",
    "activity_label",
    "dataset",
    "sensor",
]


@dataclass(slots=True)
class CanonicalSchemaDescription:
    """Machine-readable description of the internal dataset contract."""

    version: str
    ppg_columns: list[str]
    accelerometer_columns: list[str]
    reference_columns: list[str]
    timestamp_unit: str = "unix_ms"
    ppg_policy: str = "dataset-specific corrections happen at load time; canonical ppg is model input"
    reference_policy: str = "ECG or IBI references are converted to beat-interval instantaneous HR for labels"

    def to_dict(self) -> dict[str, object]:
        """Convert the schema description into a JSON-serializable dictionary."""

        return asdict(self)


def canonical_schema_description() -> CanonicalSchemaDescription:
    """Return the canonical schema used by processed manifests."""

    return CanonicalSchemaDescription(
        version=CANONICAL_SCHEMA_VERSION,
        ppg_columns=CANONICAL_PPG_COLUMNS,
        accelerometer_columns=CANONICAL_ACCELEROMETER_COLUMNS,
        reference_columns=CANONICAL_REFERENCE_COLUMNS,
    )


def add_session_labels(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach session labels to timestamped samples using ENTER/EXIT intervals."""

    result = frame.copy()
    result["session_id"] = pd.NA
    result["session_name"] = pd.NA
    result["activity_label"] = pd.NA
    if result.empty or events.empty:
        return result

    intervals = _event_intervals(events)
    for index, (session_name, start_ms, end_ms) in enumerate(intervals, start=1):
        mask = (result["timestamp_ms"] >= start_ms) & (result["timestamp_ms"] < end_ms)
        session_id = f"{session_name}#{index}"
        result.loc[mask, "session_id"] = session_id
        result.loc[mask, "session_name"] = session_name
        result.loc[mask, "activity_label"] = session_name
    return result


def canonicalize_galaxyppg_ppg(
    participant_id: str,
    ppg: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return GalaxyPPG watch PPG in the canonical internal schema."""

    result = ppg.copy()
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = "GalaxyPPG"
    result["sensor"] = "GalaxyWatch/PPG"
    result = add_session_labels(result, events)
    return _select_columns(result, CANONICAL_PPG_COLUMNS)


def canonicalize_galaxyppg_accelerometer(
    participant_id: str,
    accelerometer: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return GalaxyPPG watch accelerometer data in the canonical internal schema."""

    result = accelerometer.copy()
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = "GalaxyPPG"
    result["sensor"] = "GalaxyWatch/ACC"
    result = add_session_labels(result, events)
    return _select_columns(result, CANONICAL_ACCELEROMETER_COLUMNS)


def canonicalize_galaxyppg_reference(
    participant_id: str,
    reference: pd.DataFrame,
    events: pd.DataFrame,
    sensor: str,
) -> pd.DataFrame:
    """Return GalaxyPPG Polar ECG/IBI/HR data in the canonical reference schema."""

    result = reference.copy()
    result.insert(0, "participant_id", participant_id)
    if "ecg_uv" not in result.columns:
        result["ecg_uv"] = pd.NA
    if "rr_interval_ms" not in result.columns:
        result["rr_interval_ms"] = pd.NA
    if "hr_bpm" not in result.columns:
        result["hr_bpm"] = pd.NA
    result["dataset"] = "GalaxyPPG"
    result["sensor"] = sensor
    result = add_session_labels(result, events)
    return _select_columns(result, CANONICAL_REFERENCE_COLUMNS)


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with all canonical columns in a stable order."""

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _event_intervals(events: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Build ENTER/EXIT intervals from a canonical GalaxyPPG Event.csv table."""

    starts: dict[str, int] = {}
    intervals: list[tuple[str, int, int]] = []
    for row in events.sort_values("timestamp_ms").itertuples(index=False):
        session_name = str(row.session)
        status = str(row.status).upper()
        timestamp_ms = int(row.timestamp_ms)
        if status == "ENTER":
            starts[session_name] = timestamp_ms
        elif status == "EXIT" and session_name in starts:
            start_ms = starts.pop(session_name)
            if timestamp_ms > start_ms:
                intervals.append((session_name, start_ms, timestamp_ms))
    return intervals
