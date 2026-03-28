"""Load GalaxyPPG metadata and raw sensor files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ReferenceSource = Literal["auto", "hr", "ibi"]
POLAR_PHONE_TIMESTAMP_OFFSET_MS = 9 * 60 * 60 * 1000


@dataclass(slots=True)
class ParticipantData:
    """Raw participant-level signals after basic timestamp normalization."""

    participant_id: str
    ppg: pd.DataFrame
    reference: pd.DataFrame
    events: pd.DataFrame
    reference_source: str


def default_dataset_root() -> Path:
    """Return the default GalaxyPPG dataset root inside this repository."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "GalaxyPPG"


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the dataset root and fail fast if it does not exist."""

    root = Path(dataset_root) if dataset_root is not None else default_dataset_root()
    if not root.exists():
        raise FileNotFoundError(f"GalaxyPPG dataset root does not exist: {root}")
    return root


def list_participants(dataset_root: str | Path | None = None) -> list[str]:
    """Return all participant IDs present in the dataset."""

    root = resolve_dataset_root(dataset_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("P"))


def load_metadata(dataset_root: str | Path | None = None) -> pd.DataFrame:
    """Load the participant metadata table."""

    root = resolve_dataset_root(dataset_root)
    metadata_path = root / "Meta.csv"
    metadata = pd.read_csv(metadata_path)
    return metadata.rename(columns={"UID": "participant_id"})


def load_participant_data(
    participant_id: str,
    dataset_root: str | Path | None = None,
    reference_source: ReferenceSource = "hr",
    polar_offset_ms: int = POLAR_PHONE_TIMESTAMP_OFFSET_MS,
) -> ParticipantData:
    """Load a participant's PPG, reference HR/RR, and session event log."""

    root = resolve_dataset_root(dataset_root)
    participant_dir = root / participant_id
    if not participant_dir.exists():
        raise FileNotFoundError(f"Participant folder does not exist: {participant_dir}")

    resolved_reference_source = reference_source
    if reference_source == "auto":
        resolved_reference_source = "hr" if (participant_dir / "PolarH10" / "HR.csv").exists() else "ibi"

    ppg = load_galaxy_watch_ppg(participant_dir)
    reference = load_polar_reference(
        participant_dir=participant_dir,
        reference_source=resolved_reference_source,
        polar_offset_ms=polar_offset_ms,
    )
    events = load_event_log(participant_dir)

    return ParticipantData(
        participant_id=participant_id,
        ppg=ppg,
        reference=reference,
        events=events,
        reference_source=resolved_reference_source,
    )


def load_all_participants(
    dataset_root: str | Path | None = None,
    reference_source: ReferenceSource = "hr",
    polar_offset_ms: int = POLAR_PHONE_TIMESTAMP_OFFSET_MS,
) -> list[ParticipantData]:
    """Load all participants available under the dataset root."""

    return [
        load_participant_data(
            participant_id=participant_id,
            dataset_root=dataset_root,
            reference_source=reference_source,
            polar_offset_ms=polar_offset_ms,
        )
        for participant_id in list_participants(dataset_root)
    ]


def load_galaxy_watch_ppg(participant_dir: str | Path) -> pd.DataFrame:
    """Load Galaxy Watch PPG samples on the Galaxy Watch timestamp axis."""

    participant_path = Path(participant_dir)
    ppg_path = participant_path / "GalaxyWatch" / "PPG.csv"
    if not ppg_path.exists():
        return pd.DataFrame(
            columns=["timestamp_ms", "ppg", "ppg_status", "ppg_data_received_ms", "is_valid_ppg"]
        )

    ppg = pd.read_csv(ppg_path, usecols=["dataReceived", "timestamp", "ppg", "status"])
    ppg = ppg.rename(
        columns={
            "dataReceived": "ppg_data_received_ms",
            "timestamp": "timestamp_ms",
            "status": "ppg_status",
        }
    )
    ppg = _coerce_numeric(ppg, columns=["ppg_data_received_ms", "timestamp_ms", "ppg", "ppg_status"])
    ppg = ppg.dropna(subset=["timestamp_ms", "ppg"])
    ppg["timestamp_ms"] = ppg["timestamp_ms"].astype("int64")
    ppg["ppg"] = ppg["ppg"].astype("float64")
    ppg["ppg_status"] = ppg["ppg_status"].fillna(0).astype("int64")
    ppg["is_valid_ppg"] = ppg["ppg_status"].isin({0, 500})
    return ppg.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)


def load_polar_reference(
    participant_dir: str | Path,
    reference_source: Literal["hr", "ibi"],
    polar_offset_ms: int = POLAR_PHONE_TIMESTAMP_OFFSET_MS,
) -> pd.DataFrame:
    """Load Polar H10 reference HR or RR intervals on the Galaxy Watch time axis."""

    participant_path = Path(participant_dir)
    polar_dir = participant_path / "PolarH10"
    if reference_source == "hr":
        reference_path = polar_dir / "HR.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing Polar HR file: {reference_path}")

        reference = pd.read_csv(reference_path, usecols=["phoneTimestamp", "hr", "hrv"])
        reference = reference.rename(
            columns={
                "phoneTimestamp": "raw_timestamp_ms",
                "hr": "hr_bpm",
                "hrv": "hrv_value",
            }
        )
        reference = _coerce_numeric(reference, columns=["raw_timestamp_ms", "hr_bpm", "hrv_value"])
        reference = reference.dropna(subset=["raw_timestamp_ms", "hr_bpm"])
        reference["rr_interval_ms"] = pd.NA
    elif reference_source == "ibi":
        reference_path = polar_dir / "IBI.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing Polar IBI file: {reference_path}")

        reference = pd.read_csv(reference_path, usecols=["phoneTimestamp", "duration"])
        reference = reference.rename(columns={"phoneTimestamp": "raw_timestamp_ms", "duration": "rr_interval_ms"})
        reference = _coerce_numeric(reference, columns=["raw_timestamp_ms", "rr_interval_ms"])
        reference = reference.dropna(subset=["raw_timestamp_ms", "rr_interval_ms"])
        reference = reference[reference["rr_interval_ms"] > 0].copy()
        reference["hr_bpm"] = 60000.0 / reference["rr_interval_ms"]
        reference["hrv_value"] = pd.NA
    else:
        raise ValueError(f"Unsupported reference source: {reference_source}")

    reference["timestamp_ms"] = reference["raw_timestamp_ms"].astype("int64") - polar_offset_ms
    reference["reference_source"] = reference_source
    return (
        reference.sort_values("timestamp_ms")
        .drop_duplicates("timestamp_ms", keep="last")
        .reset_index(drop=True)
    )


def load_event_log(participant_dir: str | Path) -> pd.DataFrame:
    """Load the participant event log used as session boundaries."""

    participant_path = Path(participant_dir)
    event_path = participant_path / "Event.csv"
    if not event_path.exists():
        return pd.DataFrame(columns=["timestamp_ms", "session", "status"])

    events = pd.read_csv(event_path, usecols=["timestamp", "session", "status"])
    events = events.rename(columns={"timestamp": "timestamp_ms"})
    events = _coerce_numeric(events, columns=["timestamp_ms"])
    events = events.dropna(subset=["timestamp_ms", "session", "status"])
    events["timestamp_ms"] = events["timestamp_ms"].astype("int64")
    events["session"] = events["session"].astype(str)
    events["status"] = events["status"].astype(str).str.upper()
    return events.sort_values("timestamp_ms").reset_index(drop=True)


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values."""

    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result
