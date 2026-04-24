"""Load GalaxyPPG metadata and raw sensor files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .canonical import (
    canonicalize_galaxyppg_accelerometer,
    canonicalize_galaxyppg_ppg,
    canonicalize_galaxyppg_reference,
)

ReferenceSource = Literal["auto", "hr", "ibi", "ecg"]
POLAR_PHONE_TIMESTAMP_OFFSET_MS = 9 * 60 * 60 * 1000


@dataclass(slots=True)
class ParticipantData:
    """Raw participant-level signals after basic timestamp normalization."""

    participant_id: str
    ppg: pd.DataFrame
    accelerometer: pd.DataFrame
    reference: pd.DataFrame
    ecg: pd.DataFrame
    events: pd.DataFrame
    reference_source: str
    canonical_ppg: pd.DataFrame
    canonical_accelerometer: pd.DataFrame
    canonical_reference: pd.DataFrame


def default_dataset_root() -> Path:
    """Return the default GalaxyPPG dataset root inside this repository."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "GalaxyPPG"


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the dataset root and fail fast if it does not exist."""

    root = Path(dataset_root) if dataset_root is not None else default_dataset_root()
    if not root.exists():
        raise FileNotFoundError(f"GalaxyPPG dataset root does not exist: {root}")
    metadata_path = root / "Meta.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "GalaxyPPG dataset root is missing `Meta.csv`. "
            f"Expected structure like `{root}/Meta.csv` plus participant folders `P02/`, `P03/`, ..."
        )
    return root


def list_participants(dataset_root: str | Path | None = None) -> list[str]:
    """Return all participant IDs present in the dataset."""

    root = resolve_dataset_root(dataset_root)
    participants = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("P"))
    if not participants:
        raise FileNotFoundError(
            "No participant folders were found under the GalaxyPPG dataset root. "
            f"Expected folders like `{root}/P02`, `{root}/P03`, ..."
        )
    return participants


def load_metadata(dataset_root: str | Path | None = None) -> pd.DataFrame:
    """Load the participant metadata table."""

    root = resolve_dataset_root(dataset_root)
    metadata_path = root / "Meta.csv"
    metadata = pd.read_csv(metadata_path)
    return metadata.rename(columns={"UID": "participant_id"})


def load_participant_data(
    participant_id: str,
    dataset_root: str | Path | None = None,
    reference_source: ReferenceSource = "auto",
    polar_offset_ms: int = POLAR_PHONE_TIMESTAMP_OFFSET_MS,
) -> ParticipantData:
    """Load a participant's PPG, reference HR/RR, and session event log."""

    root = resolve_dataset_root(dataset_root)
    participant_dir = root / participant_id
    if not participant_dir.exists():
        raise FileNotFoundError(f"Participant folder does not exist: {participant_dir}")

    resolved_reference_source = reference_source
    if reference_source == "auto":
        polar_dir = participant_dir / "PolarH10"
        if (polar_dir / "IBI.csv").exists():
            resolved_reference_source = "ibi"
        elif (polar_dir / "ECG.csv").exists():
            resolved_reference_source = "ecg"
        else:
            resolved_reference_source = "hr"

    ppg = load_galaxy_watch_ppg(participant_dir)
    accelerometer = load_galaxy_watch_accelerometer(participant_dir)
    reference = load_polar_reference(
        participant_dir=participant_dir,
        reference_source=resolved_reference_source,
        polar_offset_ms=polar_offset_ms,
    )
    ecg = load_polar_ecg(participant_dir=participant_dir, polar_offset_ms=polar_offset_ms)
    events = load_event_log(participant_dir)
    canonical_ppg = canonicalize_galaxyppg_ppg(participant_id=participant_id, ppg=ppg, events=events)
    canonical_accelerometer = canonicalize_galaxyppg_accelerometer(
        participant_id=participant_id,
        accelerometer=accelerometer,
        events=events,
    )
    reference_sensor = {
        "hr": "PolarH10/HR",
        "ibi": "PolarH10/IBI",
        "ecg": "PolarH10/ECG",
    }[str(resolved_reference_source)]
    canonical_reference = canonicalize_galaxyppg_reference(
        participant_id=participant_id,
        reference=reference,
        events=events,
        sensor=reference_sensor,
    )

    return ParticipantData(
        participant_id=participant_id,
        ppg=ppg,
        accelerometer=accelerometer,
        reference=reference,
        ecg=ecg,
        events=events,
        reference_source=resolved_reference_source,
        canonical_ppg=canonical_ppg,
        canonical_accelerometer=canonical_accelerometer,
        canonical_reference=canonical_reference,
    )


def load_all_participants(
    dataset_root: str | Path | None = None,
    reference_source: ReferenceSource = "auto",
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
    """Load Galaxy Watch PPG samples on the Galaxy Watch timestamp axis.

    Galaxy Watch PPG is inverted once at load time so every downstream stage
    consumes the same canonical waveform. The raw value is kept for inspection.
    """

    participant_path = Path(participant_dir)
    ppg_path = participant_path / "GalaxyWatch" / "PPG.csv"
    if not ppg_path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_ms",
                "ppg",
                "ppg_raw",
                "ppg_status",
                "ppg_data_received_ms",
                "is_valid_ppg",
                "ppg_inverted",
                "ppg_canonical_source",
            ]
        )

    ppg = pd.read_csv(ppg_path, usecols=["dataReceived", "timestamp", "ppg", "status"])
    ppg = ppg.rename(
        columns={
            "dataReceived": "ppg_data_received_ms",
            "timestamp": "timestamp_ms",
            "ppg": "ppg_raw",
            "status": "ppg_status",
        }
    )
    ppg = _coerce_numeric(ppg, columns=["ppg_data_received_ms", "timestamp_ms", "ppg_raw", "ppg_status"])
    ppg = ppg.dropna(subset=["timestamp_ms", "ppg_raw"])
    ppg["timestamp_ms"] = ppg["timestamp_ms"].astype("int64")
    ppg["ppg_raw"] = ppg["ppg_raw"].astype("float64")
    ppg["ppg"] = -ppg["ppg_raw"]
    ppg["ppg_status"] = ppg["ppg_status"].fillna(0).astype("int64")
    ppg["is_valid_ppg"] = ppg["ppg_status"].isin({0, 500})
    ppg["ppg_inverted"] = True
    ppg["ppg_canonical_source"] = "GalaxyWatch/PPG.csv:ppg_raw_inverted"
    return ppg.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)


def load_galaxy_watch_accelerometer(participant_dir: str | Path) -> pd.DataFrame:
    """Load Galaxy Watch accelerometer samples on the watch timestamp axis."""

    participant_path = Path(participant_dir)
    acc_path = participant_path / "GalaxyWatch" / "ACC.csv"
    if not acc_path.exists():
        return pd.DataFrame(columns=["timestamp_ms", "acc_x", "acc_y", "acc_z", "acc_data_received_ms"])

    accelerometer = pd.read_csv(acc_path, usecols=["dataReceived", "timestamp", "x", "y", "z"])
    accelerometer = accelerometer.rename(
        columns={
            "dataReceived": "acc_data_received_ms",
            "timestamp": "timestamp_ms",
            "x": "acc_x",
            "y": "acc_y",
            "z": "acc_z",
        }
    )
    accelerometer = _coerce_numeric(
        accelerometer,
        columns=["acc_data_received_ms", "timestamp_ms", "acc_x", "acc_y", "acc_z"],
    )
    accelerometer = accelerometer.dropna(subset=["timestamp_ms", "acc_x", "acc_y", "acc_z"])
    accelerometer["timestamp_ms"] = accelerometer["timestamp_ms"].astype("int64")
    return (
        accelerometer.sort_values("timestamp_ms")
        .drop_duplicates("timestamp_ms", keep="last")
        .reset_index(drop=True)
    )


def load_polar_reference(
    participant_dir: str | Path,
    reference_source: Literal["hr", "ibi", "ecg"],
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
    elif reference_source == "ecg":
        reference = load_polar_ecg(participant_dir=participant_dir, polar_offset_ms=polar_offset_ms)
        reference["rr_interval_ms"] = pd.NA
        reference["hr_bpm"] = pd.NA
        reference["hrv_value"] = pd.NA
        reference["reference_source"] = reference_source
        return reference
    else:
        raise ValueError(f"Unsupported reference source: {reference_source}")

    reference["timestamp_ms"] = reference["raw_timestamp_ms"].astype("int64") - polar_offset_ms
    reference["reference_source"] = reference_source
    return (
        reference.sort_values("timestamp_ms")
        .drop_duplicates("timestamp_ms", keep="last")
        .reset_index(drop=True)
    )


def load_polar_ecg(
    participant_dir: str | Path,
    polar_offset_ms: int = POLAR_PHONE_TIMESTAMP_OFFSET_MS,
) -> pd.DataFrame:
    """Load Polar H10 ECG on the Galaxy Watch time axis."""

    participant_path = Path(participant_dir)
    ecg_path = participant_path / "PolarH10" / "ECG.csv"
    if not ecg_path.exists():
        return pd.DataFrame(columns=["raw_timestamp_ms", "timestamp_ms", "ecg_uv", "reference_source"])

    ecg = pd.read_csv(ecg_path, usecols=["phoneTimestamp", "ecg"])
    ecg = ecg.rename(columns={"phoneTimestamp": "raw_timestamp_ms", "ecg": "ecg_uv"})
    ecg = _coerce_numeric(ecg, columns=["raw_timestamp_ms", "ecg_uv"])
    ecg = ecg.dropna(subset=["raw_timestamp_ms", "ecg_uv"])
    ecg["timestamp_ms"] = ecg["raw_timestamp_ms"].astype("int64") - polar_offset_ms
    ecg["reference_source"] = "ecg"
    return ecg.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)


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
