"""Configurable WildPPG wrist loader for canonical PPG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .canonical import CANONICAL_ACCELEROMETER_COLUMNS, CANONICAL_PPG_COLUMNS, CANONICAL_REFERENCE_COLUMNS
from .loader import ParticipantData


DATASET_NAME = "WildPPG-wrist"


@dataclass(slots=True)
class WildPPGRecord:
    """One manifest row describing a participant/session recording."""

    participant_id: str
    session_id: str
    activity: str
    ppg_path: Path
    ppg_col: str
    ppg_time_col: str | None
    ppg_sampling_hz: float | None
    acc_path: Path | None
    acc_time_col: str | None
    acc_x_col: str | None
    acc_y_col: str | None
    acc_z_col: str | None
    acc_sampling_hz: float | None
    reference_path: Path
    reference_time_col: str | None
    ecg_col: str | None
    rr_col: str | None
    reference_sampling_hz: float | None
    time_unit: str


def default_wildppg_root() -> Path:
    """Return the expected WildPPG wrist raw-data root inside this repository."""

    return Path(__file__).resolve().parents[2] / "data" / "raw" / "WildPPG-wrist"


def resolve_wildppg_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the WildPPG wrist root and fail with the expected manifest path."""

    root = Path(dataset_root) if dataset_root is not None else default_wildppg_root()
    if not root.exists():
        raise FileNotFoundError(
            f"WildPPG wrist dataset root does not exist: {root}. "
            "Expected a `wildppg_wrist_manifest.csv` file under that root."
        )
    manifest_path = root / "wildppg_wrist_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing WildPPG wrist manifest: {manifest_path}. "
            "See `data/raw/WildPPG-wrist/README.md` for the required columns."
        )
    return root


def load_wildppg_manifest(dataset_root: str | Path | None = None) -> list[WildPPGRecord]:
    """Load the file/column mapping manifest for WildPPG wrist."""

    root = resolve_wildppg_root(dataset_root)
    manifest = pd.read_csv(root / "wildppg_wrist_manifest.csv").fillna("")
    required = {"participant_id", "ppg_path", "ppg_col", "reference_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"WildPPG manifest is missing required columns: {missing}")
    records: list[WildPPGRecord] = []
    for row in manifest.to_dict(orient="records"):
        participant_id = str(row["participant_id"])
        session_id = str(row.get("session_id") or "recording")
        records.append(
            WildPPGRecord(
                participant_id=participant_id,
                session_id=session_id,
                activity=str(row.get("activity") or "full_recording"),
                ppg_path=resolve_relative(root, row["ppg_path"]),
                ppg_col=str(row["ppg_col"]),
                ppg_time_col=optional_str(row.get("ppg_time_col")),
                ppg_sampling_hz=optional_float(row.get("ppg_sampling_hz")),
                acc_path=resolve_relative(root, row["acc_path"]) if optional_str(row.get("acc_path")) else None,
                acc_time_col=optional_str(row.get("acc_time_col")),
                acc_x_col=optional_str(row.get("acc_x_col")),
                acc_y_col=optional_str(row.get("acc_y_col")),
                acc_z_col=optional_str(row.get("acc_z_col")),
                acc_sampling_hz=optional_float(row.get("acc_sampling_hz")),
                reference_path=resolve_relative(root, row["reference_path"]),
                reference_time_col=optional_str(row.get("reference_time_col")),
                ecg_col=optional_str(row.get("ecg_col")),
                rr_col=optional_str(row.get("rr_col")),
                reference_sampling_hz=optional_float(row.get("reference_sampling_hz")),
                time_unit=str(row.get("time_unit") or "ms").lower(),
            )
        )
    return records


def list_wildppg_participants(dataset_root: str | Path | None = None) -> list[str]:
    """Return participant IDs present in the WildPPG manifest."""

    return sorted({record.participant_id for record in load_wildppg_manifest(dataset_root)})


def load_wildppg_participant_data(
    participant_id: str,
    dataset_root: str | Path | None = None,
    reference_source: str = "ecg",
) -> ParticipantData:
    """Load one WildPPG participant from all manifest rows."""

    if reference_source != "ecg":
        raise ValueError("WildPPG wrist currently supports ECG or RR-reference labels only.")
    records = [record for record in load_wildppg_manifest(dataset_root) if record.participant_id == participant_id]
    if not records:
        raise FileNotFoundError(f"WildPPG participant `{participant_id}` was not found in the manifest.")

    ppg_frames: list[pd.DataFrame] = []
    acc_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    offset_ms = 0
    for record_index, record in enumerate(records, start=1):
        ppg = load_ppg_frame(record, offset_ms)
        acc = load_acc_frame(record, offset_ms)
        reference = load_reference_frame(record, offset_ms)
        start_ms = int(ppg["timestamp_ms"].min()) if not ppg.empty else offset_ms
        end_ms = int(ppg["timestamp_ms"].max()) + 1 if not ppg.empty else start_ms + 1
        session_name = record.activity or record.session_id
        session_id = f"{record.session_id}#{record_index}"
        for frame in [ppg, acc, reference]:
            if not frame.empty:
                frame["session_id"] = session_id
                frame["session_name"] = session_name
                frame["activity_label"] = session_name
        event_rows.extend(
            [
                {"timestamp_ms": start_ms, "session": session_name, "status": "ENTER"},
                {"timestamp_ms": end_ms, "session": session_name, "status": "EXIT"},
            ]
        )
        ppg_frames.append(ppg)
        acc_frames.append(acc)
        reference_frames.append(reference)
        offset_ms = max(end_ms + 1000, offset_ms)

    ppg_all = concat_frames(ppg_frames)
    acc_all = concat_frames(acc_frames)
    reference_all = concat_frames(reference_frames)
    events = pd.DataFrame(event_rows, columns=["timestamp_ms", "session", "status"])

    canonical_ppg = canonicalize_wildppg_ppg(participant_id, ppg_all)
    canonical_acc = canonicalize_wildppg_accelerometer(participant_id, acc_all)
    canonical_ref = canonicalize_wildppg_reference(participant_id, reference_all)
    return ParticipantData(
        participant_id=participant_id,
        ppg=canonical_ppg,
        accelerometer=canonical_acc,
        reference=canonical_ref,
        ecg=canonical_ref[["timestamp_ms", "ecg_uv", "reference_source"]].copy(),
        events=events,
        reference_source="ecg",
        canonical_ppg=canonical_ppg,
        canonical_accelerometer=canonical_acc,
        canonical_reference=canonical_ref,
    )


def load_ppg_frame(record: WildPPGRecord, offset_ms: int) -> pd.DataFrame:
    """Load one PPG CSV according to the manifest row."""

    frame = pd.read_csv(record.ppg_path)
    timestamps = resolve_timestamps(frame, record.ppg_time_col, record.ppg_sampling_hz, record.time_unit, offset_ms)
    ppg = pd.to_numeric(frame[record.ppg_col], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "ppg": ppg,
            "ppg_raw": ppg,
            "is_valid_ppg": ppg.notna(),
            "ppg_inverted": False,
            "ppg_canonical_source": f"{record.ppg_path.name}:{record.ppg_col}",
        }
    ).dropna(subset=["ppg"]).reset_index(drop=True)


def load_acc_frame(record: WildPPGRecord, offset_ms: int) -> pd.DataFrame:
    """Load one accelerometer CSV when provided."""

    if record.acc_path is None:
        return pd.DataFrame(columns=["timestamp_ms", "acc_x", "acc_y", "acc_z"])
    required = [record.acc_x_col, record.acc_y_col, record.acc_z_col]
    if any(column is None for column in required):
        return pd.DataFrame(columns=["timestamp_ms", "acc_x", "acc_y", "acc_z"])
    frame = pd.read_csv(record.acc_path)
    timestamps = resolve_timestamps(frame, record.acc_time_col, record.acc_sampling_hz, record.time_unit, offset_ms)
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "acc_x": pd.to_numeric(frame[str(record.acc_x_col)], errors="coerce"),
            "acc_y": pd.to_numeric(frame[str(record.acc_y_col)], errors="coerce"),
            "acc_z": pd.to_numeric(frame[str(record.acc_z_col)], errors="coerce"),
        }
    ).dropna(subset=["acc_x", "acc_y", "acc_z"]).reset_index(drop=True)


def load_reference_frame(record: WildPPGRecord, offset_ms: int) -> pd.DataFrame:
    """Load ECG samples or RR intervals according to the manifest row."""

    frame = pd.read_csv(record.reference_path)
    if record.reference_time_col is None and record.reference_sampling_hz is None:
        raise ValueError("WildPPG reference rows need `reference_time_col` or `reference_sampling_hz`.")
    timestamps = resolve_timestamps(
        frame,
        record.reference_time_col,
        record.reference_sampling_hz,
        record.time_unit,
        offset_ms,
    )
    result = pd.DataFrame({"timestamp_ms": timestamps, "reference_source": "ecg"})
    if record.ecg_col:
        result["ecg_uv"] = pd.to_numeric(frame[record.ecg_col], errors="coerce")
    else:
        result["ecg_uv"] = pd.NA
    if record.rr_col:
        result["rr_interval_ms"] = pd.to_numeric(frame[record.rr_col], errors="coerce")
        result["hr_bpm"] = 60000.0 / result["rr_interval_ms"]
    else:
        result["rr_interval_ms"] = pd.NA
        result["hr_bpm"] = pd.NA
    keep_columns = ["timestamp_ms", "ecg_uv", "rr_interval_ms", "hr_bpm", "reference_source"]
    subset = ["rr_interval_ms"] if record.rr_col and not record.ecg_col else ["ecg_uv"]
    return result.loc[:, keep_columns].dropna(subset=subset).reset_index(drop=True)


def resolve_timestamps(
    frame: pd.DataFrame,
    time_col: str | None,
    sampling_hz: float | None,
    time_unit: str,
    offset_ms: int,
) -> np.ndarray:
    """Resolve timestamps from a column or generate them from a sampling rate."""

    if time_col:
        values = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
        if time_unit in {"s", "sec", "second", "seconds"}:
            values = values * 1000.0
        elif time_unit in {"us", "microsecond", "microseconds"}:
            values = values / 1000.0
        elif time_unit in {"ns", "nanosecond", "nanoseconds"}:
            values = values / 1_000_000.0
        elif time_unit not in {"ms", "millisecond", "milliseconds"}:
            raise ValueError(f"Unsupported time unit: {time_unit}")
        values = values - np.nanmin(values)
        return np.round(values + offset_ms).astype("int64")
    if sampling_hz is None or sampling_hz <= 0:
        raise ValueError("A sampling rate is required when no timestamp column is provided.")
    return np.round(np.arange(len(frame), dtype=float) * 1000.0 / sampling_hz + offset_ms).astype("int64")


def canonicalize_wildppg_ppg(participant_id: str, ppg: pd.DataFrame) -> pd.DataFrame:
    """Return WildPPG wrist PPG in canonical schema."""

    result = ppg.copy()
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "WildPPG/wrist/PPG"
    return select_columns(result, CANONICAL_PPG_COLUMNS)


def canonicalize_wildppg_accelerometer(participant_id: str, acc: pd.DataFrame) -> pd.DataFrame:
    """Return WildPPG wrist accelerometer data in canonical schema."""

    result = acc.copy()
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "WildPPG/wrist/ACC"
    return select_columns(result, CANONICAL_ACCELEROMETER_COLUMNS)


def canonicalize_wildppg_reference(participant_id: str, reference: pd.DataFrame) -> pd.DataFrame:
    """Return WildPPG ECG/RR references in canonical schema."""

    result = reference.copy()
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "WildPPG/ECG"
    return select_columns(result, CANONICAL_REFERENCE_COLUMNS)


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with all requested columns in stable order."""

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames without failing on all-empty inputs."""

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)


def optional_str(value: Any) -> str | None:
    """Return a non-empty string or None."""

    text = "" if value is None else str(value).strip()
    return text or None


def optional_float(value: Any) -> float | None:
    """Return a finite float or None."""

    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def resolve_relative(root: Path, value: Any) -> Path:
    """Resolve manifest paths relative to the dataset root."""

    path = Path(str(value))
    return path if path.is_absolute() else root / path
