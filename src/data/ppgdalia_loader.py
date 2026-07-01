"""Load PPG-DaLiA participants into the canonical PPG schema."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .canonical import CANONICAL_ACCELEROMETER_COLUMNS, CANONICAL_PPG_COLUMNS, CANONICAL_REFERENCE_COLUMNS
from .loader import ParticipantData


DATASET_NAME = "PPG-DaLiA"
WRIST_BVP_HZ = 64.0
WRIST_ACC_HZ = 32.0
CHEST_ECG_HZ = 700.0
ACTIVITY_HZ = 4.0

ACTIVITY_LABELS = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}


@dataclass(slots=True)
class PPGDaLiAParticipantPaths:
    """Resolved on-disk files for one PPG-DaLiA participant."""

    participant_id: str
    pickle_path: Path


def default_ppgdalia_root() -> Path:
    """Return the expected PPG-DaLiA raw-data root inside this repository."""

    return Path(__file__).resolve().parents[2] / "data" / "raw" / "PPG-DaLiA"


def resolve_ppgdalia_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve the PPG-DaLiA root and fail with the expected placement."""

    root = Path(dataset_root) if dataset_root is not None else default_ppgdalia_root()
    if not root.exists():
        raise FileNotFoundError(
            f"PPG-DaLiA dataset root does not exist: {root}. "
            "Expected files like `data/raw/PPG-DaLiA/S1.pkl` or `data/raw/PPG-DaLiA/S1/S1.pkl`."
        )
    return root


def list_ppgdalia_participants(dataset_root: str | Path | None = None) -> list[str]:
    """Return PPG-DaLiA participant IDs discovered from S*.pkl files."""

    root = resolve_ppgdalia_root(dataset_root)
    participants = [item.participant_id for item in discover_ppgdalia_participant_paths(root)]
    if not participants:
        raise FileNotFoundError(
            f"No PPG-DaLiA participant pickle files were found under {root}. "
            "Expected files like `S1.pkl`, `S2.pkl`, ... or nested `S1/S1.pkl`."
        )
    return participants


def discover_ppgdalia_participant_paths(dataset_root: str | Path | None = None) -> list[PPGDaLiAParticipantPaths]:
    """Discover participant pickle files in common PPG-DaLiA layouts."""

    root = resolve_ppgdalia_root(dataset_root)
    candidates = list(root.glob("S*.pkl")) + list(root.glob("S*/S*.pkl"))
    rows: list[PPGDaLiAParticipantPaths] = []
    for path in sorted(candidates, key=lambda item: participant_sort_key(item.stem)):
        if not path.stem.startswith("S"):
            continue
        rows.append(PPGDaLiAParticipantPaths(participant_id=path.stem, pickle_path=path))
    deduplicated: dict[str, PPGDaLiAParticipantPaths] = {}
    for row in rows:
        deduplicated.setdefault(row.participant_id, row)
    return [deduplicated[key] for key in sorted(deduplicated, key=participant_sort_key)]


def load_ppgdalia_participant_data(
    participant_id: str,
    dataset_root: str | Path | None = None,
    reference_source: str = "ecg",
) -> ParticipantData:
    """Load one PPG-DaLiA participant and map it to canonical frames."""

    if reference_source != "ecg":
        raise ValueError("PPG-DaLiA currently supports ECG/rpeak-derived labels only.")

    paths = {item.participant_id: item for item in discover_ppgdalia_participant_paths(dataset_root)}
    if participant_id not in paths:
        raise FileNotFoundError(f"PPG-DaLiA participant `{participant_id}` was not found.")

    payload = read_ppgdalia_pickle(paths[participant_id].pickle_path)
    wrist = payload.get("signal", {}).get("wrist", {})
    chest = payload.get("signal", {}).get("chest", {})

    ppg = build_wrist_bvp_frame(wrist.get("BVP"))
    accelerometer = build_wrist_acc_frame(wrist.get("ACC"))
    ecg = build_chest_ecg_frame(chest.get("ECG"))
    reference = build_reference_frame_from_rpeaks(payload.get("rpeaks"), ecg)
    events = build_activity_events(payload.get("activity"), ppg)

    canonical_ppg = canonicalize_ppgdalia_ppg(participant_id, ppg, events)
    canonical_accelerometer = canonicalize_ppgdalia_accelerometer(participant_id, accelerometer, events)
    canonical_reference = canonicalize_ppgdalia_reference(participant_id, reference, events)

    return ParticipantData(
        participant_id=participant_id,
        ppg=canonical_ppg,
        accelerometer=canonical_accelerometer,
        reference=canonical_reference,
        ecg=ecg,
        events=events,
        reference_source="ecg",
        canonical_ppg=canonical_ppg,
        canonical_accelerometer=canonical_accelerometer,
        canonical_reference=canonical_reference,
    )


def read_ppgdalia_pickle(path: Path) -> dict[str, Any]:
    """Read a PPG-DaLiA pickle file."""

    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}.")
    return payload


def build_wrist_bvp_frame(values: Any) -> pd.DataFrame:
    """Build a timestamped wrist BVP frame."""

    signal = as_1d_float_array(values)
    timestamps_ms = sample_timestamps_ms(len(signal), WRIST_BVP_HZ)
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps_ms,
            "ppg": signal,
            "ppg_raw": signal,
            "is_valid_ppg": np.isfinite(signal),
            "ppg_inverted": False,
            "ppg_canonical_source": "wrist/BVP",
        }
    ).dropna(subset=["ppg"]).reset_index(drop=True)


def build_wrist_acc_frame(values: Any) -> pd.DataFrame:
    """Build a timestamped wrist accelerometer frame."""

    acc = as_2d_float_array(values, expected_columns=3)
    timestamps_ms = sample_timestamps_ms(len(acc), WRIST_ACC_HZ)
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps_ms,
            "acc_x": acc[:, 0],
            "acc_y": acc[:, 1],
            "acc_z": acc[:, 2],
        }
    ).dropna(subset=["acc_x", "acc_y", "acc_z"]).reset_index(drop=True)


def build_chest_ecg_frame(values: Any) -> pd.DataFrame:
    """Build a timestamped chest ECG frame."""

    ecg = as_1d_float_array(values)
    timestamps_ms = sample_timestamps_ms(len(ecg), CHEST_ECG_HZ)
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps_ms,
            "ecg_uv": ecg,
            "reference_source": "ecg",
        }
    ).dropna(subset=["ecg_uv"]).reset_index(drop=True)


def build_reference_frame_from_rpeaks(rpeaks: Any, ecg: pd.DataFrame) -> pd.DataFrame:
    """Build ECG reference rows, preferring supplied R-peak indices when present."""

    reference = ecg.copy()
    peaks = pd.to_numeric(pd.Series(np.asarray(rpeaks).reshape(-1)), errors="coerce").dropna().to_numpy(dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(ecg))]
    if len(peaks) >= 2:
        peak_timestamps = ecg["timestamp_ms"].iloc[peaks].to_numpy(dtype=float)
        rr_ms = np.diff(peak_timestamps)
        valid = (rr_ms >= 60000.0 / 220.0) & (rr_ms <= 60000.0 / 35.0)
        beat_timestamps = peak_timestamps[1:][valid]
        rr_ms = rr_ms[valid]
        ibi = pd.DataFrame(
            {
                "timestamp_ms": beat_timestamps.astype("int64"),
                "ecg_uv": pd.NA,
                "rr_interval_ms": rr_ms,
                "hr_bpm": 60000.0 / rr_ms,
                "reference_source": "ecg",
            }
        )
        return ibi.reset_index(drop=True)

    reference["rr_interval_ms"] = pd.NA
    reference["hr_bpm"] = pd.NA
    return reference


def build_activity_events(activity: Any, ppg: pd.DataFrame) -> pd.DataFrame:
    """Convert PPG-DaLiA activity IDs into ENTER/EXIT session events when available."""

    if ppg.empty:
        return pd.DataFrame(columns=["timestamp_ms", "session", "status"])
    activities = pd.to_numeric(pd.Series(np.asarray(activity).reshape(-1)), errors="coerce").dropna()
    if activities.empty:
        start_ms = int(ppg["timestamp_ms"].min())
        end_ms = int(ppg["timestamp_ms"].max()) + int(round(1000.0 / WRIST_BVP_HZ))
        return pd.DataFrame(
            [
                {"timestamp_ms": start_ms, "session": "full_recording", "status": "ENTER"},
                {"timestamp_ms": end_ms, "session": "full_recording", "status": "EXIT"},
            ]
        )

    rows: list[dict[str, Any]] = []
    values = activities.astype(int).to_numpy()
    start_index = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and values[index] == values[start_index]:
            continue
        label = ACTIVITY_LABELS.get(int(values[start_index]), f"activity_{int(values[start_index])}")
        start_ms = int(round(start_index * 1000.0 / ACTIVITY_HZ))
        end_ms = int(round(index * 1000.0 / ACTIVITY_HZ))
        if end_ms > start_ms:
            rows.append({"timestamp_ms": start_ms, "session": label, "status": "ENTER"})
            rows.append({"timestamp_ms": end_ms, "session": label, "status": "EXIT"})
        start_index = index
    return pd.DataFrame(rows, columns=["timestamp_ms", "session", "status"])


def canonicalize_ppgdalia_ppg(participant_id: str, ppg: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return PPG-DaLiA wrist BVP in the canonical PPG schema."""

    result = attach_session_labels(ppg, events)
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "EmpaticaE4/BVP"
    return select_columns(result, CANONICAL_PPG_COLUMNS)


def canonicalize_ppgdalia_accelerometer(
    participant_id: str,
    accelerometer: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return PPG-DaLiA wrist accelerometer data in canonical schema."""

    result = attach_session_labels(accelerometer, events)
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "EmpaticaE4/ACC"
    return select_columns(result, CANONICAL_ACCELEROMETER_COLUMNS)


def canonicalize_ppgdalia_reference(
    participant_id: str,
    reference: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return PPG-DaLiA ECG/rpeak references in canonical schema."""

    result = attach_session_labels(reference, events)
    result.insert(0, "participant_id", participant_id)
    result["dataset"] = DATASET_NAME
    result["sensor"] = "RespiBAN/ECG"
    return select_columns(result, CANONICAL_REFERENCE_COLUMNS)


def attach_session_labels(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach activity labels using ENTER/EXIT event rows."""

    result = frame.copy()
    result["session_id"] = pd.NA
    result["session_name"] = pd.NA
    result["activity_label"] = pd.NA
    if result.empty or events.empty:
        return result
    starts: dict[str, int] = {}
    counts: dict[str, int] = {}
    for row in events.sort_values("timestamp_ms").itertuples(index=False):
        status = str(row.status).upper()
        session = str(row.session)
        timestamp_ms = int(row.timestamp_ms)
        if status == "ENTER":
            starts[session] = timestamp_ms
        elif status == "EXIT" and session in starts:
            start_ms = starts.pop(session)
            if timestamp_ms <= start_ms:
                continue
            counts[session] = counts.get(session, 0) + 1
            session_id = f"{session}#{counts[session]}"
            mask = (result["timestamp_ms"] >= start_ms) & (result["timestamp_ms"] < timestamp_ms)
            result.loc[mask, "session_id"] = session_id
            result.loc[mask, "session_name"] = session
            result.loc[mask, "activity_label"] = session
    return result


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with all requested columns in stable order."""

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def as_1d_float_array(values: Any) -> np.ndarray:
    """Coerce a payload value into a finite-aware 1D float array."""

    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def as_2d_float_array(values: Any, expected_columns: int) -> np.ndarray:
    """Coerce a payload value into a 2D float matrix."""

    if values is None:
        return np.empty((0, expected_columns), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(f"Expected a matrix with {expected_columns} columns, got shape {array.shape}.")
    return array


def sample_timestamps_ms(length: int, sampling_hz: float) -> np.ndarray:
    """Build relative millisecond timestamps for uniformly sampled signals."""

    return np.round(np.arange(length, dtype=float) * 1000.0 / sampling_hz).astype("int64")


def participant_sort_key(value: str) -> tuple[int, str]:
    """Sort S1, S2, ... numerically when possible."""

    suffix = value[1:] if value.startswith("S") else value
    return (int(suffix), value) if suffix.isdigit() else (10_000, value)
