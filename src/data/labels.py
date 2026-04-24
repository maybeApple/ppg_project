"""Reusable target-generation rules for windowed heart-rate datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

LabelAggregation = Literal["mean", "median"]
LabelMethod = Literal["beat_interval_instant_hr", "provided_hr_samples"]


@dataclass(slots=True)
class LabelGenerationConfig:
    """Explicit label-generation settings shared across datasets."""

    method: LabelMethod = "beat_interval_instant_hr"
    aggregation: LabelAggregation = "median"
    min_valid_beats: int = 2
    min_reference_samples: int = 1

    def to_dict(self) -> dict[str, object]:
        """Convert the config into a JSON-serializable, auditable dictionary."""

        payload = asdict(self)
        if self.method == "beat_interval_instant_hr":
            payload.update(
                {
                    "instant_hr_formula": "instant_hr_bpm = 60000 / rr_interval_ms",
                    "ibi_reference": "use provided positive rr_interval_ms values from IBI files",
                    "ecg_reference": (
                        "detect ECG R peaks inside each window, compute adjacent R-R intervals, "
                        "and assign each interval timestamp to the second beat"
                    ),
                    "window_target": f"{self.aggregation} instantaneous HR inside each window",
                    "discard_rule": f"drop windows with fewer than {self.min_valid_beats} valid beat intervals",
                }
            )
        else:
            payload.update(
                {
                    "window_target": f"{self.aggregation} provided HR samples inside each window",
                    "discard_rule": (
                        f"drop windows with fewer than {self.min_reference_samples} provided HR samples"
                    ),
                }
            )
        return payload


@dataclass(slots=True)
class WindowLabel:
    """The scalar target and audit metadata for one window."""

    label_hr_bpm: float
    reference_sample_count: int
    valid_beat_count: int
    label_method: str
    label_aggregation: str
    reference_timestamps_ms: list[int]
    reference_hr_bpm_values: list[float]
    reference_rr_interval_ms_values: list[float]


def compute_window_label(
    reference_window: pd.DataFrame,
    config: LabelGenerationConfig,
) -> WindowLabel | None:
    """Compute one window target or return None when the window is invalid."""

    if config.aggregation not in {"mean", "median"}:
        raise ValueError(f"Unsupported label aggregation: {config.aggregation}")
    if config.min_valid_beats < 1:
        raise ValueError("min_valid_beats must be at least 1")
    if config.min_reference_samples < 1:
        raise ValueError("min_reference_samples must be at least 1")

    if config.method == "beat_interval_instant_hr":
        target_frame = build_instant_hr_from_beat_intervals(reference_window)
        if len(target_frame) < config.min_valid_beats:
            return None
        label_method = "beat_interval_instant_hr"
    elif config.method == "provided_hr_samples":
        target_frame = build_instant_hr_from_provided_hr(reference_window)
        if len(target_frame) < config.min_reference_samples:
            return None
        label_method = "provided_hr_samples"
    else:
        raise ValueError(f"Unsupported label method: {config.method}")

    values = target_frame["instant_hr_bpm"].to_numpy(dtype=float, copy=True)
    if config.aggregation == "median":
        label_hr_bpm = float(np.median(values))
    else:
        label_hr_bpm = float(np.mean(values))

    return WindowLabel(
        label_hr_bpm=label_hr_bpm,
        reference_sample_count=int(len(reference_window)),
        valid_beat_count=int(len(target_frame)) if config.method == "beat_interval_instant_hr" else 0,
        label_method=label_method,
        label_aggregation=config.aggregation,
        reference_timestamps_ms=target_frame["timestamp_ms"].astype("int64").tolist(),
        reference_hr_bpm_values=target_frame["instant_hr_bpm"].astype(float).tolist(),
        reference_rr_interval_ms_values=(
            target_frame["rr_interval_ms"].astype(float).tolist()
            if "rr_interval_ms" in target_frame.columns
            else []
        ),
    )


def build_instant_hr_from_beat_intervals(reference: pd.DataFrame) -> pd.DataFrame:
    """Derive instantaneous HR from valid beat intervals or ECG samples."""

    if "rr_interval_ms" not in reference.columns:
        return build_instant_hr_from_ecg(reference)

    result = reference.loc[:, ["timestamp_ms", "rr_interval_ms"]].copy()
    result["timestamp_ms"] = pd.to_numeric(result["timestamp_ms"], errors="coerce")
    result["rr_interval_ms"] = pd.to_numeric(result["rr_interval_ms"], errors="coerce")
    result = result.dropna(subset=["timestamp_ms", "rr_interval_ms"])
    result = result[result["rr_interval_ms"] > 0].copy()
    if result.empty:
        return build_instant_hr_from_ecg(reference)

    result["instant_hr_bpm"] = 60000.0 / result["rr_interval_ms"]
    result = result[np.isfinite(result["instant_hr_bpm"])].copy()
    return result.sort_values("timestamp_ms").reset_index(drop=True)


def build_instant_hr_from_ecg(reference: pd.DataFrame) -> pd.DataFrame:
    """Detect ECG beats and convert adjacent R-R intervals to instantaneous HR."""

    if "ecg_uv" not in reference.columns:
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])

    ecg = reference.loc[:, ["timestamp_ms", "ecg_uv"]].copy()
    ecg["timestamp_ms"] = pd.to_numeric(ecg["timestamp_ms"], errors="coerce")
    ecg["ecg_uv"] = pd.to_numeric(ecg["ecg_uv"], errors="coerce")
    ecg = ecg.dropna(subset=["timestamp_ms", "ecg_uv"]).sort_values("timestamp_ms")
    if len(ecg) < 3:
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])

    timestamps_ms = ecg["timestamp_ms"].to_numpy(dtype=float, copy=True)
    signal = ecg["ecg_uv"].to_numpy(dtype=float, copy=True)
    sampling_hz = _estimate_sampling_rate_hz(timestamps_ms)
    if sampling_hz is None or sampling_hz <= 0:
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])

    peak_indices = _detect_ecg_peak_indices(signal=signal, sampling_hz=sampling_hz)
    if len(peak_indices) < 3:
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])

    peak_timestamps_ms = timestamps_ms[peak_indices]
    rr_interval_ms = np.diff(peak_timestamps_ms)
    valid_mask = (rr_interval_ms >= 60000.0 / 220.0) & (rr_interval_ms <= 60000.0 / 35.0)
    if not valid_mask.any():
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])

    rr_interval_ms = rr_interval_ms[valid_mask]
    interval_timestamps_ms = peak_timestamps_ms[1:][valid_mask]
    instant_hr_bpm = 60000.0 / rr_interval_ms
    result = pd.DataFrame(
        {
            "timestamp_ms": interval_timestamps_ms.astype("int64"),
            "instant_hr_bpm": instant_hr_bpm,
            "rr_interval_ms": rr_interval_ms,
        }
    )
    return result[np.isfinite(result["instant_hr_bpm"])].reset_index(drop=True)


def _detect_ecg_peak_indices(signal: np.ndarray, sampling_hz: float) -> np.ndarray:
    """Return likely R-peak sample indices for one ECG window."""

    filtered_signal = _filter_ecg(signal=signal, sampling_hz=sampling_hz)
    positive_peaks = _find_ecg_peaks(filtered_signal, sampling_hz)
    negative_peaks = _find_ecg_peaks(-filtered_signal, sampling_hz)
    if len(negative_peaks) > len(positive_peaks):
        return negative_peaks
    return positive_peaks


def _filter_ecg(signal: np.ndarray, sampling_hz: float) -> np.ndarray:
    """Band-pass ECG before simple R-peak detection."""

    centered = signal - np.nanmedian(signal)
    nyquist_hz = sampling_hz / 2.0
    low_hz = min(5.0, nyquist_hz * 0.25)
    high_hz = min(30.0, nyquist_hz * 0.9)
    if low_hz >= high_hz:
        return centered
    try:
        sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=sampling_hz, output="sos")
        return sosfiltfilt(sos, centered)
    except ValueError:
        return centered


def _find_ecg_peaks(signal: np.ndarray, sampling_hz: float) -> np.ndarray:
    """Find peaks with ECG-compatible spacing and adaptive prominence."""

    signal_std = float(np.nanstd(signal))
    signal_range = float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))
    if not np.isfinite(signal_std) or signal_std <= 0:
        return np.asarray([], dtype=int)
    prominence = max(signal_std * 1.0, signal_range * 0.12, 1e-6)
    min_distance_samples = max(1, int(sampling_hz * 60.0 / 220.0))
    peaks, _ = find_peaks(signal, distance=min_distance_samples, prominence=prominence)
    return peaks.astype(int, copy=False)


def _estimate_sampling_rate_hz(timestamps_ms: np.ndarray) -> float | None:
    """Estimate sample rate from monotonically increasing millisecond timestamps."""

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


def build_instant_hr_from_provided_hr(reference: pd.DataFrame) -> pd.DataFrame:
    """Use already sampled HR values as a legacy target source."""

    if "hr_bpm" not in reference.columns:
        return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm"])

    result = reference.loc[:, ["timestamp_ms", "hr_bpm"]].copy()
    result["timestamp_ms"] = pd.to_numeric(result["timestamp_ms"], errors="coerce")
    result["instant_hr_bpm"] = pd.to_numeric(result["hr_bpm"], errors="coerce")
    result = result.dropna(subset=["timestamp_ms", "instant_hr_bpm"])
    result = result[np.isfinite(result["instant_hr_bpm"])].copy()
    return result.loc[:, ["timestamp_ms", "instant_hr_bpm"]].sort_values("timestamp_ms").reset_index(drop=True)
