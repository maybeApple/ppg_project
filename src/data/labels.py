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
    min_hr_bpm: float = 35.0
    max_hr_bpm: float = 220.0
    rr_relative_deviation_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.aggregation not in {"mean", "median"}:
            raise ValueError(f"Unsupported label aggregation: {self.aggregation}")
        if self.min_valid_beats < 1 or self.min_reference_samples < 1:
            raise ValueError("Minimum reference and valid interval counts must be at least 1")
        if not (np.isfinite(self.min_hr_bpm) and np.isfinite(self.max_hr_bpm)
                and 0 < self.min_hr_bpm < self.max_hr_bpm):
            raise ValueError("Heart-rate bounds must be finite with 0 < min_hr_bpm < max_hr_bpm")
        if self.rr_relative_deviation_threshold is not None and not (
            np.isfinite(self.rr_relative_deviation_threshold) and self.rr_relative_deviation_threshold > 0
        ):
            raise ValueError("RR relative deviation threshold must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        """Convert the config into a JSON-serializable, auditable dictionary."""

        payload = asdict(self)
        if self.method == "beat_interval_instant_hr":
            payload.update(
                {
                    "instant_hr_formula": "instant_hr_bpm = 60000 / rr_interval_ms",
                    "label_schema_version": "beat_interval_v2",
                    "ibi_reference": "use finite RR intervals under the same physiologic rules as ECG",
                    "ecg_reference": (
                        "detect ECG R peaks once per aligned session, compute adjacent R-R intervals, "
                        "and assign each interval timestamp to the second beat"
                    ),
                    "window_target": f"{self.aggregation} instantaneous HR inside each window",
                    "discard_rule": f"drop windows with fewer than {self.min_valid_beats} valid beat intervals",
                    "valid_rr_interval_ms": [60000.0 / self.max_hr_bpm, 60000.0 / self.min_hr_bpm],
                    "rr_anomaly_policy": (
                        "drop outside physiologic range; no interpolation or smoothing"
                        if self.rr_relative_deviation_threshold is None else
                        "drop outside physiologic range and configured relative deviation from window median; no interpolation"
                    ),
                    "timestamp_policy": "finite strictly increasing timestamps; interval belongs to second beat; [start, end)",
                    "min_valid_beats_semantics": "number of valid beat intervals (not R-peak count)",
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
    audit: dict[str, object] | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> WindowLabel | None:
    """Compute one window target or return None when the window is invalid."""

    details = audit if audit is not None else {}
    details.clear()
    if (window_start_ms is None) != (window_end_ms is None):
        raise ValueError("Both window bounds must be supplied together")
    if window_start_ms is not None:
        if not (np.isfinite(window_start_ms) and np.isfinite(window_end_ms) and window_start_ms < window_end_ms):
            raise ValueError("Window bounds must be finite and increasing")
        timestamps = pd.to_numeric(reference_window.get("timestamp_ms", pd.Series(dtype=float)), errors="coerce")
        if ((timestamps < window_start_ms) | (timestamps >= window_end_ms)).any():
            details["drop_reason"] = "reference_outside_window"
            return None

    if config.method == "beat_interval_instant_hr":
        target_frame = build_instant_hr_from_beat_intervals(reference_window, config=config, audit=details)
        if len(target_frame) < config.min_valid_beats:
            details.setdefault("drop_reason", "insufficient_valid_beat_intervals")
            return None
        label_method = "beat_interval_instant_hr"
    elif config.method == "provided_hr_samples":
        target_frame = build_instant_hr_from_provided_hr(reference_window)
        if len(target_frame) < config.min_reference_samples:
            details["drop_reason"] = "insufficient_reference_samples"
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


def build_instant_hr_from_beat_intervals(
    reference: pd.DataFrame,
    config: LabelGenerationConfig | None = None,
    audit: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Derive instantaneous HR from valid beat intervals or ECG samples."""

    config = config or LabelGenerationConfig()
    if "rr_interval_ms" not in reference.columns or (
        reference["rr_interval_ms"].isna().all() and "ecg_uv" in reference.columns
    ):
        return build_instant_hr_from_ecg(reference, config=config, audit=audit)

    return _validate_rr_intervals(reference, config, audit)


def _validate_rr_intervals(
    reference: pd.DataFrame, config: LabelGenerationConfig, audit: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Reject invalid observations; never replace, interpolate, or smooth intervals."""

    details = audit if audit is not None else {}
    if "timestamp_ms" not in reference or "rr_interval_ms" not in reference:
        details["drop_reason"] = "missing_reference_columns"
        return _empty_intervals()

    result = reference.loc[:, ["timestamp_ms", "rr_interval_ms"]].copy()
    result["timestamp_ms"] = pd.to_numeric(result["timestamp_ms"], errors="coerce")
    result["rr_interval_ms"] = pd.to_numeric(result["rr_interval_ms"], errors="coerce")
    finite_timestamps = np.isfinite(result["timestamp_ms"].to_numpy(dtype=float))
    finite_rr = np.isfinite(result["rr_interval_ms"].to_numpy(dtype=float))
    details["nonfinite_timestamp_count"] = int((~finite_timestamps).sum())
    details["nonfinite_rr_count"] = int((~finite_rr).sum())
    result = result.loc[finite_timestamps & finite_rr].copy()
    if len(result) > 1 and (np.diff(result["timestamp_ms"].to_numpy(dtype=float)) <= 0).any():
        details["drop_reason"] = "nonmonotonic_reference_timestamps"
        return _empty_intervals()
    valid = result["rr_interval_ms"].between(60000.0 / config.max_hr_bpm, 60000.0 / config.min_hr_bpm)
    details["physiologic_rr_rejection_count"] = int((~valid).sum())
    result = result.loc[valid].copy()
    if config.rr_relative_deviation_threshold is not None and not result.empty:
        median_rr = float(result["rr_interval_ms"].median())
        keep = (result["rr_interval_ms"] / median_rr - 1.0).abs() <= config.rr_relative_deviation_threshold
        details["relative_rr_rejection_count"] = int((~keep).sum())
        result = result.loc[keep].copy()

    result["instant_hr_bpm"] = 60000.0 / result["rr_interval_ms"]
    result = result[np.isfinite(result["instant_hr_bpm"])].copy()
    return result.sort_values("timestamp_ms").reset_index(drop=True)


def build_instant_hr_from_ecg(
    reference: pd.DataFrame,
    config: LabelGenerationConfig | None = None,
    audit: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Detect ECG beats and convert adjacent R-R intervals to instantaneous HR."""

    config = config or LabelGenerationConfig()
    details = audit if audit is not None else {}
    if "ecg_uv" not in reference.columns or "timestamp_ms" not in reference.columns:
        details["drop_reason"] = "missing_ecg_reference"
        return _empty_intervals()

    ecg = reference.loc[:, ["timestamp_ms", "ecg_uv"]].copy()
    ecg["timestamp_ms"] = pd.to_numeric(ecg["timestamp_ms"], errors="coerce")
    ecg["ecg_uv"] = pd.to_numeric(ecg["ecg_uv"], errors="coerce")
    if not np.isfinite(ecg.to_numpy(dtype=float)).all():
        # Removing ECG samples would bridge gaps and fabricate RR intervals.
        details["drop_reason"] = "nonfinite_ecg_samples"
        return _empty_intervals()
    if len(ecg) < 3:
        details["drop_reason"] = "insufficient_ecg_samples"
        return _empty_intervals()

    timestamps_ms = ecg["timestamp_ms"].to_numpy(dtype=float, copy=True)
    signal = ecg["ecg_uv"].to_numpy(dtype=float, copy=True)
    sampling_hz = _estimate_sampling_rate_hz(timestamps_ms)
    if sampling_hz is None or sampling_hz <= 0:
        details["drop_reason"] = "invalid_ecg_timestamps"
        return _empty_intervals()
    if (np.diff(timestamps_ms) <= 0).any():
        details["drop_reason"] = "nonmonotonic_reference_timestamps"
        return _empty_intervals()
    # Reject gaps rather than silently treat an irregular stream as uniform ECG.
    if np.max(np.diff(timestamps_ms)) > max(3.0 * 1000.0 / sampling_hz, 5.0):
        details["drop_reason"] = "ecg_timestamp_gap"
        return _empty_intervals()

    try:
        peak_indices = _detect_ecg_peak_indices(signal=signal, sampling_hz=sampling_hz, max_hr_bpm=config.max_hr_bpm)
    except (ValueError, FloatingPointError):
        details["drop_reason"] = "ecg_peak_detection_failed"
        return _empty_intervals()
    details["ecg_detected_peak_count"] = int(len(peak_indices))
    if len(peak_indices) < 2:
        details["drop_reason"] = "ecg_peak_detection_failed"
        return _empty_intervals()

    peak_timestamps_ms = timestamps_ms[peak_indices]
    rr_interval_ms = np.diff(peak_timestamps_ms)
    result = pd.DataFrame(
        {
            "timestamp_ms": peak_timestamps_ms[1:],
            "rr_interval_ms": rr_interval_ms,
        }
    )
    return _validate_rr_intervals(result, config, details)


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_ms", "instant_hr_bpm", "rr_interval_ms"])


def _detect_ecg_peak_indices(signal: np.ndarray, sampling_hz: float, max_hr_bpm: float = 220.0) -> np.ndarray:
    """Return likely R-peak sample indices for one ECG window."""

    filtered_signal = _filter_ecg(signal=signal, sampling_hz=sampling_hz)
    positive_peaks = _find_ecg_peaks(filtered_signal, sampling_hz, max_hr_bpm)
    negative_peaks = _find_ecg_peaks(-filtered_signal, sampling_hz, max_hr_bpm)
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


def _find_ecg_peaks(signal: np.ndarray, sampling_hz: float, max_hr_bpm: float = 220.0) -> np.ndarray:
    """Find peaks with ECG-compatible spacing and adaptive prominence."""

    signal_std = float(np.nanstd(signal))
    signal_range = float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))
    if not np.isfinite(signal_std) or signal_std <= 0:
        return np.asarray([], dtype=int)
    prominence = max(signal_std * 1.0, signal_range * 0.12, 1e-6)
    min_distance_samples = max(1, int(sampling_hz * 60.0 / max_hr_bpm))
    peaks, _ = find_peaks(signal, distance=min_distance_samples, prominence=prominence)
    return peaks.astype(int, copy=False)


def _estimate_sampling_rate_hz(timestamps_ms: np.ndarray) -> float | None:
    """Estimate sample rate from monotonically increasing millisecond timestamps."""

    if len(timestamps_ms) < 2:
        return None
    diffs = np.diff(timestamps_ms.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
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
