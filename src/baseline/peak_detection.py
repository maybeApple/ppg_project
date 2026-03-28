"""Peak-detection baseline for heart rate estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from src.data.preprocessing import estimate_sampling_rate_hz


@dataclass(slots=True)
class PeakDetectionResult:
    """Result of one peak-based HR estimate."""

    predicted_hr_bpm: float | None
    sampling_rate_hz: float | None
    peak_count: int
    median_interval_s: float | None
    prominence_used: float | None
    is_valid_prediction: bool
    error_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return asdict(self)


def estimate_hr_from_peaks(
    ppg_values: list[float] | np.ndarray,
    timestamps_ms: list[int] | np.ndarray | None = None,
    sampling_rate_hz: float | None = None,
    min_hr_bpm: float = 42.0,
    max_hr_bpm: float = 210.0,
    filter_order: int = 3,
    prominence_scale: float = 0.5,
) -> PeakDetectionResult:
    """Estimate HR by band-pass filtering, peak detection, and peak intervals."""

    signal = np.asarray(ppg_values, dtype=float)
    #There needs to be at least some decent waveform; if it's too short, it will fail immediately.
    if len(signal) < 3:
        return PeakDetectionResult(None, None, 0, None, None, False, "too_few_samples")

    fs = _resolve_sampling_rate_hz(signal, timestamps_ms, sampling_rate_hz)
    if fs is None or fs <= 0:
        return PeakDetectionResult(None, None, 0, None, None, False, "invalid_sampling_rate")

    #min_hr_bpm = 42max_hr_bpm = 210
    #换成 Hz 就是：42 / 60 = 0.7 Hz 210 / 60 = 3.5 Hz
    #滤波保留的是大致 0.7–3.5 Hz 的频段，也就是常见心率范围。
    filtered_signal = _bandpass_filter_ppg(
        signal=signal,
        sampling_rate_hz=fs,
        low_hz=min_hr_bpm / 60.0,
        high_hz=max_hr_bpm / 60.0,
        filter_order=filter_order,
    )
    if np.isnan(filtered_signal).all():
        return PeakDetectionResult(None, fs, 0, None, None, False, "filter_failed")

    #找峰的阈值不是写死常数，而是根据当前窗口滤波后信号的标准差Standard deviation自动设定。
    signal_std = float(np.nanstd(filtered_signal))
    if not np.isfinite(signal_std) or signal_std == 0:
        return PeakDetectionResult(None, fs, 0, None, None, False, "flat_signal")

    prominence = max(signal_std * prominence_scale, 1e-6)
    min_peak_distance_samples = max(1, int(fs * 60.0 / max_hr_bpm))

    peaks, _ = find_peaks(
        filtered_signal,
        distance=min_peak_distance_samples,
        prominence=prominence,
    )
    if len(peaks) < 2:
        return PeakDetectionResult(None, fs, int(len(peaks)), None, prominence, False, "not_enough_peaks")

    intervals_s = np.diff(peaks) / fs
    min_interval_s = 60.0 / max_hr_bpm
    max_interval_s = 60.0 / min_hr_bpm
    valid_intervals_s = intervals_s[(intervals_s >= min_interval_s) & (intervals_s <= max_interval_s)]
    if len(valid_intervals_s) == 0:
        return PeakDetectionResult(None, fs, int(len(peaks)), None, prominence, False, "invalid_intervals")

    median_interval_s = float(np.median(valid_intervals_s))
    predicted_hr_bpm = 60.0 / median_interval_s
    return PeakDetectionResult(
        predicted_hr_bpm=float(predicted_hr_bpm),
        sampling_rate_hz=float(fs),
        peak_count=int(len(peaks)),
        median_interval_s=median_interval_s,
        prominence_used=prominence,
        is_valid_prediction=True,
        error_reason=None,
    )


def apply_peak_detection_baseline(windows: pd.DataFrame) -> pd.DataFrame:
    """Apply the peak-detection baseline to a window table."""

    prediction_rows: list[dict[str, object]] = []
    for row in windows.itertuples(index=False):
        result = estimate_hr_from_peaks(
            ppg_values=row.ppg_values,
            timestamps_ms=row.ppg_timestamps_ms,
            sampling_rate_hz=_as_optional_float(getattr(row, "ppg_sampling_hz", None)),
        )
        prediction_rows.append(
            {
                "participant_id": row.participant_id,
                "session_id": row.session_id,
                "session_name": row.session_name,
                "window_index": row.window_index,
                "window_start_ms": row.window_start_ms,
                "window_end_ms": row.window_end_ms,
                "label_hr_bpm": row.label_hr_bpm,
                "reference_source": row.reference_source,
                **result.to_dict(),
            }
        )

    return pd.DataFrame(prediction_rows)


def _resolve_sampling_rate_hz(
    signal: np.ndarray,
    timestamps_ms: list[int] | np.ndarray | None,
    sampling_rate_hz: float | None,
) -> float | None:
    """Resolve the sampling rate from explicit input or timestamps."""

    if sampling_rate_hz is not None and np.isfinite(sampling_rate_hz) and sampling_rate_hz > 0:
        return float(sampling_rate_hz)
    if timestamps_ms is None:
        return None
    timestamps = np.asarray(timestamps_ms, dtype=float)
    if len(timestamps) != len(signal):
        return None
    return estimate_sampling_rate_hz(timestamps)


def _bandpass_filter_ppg(
    signal: np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter after removing the DC offset."""

    #Remove the DC component to avoid baseline offset affecting the filtering.
    demeaned_signal = signal - np.nanmean(signal)
    nyquist_hz = sampling_rate_hz / 2.0
    low_hz = max(low_hz, 1e-3)
    #Boundary protection
    high_hz = min(high_hz, nyquist_hz * 0.99)
    if low_hz >= high_hz:
        return np.full_like(signal, np.nan, dtype=float)

    try:
        #Butterworth 带通滤波器 filtfilt(...)：双向滤波bidirectional filtering，避免相位延迟
        b_coeff, a_coeff = butter(filter_order, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz)
        return filtfilt(b_coeff, a_coeff, demeaned_signal)
    except ValueError:
        return np.full_like(signal, np.nan, dtype=float)


def _as_optional_float(value: object) -> float | None:
    """Convert a possibly missing scalar into float or None."""

    if value is None or pd.isna(value):
        return None
    return float(value)
