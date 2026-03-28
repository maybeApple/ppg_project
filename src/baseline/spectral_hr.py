"""Spectral baseline for heart rate estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.data.preprocessing import estimate_sampling_rate_hz


@dataclass(slots=True)
class SpectralResult:
    """Result of one spectrum-based HR estimate."""

    predicted_hr_bpm: float | None
    sampling_rate_hz: float | None
    dominant_frequency_hz: float | None
    dominant_power: float | None
    is_valid_prediction: bool
    error_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return asdict(self)


def estimate_hr_from_spectrum(
    ppg_values: list[float] | np.ndarray,
    timestamps_ms: list[int] | np.ndarray | None = None,
    sampling_rate_hz: float | None = None,
    min_hr_bpm: float = 42.0,
    max_hr_bpm: float = 210.0,
) -> SpectralResult:
    """Estimate HR with mean removal and Welch PSD peak search."""

    signal = np.asarray(ppg_values, dtype=float)
    if len(signal) < 3:
        return SpectralResult(None, None, None, None, False, "too_few_samples")

    fs = _resolve_sampling_rate_hz(signal, timestamps_ms, sampling_rate_hz)
    if fs is None or fs <= 0:
        return SpectralResult(None, None, None, None, False, "invalid_sampling_rate")

    demeaned_signal = signal - np.nanmean(signal)
    signal_std = float(np.nanstd(demeaned_signal))
    if not np.isfinite(signal_std) or signal_std == 0:
        return SpectralResult(None, fs, None, None, False, "flat_signal")

    frequencies_hz, power_density = welch(
        demeaned_signal,
        fs=fs,
        nperseg=min(len(demeaned_signal), max(64, int(round(fs * 4)))),
        detrend="constant",
    )
    band_mask = (frequencies_hz >= min_hr_bpm / 60.0) & (frequencies_hz <= max_hr_bpm / 60.0)
    if not np.any(band_mask):
        return SpectralResult(None, fs, None, None, False, "empty_frequency_band")

    band_frequencies = frequencies_hz[band_mask]
    band_power = power_density[band_mask]
    peak_index = int(np.argmax(band_power))
    dominant_frequency_hz = float(band_frequencies[peak_index])
    dominant_power = float(band_power[peak_index])
    predicted_hr_bpm = dominant_frequency_hz * 60.0

    return SpectralResult(
        predicted_hr_bpm=predicted_hr_bpm,
        sampling_rate_hz=float(fs),
        dominant_frequency_hz=dominant_frequency_hz,
        dominant_power=dominant_power,
        is_valid_prediction=True,
        error_reason=None,
    )


def apply_spectral_baseline(windows: pd.DataFrame) -> pd.DataFrame:
    """Apply the spectral baseline to a window table."""

    prediction_rows: list[dict[str, object]] = []
    for row in windows.itertuples(index=False):
        result = estimate_hr_from_spectrum(
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


def _as_optional_float(value: object) -> float | None:
    """Convert a possibly missing scalar into float or None."""

    if value is None or pd.isna(value):
        return None
    return float(value)
