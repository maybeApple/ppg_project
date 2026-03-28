"""Baseline heart rate estimation methods."""

from .peak_detection import PeakDetectionResult, apply_peak_detection_baseline, estimate_hr_from_peaks
from .spectral_hr import SpectralResult, apply_spectral_baseline, estimate_hr_from_spectrum

__all__ = [
    "PeakDetectionResult",
    "SpectralResult",
    "apply_peak_detection_baseline",
    "apply_spectral_baseline",
    "estimate_hr_from_peaks",
    "estimate_hr_from_spectrum",
]
