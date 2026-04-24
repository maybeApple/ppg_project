"""Foundation-model feature extraction adapters."""

from .common import (
    EmbeddingExportSummary,
    SignalPreprocessingConfig,
    build_signal_preprocessing_config,
    build_window_signal_matrix,
    load_signal_preprocessing_config,
    load_windows_from_manifest,
)

__all__ = [
    "EmbeddingExportSummary",
    "SignalPreprocessingConfig",
    "build_signal_preprocessing_config",
    "build_window_signal_matrix",
    "load_signal_preprocessing_config",
    "load_windows_from_manifest",
]
