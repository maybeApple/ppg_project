"""Foundation-model feature extraction adapters."""

from .common import EmbeddingExportSummary, build_window_signal_matrix, load_windows_from_manifest

__all__ = [
    "EmbeddingExportSummary",
    "build_window_signal_matrix",
    "load_windows_from_manifest",
]
