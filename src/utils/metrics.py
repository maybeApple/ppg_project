"""Common regression metrics used across experiments."""

from __future__ import annotations

import numpy as np


def _coerce_valid_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to aligned float arrays and drop NaN pairs."""

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    valid_mask = ~(np.isnan(true_values) | np.isnan(pred_values))
    return true_values[valid_mask], pred_values[valid_mask]


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    """Return MAE after dropping invalid pairs."""

    true_values, pred_values = _coerce_valid_arrays(y_true, y_pred)
    if len(true_values) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred_values - true_values)))


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    """Return RMSE after dropping invalid pairs."""

    true_values, pred_values = _coerce_valid_arrays(y_true, y_pred)
    if len(true_values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_values - true_values))))


def prediction_coverage(y_true: object, y_pred: object) -> float:
    """Return the fraction of valid prediction-target pairs."""

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    if len(true_values) == 0:
        return float("nan")
    valid_mask = ~(np.isnan(true_values) | np.isnan(pred_values))
    return float(np.mean(valid_mask))
