"""Evaluate heart rate predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.utils.metrics import mean_absolute_error, prediction_coverage, root_mean_squared_error


@dataclass(slots=True)
class EvaluationSummary:
    """Container for aggregate prediction metrics."""

    num_windows: int
    num_valid_predictions: int
    coverage: float
    mae: float
    rmse: float

    def to_dict(self) -> dict[str, float | int]:
        """Convert the summary into a JSON-serializable dictionary."""

        return asdict(self)


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    truth_col: str = "label_hr_bpm",
    prediction_col: str = "predicted_hr_bpm",
) -> EvaluationSummary:
    """Evaluate a prediction frame against its reference labels."""

    if truth_col not in predictions.columns or prediction_col not in predictions.columns:
        raise KeyError(f"Prediction frame must include `{truth_col}` and `{prediction_col}`.")

    y_true = predictions[truth_col].to_numpy(dtype=float, copy=True)
    y_pred = predictions[prediction_col].to_numpy(dtype=float, copy=True)
    valid_mask = ~(pd.isna(predictions[truth_col]) | pd.isna(predictions[prediction_col]))

    return EvaluationSummary(
        num_windows=int(len(predictions)),
        num_valid_predictions=int(valid_mask.sum()),
        coverage=prediction_coverage(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
    )
