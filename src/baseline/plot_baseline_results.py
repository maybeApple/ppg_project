"""Generate baseline evaluation plots from saved prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["peak", "spectral"], nargs="*")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def generate_plots_for_method(
    result_dir: str | Path,
    method: str,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate the required evaluation plots for one saved baseline run."""

    result_path = Path(result_dir)
    predictions_path = result_path / f"{method}_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    valid_predictions = _prepare_valid_predictions(predictions)
    metrics = _load_metrics(result_path / f"{method}_metrics.json")

    target_dir = Path(output_dir) if output_dir is not None else result_path / "plots"
    target_dir.mkdir(parents=True, exist_ok=True)

    prediction_plot_path = target_dir / f"{method}_prediction_vs_truth.png"
    error_plot_path = target_dir / f"{method}_error_distribution.png"
    bland_altman_path = target_dir / f"{method}_bland_altman.png"

    _plot_prediction_vs_truth(valid_predictions, prediction_plot_path, method, metrics)
    _plot_error_distribution(valid_predictions, error_plot_path, method, metrics)
    _plot_bland_altman(valid_predictions, bland_altman_path, method, metrics)

    return {
        "prediction_vs_truth": str(prediction_plot_path),
        "error_distribution": str(error_plot_path),
        "bland_altman": str(bland_altman_path),
    }


def discover_methods(result_dir: str | Path) -> list[str]:
    """Discover saved baseline methods in a result directory."""

    result_path = Path(result_dir)
    methods = []
    for predictions_path in sorted(result_path.glob("*_predictions.csv")):
        methods.append(predictions_path.stem.removesuffix("_predictions"))
    return methods


def _prepare_valid_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only valid rows and derive error columns used by the plots."""

    required_columns = {"label_hr_bpm", "predicted_hr_bpm"}
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise KeyError(f"Prediction file is missing columns: {missing_columns}")

    valid_predictions = predictions.dropna(subset=["label_hr_bpm", "predicted_hr_bpm"]).copy()
    if valid_predictions.empty:
        raise ValueError("Prediction file does not contain any valid label/prediction pairs.")

    valid_predictions["error_bpm"] = (
        valid_predictions["predicted_hr_bpm"] - valid_predictions["label_hr_bpm"]
    )
    valid_predictions["mean_hr_bpm"] = (
        valid_predictions["predicted_hr_bpm"] + valid_predictions["label_hr_bpm"]
    ) / 2.0
    return valid_predictions


def _load_metrics(metrics_path: Path) -> dict[str, object]:
    """Load the saved metrics JSON when it exists."""

    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _plot_prediction_vs_truth(
    predictions: pd.DataFrame,
    output_path: Path,
    method: str,
    metrics: dict[str, object],
) -> None:
    """Plot prediction against ground truth with an identity line."""

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        predictions["label_hr_bpm"],
        predictions["predicted_hr_bpm"],
        s=10,
        alpha=0.35,
        color="#1f77b4",
        edgecolors="none",
    )

    lower = float(min(predictions["label_hr_bpm"].min(), predictions["predicted_hr_bpm"].min()))
    upper = float(max(predictions["label_hr_bpm"].max(), predictions["predicted_hr_bpm"].max()))
    margin = max((upper - lower) * 0.05, 3.0)
    axis.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        linewidth=1.2,
        color="#222222",
    )

    axis.set_xlabel("Ground Truth HR (bpm)")
    axis.set_ylabel("Predicted HR (bpm)")
    axis.set_title(_build_title(method, "Prediction vs Ground Truth", metrics))
    axis.grid(alpha=0.25)
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_error_distribution(
    predictions: pd.DataFrame,
    output_path: Path,
    method: str,
    metrics: dict[str, object],
) -> None:
    """Plot the distribution of prediction errors."""

    errors = predictions["error_bpm"].to_numpy(dtype=float, copy=True)
    mean_error = float(np.mean(errors))

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(errors, bins=40, color="#ff7f0e", alpha=0.8, edgecolor="white")
    axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="Zero error")
    axis.axvline(mean_error, color="#d62728", linewidth=1.2, label=f"Mean error = {mean_error:.2f}")

    axis.set_xlabel("Prediction Error (bpm)")
    axis.set_ylabel("Window Count")
    axis.set_title(_build_title(method, "Error Distribution", metrics))
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_bland_altman(
    predictions: pd.DataFrame,
    output_path: Path,
    method: str,
    metrics: dict[str, object],
) -> None:
    """Plot Bland-Altman agreement limits for one baseline."""

    means = predictions["mean_hr_bpm"].to_numpy(dtype=float, copy=True)
    errors = predictions["error_bpm"].to_numpy(dtype=float, copy=True)
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    upper_limit = mean_error + 1.96 * std_error
    lower_limit = mean_error - 1.96 * std_error

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(means, errors, s=10, alpha=0.35, color="#2ca02c", edgecolors="none")
    axis.axhline(mean_error, color="#222222", linewidth=1.2, label=f"Mean diff = {mean_error:.2f}")
    axis.axhline(upper_limit, color="#d62728", linestyle="--", linewidth=1.2, label=f"+1.96 SD = {upper_limit:.2f}")
    axis.axhline(lower_limit, color="#d62728", linestyle="--", linewidth=1.2, label=f"-1.96 SD = {lower_limit:.2f}")

    axis.set_xlabel("Mean HR of Prediction and Truth (bpm)")
    axis.set_ylabel("Prediction - Truth (bpm)")
    axis.set_title(_build_title(method, "Bland-Altman Plot", metrics))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _build_title(method: str, plot_name: str, metrics: dict[str, object]) -> str:
    """Build a compact plot title that includes headline metrics when available."""

    method_name = method.capitalize()
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    if isinstance(mae, (int, float)) and isinstance(rmse, (int, float)):
        return f"{method_name} {plot_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}"
    return f"{method_name} {plot_name}"


def main() -> None:
    """Generate plots for all or selected baseline methods in one result directory."""

    args = parse_args()
    methods = args.method or discover_methods(args.result_dir)
    if not methods:
        raise FileNotFoundError(f"No *_predictions.csv files were found in {args.result_dir}")

    outputs = {
        method: generate_plots_for_method(
            result_dir=args.result_dir,
            method=method,
            output_dir=args.output_dir,
        )
        for method in methods
    }
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
