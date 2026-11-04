"""Generate evaluation plots from saved regression prediction files."""

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
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_prediction_file(result_dir: Path) -> Path:
    """Infer the single saved prediction file inside one regression result directory."""

    matches = sorted(result_dir.glob("*_predictions.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_predictions.csv in {result_dir.as_posix()}, found {len(matches)}."
        )
    return matches[0]


def infer_metrics_file(result_dir: Path, prediction_file: Path) -> Path | None:
    """Infer the metrics file paired with one prediction file."""

    candidate = result_dir / prediction_file.name.replace("_predictions.csv", "_metrics.json")
    return candidate if candidate.exists() else None


def prepare_valid_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
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


def load_metrics(metrics_path: Path | None) -> dict[str, object]:
    """Load metrics JSON when it exists."""

    if metrics_path is None or not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_title(method_name: str, plot_name: str, metrics: dict[str, object]) -> str:
    """Build a compact plot title with headline metrics when available."""

    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    if isinstance(mae, (int, float)) and isinstance(rmse, (int, float)):
        return f"{method_name} {plot_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}"
    return f"{method_name} {plot_name}"


def plot_prediction_vs_truth(
    predictions: pd.DataFrame,
    output_path: Path,
    method_name: str,
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
    axis.set_title(build_title(method_name, "Prediction vs Ground Truth", metrics))
    axis.grid(alpha=0.25)
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_error_distribution(
    predictions: pd.DataFrame,
    output_path: Path,
    method_name: str,
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
    axis.set_title(build_title(method_name, "Error Distribution", metrics))
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_bland_altman(
    predictions: pd.DataFrame,
    output_path: Path,
    method_name: str,
    metrics: dict[str, object],
) -> None:
    """Plot Bland-Altman agreement limits for one saved regressor result."""

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
    axis.set_title(build_title(method_name, "Bland-Altman Plot", metrics))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_plots(
    result_dir: Path,
    prediction_file: Path | None = None,
    metrics_file: Path | None = None,
    method_name: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate the required evaluation plots for one saved regression run."""

    resolved_prediction_file = prediction_file or infer_prediction_file(result_dir)
    resolved_metrics_file = metrics_file or infer_metrics_file(result_dir, resolved_prediction_file)
    resolved_method_name = method_name or resolved_prediction_file.stem.removesuffix("_predictions")

    predictions = pd.read_csv(resolved_prediction_file)
    valid_predictions = prepare_valid_predictions(predictions)
    metrics = load_metrics(resolved_metrics_file)

    target_dir = output_dir or (result_dir / "plots")
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = resolved_prediction_file.stem.removesuffix("_predictions")
    prediction_plot_path = target_dir / f"{slug}_prediction_vs_truth.png"
    error_plot_path = target_dir / f"{slug}_error_distribution.png"
    bland_altman_path = target_dir / f"{slug}_bland_altman.png"

    plot_prediction_vs_truth(valid_predictions, prediction_plot_path, resolved_method_name, metrics)
    plot_error_distribution(valid_predictions, error_plot_path, resolved_method_name, metrics)
    plot_bland_altman(valid_predictions, bland_altman_path, resolved_method_name, metrics)

    return {
        "prediction_vs_truth": str(prediction_plot_path),
        "error_distribution": str(error_plot_path),
        "bland_altman": str(bland_altman_path),
    }


def main() -> None:
    """Generate plots for one saved regression result directory."""

    args = parse_args()
    outputs = generate_plots(
        result_dir=args.result_dir,
        prediction_file=args.prediction_file,
        metrics_file=args.metrics_file,
        method_name=args.method_name,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
