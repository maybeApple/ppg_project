"""Build Week 2 GalaxyPPG benchmark artifacts from completed run folders."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "median_absolute_error",
    "p95_absolute_error",
    "catastrophic_error_rate_20bpm",
]
IDENTITY_COLUMNS = [
    "dataset",
    "model_name",
    "preprocessing_mode",
    "inversion",
    "normalization_strategy",
    "probe_type",
    "regressor_type",
]
PREDICTION_COLUMNS = [
    "dataset",
    "participant_id",
    "session_id",
    "activity",
    "window_id",
    "window_start_time",
    "window_end_time",
    "y_true_hr",
    "y_pred_hr",
    "abs_error",
    "squared_error",
    "model_name",
    "preprocessing_mode",
    "split_id",
    "inversion",
    "normalization_strategy",
    "probe_type",
    "regressor_type",
    "peak_count",
    "estimated_peak_hr",
    "spectral_peak_frequency",
    "estimated_spectral_hr",
    "ppg_quality_flag_if_available",
    "acc_motion_summary_if_available",
]


@dataclass(slots=True)
class RunBundle:
    """One prediction/metric pair discovered in a completed result directory."""

    predictions_path: Path
    metrics_path: Path
    run_id: str
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag-name", type=str, default="")
    parser.add_argument("--include-raw-corrected-results", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    bundles = discover_run_bundles(args.search_root)
    if not bundles:
        raise RuntimeError(f"No prediction/metrics pairs found under {args.search_root}.")

    predictions_dir = args.output_root / "predictions"
    metrics_dir = args.output_root / "metrics"
    tables_dir = args.output_root / "tables"
    figures_dir = args.output_root / "figures"
    for path in [predictions_dir, metrics_dir, tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    standardized_frames: list[pd.DataFrame] = []
    run_index_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        frame = standardize_prediction_frame(bundle)
        output_path = predictions_dir / f"{bundle.run_id}_predictions.csv"
        frame.to_csv(output_path, index=False)
        standardized_frames.append(frame)
        run_index_rows.append(
            {
                "run_id": bundle.run_id,
                "raw_predictions_path": str(bundle.predictions_path),
                "raw_metrics_path": str(bundle.metrics_path),
                "standardized_predictions_path": str(output_path),
                "model_name": frame["model_name"].iloc[0],
                "preprocessing_mode": frame["preprocessing_mode"].iloc[0],
                "inversion": bool(frame["inversion"].iloc[0]),
                "normalization_strategy": frame["normalization_strategy"].iloc[0],
                "probe_type": frame["probe_type"].iloc[0],
                "regressor_type": frame["regressor_type"].iloc[0],
                "n_predictions": int(len(frame)),
            }
        )

    all_predictions = pd.concat(standardized_frames, ignore_index=True)
    all_predictions_path = predictions_dir / "week2_all_standardized_predictions.csv"
    all_predictions.to_csv(all_predictions_path, index=False)
    pd.DataFrame(run_index_rows).sort_values(["model_name", "preprocessing_mode", "regressor_type"]).to_csv(
        args.output_root / "run_index.csv",
        index=False,
    )

    overall = aggregate_metrics(all_predictions, IDENTITY_COLUMNS)
    participant = aggregate_metrics(all_predictions, IDENTITY_COLUMNS + ["participant_id"])
    activity = aggregate_metrics(all_predictions, IDENTITY_COLUMNS + ["activity"])
    overall.to_csv(metrics_dir / "overall_metrics.csv", index=False)
    participant.to_csv(metrics_dir / "participant_level_metrics.csv", index=False)
    activity.to_csv(metrics_dir / "activity_level_metrics.csv", index=False)
    (metrics_dir / "overall_metrics.json").write_text(
        json.dumps(overall.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_tables(overall, tables_dir)
    write_figures(all_predictions, overall, participant, activity, figures_dir)
    write_memo(
        output_root=args.output_root,
        overall=overall,
        participant=participant,
        activity=activity,
        tag_name=args.tag_name,
        all_predictions_path=all_predictions_path,
    )

    print(f"runs={len(bundles)}")
    print(f"all_predictions={all_predictions_path}")
    print(f"overall_metrics={metrics_dir / 'overall_metrics.csv'}")
    print(f"memo={args.output_root / 'week2_memo.md'}")


def discover_run_bundles(search_root: Path) -> list[RunBundle]:
    bundles: list[RunBundle] = []
    for predictions_path in sorted(search_root.rglob("*_predictions.csv")):
        if "week2_all_standardized_predictions" in predictions_path.name:
            continue
        if predictions_path.parent.name in {"predictions", "metrics", "tables", "figures"}:
            continue
        stem = predictions_path.name.removesuffix("_predictions.csv")
        metrics_path = predictions_path.with_name(f"{stem}_metrics.json")
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = sanitize_run_id(str(predictions_path.relative_to(search_root)).removesuffix("_predictions.csv"))
        bundles.append(RunBundle(predictions_path, metrics_path, run_id, metrics))
    return bundles


def standardize_prediction_frame(bundle: RunBundle) -> pd.DataFrame:
    raw = pd.read_csv(bundle.predictions_path)
    metrics = bundle.metrics
    model_name = infer_model_name(bundle, raw)
    regressor = infer_regressor(bundle, metrics)
    preprocessing_mode = infer_preprocessing_mode(bundle, metrics)
    normalization = infer_normalization(bundle, metrics)
    inversion = infer_inversion(raw, metrics)
    split_id = str(metrics.get("split_name") or metrics.get("cv_strategy") or "subject_independent_fixed_split")

    y_true = raw["label_hr_bpm"].astype(float)
    y_pred = raw["predicted_hr_bpm"].astype(float)
    activity = raw["session_name"] if "session_name" in raw.columns else pd.Series("NA", index=raw.index)
    window_id = (
        raw["window_uid"].astype(str)
        if "window_uid" in raw.columns
        else raw["participant_id"].astype(str)
        + ":"
        + raw.get("session_id", pd.Series("NA", index=raw.index)).astype(str)
        + ":"
        + raw["window_index"].astype(str)
    )

    frame = pd.DataFrame(
        {
            "dataset": "GalaxyPPG",
            "participant_id": raw["participant_id"].astype(str),
            "session_id": raw.get("session_id", pd.Series("NA", index=raw.index)).astype(str),
            "activity": activity.fillna("NA").astype(str),
            "window_id": window_id,
            "window_start_time": raw.get("window_start_ms", pd.Series("NA", index=raw.index)),
            "window_end_time": raw.get("window_end_ms", pd.Series("NA", index=raw.index)),
            "y_true_hr": y_true,
            "y_pred_hr": y_pred,
            "abs_error": (y_pred - y_true).abs(),
            "squared_error": np.square(y_pred - y_true),
            "model_name": model_name,
            "preprocessing_mode": preprocessing_mode,
            "split_id": split_id,
            "inversion": inversion,
            "normalization_strategy": normalization,
            "probe_type": regressor if regressor in {"linear", "ridge"} else "NA",
            "regressor_type": regressor,
            "peak_count": raw["peak_count"] if "peak_count" in raw.columns else "NA",
            "estimated_peak_hr": raw["predicted_hr_bpm"] if model_name == "peak_based" else "NA",
            "spectral_peak_frequency": raw["dominant_frequency_hz"] if "dominant_frequency_hz" in raw.columns else "NA",
            "estimated_spectral_hr": raw["predicted_hr_bpm"] if model_name == "spectral" else "NA",
            "ppg_quality_flag_if_available": raw.get("is_valid_prediction", pd.Series("NA", index=raw.index)),
            "acc_motion_summary_if_available": "NA",
        }
    )
    return frame.loc[:, PREDICTION_COLUMNS]


def infer_model_name(bundle: RunBundle, raw: pd.DataFrame) -> str:
    method = str(bundle.metrics.get("method", "")).lower()
    if method == "peak" or bundle.predictions_path.name.startswith("peak_"):
        return "peak_based"
    if method == "spectral" or bundle.predictions_path.name.startswith("spectral_"):
        return "spectral"
    model = str(bundle.metrics.get("model_name", "")).lower()
    if model:
        return model
    name = bundle.predictions_path.name.lower()
    if "pulseppg" in name:
        return "pulseppg"
    if "papagei" in name:
        return "papagei"
    return "unknown"


def infer_regressor(bundle: RunBundle, metrics: dict[str, Any]) -> str:
    regressor = str(metrics.get("regressor") or "").lower()
    if regressor:
        return regressor
    if bundle.predictions_path.name.startswith("peak_"):
        return "classical_peak"
    if bundle.predictions_path.name.startswith("spectral_"):
        return "classical_spectral"
    return "NA"


def infer_preprocessing_mode(bundle: RunBundle, metrics: dict[str, Any]) -> str:
    mode = str(metrics.get("preprocessing_mode") or "").lower()
    if mode:
        return mode
    path_text = str(bundle.predictions_path).lower()
    if "model_faithful" in path_text:
        return "model_faithful"
    return "harmonized"


def infer_normalization(bundle: RunBundle, metrics: dict[str, Any]) -> str:
    value = metrics.get("feature_normalization")
    if value:
        return str(value)
    preprocessing = metrics.get("feature_preprocessing")
    if isinstance(preprocessing, dict) and preprocessing.get("normalization"):
        return str(preprocessing["normalization"])
    if bundle.metrics.get("method"):
        return "NA"
    return "unknown"


def infer_inversion(raw: pd.DataFrame, metrics: dict[str, Any]) -> bool:
    if "inversion" in raw.columns and raw["inversion"].notna().any():
        return bool(raw["inversion"].dropna().iloc[0])
    if "ppg_inverted" in raw.columns and raw["ppg_inverted"].notna().any():
        return bool(raw["ppg_inverted"].dropna().iloc[0])
    return bool(metrics.get("ppg_inverted", True))


def aggregate_metrics(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        valid = group.dropna(subset=["y_true_hr", "y_pred_hr"]).copy()
        row = {column: value for column, value in zip(group_columns, key_values)}
        if valid.empty:
            row.update({metric: math.nan for metric in METRIC_COLUMNS})
            row["n_windows"] = 0
            row["n_participants"] = 0
        else:
            row.update(
                {
                    "MAE": float(valid["abs_error"].mean()),
                    "RMSE": float(np.sqrt(valid["squared_error"].mean())),
                    "median_absolute_error": float(valid["abs_error"].median()),
                    "p95_absolute_error": float(valid["abs_error"].quantile(0.95)),
                    "catastrophic_error_rate_20bpm": float((valid["abs_error"] > 20).mean()),
                    "n_windows": int(len(valid)),
                    "n_participants": int(valid["participant_id"].nunique()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["MAE", "model_name"], na_position="last").reset_index(drop=True)


def write_tables(overall: pd.DataFrame, tables_dir: Path) -> None:
    benchmark = format_benchmark_table(overall[overall["inversion"] == True].copy())  # noqa: E712
    benchmark.to_csv(tables_dir / "main_benchmark_table.csv", index=False)
    write_markdown_table(benchmark, tables_dir / "main_benchmark_table.md")

    harmonized = format_benchmark_table(
        overall[(overall["preprocessing_mode"] == "harmonized") & (overall["inversion"] == True)].copy()  # noqa: E712
    )
    harmonized.to_csv(tables_dir / "harmonized_preprocessing_table.csv", index=False)
    write_markdown_table(harmonized, tables_dir / "harmonized_preprocessing_table.md")

    model_faithful = format_benchmark_table(
        overall[
            (overall["preprocessing_mode"] == "model_faithful") & (overall["inversion"] == True)  # noqa: E712
        ].copy()
    )
    model_faithful.to_csv(tables_dir / "model_faithful_preprocessing_table.csv", index=False)
    write_markdown_table(model_faithful, tables_dir / "model_faithful_preprocessing_table.md")

    inversion = build_inversion_table(overall)
    inversion.to_csv(tables_dir / "inversion_ablation_table.csv", index=False)
    write_markdown_table(inversion, tables_dir / "inversion_ablation_table.md")


def format_benchmark_table(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame(
            columns=[
                "Model",
                "Preprocessing mode",
                "Normalization",
                "Probe / Regressor",
                "MAE",
                "RMSE",
                "Median AE",
                "P95 AE",
                "Catastrophic error >20 bpm",
                "n_windows",
            ]
        )
    table = overall.copy()
    table["Model"] = table["model_name"]
    table["Preprocessing mode"] = table["preprocessing_mode"]
    table["Normalization"] = table["normalization_strategy"]
    table["Probe / Regressor"] = table["regressor_type"]
    table["Median AE"] = table["median_absolute_error"]
    table["P95 AE"] = table["p95_absolute_error"]
    table["Catastrophic error >20 bpm"] = table["catastrophic_error_rate_20bpm"]
    return table[
        [
            "Model",
            "Preprocessing mode",
            "Normalization",
            "Probe / Regressor",
            "MAE",
            "RMSE",
            "Median AE",
            "P95 AE",
            "Catastrophic error >20 bpm",
            "n_windows",
        ]
    ].sort_values("MAE")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Return a small GitHub-style markdown table without optional dependencies."""

    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    return "\n".join("| " + " | ".join(values) + " |" for values in rows)


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(dataframe_to_markdown(frame) + "\n", encoding="utf-8")


def build_inversion_table(overall: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "preprocessing_mode", "normalization_strategy", "probe_type", "regressor_type"]
    rows: list[dict[str, Any]] = []
    for key_values, group in overall.groupby(keys, dropna=False):
        inverted = group[group["inversion"] == True]  # noqa: E712
        raw = group[group["inversion"] == False]  # noqa: E712
        if inverted.empty or raw.empty:
            continue
        inv = inverted.iloc[0]
        noinv = raw.iloc[0]
        row = {column: value for column, value in zip(keys, key_values)}
        for metric in METRIC_COLUMNS:
            row[f"{metric}_with_inversion"] = inv[metric]
            row[f"{metric}_without_inversion"] = noinv[metric]
            row[f"delta_{metric}"] = noinv[metric] - inv[metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_MAE", ascending=False) if rows else pd.DataFrame()


def write_figures(
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    participant: pd.DataFrame,
    activity: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plot_frame = overall[overall["inversion"] == True].copy()  # noqa: E712
    if plot_frame.empty:
        return
    top_models = plot_frame.sort_values("MAE").head(8)

    activity_plot = activity[
        (activity["inversion"] == True)  # noqa: E712
        & (activity["model_name"].isin(top_models["model_name"]))
        & (activity["regressor_type"].isin(top_models["regressor_type"]))
    ].copy()
    if not activity_plot.empty:
        pivot = activity_plot.pivot_table(index="activity", columns="model_name", values="MAE", aggfunc="min")
        pivot.sort_index().plot(kind="bar", figsize=(12, 5))
        plt.ylabel("MAE (bpm)")
        plt.tight_layout()
        plt.savefig(figures_dir / "activity_level_mae_by_model.png", dpi=180)
        plt.close()

    participant_plot = participant[
        (participant["inversion"] == True)  # noqa: E712
        & (participant["model_name"].isin(top_models["model_name"]))
        & (participant["regressor_type"].isin(top_models["regressor_type"]))
    ].copy()
    if not participant_plot.empty:
        pivot = participant_plot.pivot_table(index="participant_id", columns="model_name", values="MAE", aggfunc="min")
        plt.figure(figsize=(12, 5))
        plt.imshow(pivot.fillna(np.nan).to_numpy(), aspect="auto", interpolation="nearest")
        plt.colorbar(label="MAE (bpm)")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.tight_layout()
        plt.savefig(figures_dir / "participant_level_mae_by_model.png", dpi=180)
        plt.close()

    pred_plot = predictions[
        (predictions["inversion"] == True)  # noqa: E712
        & (predictions["model_name"].isin(top_models["model_name"]))
        & (predictions["regressor_type"].isin(top_models["regressor_type"]))
    ].copy()
    if not pred_plot.empty:
        groups = [group["abs_error"].dropna().to_numpy() for _, group in pred_plot.groupby("model_name")]
        labels = [name for name, _ in pred_plot.groupby("model_name")]
        plt.figure(figsize=(10, 5))
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.ylabel("Absolute error (bpm)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "error_distribution_by_model.png", dpi=180)
        plt.close()

    inversion = build_inversion_table(overall)
    if not inversion.empty:
        plt.figure(figsize=(9, 4))
        labels = inversion["model_name"] + "/" + inversion["regressor_type"]
        plt.bar(labels, inversion["delta_MAE"])
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Delta MAE: non-inverted minus inverted")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "inversion_ablation_summary.png", dpi=180)
        plt.close()


def write_memo(
    output_root: Path,
    overall: pd.DataFrame,
    participant: pd.DataFrame,
    activity: pd.DataFrame,
    tag_name: str,
    all_predictions_path: Path,
) -> None:
    inverted = overall[overall["inversion"] == True].copy()  # noqa: E712
    best_mae = safe_best(inverted, "MAE")
    best_p95 = safe_best(inverted, "p95_absolute_error")
    best_cat = safe_best(inverted, "catastrophic_error_rate_20bpm")
    inversion_table = build_inversion_table(overall)
    branch, commit = git_state()
    native_note = "Native watch HR was unavailable in the processed GalaxyPPG cache used here; no native-HR result was fabricated."
    skipped = []
    observed = set((overall["model_name"] + ":" + overall["regressor_type"]).tolist())
    if not any(item.startswith("pulseppg:") for item in observed):
        skipped.append("PulsePPG runs are absent from the discovered result set.")
    if not any(item.startswith("papagei:") for item in observed):
        skipped.append("PaPaGei runs are absent from the discovered result set.")

    memo = f"""# Week 2 Memo: GalaxyPPG Corrected Benchmark

## 1. Objective

Week 2 evaluates GalaxyPPG only, using subject-independent participant splits, corrected loader-level PPG inversion, ECG/IBI-derived beat-interval labels, and the configured harmonized/model-faithful preprocessing modes. No PPG-DaLiA or WildPPG experiments are included.

## 2. Repository and pipeline status

Week 1 prerequisites were present before this benchmark: GalaxyPPG inversion is explicit in the loader, the `canonical_ppg_v1` schema exists, labels use 10-second windows with 2-second stride and median instantaneous HR from beat intervals, and `configs/experiment_modes.json` defines `harmonized` and `model_faithful` modes.

Outputs are organized under `{output_root.as_posix()}`. Standardized prediction CSVs are stored in `predictions/`, metrics in `metrics/`, benchmark tables in `tables/`, and figures in `figures/`.

## 3. Experimental design

- Dataset: GalaxyPPG
- Evaluation: subject-independent fixed held-out participant split with participant-level validation folds
- Window: 10 seconds, 2 seconds stride
- Label: median instantaneous HR from IBI/ECG beat intervals
- Metrics: MAE, RMSE, median absolute error, 95th percentile absolute error, catastrophic error rate above 20 bpm
- Prediction export: `{all_predictions_path.as_posix()}`

## 4. Methods compared

Completed runs discovered in the result tree:

{dataframe_to_markdown(overall[['model_name', 'preprocessing_mode', 'normalization_strategy', 'probe_type', 'regressor_type', 'inversion', 'n_windows']])}

{native_note}

## 5. Main benchmark results

Best by MAE: {best_mae}

Best by P95 AE: {best_p95}

Best by catastrophic error rate: {best_cat}

The paper-facing tables are:

- `tables/main_benchmark_table.csv`
- `tables/harmonized_preprocessing_table.csv`
- `tables/model_faithful_preprocessing_table.csv`

## 6. Inversion ablation

The inversion ablation table is `tables/inversion_ablation_table.csv`. Positive delta values mean the non-inverted raw signal was worse than the corrected inverted signal.

{dataframe_to_markdown(inversion_table) if not inversion_table.empty else 'No paired inversion ablation rows were available.'}

## 7. Preprocessing-mode comparison

The harmonized and model-faithful tables are stored separately. Interpret ranking changes only among completed runs with the same model/checkpoint availability.

## 8. Classical baseline regime analysis

Week 2 reports activity-level summaries only; deeper regime maps and oracle routing are intentionally left for Week 3. Activity-level metrics are stored in `metrics/activity_level_metrics.csv`.

## 9. PaPaGei assessment

PaPaGei should be retained as a benchmark when completed runs are available, but Week 2 treats it as a frozen-embedding comparator rather than a routing expert decision. Routing decisions are outside the Week 2 boundary.

## 10. Participant-level observations

Participant-level metrics are stored in `metrics/participant_level_metrics.csv`. These participant aggregates are the correct unit for later confidence intervals and paired significance tests.

## 11. Limitations

- {native_note}
- {' '.join(skipped) if skipped else 'No completed foundation-model run was missing from the discovered result set.'}
- Stronger regressors are reported only as practical upper bounds and should not be framed as the core novelty.
- The fixed split is subject-independent, but it is not leave-one-subject-out.

## 12. Conclusion

1. Did inversion materially change performance? Use `tables/inversion_ablation_table.csv`; positive deltas show the corrected inversion improved the metric.
2. Did model-faithful preprocessing change the ranking? Use the harmonized and model-faithful tables; ranking should be interpreted among matched completed runs.
3. Does the classical baseline remain competitive in any regimes? Use `metrics/activity_level_metrics.csv`; detailed regime analysis is Week 3.
4. Is PaPaGei still useful as a benchmark after the comparison is corrected? Yes, if completed frozen-embedding linear/Ridge runs are available; it should remain a benchmark comparator, not the main routing claim.

## 13. Reproducibility

- branch: `{branch}`
- commit hash: `{commit}`
- tag name: `{tag_name or 'not_created_by_builder'}`
- configs used: `configs/galaxyppg_submission_split.json`, `configs/experiment_modes.json`
- main output location: `{output_root.as_posix()}`
"""
    (output_root / "week2_memo.md").write_text(memo, encoding="utf-8")


def safe_best(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty or metric not in frame.columns:
        return "NA"
    row = frame.sort_values(metric, na_position="last").iloc[0]
    return (
        f"{row['model_name']} / {row['preprocessing_mode']} / {row['normalization_strategy']} / "
        f"{row['regressor_type']} ({metric}={row[metric]:.3f})"
    )


def git_state() -> tuple[str, str]:
    def run(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return run(["git", "branch", "--show-current"]), run(["git", "rev-parse", "HEAD"])


def sanitize_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
