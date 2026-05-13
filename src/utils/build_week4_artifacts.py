"""Build Week 4 lightweight routing artifacts from Week 3 regime features."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "median_absolute_error",
    "p95_absolute_error",
    "catastrophic_error_rate_20bpm",
]
MOTION_FEATURES = [
    "acc_norm_mean",
    "acc_norm_std",
    "acc_dominant_frequency_hz",
    "acc_cadence_band_power_fraction",
]
QUALITY_FEATURES = [
    "ppg_amplitude_range",
    "ppg_clipping_rate",
    "ppg_flatline_rate",
    "ppg_autocorr_peak_strength",
    "ppg_spectral_entropy",
    "ppg_spectral_peak_sharpness",
    "beat_count_consistency",
    "peak_spectral_disagreement_bpm",
]
FEATURE_SETS = {
    "motion_only": MOTION_FEATURES,
    "quality_only": QUALITY_FEATURES,
    "motion_quality": MOTION_FEATURES + QUALITY_FEATURES,
}


@dataclass(slots=True)
class FoldResult:
    """Out-of-fold gate predictions for one held-out participant."""

    heldout_participant: str
    n_train: int
    n_test: int
    positive_rate_train: float
    estimator_type: str
    coefficients: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week3-root",
        type=Path,
        default=Path("experiments/week3_galaxyppg_regime_oracle_2026-05-13"),
    )
    parser.add_argument(
        "--week2-predictions",
        type=Path,
        default=Path("experiments/week2_galaxyppg_corrected_2026-05-01/predictions/week2_all_standardized_predictions.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/week4_galaxyppg_lightweight_router_2026-05-13"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    for path in [predictions_dir, metrics_dir, tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    week3_path = args.week3_root / "predictions" / "week3_window_regime_expert_errors.csv"
    windows = pd.read_csv(week3_path, low_memory=False)
    windows = augment_quality_features(windows, args.week2_predictions)
    windows = windows.dropna(
        subset=["y_true_hr", "classical_pred_hr", "foundation_pred_hr", "participant_id"]
    ).reset_index(drop=True)
    if windows.empty:
        raise RuntimeError("No valid paired windows were available for Week 4 routing.")

    routed_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for feature_set_name, feature_columns in FEATURE_SETS.items():
        routed, fold_summary, coefficients = train_out_of_fold_router(
            windows=windows,
            feature_set_name=feature_set_name,
            feature_columns=feature_columns,
            random_state=args.random_state,
        )
        routed_frames.append(routed)
        fold_frames.append(fold_summary)
        coefficient_frames.append(coefficients)

    routed_predictions = pd.concat(routed_frames, ignore_index=True)
    routed_predictions.to_csv(predictions_dir / "week4_routed_predictions.csv", index=False)

    fold_table = pd.concat(fold_frames, ignore_index=True)
    fold_table.to_csv(tables_dir / "gate_fold_summary.csv", index=False)
    write_markdown_table(fold_table, tables_dir / "gate_fold_summary.md")

    coefficient_table = pd.concat(coefficient_frames, ignore_index=True)
    coefficient_table.to_csv(tables_dir / "gate_feature_coefficients.csv", index=False)
    write_markdown_table(coefficient_table, tables_dir / "gate_feature_coefficients.md")

    summary = summarize_methods(windows, routed_predictions)
    summary.to_csv(tables_dir / "routing_summary.csv", index=False)
    write_markdown_table(summary, tables_dir / "routing_summary.md")
    (metrics_dir / "routing_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    participant = summarize_by_group(windows, routed_predictions, ["participant_id"])
    participant.to_csv(tables_dir / "participant_level_routing_metrics.csv", index=False)
    write_markdown_table(participant, tables_dir / "participant_level_routing_metrics.md")

    activity = summarize_by_group(windows, routed_predictions, ["activity"])
    activity.to_csv(tables_dir / "activity_level_routing_metrics.csv", index=False)
    write_markdown_table(activity, tables_dir / "activity_level_routing_metrics.md")

    write_figures(windows, routed_predictions, summary, coefficient_table, figures_dir)
    write_memo(
        output_root=output_root,
        week3_root=args.week3_root,
        week2_predictions=args.week2_predictions,
        summary=summary,
        coefficient_table=coefficient_table,
    )

    print(f"routed_predictions={predictions_dir / 'week4_routed_predictions.csv'}")
    print(f"summary={tables_dir / 'routing_summary.csv'}")
    print(f"memo={output_root / 'week4_lightweight_router.md'}")


def augment_quality_features(windows: pd.DataFrame, week2_predictions_path: Path) -> pd.DataFrame:
    result = windows.copy()
    if "beat_count_consistency" in result.columns:
        return result
    week2 = pd.read_csv(week2_predictions_path, low_memory=False)
    week2["window_key"] = make_window_key(
        week2["participant_id"],
        week2["session_id"],
        week2["window_start_time"],
        week2["window_end_time"],
    )
    peak = week2[
        (week2["model_name"] == "peak_based")
        & (week2["inversion"] == True)  # noqa: E712
        & (week2["preprocessing_mode"] == "harmonized")
    ][["window_key", "peak_count", "estimated_peak_hr"]].copy()
    spectral = week2[
        (week2["model_name"] == "spectral")
        & (week2["inversion"] == True)  # noqa: E712
        & (week2["preprocessing_mode"] == "harmonized")
    ][["window_key", "estimated_spectral_hr"]].copy()
    aux = peak.merge(spectral, on="window_key", how="outer")
    aux["peak_count"] = pd.to_numeric(aux["peak_count"], errors="coerce")
    aux["estimated_spectral_hr"] = pd.to_numeric(aux["estimated_spectral_hr"], errors="coerce")
    expected_beats = aux["estimated_spectral_hr"] * 10.0 / 60.0
    aux["beat_count_consistency"] = (aux["peak_count"] - expected_beats).abs()
    aux["peak_spectral_disagreement_bpm_from_week2"] = (
        pd.to_numeric(aux["estimated_peak_hr"], errors="coerce") - aux["estimated_spectral_hr"]
    ).abs()
    result = result.merge(
        aux[["window_key", "beat_count_consistency", "peak_spectral_disagreement_bpm_from_week2"]],
        on="window_key",
        how="left",
    )
    if "peak_spectral_disagreement_bpm" not in result.columns:
        result["peak_spectral_disagreement_bpm"] = result["peak_spectral_disagreement_bpm_from_week2"]
    else:
        result["peak_spectral_disagreement_bpm"] = result["peak_spectral_disagreement_bpm"].fillna(
            result["peak_spectral_disagreement_bpm_from_week2"]
        )
    result = result.drop(columns=["peak_spectral_disagreement_bpm_from_week2"])
    return result


def make_window_key(
    participant_id: pd.Series,
    session_id: pd.Series,
    window_start: pd.Series,
    window_end: pd.Series,
) -> pd.Series:
    start = pd.to_numeric(window_start, errors="coerce").round().astype("Int64").astype(str)
    end = pd.to_numeric(window_end, errors="coerce").round().astype("Int64").astype(str)
    return participant_id.astype(str) + "|" + session_id.astype(str) + "|" + start + "|" + end


def train_out_of_fold_router(
    windows: pd.DataFrame,
    feature_set_name: str,
    feature_columns: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = windows.copy()
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = (frame["foundation_abs_error"] < frame["classical_abs_error"]).astype(int).to_numpy()
    groups = frame["participant_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    hard_pred = np.full(len(frame), np.nan, dtype=float)
    soft_weight = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
        heldout = sorted(set(groups[test_idx]))
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            positive_rate = float(np.mean(y_train))
            probability = np.full(len(test_idx), positive_rate, dtype=float)
            fold_result = FoldResult(
                heldout_participant=",".join(heldout),
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                positive_rate_train=positive_rate,
                estimator_type="majority_fallback",
                coefficients={column: math.nan for column in feature_columns},
            )
        else:
            estimator = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "gate",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            estimator.fit(x.iloc[train_idx], y_train)
            probability = estimator.predict_proba(x.iloc[test_idx])[:, 1]
            gate = estimator.named_steps["gate"]
            fold_result = FoldResult(
                heldout_participant=",".join(heldout),
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                positive_rate_train=float(np.mean(y_train)),
                estimator_type="logistic_regression",
                coefficients={column: float(value) for column, value in zip(feature_columns, gate.coef_[0])},
            )

        soft_weight[test_idx] = probability
        hard_pred[test_idx] = (probability >= 0.5).astype(float)
        fold_row = {
            "feature_set": feature_set_name,
            "fold": fold_index,
            "heldout_participant": fold_result.heldout_participant,
            "n_train": fold_result.n_train,
            "n_test": fold_result.n_test,
            "positive_rate_train": fold_result.positive_rate_train,
            "estimator_type": fold_result.estimator_type,
        }
        fold_rows.append(fold_row)
        for feature_name, coefficient in fold_result.coefficients.items():
            coefficient_rows.append(
                {
                    "feature_set": feature_set_name,
                    "fold": fold_index,
                    "heldout_participant": fold_result.heldout_participant,
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient) if not math.isnan(coefficient) else math.nan,
                }
            )

    routed = build_routed_prediction_rows(
        frame=frame,
        feature_set_name=feature_set_name,
        hard_choose_foundation=hard_pred.astype(bool),
        soft_foundation_weight=soft_weight,
    )
    return routed, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def build_routed_prediction_rows(
    frame: pd.DataFrame,
    feature_set_name: str,
    hard_choose_foundation: np.ndarray,
    soft_foundation_weight: np.ndarray,
) -> pd.DataFrame:
    common = frame[
        [
            "window_key",
            "dataset",
            "participant_id",
            "session_id",
            "activity",
            "window_id",
            "window_start_time",
            "window_end_time",
            "y_true_hr",
            "classical_pred_hr",
            "foundation_pred_hr",
            "winner",
        ]
    ].copy()

    hard = common.copy()
    hard["feature_set"] = feature_set_name
    hard["routing_type"] = "hard_gate"
    hard["foundation_weight"] = hard_choose_foundation.astype(float)
    hard["selected_expert"] = np.where(hard_choose_foundation, "foundation", "classical")
    hard["routed_pred_hr"] = np.where(
        hard_choose_foundation,
        hard["foundation_pred_hr"],
        hard["classical_pred_hr"],
    )

    soft = common.copy()
    soft["feature_set"] = feature_set_name
    soft["routing_type"] = "soft_gate"
    soft["foundation_weight"] = soft_foundation_weight
    soft["selected_expert"] = "weighted_average"
    soft["routed_pred_hr"] = (
        soft_foundation_weight * soft["foundation_pred_hr"]
        + (1.0 - soft_foundation_weight) * soft["classical_pred_hr"]
    )

    routed = pd.concat([hard, soft], ignore_index=True)
    routed["routed_abs_error"] = (routed["routed_pred_hr"] - routed["y_true_hr"]).abs()
    routed["routed_squared_error"] = np.square(routed["routed_pred_hr"] - routed["y_true_hr"])
    routed["gate_correct"] = np.where(
        routed["routing_type"] == "hard_gate",
        routed["selected_expert"] == routed["winner"],
        pd.NA,
    )
    return routed


def summarize_methods(windows: pd.DataFrame, routed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, pred_col in [
        ("classical_expert", "classical_pred_hr"),
        ("foundation_expert", "foundation_pred_hr"),
        ("oracle_router", None),
    ]:
        if method == "oracle_router":
            pred = np.where(
                windows["classical_abs_error"] <= windows["foundation_abs_error"],
                windows["classical_pred_hr"],
                windows["foundation_pred_hr"],
            )
        else:
            pred = windows[pred_col]
        row = {
            "method": method,
            "feature_set": "NA",
            "routing_type": "NA",
            "n_windows": int(len(windows)),
            "gate_accuracy": math.nan,
            "mean_foundation_weight": math.nan,
        }
        row.update(compute_metrics(windows["y_true_hr"], pred))
        rows.append(row)

    for keys, group in routed.groupby(["feature_set", "routing_type"], dropna=False):
        feature_set, routing_type = keys
        row = {
            "method": "learned_router",
            "feature_set": feature_set,
            "routing_type": routing_type,
            "n_windows": int(len(group)),
            "gate_accuracy": float(group["gate_correct"].dropna().mean()) if routing_type == "hard_gate" else math.nan,
            "mean_foundation_weight": float(group["foundation_weight"].mean()),
        }
        row.update(compute_metrics(group["y_true_hr"], group["routed_pred_hr"]))
        rows.append(row)

    summary = pd.DataFrame(rows)
    best_single_mae = summary[summary["method"].isin(["classical_expert", "foundation_expert"])]["MAE"].min()
    best_single_p95 = summary[summary["method"].isin(["classical_expert", "foundation_expert"])][
        "p95_absolute_error"
    ].min()
    best_single_cat = summary[summary["method"].isin(["classical_expert", "foundation_expert"])][
        "catastrophic_error_rate_20bpm"
    ].min()
    oracle_mae = float(summary.loc[summary["method"] == "oracle_router", "MAE"].iloc[0])
    oracle_gain = best_single_mae - oracle_mae
    summary["gain_vs_best_single_MAE"] = best_single_mae - summary["MAE"]
    summary["gain_vs_best_single_p95"] = best_single_p95 - summary["p95_absolute_error"]
    summary["gain_vs_best_single_catastrophic_rate"] = best_single_cat - summary[
        "catastrophic_error_rate_20bpm"
    ]
    summary["oracle_gain_recovered_MAE"] = np.where(
        oracle_gain > 0,
        summary["gain_vs_best_single_MAE"] / oracle_gain,
        math.nan,
    )
    return summary.sort_values(["method", "MAE"], na_position="last").reset_index(drop=True)


def summarize_by_group(windows: pd.DataFrame, routed: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in windows.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        for method, pred_col in [
            ("classical_expert", "classical_pred_hr"),
            ("foundation_expert", "foundation_pred_hr"),
        ]:
            row = {column: value for column, value in zip(group_columns, key_values)}
            row.update({"method": method, "feature_set": "NA", "routing_type": "NA", "n_windows": int(len(group))})
            row.update(compute_metrics(group["y_true_hr"], group[pred_col]))
            rows.append(row)
    for keys, group in routed.groupby(group_columns + ["feature_set", "routing_type"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_columns + ["feature_set", "routing_type"], key_values)}
        row.update({"method": "learned_router", "n_windows": int(len(group))})
        row.update(compute_metrics(group["y_true_hr"], group["routed_pred_hr"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns + ["MAE"]).reset_index(drop=True)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(true) | np.isnan(pred))
    true = true[valid]
    pred = pred[valid]
    if len(true) == 0:
        return {metric: math.nan for metric in METRIC_COLUMNS}
    abs_error = np.abs(pred - true)
    return {
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(np.square(pred - true)))),
        "median_absolute_error": float(np.median(abs_error)),
        "p95_absolute_error": float(np.percentile(abs_error, 95)),
        "catastrophic_error_rate_20bpm": float(np.mean(abs_error > 20.0)),
    }


def write_figures(
    windows: pd.DataFrame,
    routed: pd.DataFrame,
    summary: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    figures_dir: Path,
) -> None:
    plot_rows = summary[
        (summary["method"].isin(["classical_expert", "foundation_expert", "oracle_router"]))
        | ((summary["method"] == "learned_router") & (summary["routing_type"].isin(["hard_gate", "soft_gate"])))
    ].copy()
    plot_rows["label"] = np.where(
        plot_rows["method"] == "learned_router",
        plot_rows["feature_set"] + "/" + plot_rows["routing_type"],
        plot_rows["method"],
    )
    plt.figure(figsize=(11, 4.8))
    plt.bar(plot_rows["label"], plot_rows["MAE"])
    plt.ylabel("MAE (bpm)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "hard_soft_router_mae.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 4.8))
    best_soft = summary[(summary["method"] == "learned_router") & (summary["routing_type"] == "soft_gate")]
    if not best_soft.empty:
        best = best_soft.sort_values("MAE").iloc[0]
        routed_best = routed[
            (routed["feature_set"] == best["feature_set"]) & (routed["routing_type"] == best["routing_type"])
        ]
        curves = {
            "classical": windows["classical_abs_error"].to_numpy(dtype=float),
            "foundation": windows["foundation_abs_error"].to_numpy(dtype=float),
            "best_soft_router": routed_best["routed_abs_error"].to_numpy(dtype=float),
        }
        for label, values in curves.items():
            values = np.sort(values[np.isfinite(values)])
            y = np.arange(1, len(values) + 1) / len(values)
            plt.plot(values, y, label=label)
        plt.xlim(0, np.nanpercentile(windows[["classical_abs_error", "foundation_abs_error"]].to_numpy(), 98))
        plt.xlabel("Absolute error (bpm)")
        plt.ylabel("Cumulative share")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "best_router_error_cdf.png", dpi=180)
    plt.close()

    combined = coefficient_table[coefficient_table["feature_set"] == "motion_quality"].copy()
    if not combined.empty:
        coef = (
            combined.groupby("feature", as_index=False)["coefficient"]
            .mean()
            .assign(abs_mean=lambda x: x["coefficient"].abs())
            .sort_values("abs_mean", ascending=True)
        )
        plt.figure(figsize=(8, 5))
        plt.barh(coef["feature"], coef["coefficient"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Mean standardized logistic coefficient")
        plt.tight_layout()
        plt.savefig(figures_dir / "combined_gate_feature_coefficients.png", dpi=180)
        plt.close()


def write_memo(
    output_root: Path,
    week3_root: Path,
    week2_predictions: Path,
    summary: pd.DataFrame,
    coefficient_table: pd.DataFrame,
) -> None:
    best_router = summary[summary["method"] == "learned_router"].sort_values("MAE").iloc[0]
    best_hard = summary[
        (summary["method"] == "learned_router") & (summary["routing_type"] == "hard_gate")
    ].sort_values("MAE").iloc[0]
    best_soft = summary[
        (summary["method"] == "learned_router") & (summary["routing_type"] == "soft_gate")
    ].sort_values("MAE").iloc[0]
    branch, commit = git_state()
    top_coefficients = (
        coefficient_table[coefficient_table["feature_set"] == "motion_quality"]
        .groupby("feature", as_index=False)["abs_coefficient"]
        .mean()
        .sort_values("abs_coefficient", ascending=False)
        .head(8)
    )
    memo = f"""# Week 4 Memo: Lightweight Motion- and Quality-Aware Routing

## Objective

Week 4 implements the first deployable-style lightweight router between the selected classical expert and PulsePPG-style foundation expert from Week 3. The gate uses only inference-time motion and PPG-quality features, and is evaluated with participant-level leave-one-participant-out routing predictions across the available corrected GalaxyPPG held-out participants.

## Inputs

- Week 3 paired window features: `{(week3_root / 'predictions' / 'week3_window_regime_expert_errors.csv').as_posix()}`
- Week 2 standardized predictions for peak/spectral auxiliary quality features: `{week2_predictions.as_posix()}`
- Output root: `{output_root.as_posix()}`

## Gate Variants

- `motion_only`: accelerometer norm mean/std, dominant accelerometer frequency, cadence-band power fraction
- `quality_only`: PPG amplitude range, clipping rate, flatline rate, autocorrelation peak strength, spectral entropy, spectral peak sharpness, beat-count consistency, peak-vs-spectral HR disagreement
- `motion_quality`: all motion and quality features combined

For each feature set, the builder trains:

- `hard_gate`: logistic gate chooses either the classical or foundation expert.
- `soft_gate`: logistic probability is used as the foundation weight, and predictions are averaged.

## Main Result

Best learned router by MAE:

{dataframe_to_markdown(pd.DataFrame([best_router]))}

Best hard gate:

{dataframe_to_markdown(pd.DataFrame([best_hard]))}

Best soft gate:

{dataframe_to_markdown(pd.DataFrame([best_soft]))}

Full routing summary:

{dataframe_to_markdown(summary)}

## Gate Interpretation

Largest mean absolute standardized coefficients for the combined gate:

{dataframe_to_markdown(top_coefficients)}

Positive coefficients push the hard gate toward the foundation expert. Negative coefficients push it toward the classical expert.

## Expert Decision

The Week 3 oracle table showed the main routing pair should remain `peak_based` plus `PulsePPG` for the first routed system. PaPaGei remains useful as a benchmark, but the current routed implementation keeps the paper's main system focused on one classical expert and one PulsePPG-based expert.

## Artifacts

- Routed predictions: `predictions/week4_routed_predictions.csv`
- Main comparison table: `tables/routing_summary.csv`
- Gate fold summary: `tables/gate_fold_summary.csv`
- Gate coefficients: `tables/gate_feature_coefficients.csv`
- Participant-level routing metrics: `tables/participant_level_routing_metrics.csv`
- Activity-level routing metrics: `tables/activity_level_routing_metrics.csv`
- Figures: `figures/hard_soft_router_mae.png`, `figures/best_router_error_cdf.png`, `figures/combined_gate_feature_coefficients.png`

## Limitations

This is a first GalaxyPPG router trained and evaluated only across the Week 2 held-out participants using participant-level out-of-fold gates. It is suitable for deciding whether hard/soft lightweight routing is promising, but final claims still require the later external-validation weeks.

## Reproducibility

- branch: `{branch}`
- commit hash: `{commit}`
- command: `python -m src.utils.build_week4_artifacts --week3-root {week3_root.as_posix()} --output-root {output_root.as_posix()}`
"""
    (output_root / "week4_lightweight_router.md").write_text(memo, encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
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


def git_state() -> tuple[str, str]:
    def run(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return run(["git", "branch", "--show-current"]), run(["git", "rev-parse", "HEAD"])


if __name__ == "__main__":
    main()
