"""Build Week 3 GalaxyPPG regime and oracle-routing artifacts."""

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


IDENTITY_COLUMNS = [
    "model_name",
    "preprocessing_mode",
    "normalization_strategy",
    "probe_type",
    "regressor_type",
    "inversion",
]
METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "median_absolute_error",
    "p95_absolute_error",
    "catastrophic_error_rate_20bpm",
]


@dataclass(slots=True)
class ExpertSpec:
    """A concrete completed run selected as an expert."""

    model_name: str
    preprocessing_mode: str
    normalization_strategy: str
    probe_type: str
    regressor_type: str
    inversion: bool

    @classmethod
    def from_row(cls, row: pd.Series) -> "ExpertSpec":
        return cls(
            model_name=str(row["model_name"]),
            preprocessing_mode=str(row["preprocessing_mode"]),
            normalization_strategy=normalize_text(row["normalization_strategy"]),
            probe_type=normalize_text(row["probe_type"]),
            regressor_type=str(row["regressor_type"]),
            inversion=bool(row["inversion"]),
        )

    def label(self) -> str:
        return (
            f"{self.model_name}/{self.preprocessing_mode}/"
            f"{self.normalization_strategy}/{self.regressor_type}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week2-root",
        type=Path,
        default=Path("experiments/week2_galaxyppg_corrected_2026-05-01"),
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Defaults to <week2-root>/predictions/week2_all_standardized_predictions.csv.",
    )
    parser.add_argument(
        "--processed-windows",
        type=Path,
        default=Path("data/processed/windows/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_windows.jsonl.gz"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/GalaxyPPG"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/week3_galaxyppg_regime_oracle_2026-05-13"),
    )
    parser.add_argument("--classical-model", choices=["peak_based", "spectral"], default=None)
    parser.add_argument("--foundation-model", choices=["pulseppg", "papagei"], default=None)
    parser.add_argument("--foundation-regressor", default=None)
    parser.add_argument("--foundation-preprocessing-mode", default=None)
    parser.add_argument("--foundation-normalization", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions_path or (
        args.week2_root / "predictions" / "week2_all_standardized_predictions.csv"
    )
    output_root = args.output_root
    tables_dir = output_root / "tables"
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    predictions_dir = output_root / "predictions"
    for path in [tables_dir, metrics_dir, figures_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path)
    run_metrics = aggregate_run_metrics(predictions)
    classical_spec = select_classical_expert(run_metrics, args.classical_model)
    foundation_spec = select_foundation_expert(
        run_metrics=run_metrics,
        foundation_model=args.foundation_model,
        regressor=args.foundation_regressor,
        preprocessing_mode=args.foundation_preprocessing_mode,
        normalization=args.foundation_normalization,
    )

    classical = extract_expert_predictions(predictions, classical_spec, "classical")
    foundation = extract_expert_predictions(predictions, foundation_spec, "foundation")
    paired = pair_experts(classical, foundation)
    if paired.empty:
        raise RuntimeError("Selected experts do not share any comparable windows.")

    feature_frame, feature_notes = build_regime_features(
        processed_windows_path=args.processed_windows,
        dataset_root=args.dataset_root,
        keys=paired[["window_key", "participant_id", "session_id", "window_start_time", "window_end_time"]],
        predictions=predictions,
    )
    analysis = paired.merge(feature_frame, on="window_key", how="left", suffixes=("", "_feature"))
    analysis = add_regime_bins(analysis)
    oracle = build_oracle_predictions(analysis)

    analysis.to_csv(predictions_dir / "week3_window_regime_expert_errors.csv", index=False)
    oracle.to_csv(predictions_dir / "week3_oracle_router_predictions.csv", index=False)

    selected_summary = summarize_selected_pair(analysis, classical_spec, foundation_spec)
    selected_summary.to_csv(tables_dir / "selected_expert_oracle_summary.csv", index=False)
    write_markdown_table(selected_summary, tables_dir / "selected_expert_oracle_summary.md")

    all_pairs = build_all_pair_oracle_table(predictions)
    all_pairs.to_csv(tables_dir / "oracle_all_pairs_table.csv", index=False)
    write_markdown_table(all_pairs, tables_dir / "oracle_all_pairs_table.md")

    regime_tables = {
        "regime_by_activity": summarize_regime(analysis, ["activity"]),
        "regime_by_participant": summarize_regime(analysis, ["participant_id"]),
        "regime_by_hr_range": summarize_regime(analysis, ["hr_range"]),
        "regime_by_motion_intensity": summarize_regime(analysis, ["motion_intensity_bin"]),
        "regime_by_ppg_quality": summarize_regime(analysis, ["ppg_autocorr_quality_bin"]),
        "regime_by_motion_and_quality": summarize_regime(
            analysis,
            ["motion_intensity_bin", "ppg_autocorr_quality_bin"],
        ),
    }
    for name, frame in regime_tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
        write_markdown_table(frame, tables_dir / f"{name}.md")

    write_metric_json(selected_summary, metrics_dir / "selected_expert_oracle_summary.json")
    write_figures(analysis, selected_summary, regime_tables, figures_dir)
    write_memo(
        output_root=output_root,
        predictions_path=predictions_path,
        processed_windows_path=args.processed_windows,
        classical_spec=classical_spec,
        foundation_spec=foundation_spec,
        selected_summary=selected_summary,
        all_pairs=all_pairs,
        regime_tables=regime_tables,
        feature_notes=feature_notes,
    )

    print(f"selected_classical={classical_spec.label()}")
    print(f"selected_foundation={foundation_spec.label()}")
    print(f"paired_windows={len(analysis)}")
    print(f"output_root={output_root}")


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    for column in ["normalization_strategy", "probe_type", "regressor_type", "preprocessing_mode", "model_name"]:
        frame[column] = frame[column].map(normalize_text)
    frame["inversion"] = frame["inversion"].astype(bool)
    frame["window_key"] = make_window_key(
        frame["participant_id"],
        frame["session_id"],
        frame["window_start_time"],
        frame["window_end_time"],
    )
    return frame


def make_window_key(
    participant_id: pd.Series,
    session_id: pd.Series,
    window_start: pd.Series,
    window_end: pd.Series,
) -> pd.Series:
    start = pd.to_numeric(window_start, errors="coerce").round().astype("Int64").astype(str)
    end = pd.to_numeric(window_end, errors="coerce").round().astype("Int64").astype(str)
    return participant_id.astype(str) + "|" + session_id.astype(str) + "|" + start + "|" + end


def normalize_text(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    text = str(value)
    return "NA" if text.lower() in {"nan", "none", ""} else text


def aggregate_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(IDENTITY_COLUMNS, dropna=False):
        valid = group.dropna(subset=["y_true_hr", "y_pred_hr"])
        row = {column: value for column, value in zip(IDENTITY_COLUMNS, keys)}
        row.update(compute_metrics(valid["y_true_hr"], valid["y_pred_hr"]))
        row["n_windows"] = int(len(valid))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE", na_position="last").reset_index(drop=True)


def select_classical_expert(run_metrics: pd.DataFrame, model_name: str | None) -> ExpertSpec:
    candidates = run_metrics[
        (run_metrics["inversion"] == True)  # noqa: E712
        & (run_metrics["model_name"].isin(["peak_based", "spectral"]))
    ].copy()
    if model_name is not None:
        candidates = candidates[candidates["model_name"] == model_name]
    if candidates.empty:
        raise RuntimeError("No inverted classical expert run was found.")
    return ExpertSpec.from_row(candidates.sort_values("MAE").iloc[0])


def select_foundation_expert(
    run_metrics: pd.DataFrame,
    foundation_model: str | None,
    regressor: str | None,
    preprocessing_mode: str | None,
    normalization: str | None,
) -> ExpertSpec:
    candidates = run_metrics[
        (run_metrics["inversion"] == True)  # noqa: E712
        & (run_metrics["model_name"].isin(["pulseppg", "papagei"]))
    ].copy()
    if foundation_model is not None:
        candidates = candidates[candidates["model_name"] == foundation_model]
    if regressor is not None:
        candidates = candidates[candidates["regressor_type"] == regressor]
    if preprocessing_mode is not None:
        candidates = candidates[candidates["preprocessing_mode"] == preprocessing_mode]
    if normalization is not None:
        candidates = candidates[candidates["normalization_strategy"] == normalization]
    if candidates.empty:
        raise RuntimeError("No inverted foundation-model expert run matched the requested filters.")
    return ExpertSpec.from_row(candidates.sort_values("MAE").iloc[0])


def extract_expert_predictions(predictions: pd.DataFrame, spec: ExpertSpec, prefix: str) -> pd.DataFrame:
    mask = (
        (predictions["model_name"] == spec.model_name)
        & (predictions["preprocessing_mode"] == spec.preprocessing_mode)
        & (predictions["normalization_strategy"] == spec.normalization_strategy)
        & (predictions["probe_type"] == spec.probe_type)
        & (predictions["regressor_type"] == spec.regressor_type)
        & (predictions["inversion"] == spec.inversion)
    )
    frame = predictions.loc[mask].copy()
    if frame.empty:
        raise RuntimeError(f"No predictions found for {spec.label()}.")
    columns = [
        "window_key",
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
    ]
    frame = frame[columns].rename(
        columns={
            "y_pred_hr": f"{prefix}_pred_hr",
            "abs_error": f"{prefix}_abs_error",
            "squared_error": f"{prefix}_squared_error",
        }
    )
    return frame


def pair_experts(classical: pd.DataFrame, foundation: pd.DataFrame) -> pd.DataFrame:
    paired = classical.merge(
        foundation[
            [
                "window_key",
                "foundation_pred_hr",
                "foundation_abs_error",
                "foundation_squared_error",
            ]
        ],
        on="window_key",
        how="inner",
    )
    paired = paired.dropna(
        subset=[
            "y_true_hr",
            "classical_pred_hr",
            "foundation_pred_hr",
            "classical_abs_error",
            "foundation_abs_error",
        ]
    ).copy()
    paired["error_gap_classical_minus_foundation"] = (
        paired["classical_abs_error"] - paired["foundation_abs_error"]
    )
    paired["winner"] = np.select(
        [
            paired["classical_abs_error"] < paired["foundation_abs_error"],
            paired["foundation_abs_error"] < paired["classical_abs_error"],
        ],
        ["classical", "foundation"],
        default="tie",
    )
    return paired


def build_regime_features(
    processed_windows_path: Path,
    dataset_root: Path,
    keys: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    processed = pd.read_json(processed_windows_path, lines=True)
    processed["window_key"] = make_window_key(
        processed["participant_id"],
        processed["session_id"],
        processed["window_start_ms"],
        processed["window_end_ms"],
    )
    needed = set(keys["window_key"].astype(str))
    processed = processed[processed["window_key"].isin(needed)].copy()

    ppg_features = processed.apply(compute_ppg_quality_features, axis=1, result_type="expand")
    base = pd.concat(
        [
            processed[
                [
                    "window_key",
                    "participant_id",
                    "session_id",
                    "session_name",
                    "window_start_ms",
                    "window_end_ms",
                    "valid_beat_count",
                    "ppg_sample_count",
                    "label_hr_bpm",
                ]
            ].reset_index(drop=True),
            ppg_features.reset_index(drop=True),
        ],
        axis=1,
    )

    acc_features, acc_note = compute_accelerometer_features(dataset_root, keys)
    if acc_note:
        notes.append(acc_note)
    base = base.merge(acc_features, on="window_key", how="left")

    disagreement = build_peak_spectral_disagreement(predictions)
    base = base.merge(disagreement, on="window_key", how="left")
    return base, notes


def compute_ppg_quality_features(row: pd.Series) -> dict[str, float]:
    values = safe_float_array(row.get("ppg_values", []))
    timestamps = safe_float_array(row.get("ppg_timestamps_ms", []))
    if len(values) == 0:
        return {
            "ppg_amplitude_range": math.nan,
            "ppg_std": math.nan,
            "ppg_flatline_rate": math.nan,
            "ppg_clipping_rate": math.nan,
            "ppg_autocorr_peak_strength": math.nan,
            "ppg_spectral_entropy": math.nan,
            "ppg_spectral_peak_sharpness": math.nan,
            "ppg_dominant_frequency_hz": math.nan,
        }

    values = values[np.isfinite(values)]
    if len(values) == 0:
        return compute_ppg_quality_features(pd.Series({"ppg_values": [], "ppg_timestamps_ms": []}))
    centered = values - np.nanmean(values)
    diffs = np.diff(values)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    clipping_rate = float(np.mean((values == min_value) | (values == max_value))) if len(values) else math.nan
    flatline_rate = float(np.mean(np.abs(diffs) <= 1e-9)) if len(diffs) else math.nan
    sampling_hz = estimate_sampling_hz(timestamps)
    autocorr = autocorr_peak_strength(centered, sampling_hz)
    spectral = spectral_quality(centered, sampling_hz)
    return {
        "ppg_amplitude_range": float(np.nanmax(values) - np.nanmin(values)),
        "ppg_std": float(np.nanstd(values)),
        "ppg_flatline_rate": flatline_rate,
        "ppg_clipping_rate": clipping_rate,
        "ppg_autocorr_peak_strength": autocorr,
        "ppg_spectral_entropy": spectral["entropy"],
        "ppg_spectral_peak_sharpness": spectral["peak_sharpness"],
        "ppg_dominant_frequency_hz": spectral["dominant_frequency_hz"],
    }


def safe_float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, list):
        arr = np.asarray(value)
    else:
        return np.asarray([], dtype=float)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)


def estimate_sampling_hz(timestamps_ms: np.ndarray) -> float | None:
    timestamps_ms = timestamps_ms[np.isfinite(timestamps_ms)]
    if len(timestamps_ms) < 2:
        return None
    diffs = np.diff(timestamps_ms)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return 1000.0 / median_dt


def autocorr_peak_strength(values: np.ndarray, sampling_hz: float | None) -> float:
    if sampling_hz is None or len(values) < 8:
        return math.nan
    denom = float(np.dot(values, values))
    if denom <= 0:
        return math.nan
    min_lag = max(1, int(round(sampling_hz * 60.0 / 180.0)))
    max_lag = min(len(values) - 1, int(round(sampling_hz * 60.0 / 40.0)))
    if max_lag <= min_lag:
        return math.nan
    scores = []
    for lag in range(min_lag, max_lag + 1):
        scores.append(float(np.dot(values[:-lag], values[lag:]) / denom))
    return float(np.nanmax(scores)) if scores else math.nan


def spectral_quality(values: np.ndarray, sampling_hz: float | None) -> dict[str, float]:
    if sampling_hz is None or len(values) < 8 or np.nanstd(values) == 0:
        return {"entropy": math.nan, "peak_sharpness": math.nan, "dominant_frequency_hz": math.nan}
    window = np.hanning(len(values))
    spectrum = np.fft.rfft(values * window)
    freqs = np.fft.rfftfreq(len(values), d=1.0 / sampling_hz)
    power = np.abs(spectrum) ** 2
    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]
    if len(power) == 0 or np.nansum(power) <= 0:
        return {"entropy": math.nan, "peak_sharpness": math.nan, "dominant_frequency_hz": math.nan}
    prob = power / np.nansum(power)
    entropy = -float(np.nansum(prob * np.log(prob + 1e-12)) / np.log(len(prob)))
    dominant_index = int(np.nanargmax(power))
    median_power = float(np.nanmedian(power))
    peak_sharpness = float(power[dominant_index] / (median_power + 1e-12))
    return {
        "entropy": entropy,
        "peak_sharpness": peak_sharpness,
        "dominant_frequency_hz": float(freqs[dominant_index]),
    }


def compute_accelerometer_features(dataset_root: Path, keys: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if not dataset_root.exists():
        return pd.DataFrame({"window_key": keys["window_key"]}), f"Dataset root missing: {dataset_root}"

    for participant_id, group in keys.groupby("participant_id"):
        acc_path = dataset_root / str(participant_id) / "GalaxyWatch" / "ACC.csv"
        if not acc_path.exists():
            missing.append(str(participant_id))
            for window in group.itertuples(index=False):
                rows.append(empty_acc_features(window.window_key))
            continue
        acc = pd.read_csv(acc_path, usecols=["timestamp", "x", "y", "z"])
        acc = acc.rename(columns={"timestamp": "timestamp_ms", "x": "acc_x", "y": "acc_y", "z": "acc_z"})
        for column in ["timestamp_ms", "acc_x", "acc_y", "acc_z"]:
            acc[column] = pd.to_numeric(acc[column], errors="coerce")
        acc = acc.dropna(subset=["timestamp_ms", "acc_x", "acc_y", "acc_z"]).sort_values("timestamp_ms")
        timestamps = acc["timestamp_ms"].to_numpy(dtype=float)
        norm = np.sqrt(
            np.square(acc["acc_x"].to_numpy(dtype=float))
            + np.square(acc["acc_y"].to_numpy(dtype=float))
            + np.square(acc["acc_z"].to_numpy(dtype=float))
        )
        for window in group.itertuples(index=False):
            start = float(window.window_start_time)
            end = float(window.window_end_time)
            left = int(np.searchsorted(timestamps, start, side="left"))
            right = int(np.searchsorted(timestamps, end, side="left"))
            rows.append(compute_single_acc_window(window.window_key, timestamps[left:right], norm[left:right]))
    note = None
    if missing:
        note = f"Missing Galaxy Watch ACC.csv for participants: {', '.join(sorted(set(missing)))}"
    return pd.DataFrame(rows), note


def empty_acc_features(window_key: str) -> dict[str, Any]:
    return {
        "window_key": window_key,
        "acc_sample_count": 0,
        "acc_norm_mean": math.nan,
        "acc_norm_std": math.nan,
        "acc_dominant_frequency_hz": math.nan,
        "acc_cadence_band_power_fraction": math.nan,
    }


def compute_single_acc_window(window_key: str, timestamps: np.ndarray, norm: np.ndarray) -> dict[str, Any]:
    if len(norm) == 0:
        return empty_acc_features(window_key)
    centered = norm - np.nanmean(norm)
    sampling_hz = estimate_sampling_hz(timestamps)
    dominant_frequency = math.nan
    cadence_fraction = math.nan
    if sampling_hz is not None and len(centered) >= 8 and np.nanstd(centered) > 0:
        window = np.hanning(len(centered))
        spectrum = np.fft.rfft(centered * window)
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / sampling_hz)
        power = np.abs(spectrum) ** 2
        nonzero = freqs > 0
        if np.any(nonzero) and np.nansum(power[nonzero]) > 0:
            dominant_frequency = float(freqs[nonzero][np.nanargmax(power[nonzero])])
            cadence = (freqs >= 0.5) & (freqs <= 3.0)
            cadence_fraction = float(np.nansum(power[cadence]) / np.nansum(power[nonzero]))
    return {
        "window_key": window_key,
        "acc_sample_count": int(len(norm)),
        "acc_norm_mean": float(np.nanmean(norm)),
        "acc_norm_std": float(np.nanstd(norm)),
        "acc_dominant_frequency_hz": dominant_frequency,
        "acc_cadence_band_power_fraction": cadence_fraction,
    }


def build_peak_spectral_disagreement(predictions: pd.DataFrame) -> pd.DataFrame:
    source = predictions[
        (predictions["inversion"] == True)  # noqa: E712
        & (predictions["preprocessing_mode"] == "harmonized")
        & (predictions["model_name"].isin(["peak_based", "spectral"]))
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=["window_key", "peak_spectral_disagreement_bpm"])
    pivot = source.pivot_table(index="window_key", columns="model_name", values="y_pred_hr", aggfunc="first")
    if "peak_based" not in pivot.columns or "spectral" not in pivot.columns:
        return pd.DataFrame(columns=["window_key", "peak_spectral_disagreement_bpm"])
    result = pd.DataFrame(
        {
            "window_key": pivot.index,
            "peak_spectral_disagreement_bpm": (pivot["peak_based"] - pivot["spectral"]).abs().to_numpy(),
        }
    )
    return result.reset_index(drop=True)


def add_regime_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hr_range"] = pd.cut(
        result["y_true_hr"],
        bins=[0, 70, 90, 110, 130, np.inf],
        labels=["<70", "70-90", "90-110", "110-130", ">=130"],
        include_lowest=True,
    ).astype(str)
    result["motion_intensity_bin"] = quantile_bin(result["acc_norm_std"], ["low_motion", "mid_motion", "high_motion"])
    result["ppg_autocorr_quality_bin"] = quantile_bin(
        result["ppg_autocorr_peak_strength"],
        ["low_quality", "mid_quality", "high_quality"],
    )
    return result


def quantile_bin(series: pd.Series, labels: list[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    output = pd.Series("unavailable", index=series.index, dtype=object)
    valid = values.dropna()
    if valid.nunique() < len(labels):
        return output
    try:
        output.loc[valid.index] = pd.qcut(valid, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return output
    return output


def build_oracle_predictions(analysis: pd.DataFrame) -> pd.DataFrame:
    oracle = analysis[
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
            "classical_abs_error",
            "foundation_abs_error",
            "winner",
        ]
    ].copy()
    choose_classical = oracle["classical_abs_error"] <= oracle["foundation_abs_error"]
    oracle["oracle_selected_expert"] = np.where(choose_classical, "classical", "foundation")
    oracle["oracle_pred_hr"] = np.where(choose_classical, oracle["classical_pred_hr"], oracle["foundation_pred_hr"])
    oracle["oracle_abs_error"] = (oracle["oracle_pred_hr"] - oracle["y_true_hr"]).abs()
    oracle["oracle_squared_error"] = np.square(oracle["oracle_pred_hr"] - oracle["y_true_hr"])
    return oracle


def summarize_selected_pair(
    analysis: pd.DataFrame,
    classical_spec: ExpertSpec,
    foundation_spec: ExpertSpec,
) -> pd.DataFrame:
    rows = []
    for expert_name, pred_col in [
        ("classical", "classical_pred_hr"),
        ("foundation", "foundation_pred_hr"),
    ]:
        row = {
            "expert": expert_name,
            "run": classical_spec.label() if expert_name == "classical" else foundation_spec.label(),
        }
        row.update(compute_metrics(analysis["y_true_hr"], analysis[pred_col]))
        row["n_windows"] = int(len(analysis))
        rows.append(row)
    oracle = build_oracle_predictions(analysis)
    row = {"expert": "oracle_router", "run": "per-window lower true error"}
    row.update(compute_metrics(oracle["y_true_hr"], oracle["oracle_pred_hr"]))
    row["n_windows"] = int(len(oracle))
    rows.append(row)
    summary = pd.DataFrame(rows)
    best_single_mae = summary.loc[summary["expert"].isin(["classical", "foundation"]), "MAE"].min()
    best_single_p95 = summary.loc[summary["expert"].isin(["classical", "foundation"]), "p95_absolute_error"].min()
    best_single_cat = summary.loc[
        summary["expert"].isin(["classical", "foundation"]),
        "catastrophic_error_rate_20bpm",
    ].min()
    summary["gain_vs_best_single_MAE"] = best_single_mae - summary["MAE"]
    summary["gain_vs_best_single_p95"] = best_single_p95 - summary["p95_absolute_error"]
    summary["gain_vs_best_single_catastrophic_rate"] = (
        best_single_cat - summary["catastrophic_error_rate_20bpm"]
    )
    return summary


def build_all_pair_oracle_table(predictions: pd.DataFrame) -> pd.DataFrame:
    run_metrics = aggregate_run_metrics(predictions)
    classical_runs = run_metrics[
        (run_metrics["inversion"] == True)  # noqa: E712
        & (run_metrics["model_name"].isin(["peak_based", "spectral"]))
    ]
    foundation_runs = run_metrics[
        (run_metrics["inversion"] == True)  # noqa: E712
        & (run_metrics["model_name"].isin(["pulseppg", "papagei"]))
    ]
    rows: list[dict[str, Any]] = []
    for _, classical_row in classical_runs.iterrows():
        classical_spec = ExpertSpec.from_row(classical_row)
        classical = extract_expert_predictions(predictions, classical_spec, "classical")
        for _, foundation_row in foundation_runs.iterrows():
            foundation_spec = ExpertSpec.from_row(foundation_row)
            foundation = extract_expert_predictions(predictions, foundation_spec, "foundation")
            paired = pair_experts(classical, foundation)
            if paired.empty:
                continue
            oracle = build_oracle_predictions(paired)
            classical_metrics = compute_metrics(paired["y_true_hr"], paired["classical_pred_hr"])
            foundation_metrics = compute_metrics(paired["y_true_hr"], paired["foundation_pred_hr"])
            oracle_metrics = compute_metrics(oracle["y_true_hr"], oracle["oracle_pred_hr"])
            rows.append(
                {
                    "classical_run": classical_spec.label(),
                    "foundation_run": foundation_spec.label(),
                    "n_windows": int(len(paired)),
                    "classical_MAE": classical_metrics["MAE"],
                    "foundation_MAE": foundation_metrics["MAE"],
                    "oracle_MAE": oracle_metrics["MAE"],
                    "oracle_gain_vs_best_single_MAE": min(
                        classical_metrics["MAE"],
                        foundation_metrics["MAE"],
                    )
                    - oracle_metrics["MAE"],
                    "classical_p95": classical_metrics["p95_absolute_error"],
                    "foundation_p95": foundation_metrics["p95_absolute_error"],
                    "oracle_p95": oracle_metrics["p95_absolute_error"],
                    "oracle_gain_vs_best_single_p95": min(
                        classical_metrics["p95_absolute_error"],
                        foundation_metrics["p95_absolute_error"],
                    )
                    - oracle_metrics["p95_absolute_error"],
                    "classical_win_rate": float((paired["winner"] == "classical").mean()),
                    "foundation_win_rate": float((paired["winner"] == "foundation").mean()),
                    "tie_rate": float((paired["winner"] == "tie").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("oracle_gain_vs_best_single_MAE", ascending=False)


def summarize_regime(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_columns, key_values)}
        row.update(
            {
                "n_windows": int(len(group)),
                "classical_MAE": float(group["classical_abs_error"].mean()),
                "foundation_MAE": float(group["foundation_abs_error"].mean()),
                "oracle_MAE": float(np.minimum(group["classical_abs_error"], group["foundation_abs_error"]).mean()),
                "mean_error_gap_classical_minus_foundation": float(
                    group["error_gap_classical_minus_foundation"].mean()
                ),
                "median_error_gap_classical_minus_foundation": float(
                    group["error_gap_classical_minus_foundation"].median()
                ),
                "classical_win_rate": float((group["winner"] == "classical").mean()),
                "foundation_win_rate": float((group["winner"] == "foundation").mean()),
                "tie_rate": float((group["winner"] == "tie").mean()),
                "oracle_gain_vs_best_single_MAE": min(
                    float(group["classical_abs_error"].mean()),
                    float(group["foundation_abs_error"].mean()),
                )
                - float(np.minimum(group["classical_abs_error"], group["foundation_abs_error"]).mean()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("mean_error_gap_classical_minus_foundation", ascending=False).reset_index(drop=True)


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


def write_metric_json(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8")


def write_figures(
    analysis: pd.DataFrame,
    selected_summary: pd.DataFrame,
    regime_tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plot = selected_summary.set_index("expert")[["MAE", "p95_absolute_error"]]
    plot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Absolute error (bpm)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "oracle_vs_selected_experts.png", dpi=180)
    plt.close()

    activity = regime_tables["regime_by_activity"].copy()
    if not activity.empty:
        top = activity.sort_values("n_windows", ascending=False).head(12)
        plt.figure(figsize=(10, 4.5))
        x = np.arange(len(top))
        plt.bar(x - 0.2, top["classical_win_rate"], width=0.4, label="Classical wins")
        plt.bar(x + 0.2, top["foundation_win_rate"], width=0.4, label="Foundation wins")
        plt.ylabel("Window share")
        plt.xticks(x, top["activity"], rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "winner_rate_by_activity.png", dpi=180)
        plt.close()

    heat = regime_tables["regime_by_motion_and_quality"].copy()
    if not heat.empty:
        pivot = heat.pivot_table(
            index="motion_intensity_bin",
            columns="ppg_autocorr_quality_bin",
            values="mean_error_gap_classical_minus_foundation",
            aggfunc="first",
        )
        ordered_rows = [item for item in ["low_motion", "mid_motion", "high_motion"] if item in pivot.index]
        ordered_cols = [item for item in ["low_quality", "mid_quality", "high_quality"] if item in pivot.columns]
        pivot = pivot.loc[ordered_rows, ordered_cols]
        if not pivot.empty:
            plt.figure(figsize=(6, 4.5))
            image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
            plt.colorbar(image, label="Classical AE - foundation AE (bpm)")
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.tight_layout()
            plt.savefig(figures_dir / "motion_quality_error_gap_heatmap.png", dpi=180)
            plt.close()

    plt.figure(figsize=(9, 4.5))
    clipped = analysis["error_gap_classical_minus_foundation"].clip(-50, 50)
    plt.hist(clipped, bins=60)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Classical absolute error - foundation absolute error (bpm)")
    plt.ylabel("Windows")
    plt.tight_layout()
    plt.savefig(figures_dir / "window_error_gap_distribution.png", dpi=180)
    plt.close()


def write_memo(
    output_root: Path,
    predictions_path: Path,
    processed_windows_path: Path,
    classical_spec: ExpertSpec,
    foundation_spec: ExpertSpec,
    selected_summary: pd.DataFrame,
    all_pairs: pd.DataFrame,
    regime_tables: dict[str, pd.DataFrame],
    feature_notes: list[str],
) -> None:
    oracle_row = selected_summary[selected_summary["expert"] == "oracle_router"].iloc[0]
    best_single_mae = selected_summary[selected_summary["expert"].isin(["classical", "foundation"])]["MAE"].min()
    best_single_p95 = selected_summary[selected_summary["expert"].isin(["classical", "foundation"])][
        "p95_absolute_error"
    ].min()
    top_activity = regime_tables["regime_by_activity"].head(8)
    top_motion_quality = regime_tables["regime_by_motion_and_quality"].head(8)
    branch, commit = git_state()
    notes = "\n".join(f"- {note}" for note in feature_notes) if feature_notes else "- No missing feature inputs detected."
    memo = f"""# Week 3 Memo: GalaxyPPG Regime Analysis and Oracle Routing

## Objective

Week 3 tests whether estimator dominance is regime-dependent on the corrected GalaxyPPG benchmark. It compares one selected classical expert with one selected foundation-model expert on the same windows, computes per-window error gaps, summarizes winners by activity, participant, HR range, motion intensity, and PPG quality, and reports an oracle router upper bound.

## Inputs

- Standardized Week 2 predictions: `{predictions_path.as_posix()}`
- Processed windows with PPG waveforms and labels: `{processed_windows_path.as_posix()}`
- Output root: `{output_root.as_posix()}`

## Selected Experts

- Classical expert: `{classical_spec.label()}`
- Foundation expert: `{foundation_spec.label()}`

The default selection chooses the lowest-MAE inverted classical run and the lowest-MAE inverted foundation-model run. The builder also writes `tables/oracle_all_pairs_table.csv` so this choice can be audited against every completed classical/foundation pair.

## Oracle Result

The selected-pair oracle chooses, for each window, the expert with the smaller true absolute error. It is not deployable; it estimates routing headroom.

{dataframe_to_markdown(selected_summary)}

Best single-expert MAE was `{best_single_mae:.3f}` bpm. The oracle MAE was `{oracle_row['MAE']:.3f}` bpm, for a headroom of `{oracle_row['gain_vs_best_single_MAE']:.3f}` bpm. Best single-expert P95 AE was `{best_single_p95:.3f}` bpm; oracle P95 AE was `{oracle_row['p95_absolute_error']:.3f}` bpm.

## Regime Findings

Positive error gap means the foundation expert had lower absolute error than the classical expert. Negative error gap means the classical expert won.

Activity summary:

{dataframe_to_markdown(top_activity)}

Motion-quality summary:

{dataframe_to_markdown(top_motion_quality)}

## Artifacts

- Window-level paired errors and regime features: `predictions/week3_window_regime_expert_errors.csv`
- Oracle predictions: `predictions/week3_oracle_router_predictions.csv`
- Selected-pair oracle table: `tables/selected_expert_oracle_summary.csv`
- All-pair oracle audit table: `tables/oracle_all_pairs_table.csv`
- Regime tables: `tables/regime_by_activity.csv`, `tables/regime_by_participant.csv`, `tables/regime_by_hr_range.csv`, `tables/regime_by_motion_intensity.csv`, `tables/regime_by_ppg_quality.csv`, `tables/regime_by_motion_and_quality.csv`
- Draft figures: `figures/oracle_vs_selected_experts.png`, `figures/winner_rate_by_activity.png`, `figures/motion_quality_error_gap_heatmap.png`, `figures/window_error_gap_distribution.png`

## Feature Notes

{notes}

Motion bins are tertiles of accelerometer-norm standard deviation. PPG quality bins are tertiles of autocorrelation peak strength in plausible HR-period lags. These are descriptive Week 3 regime features, not a trained router.

## Conclusion

The oracle gain quantifies whether routing is worth pursuing before Week 4. A meaningful oracle improvement, especially in P95 AE or catastrophic error rate, supports the paper direction; a negligible oracle improvement would weaken it. The generated all-pair table should be used to choose which experts remain in the Week 4 routed system.

## Reproducibility

- branch: `{branch}`
- commit hash: `{commit}`
- command: `python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root {output_root.as_posix()}`
"""
    (output_root / "week3_regime_analysis.md").write_text(memo, encoding="utf-8")


def git_state() -> tuple[str, str]:
    def run(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return run(["git", "branch", "--show-current"]), run(["git", "rev-parse", "HEAD"])


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


if __name__ == "__main__":
    main()
