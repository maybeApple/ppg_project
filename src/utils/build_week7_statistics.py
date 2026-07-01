"""Build participant-level Week 7 statistics for router comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


METRICS = ["MAE", "p95_absolute_error", "catastrophic_error_rate_20bpm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week4-root",
        type=Path,
        default=Path("experiments/week4_galaxyppg_lightweight_router_2026-05-13"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/week7_final_statistics"),
    )
    parser.add_argument("--router-feature-set", default="motion_quality")
    parser.add_argument("--router-type", default="hard_gate", choices=["hard_gate", "soft_gate"])
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    windows_path = args.week4_root / "predictions" / "week4_routed_predictions.csv"
    if not windows_path.exists():
        raise FileNotFoundError(windows_path)
    routed = pd.read_csv(windows_path, low_memory=False)
    selected_router = routed[
        (routed["feature_set"] == args.router_feature_set) & (routed["routing_type"] == args.router_type)
    ].copy()
    if selected_router.empty:
        raise RuntimeError("No routed rows matched the requested feature set and router type.")

    participant = build_participant_comparison(selected_router)
    participant.to_csv(args.output_root / "participant_level_router_comparison.csv", index=False)

    stats_table = build_stats_table(
        participant,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    stats_table.to_csv(args.output_root / "paired_significance_tests.csv", index=False)
    (args.output_root / "paired_significance_tests.json").write_text(
        json.dumps(stats_table.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_root / "week7_statistics.md", participant, stats_table, args)
    print(f"participant={args.output_root / 'participant_level_router_comparison.csv'}")
    print(f"stats={args.output_root / 'paired_significance_tests.csv'}")


def build_participant_comparison(routed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate classical, foundation, best-single, and router metrics by participant."""

    rows: list[dict[str, Any]] = []
    for participant_id, group in routed.groupby("participant_id", dropna=False):
        classical = compute_metrics(group["y_true_hr"], group["classical_pred_hr"])
        foundation = compute_metrics(group["y_true_hr"], group["foundation_pred_hr"])
        router = compute_metrics(group["y_true_hr"], group["routed_pred_hr"])
        best_single = choose_best_single(classical, foundation)
        row = {
            "participant_id": participant_id,
            "n_windows": int(len(group)),
            "best_single_source": best_single["source"],
        }
        for prefix, metrics in [
            ("classical", classical),
            ("foundation", foundation),
            ("best_single", best_single),
            ("router", router),
        ]:
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = metrics[metric]
        for metric in METRICS:
            row[f"delta_{metric}_best_single_minus_router"] = (
                row[f"best_single_{metric}"] - row[f"router_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def choose_best_single(classical: dict[str, float], foundation: dict[str, float]) -> dict[str, Any]:
    """Choose the lower-MAE single expert for one participant."""

    if classical["MAE"] <= foundation["MAE"]:
        return {"source": "classical", **classical}
    return {"source": "foundation", **foundation}


def build_stats_table(
    participant: pd.DataFrame,
    bootstrap_iterations: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute participant-level CIs and paired tests."""

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for metric in METRICS:
        delta = participant[f"delta_{metric}_best_single_minus_router"].to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        ci_low, ci_high = bootstrap_mean_ci(delta, bootstrap_iterations, rng)
        if len(delta) >= 2:
            t_result = stats.ttest_1samp(delta, popmean=0.0, nan_policy="omit")
            try:
                wilcoxon_result = stats.wilcoxon(delta)
                wilcoxon_p = float(wilcoxon_result.pvalue)
            except ValueError:
                wilcoxon_p = math.nan
        else:
            t_result = None
            wilcoxon_p = math.nan
        rows.append(
            {
                "metric": metric,
                "n_participants": int(len(delta)),
                "mean_delta_best_single_minus_router": float(np.mean(delta)) if len(delta) else math.nan,
                "median_delta_best_single_minus_router": float(np.median(delta)) if len(delta) else math.nan,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "paired_ttest_p_value": math.nan if t_result is None else float(t_result.pvalue),
                "wilcoxon_signed_rank_p_value": wilcoxon_p,
                "interpretation": "positive means router improved over the participant's best single expert",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for the participant-level mean."""

    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute HR error metrics."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(true) | np.isnan(pred))
    true = true[valid]
    pred = pred[valid]
    if len(true) == 0:
        return {metric: math.nan for metric in METRICS}
    abs_error = np.abs(pred - true)
    return {
        "MAE": float(np.mean(abs_error)),
        "p95_absolute_error": float(np.percentile(abs_error, 95)),
        "catastrophic_error_rate_20bpm": float(np.mean(abs_error > 20.0)),
    }


def write_markdown(path: Path, participant: pd.DataFrame, stats_table: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write a concise Week 7 statistics memo."""

    lines = [
        "# Week 7 Participant-Level Statistics",
        "",
        f"- Week 4 root: `{args.week4_root.as_posix()}`",
        f"- Router: `{args.router_feature_set}/{args.router_type}`",
        f"- Participants: `{participant['participant_id'].nunique()}`",
        "",
        "Positive deltas mean the routed system improved over the participant's best single expert.",
        "",
        stats_table.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
