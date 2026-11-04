"""Participant-cluster inference for the corrected submission results only.

The estimand is the equally weighted mean of participant metrics. Windows are
never treated as independent inferential observations. A different, explicitly
post-hoc comparison chooses the best single expert separately per participant.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOT = Path("experiments/conference_submission_2026_11_01")
METRICS = ("MAE", "RMSE", "median_absolute_error", "p95_absolute_error", "catastrophic_error_rate_20bpm")
ROUTERS = tuple(f"{features}_{gate}" for features in ("motion", "quality", "motion_quality") for gate in ("hard", "soft"))
REQUIRED_COLUMNS = {"window_key", "participant_id", "session_id", "activity", "y_true", "y_pred", "model_name"}


def metric_values(y_true: Any, y_pred: Any) -> dict[str, float]:
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Metrics require matching finite prediction/label arrays.")
    if not true.size:
        return {name: math.nan for name in METRICS}
    error = np.abs(pred - true)
    return dict(zip(METRICS, (float(error.mean()), float(np.sqrt(np.mean(error**2))),
                            float(np.median(error)), float(np.percentile(error, 95)), float(np.mean(error > 20)))))


def validate_predictions(frame: pd.DataFrame) -> None:
    """Reject silent model-dependent exclusions, label mismatches, and duplicate keys."""
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"Missing final prediction columns: {sorted(missing)}")
    if frame.empty or frame[["window_key", "participant_id", "model_name"]].isna().any().any():
        raise ValueError("Final predictions must contain nonempty, identified observations.")
    if frame.duplicated(["model_name", "window_key"]).any():
        raise ValueError("Duplicate model/window predictions.")
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite final predictions: form one common finite window set before inference.")
    keys = None
    for model_name, group in frame.groupby("model_name", sort=True):
        model_keys = set(group.window_key)
        if keys is not None and keys != model_keys:
            raise ValueError(f"Unpaired window sets for {model_name}; no model-specific window deletion is allowed.")
        keys = model_keys
    for column in ("participant_id", "session_id", "activity", "y_true"):
        if frame.groupby("window_key")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting {column} across model predictions for a shared window.")


def participant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    validate_predictions(frame)
    return pd.DataFrame([
        {"model_name": model, "participant_id": participant, "n_windows": len(group),
         **metric_values(group.y_true, group.y_pred)}
        for (model, participant), group in frame.groupby(["model_name", "participant_id"], sort=True)
    ])


def bootstrap_participant_mean(values: Any, iterations: int = 10000, random_state: int = 42) -> tuple[float, float]:
    """Resample independent participants, retaining the metric of each whole cluster."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    if not len(values):
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(random_state)
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def signed_rank_pvalue(values: Any) -> float:
    """Two-sided exact sign permutation of signed ranks for small, possibly tied samples.

    Explicit enumeration avoids treating ties/zero deltas as an asymptotic sample.
    Zero differences are discarded (Wilcox convention); the all-zero case is 1.
    For larger samples SciPy's automatic method is used and recorded in outputs.
    """
    values = np.round(np.asarray(values, dtype=float), 12)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return 1.0
    if len(values) > 20:
        return float(stats.wilcoxon(values, alternative="two-sided", method="auto").pvalue)
    ranks = stats.rankdata(np.abs(values))
    observed = abs(float(np.dot(np.sign(values), ranks)))
    extreme = sum(abs(float(np.dot(signs, ranks))) >= observed - 1e-10
                  for signs in itertools.product((-1, 1), repeat=len(values)))
    return extreme / 2**len(values)


def compare_models(participant: pd.DataFrame, baseline: str, router: str,
                   iterations: int = 10000, random_state: int = 42,
                   comparison_type: str = "primary_train_preselected") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = participant[participant.model_name.eq(baseline)].set_index("participant_id")
    routed = participant[participant.model_name.eq(router)].set_index("participant_id")
    if base.empty or set(base.index) != set(routed.index):
        raise ValueError(f"Unpaired participant comparison: {baseline} vs {router}")
    routed = routed.loc[base.index]
    detail, summary = [], []
    for metric in METRICS:
        delta = base[metric].to_numpy(float) - routed[metric].to_numpy(float)
        valid = np.isfinite(delta)
        difference = delta[valid]
        ci_low, ci_high = bootstrap_participant_mean(difference, iterations, random_state)
        n = len(difference)
        if n < 2:
            t_p = math.nan
        elif np.std(difference, ddof=1) == 0:
            t_p = 1.0 if np.all(difference == 0) else 0.0
        else:
            t_p = float(stats.ttest_1samp(difference, 0).pvalue)
        summary.append({
            "comparison_type": comparison_type, "baseline_model": baseline, "router_model": router,
            "metric": metric, "n_participants": n,
            "baseline_participant_mean": float(base.loc[valid, metric].mean()),
            "baseline_participant_sd": float(base.loc[valid, metric].std(ddof=1)),
            "router_participant_mean": float(routed.loc[valid, metric].mean()),
            "router_participant_sd": float(routed.loc[valid, metric].std(ddof=1)),
            "mean_delta_baseline_minus_router": float(np.mean(difference)) if n else math.nan,
            "median_delta_baseline_minus_router": float(np.median(difference)) if n else math.nan,
            "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
            "wilcoxon_signed_rank_p_value": signed_rank_pvalue(difference),
            "paired_ttest_p_value": t_p,
            "n_improved": int(np.sum(difference > 0)), "n_degraded": int(np.sum(difference < 0)),
            "n_unchanged": int(np.sum(difference == 0)),
            "bootstrap_unit": "participant", "bootstrap_iterations": iterations,
            "wilcoxon_method": "exact_sign_permutation_wilcox_zeros" if n <= 20 else "scipy_auto",
            "pvalue_status": "unadjusted_exploratory; primary endpoint MAE, primary router fixed before testing",
        })
        for participant_id, baseline_value, router_value, change in zip(base.index, base[metric], routed[metric], delta):
            detail.append({"comparison_type": comparison_type, "baseline_model": baseline,
                           "router_model": router, "participant_id": participant_id, "metric": metric,
                           "baseline_value": baseline_value, "router_value": router_value,
                           "delta_baseline_minus_router": change,
                           "direction": "improved" if change > 0 else "degraded" if change < 0 else "unchanged"})
    return pd.DataFrame(summary), pd.DataFrame(detail)


def select_participant_hindsight(participant: pd.DataFrame, expert_models: list[str]) -> pd.DataFrame:
    """Post-hoc upper-bound comparator; one expert per person chosen by that person's test MAE."""
    candidates = participant[participant.model_name.isin(expert_models)].copy()
    if candidates.empty:
        raise ValueError("No selected experts available for hindsight supplement.")
    best = candidates.sort_values(["participant_id", "MAE", "model_name"]).groupby("participant_id", sort=True).head(1)
    best = best.rename(columns={"model_name": "hindsight_selected_expert"})
    best["model_name"] = "participant_hindsight_best"
    best["selection_source"] = "test_participant_MAE_posthoc_not_deployable"
    return best.reset_index(drop=True)


def validate_selection(selection: dict[str, Any]) -> None:
    for key in ("best_classical", "best_foundation", "best_single", "primary_router"):
        if not isinstance(selection.get(key), str):
            raise ValueError(f"selection.json must record a train-preselected string {key}.")
    source = str(selection.get("selection_source", "")).lower()
    if not any(term in source for term in ("train", "oof")) or "test" in source:
        raise ValueError("Primary comparator must be selected from training OOF data, never test metrics.")


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Write Markdown without a hidden optional tabulate dependency."""
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join("---" for _ in frame.columns) + " |",
                       *["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def build_statistics(results_root: Path, bootstrap_iterations: int = 10000, random_state: int = 42) -> dict[str, str]:
    results_root = Path(results_root)
    selection = json.loads((results_root / "manifests/selection.json").read_text(encoding="utf-8"))
    validate_selection(selection)
    predictions = pd.read_csv(results_root / "predictions/final_test_predictions.csv", low_memory=False)
    participant = participant_metrics(predictions)
    output = results_root / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    participant.to_csv(output / "participant_metrics.csv", index=False)
    per_model = participant.groupby("model_name")[list(METRICS)].agg(["mean", "std"])
    per_model.columns = [f"participant_{metric}_{summary}" for metric, summary in per_model.columns]
    per_model.reset_index().to_csv(output / "participant_metric_summary.csv", index=False)
    hindsight = select_participant_hindsight(participant, [selection["best_classical"], selection["best_foundation"]])
    hindsight.to_csv(output / "participant_hindsight_supplement.csv", index=False)
    all_metrics = pd.concat([participant, hindsight], ignore_index=True)
    summaries, details = [], []
    for router in ROUTERS:
        if router not in set(participant.model_name):
            continue
        for baseline, comparison_type in ((selection["best_single"], "primary_train_preselected"),
                                           ("participant_hindsight_best", "supplement_test_hindsight")):
            summary, detail = compare_models(all_metrics, baseline, router, bootstrap_iterations, random_state, comparison_type)
            summary["is_prespecified_primary_endpoint"] = summary.metric.eq("MAE") & summary.router_model.eq(selection["primary_router"]) & summary.comparison_type.eq("primary_train_preselected")
            summaries.append(summary)
            details.append(detail)
    if not summaries:
        raise ValueError("No learned router predictions were available.")
    combined, detail = pd.concat(summaries, ignore_index=True), pd.concat(details, ignore_index=True)
    combined.to_csv(output / "paired_comparisons.csv", index=False)
    detail.to_csv(output / "participant_paired_differences.csv", index=False)
    (output / "paired_comparisons.json").write_text(combined.to_json(orient="records", indent=2), encoding="utf-8")
    primary = combined[combined.comparison_type.eq("primary_train_preselected") & combined.metric.eq("MAE")]
    notes = (
        "Positive differences mean lower error for the router. Each participant contributes one metric, regardless of window count. "
        "The percentile 95% bootstrap resamples participants with replacement; overlapping windows are never independently resampled. "
        "These CIs describe uncertainty for an equally weighted participant mean, not for the pooled window MAE. "
        "With five independent test participants, the smallest attainable two-sided exact Wilcoxon p-value with nonzero differences is 0.0625. "
        "The paired t-test additionally assumes approximately normal participant differences; five observations cannot establish that assumption. "
        "CIs and unadjusted secondary p-values are exploratory and are unstable at this sample size. "
        "Historical inspection of this fixed test cohort prevents treating the corrected rerun as a new confirmatory evaluation. "
        "The hindsight supplement selects one of the two preselected experts using each test participant's MAE; it is not a deployable baseline."
    )
    (output / "statistics.md").write_text("# Participant-level inference\n\n" + notes + "\n\n" + markdown_table(primary) + "\n", encoding="utf-8")
    return {name: str(output / name) for name in ("participant_metrics.csv", "paired_comparisons.csv", "participant_paired_differences.csv", "participant_hindsight_supplement.csv", "statistics.md")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_statistics(args.results_root, args.bootstrap_iterations, args.random_state), indent=2))


if __name__ == "__main__":
    main()
