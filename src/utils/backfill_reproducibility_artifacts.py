"""Backfill run logs and fixed-split metadata for saved experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data import describe_validation_folds, load_fixed_split_config


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_MANIFEST = Path("data/processed/galaxyppg_hr_w10_s2_median_manifest.json")
SPLIT_CONFIG = Path("configs/galaxyppg_submission_split.json")
FIXED_SPLIT = load_fixed_split_config(REPO_ROOT / SPLIT_CONFIG)


@dataclass(slots=True)
class BaselineArtifact:
    method: str
    result_dir: Path


@dataclass(slots=True)
class RegressionArtifact:
    model_name: str
    regressor: str
    feature_manifest: Path
    result_dir: Path


BASELINE_ARTIFACTS = [
    BaselineArtifact(method="peak", result_dir=Path("experiments/baseline_results/2026-03-11")),
    BaselineArtifact(method="spectral", result_dir=Path("experiments/baseline_results/2026-03-11")),
    BaselineArtifact(method="peak", result_dir=Path("experiments/reproduced_submission/baseline_peak")),
]

REGRESSION_ARTIFACTS = [
    RegressionArtifact(
        model_name="pulseppg",
        regressor="linear",
        feature_manifest=Path("experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json"),
        result_dir=Path("experiments/pulseppg_results/2026-03-18/regression_linear"),
    ),
    RegressionArtifact(
        model_name="pulseppg",
        regressor="ridge",
        feature_manifest=Path("experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json"),
        result_dir=Path("experiments/pulseppg_results/2026-03-18/regression_ridge"),
    ),
    RegressionArtifact(
        model_name="pulseppg",
        regressor="gradient_boosting",
        feature_manifest=Path("experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json"),
        result_dir=Path("experiments/pulseppg_results/2026-03-23/regression_gradient_boosting"),
    ),
    RegressionArtifact(
        model_name="pulseppg",
        regressor="random_forest",
        feature_manifest=Path("experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json"),
        result_dir=Path("experiments/pulseppg_results/2026-03-23/regression_random_forest"),
    ),
    RegressionArtifact(
        model_name="pulseppg",
        regressor="random_forest",
        feature_manifest=Path("experiments/pulseppg_results/2026-03-18/full/pulseppg_manifest.json"),
        result_dir=Path("experiments/reproduced_submission/pulseppg_random_forest"),
    ),
    RegressionArtifact(
        model_name="papagei",
        regressor="linear",
        feature_manifest=Path("experiments/papagei_results/2026-03-18/full/papagei_manifest.json"),
        result_dir=Path("experiments/papagei_results/2026-03-18/regression_linear"),
    ),
    RegressionArtifact(
        model_name="papagei",
        regressor="ridge",
        feature_manifest=Path("experiments/papagei_results/2026-03-18/full/papagei_manifest.json"),
        result_dir=Path("experiments/papagei_results/2026-03-18/regression_ridge"),
    ),
    RegressionArtifact(
        model_name="papagei",
        regressor="gradient_boosting",
        feature_manifest=Path("experiments/papagei_results/2026-03-18/full/papagei_manifest.json"),
        result_dir=Path("experiments/papagei_results/2026-03-23/regression_gradient_boosting"),
    ),
    RegressionArtifact(
        model_name="papagei",
        regressor="random_forest",
        feature_manifest=Path("experiments/papagei_results/2026-03-18/full/papagei_manifest.json"),
        result_dir=Path("experiments/papagei_results/2026-03-23/regression_random_forest"),
    ),
]


def main() -> None:
    """Write run logs, augment metrics, and regenerate the experiment summary."""

    summary_rows: list[dict[str, Any]] = []
    for artifact in BASELINE_ARTIFACTS:
        summary_rows.append(backfill_baseline_artifact(artifact))
    for artifact in REGRESSION_ARTIFACTS:
        summary_rows.append(backfill_regression_artifact(artifact))
    write_summary(summary_rows)


def backfill_baseline_artifact(artifact: BaselineArtifact) -> dict[str, Any]:
    """Write a baseline run log and augment its metrics metadata."""

    result_dir = REPO_ROOT / artifact.result_dir
    metrics_path = result_dir / f"{artifact.method}_metrics.json"
    predictions_path = result_dir / f"{artifact.method}_predictions.csv"
    run_log_path = result_dir / f"{artifact.method}_run_log.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "split_config_path": SPLIT_CONFIG.as_posix(),
            "split_name": FIXED_SPLIT.split_name,
            "validation_strategy": FIXED_SPLIT.validation_strategy,
            "validation_folds": describe_validation_folds(FIXED_SPLIT),
            "source_processed_manifest_path": PROCESSED_MANIFEST.as_posix(),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    command = (
        "python -m src.baseline.run_baseline "
        f"--processed-manifest {PROCESSED_MANIFEST.as_posix()} "
        f"--method {artifact.method} "
        f"--output-dir {artifact.result_dir.as_posix()}"
    )
    run_log = {
        "module": "src.baseline.run_baseline",
        "argv": command.split(),
        "input_source": {
            "mode": "processed_manifest",
            "processed_manifest_path": PROCESSED_MANIFEST.as_posix(),
            "split_config_path": SPLIT_CONFIG.as_posix(),
        },
        "predictions_path": artifact_relative_path(predictions_path),
        "metrics_path": artifact_relative_path(metrics_path),
        "metrics": metrics,
    }
    run_log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "kind": "baseline",
        "name": artifact.method,
        "result_dir": artifact.result_dir.as_posix(),
        "command": command,
        "metrics_path": artifact_relative_path(metrics_path),
        "predictions_path": artifact_relative_path(predictions_path),
        "run_log_path": artifact_relative_path(run_log_path),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
    }


def backfill_regression_artifact(artifact: RegressionArtifact) -> dict[str, Any]:
    """Write a regression run log and augment its metrics metadata."""

    result_dir = REPO_ROOT / artifact.result_dir
    metrics_path = result_dir / f"{artifact.model_name}_{artifact.regressor}_metrics.json"
    predictions_path = result_dir / f"{artifact.model_name}_{artifact.regressor}_predictions.csv"
    estimator_path = result_dir / f"{artifact.model_name}_{artifact.regressor}_estimator.joblib"
    run_log_path = result_dir / f"{artifact.model_name}_{artifact.regressor}_run_log.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "split_config_path": SPLIT_CONFIG.as_posix(),
            "split_name": FIXED_SPLIT.split_name,
            "validation_strategy": FIXED_SPLIT.validation_strategy,
            "cv_strategy": "fixed_validation_folds",
            "cv_fold_assignments": describe_validation_folds(FIXED_SPLIT),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    command = (
        "python -m src.regression.train_regressor "
        f"--feature-manifest {artifact.feature_manifest.as_posix()} "
        f"--regressor {artifact.regressor} "
        "--random-state 42 "
        f"--split-config {SPLIT_CONFIG.as_posix()} "
        f"--output-dir {artifact.result_dir.as_posix()}"
    )
    run_log = {
        "module": "src.regression.train_regressor",
        "argv": command.split(),
        "feature_manifest_path": artifact.feature_manifest.as_posix(),
        "artifacts": {
            "model_name": artifact.model_name,
            "regressor": artifact.regressor,
            "output_dir": artifact.result_dir.as_posix(),
            "predictions_path": artifact_relative_path(predictions_path),
            "metrics_path": artifact_relative_path(metrics_path),
            "estimator_path": artifact_relative_path(estimator_path) if estimator_path.exists() else None,
        },
        "metrics": metrics,
    }
    run_log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "kind": "regression",
        "name": f"{artifact.model_name}_{artifact.regressor}",
        "result_dir": artifact.result_dir.as_posix(),
        "command": command,
        "metrics_path": artifact_relative_path(metrics_path),
        "predictions_path": artifact_relative_path(predictions_path),
        "estimator_path": artifact_relative_path(estimator_path) if estimator_path.exists() else None,
        "run_log_path": artifact_relative_path(run_log_path),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
    }


def write_summary(summary_rows: list[dict[str, Any]]) -> None:
    """Write machine-readable and Markdown summaries of the saved experiments."""

    summary_json_path = REPO_ROOT / "experiments" / "reported_results_summary.json"
    summary_md_path = REPO_ROOT / "experiments" / "reported_results_summary.md"
    payload = {
        "split_config": SPLIT_CONFIG.as_posix(),
        "processed_manifest": PROCESSED_MANIFEST.as_posix(),
        "experiments": summary_rows,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Reported Results Summary",
        "",
        f"- Fixed split config: `{SPLIT_CONFIG.as_posix()}`",
        f"- Processed manifest: `{PROCESSED_MANIFEST.as_posix()}`",
        "",
        "| Name | Kind | MAE | RMSE | Result Dir |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['mae']:.6f} | {row['rmse']:.6f} | `{row['result_dir']}` |"
        )
    lines.append("")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def artifact_relative_path(path: Path) -> str:
    """Render a repository-relative path."""

    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    main()
