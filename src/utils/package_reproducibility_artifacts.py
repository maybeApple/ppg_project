"""Package existing Week 2-4 artifacts into fixed reproducibility filenames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


CORRECTED_MANIFEST = "galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json"
LEGACY_HR_MANIFEST = "galaxyppg_hr_w10_s2_median_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week2-root",
        type=Path,
        default=Path("experiments/week2_galaxyppg_corrected_2026-05-01"),
    )
    parser.add_argument(
        "--week3-root",
        type=Path,
        default=Path("experiments/week3_galaxyppg_regime_oracle_2026-05-13"),
    )
    parser.add_argument(
        "--week4-root",
        type=Path,
        default=Path("experiments/week4_galaxyppg_lightweight_router_2026-05-13"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("experiments/reproducibility_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "schema_version": "reproducibility_package_v1",
        "corrected_processed_manifest": f"data/processed/{CORRECTED_MANIFEST}",
        "week2": package_week2(args.week2_root),
        "week3": package_week3(args.week3_root, args.week2_root),
        "week4": package_week4(args.week4_root, args.week3_root),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_summary(args.output_manifest.with_suffix(".md"), manifest)
    print(f"manifest={args.output_manifest}")
    print(f"week2_runs={len(manifest['week2']['runs'])}")
    print(f"week2_embedding_bundles={len(manifest['week2']['embedding_bundles'])}")


def package_week2(week2_root: Path) -> dict[str, Any]:
    runs_root = week2_root / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    run_rows: list[dict[str, Any]] = []
    for predictions_path in sorted(runs_root.rglob("*_predictions.csv")):
        if predictions_path.name == "predictions.csv":
            continue
        stem = predictions_path.name.removesuffix("_predictions.csv")
        metrics_path = predictions_path.with_name(f"{stem}_metrics.json")
        if not metrics_path.exists():
            continue
        run_dir = predictions_path.parent
        run_log_path = predictions_path.with_name(f"{stem}_run_log.json")
        generic_predictions = copy_artifact(predictions_path, run_dir / "predictions.csv")
        generic_metrics = copy_artifact(metrics_path, run_dir / "metrics.json")
        generic_run_log = run_dir / "run_log.json"
        if run_log_path.exists():
            copy_artifact(run_log_path, generic_run_log)
        else:
            generic_run_log.write_text(
                json.dumps({"predictions_path": str(predictions_path), "metrics_path": str(metrics_path)}, indent=2),
                encoding="utf-8",
            )
        metrics = read_json(metrics_path)
        run_config_path = write_run_config(
            run_dir=run_dir,
            run_type="week2_experiment_run",
            module=read_json(generic_run_log).get("module", "unknown"),
            argv=read_json(generic_run_log).get("argv", []),
            inputs=extract_inputs(metrics, read_json(generic_run_log)),
            artifacts={
                "predictions_csv": portable(generic_predictions),
                "metrics_json": portable(generic_metrics),
                "run_log_json": portable(generic_run_log),
                "source_predictions_csv": portable(predictions_path),
                "source_metrics_json": portable(metrics_path),
            },
        )
        run_rows.append(
            {
                "run_dir": portable(run_dir),
                "run_id": str(run_dir.relative_to(runs_root)).replace("\\", "/"),
                "predictions_csv": portable(generic_predictions),
                "metrics_json": portable(generic_metrics),
                "run_config_json": portable(run_config_path),
                "run_log_json": portable(generic_run_log),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
            }
        )

    embedding_rows = package_week2_embeddings(week2_root)
    write_csv(week2_root / "run_manifest.csv", run_rows)
    write_csv(week2_root / "embedding_manifest.csv", embedding_rows)
    return {
        "root": portable(week2_root),
        "runs": run_rows,
        "embedding_bundles": embedding_rows,
        "summary_artifacts": {
            "all_predictions": portable(week2_root / "predictions/week2_all_standardized_predictions.csv"),
            "overall_metrics": portable(week2_root / "metrics/overall_metrics.csv"),
            "participant_metrics": portable(week2_root / "metrics/participant_level_metrics.csv"),
            "activity_metrics": portable(week2_root / "metrics/activity_level_metrics.csv"),
        },
    }


def package_week2_embeddings(week2_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((week2_root / "runs").rglob("*_manifest.json")):
        manifest = read_json(manifest_path)
        model_name = str(manifest.get("model_name", "unknown"))
        if model_name not in {"pulseppg", "papagei"}:
            continue
        processed_manifest = str(manifest.get("processed_manifest_path", ""))
        source_windows = str(manifest.get("source_windows_path", ""))
        reference_text = f"{processed_manifest} {source_windows}"
        if LEGACY_HR_MANIFEST in reference_text:
            raise RuntimeError(f"Legacy HR manifest found in embedding manifest: {manifest_path}")
        if CORRECTED_MANIFEST not in reference_text:
            raise RuntimeError(f"Corrected IBI manifest is not recorded in embedding manifest: {manifest_path}")
        bundle_dir = manifest_path.parent
        features_path = resolve_bundle_path(manifest_path, manifest["features_path"])
        metadata_path = resolve_bundle_path(manifest_path, manifest["metadata_path"])
        run_config_path = write_run_config(
            run_dir=bundle_dir,
            run_type="week2_embedding_export",
            module=f"src.models.{model_name}_feature",
            argv=[],
            inputs={
                "processed_manifest_path": processed_manifest,
                "source_windows_path": source_windows,
                "checkpoint_path": manifest.get("checkpoint_path"),
            },
            artifacts={
                "features_npy": portable(features_path),
                "metadata_csv": portable(metadata_path),
                "embedding_manifest_json": portable(manifest_path),
            },
        )
        rows.append(
            {
                "bundle_dir": portable(bundle_dir),
                "model_name": model_name,
                "features_npy": portable(features_path),
                "metadata_csv": portable(metadata_path),
                "manifest_json": portable(manifest_path),
                "run_config_json": portable(run_config_path),
                "processed_manifest_path": processed_manifest,
                "num_windows": manifest.get("num_windows"),
                "embedding_dim": manifest.get("embedding_dim"),
                "preprocessing_mode": manifest.get("preprocessing_mode"),
                "normalization": manifest.get("normalization"),
            }
        )
    return rows


def package_week3(week3_root: Path, week2_root: Path) -> dict[str, Any]:
    if not week3_root.exists():
        raise FileNotFoundError(week3_root)
    predictions_path = week3_root / "predictions/week3_oracle_router_predictions.csv"
    metrics_path = week3_root / "metrics/selected_expert_oracle_summary.json"
    run_log_path = week3_root / "run_log.json"
    run_config_path = write_run_config(
        run_dir=week3_root,
        run_type="week3_oracle_routing",
        module="src.utils.build_week3_artifacts",
        argv=[
            "python",
            "-m",
            "src.utils.build_week3_artifacts",
            "--week2-root",
            week2_root.as_posix(),
            "--output-root",
            week3_root.as_posix(),
        ],
        inputs={
            "week2_root": portable(week2_root),
            "week2_predictions": portable(week2_root / "predictions/week2_all_standardized_predictions.csv"),
        },
        artifacts={
            "expert_pairing_csv": portable(week3_root / "predictions/week3_window_regime_expert_errors.csv"),
            "oracle_predictions_csv": portable(predictions_path),
            "metrics_json": portable(metrics_path),
            "all_pair_oracle_csv": portable(week3_root / "tables/oracle_all_pairs_table.csv"),
            "per_activity_csv": portable(week3_root / "tables/regime_by_activity.csv"),
            "per_participant_csv": portable(week3_root / "tables/regime_by_participant.csv"),
        },
    )
    copy_artifact(predictions_path, week3_root / "predictions.csv")
    copy_artifact(metrics_path, week3_root / "metrics.json")
    run_log_path.write_text(
        json.dumps({**read_json(run_config_path), "metrics": read_json(metrics_path)}, indent=2),
        encoding="utf-8",
    )
    return {
        "root": portable(week3_root),
        "predictions_csv": portable(week3_root / "predictions.csv"),
        "metrics_json": portable(week3_root / "metrics.json"),
        "run_config_json": portable(run_config_path),
        "run_log_json": portable(run_log_path),
    }


def package_week4(week4_root: Path, week3_root: Path) -> dict[str, Any]:
    if not week4_root.exists():
        raise FileNotFoundError(week4_root)
    predictions_path = week4_root / "predictions/week4_routed_predictions.csv"
    metrics_path = week4_root / "metrics/routing_summary.json"
    if predictions_path.exists():
        copy_artifact(predictions_path, week4_root / "predictions.csv")
    if metrics_path.exists():
        copy_artifact(metrics_path, week4_root / "metrics.json")
    run_config_path = week4_root / "run_config.json"
    if not run_config_path.exists():
        run_config_path = write_run_config(
            run_dir=week4_root,
            run_type="week4_lightweight_router",
            module="src.utils.build_week4_artifacts",
            argv=[
                "python",
                "-m",
                "src.utils.build_week4_artifacts",
                "--week3-root",
                week3_root.as_posix(),
                "--output-root",
                week4_root.as_posix(),
            ],
            inputs={"week3_root": portable(week3_root)},
            artifacts={
                "router_predictions_csv": portable(predictions_path),
                "router_metrics_json": portable(metrics_path),
                "motion_quality_features_csv": portable(week4_root / "features/week4_motion_quality_features.csv"),
                "fold_assignments_csv": portable(week4_root / "tables/router_fold_assignments.csv"),
                "router_model_manifest_json": portable(week4_root / "models/router_model_manifest.json"),
            },
        )
    run_log_path = week4_root / "run_log.json"
    if not run_log_path.exists():
        run_log_path.write_text(
            json.dumps({**read_json(run_config_path), "metrics": read_json(metrics_path)}, indent=2),
            encoding="utf-8",
        )
    return {
        "root": portable(week4_root),
        "predictions_csv": portable(week4_root / "predictions.csv"),
        "metrics_json": portable(week4_root / "metrics.json"),
        "run_config_json": portable(run_config_path),
        "run_log_json": portable(run_log_path),
        "feature_csv": portable(week4_root / "features/week4_motion_quality_features.csv"),
        "fold_assignments_csv": portable(week4_root / "tables/router_fold_assignments.csv"),
        "router_model_manifest_json": portable(week4_root / "models/router_model_manifest.json"),
    }


def write_run_config(
    run_dir: Path,
    run_type: str,
    module: str,
    argv: list[Any],
    inputs: dict[str, Any],
    artifacts: dict[str, Any],
) -> Path:
    path = run_dir / "run_config.json"
    payload = {
        "schema_version": "run_config_v1",
        "run_type": run_type,
        "module": module,
        "argv": argv,
        "inputs": inputs,
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def extract_inputs(metrics: dict[str, Any], run_log: dict[str, Any]) -> dict[str, Any]:
    return {
        "processed_manifest_path": first_present(
            metrics.get("processed_manifest_path"),
            metrics.get("source_processed_manifest_path"),
            run_log.get("input_source", {}).get("processed_manifest_path"),
            metrics.get("feature_manifest_path"),
        ),
        "split_config_path": first_present(
            metrics.get("split_config_path"),
            run_log.get("input_source", {}).get("split_config_path"),
        ),
        "feature_manifest_path": first_present(metrics.get("feature_manifest_path"), run_log.get("feature_manifest_path")),
        "ppg_source": first_present(metrics.get("ppg_source"), run_log.get("input_source", {}).get("ppg_source")),
    }


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def copy_artifact(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bundle_path(manifest_path: Path, stored_path: str) -> Path:
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return candidate
    direct = Path(stored_path)
    if direct.exists():
        return direct
    return manifest_path.parent / candidate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Reproducibility Package Manifest",
        "",
        f"- Corrected processed manifest: `{manifest['corrected_processed_manifest']}`",
        f"- Week 2 run folders: `{len(manifest['week2']['runs'])}`",
        f"- Week 2 embedding bundles: `{len(manifest['week2']['embedding_bundles'])}`",
        f"- Week 3 root: `{manifest['week3']['root']}`",
        f"- Week 4 root: `{manifest['week4']['root']}`",
        "",
        "This manifest indexes the fixed filenames added for reproducible Week 2-4 reruns.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def portable(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    main()
