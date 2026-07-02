"""Freeze selected experiment artifacts into a final reproducibility package."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any


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
    parser.add_argument("--week7-root", type=Path, default=Path("experiments/week7_final_statistics"))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or Path("experiments") / f"final_frozen_results_{date.today().isoformat()}"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "final_frozen_results_v1",
        "output_root": output_root.as_posix(),
        "sources": {
            "week2_root": args.week2_root.as_posix(),
            "week3_root": args.week3_root.as_posix(),
            "week4_root": args.week4_root.as_posix(),
            "week7_root": args.week7_root.as_posix(),
        },
        "artifacts": [],
        "artifact_files": [],
        "commands": reproduction_commands(args),
    }
    copy_groups = [
        ("configs", Path("configs"), output_root / "configs"),
        ("week2_runs", args.week2_root / "runs", output_root / "week2" / "runs"),
        ("week2_tables", args.week2_root / "tables", output_root / "week2" / "tables"),
        ("week2_metrics", args.week2_root / "metrics", output_root / "week2" / "metrics"),
        ("week2_predictions", args.week2_root / "predictions", output_root / "week2" / "predictions"),
        ("week2_figures", args.week2_root / "figures", output_root / "week2" / "figures"),
        ("week3_tables", args.week3_root / "tables", output_root / "week3" / "tables"),
        ("week3_predictions", args.week3_root / "predictions", output_root / "week3" / "predictions"),
        ("week3_metrics", args.week3_root / "metrics", output_root / "week3" / "metrics"),
        ("week3_figures", args.week3_root / "figures", output_root / "week3" / "figures"),
        ("week4_tables", args.week4_root / "tables", output_root / "week4" / "tables"),
        ("week4_predictions", args.week4_root / "predictions", output_root / "week4" / "predictions"),
        ("week4_metrics", args.week4_root / "metrics", output_root / "week4" / "metrics"),
        ("week4_features", args.week4_root / "features", output_root / "week4" / "features"),
        ("week4_models", args.week4_root / "models", output_root / "week4" / "models"),
        ("week4_figures", args.week4_root / "figures", output_root / "week4" / "figures"),
        ("week7_statistics", args.week7_root, output_root / "week7_statistics"),
    ]
    for label, source, target in copy_groups:
        manifest["artifacts"].append(copy_tree_if_exists(label, source, target))

    copy_files = [
        ("reproducibility_manifest_json", Path("experiments/reproducibility_manifest.json"), output_root / "reproducibility_manifest.json"),
        ("reproducibility_manifest_md", Path("experiments/reproducibility_manifest.md"), output_root / "reproducibility_manifest.md"),
        ("week2_run_index", args.week2_root / "run_index.csv", output_root / "week2" / "run_index.csv"),
        ("week2_run_manifest", args.week2_root / "run_manifest.csv", output_root / "week2" / "run_manifest.csv"),
        ("week2_embedding_manifest", args.week2_root / "embedding_manifest.csv", output_root / "week2" / "embedding_manifest.csv"),
        ("week2_memo", args.week2_root / "week2_memo.md", output_root / "week2" / "week2_memo.md"),
        ("week3_memo", args.week3_root / "week3_regime_analysis.md", output_root / "week3" / "week3_regime_analysis.md"),
        ("week3_predictions_csv", args.week3_root / "predictions.csv", output_root / "week3" / "predictions.csv"),
        ("week3_metrics_json", args.week3_root / "metrics.json", output_root / "week3" / "metrics.json"),
        ("week3_run_config", args.week3_root / "run_config.json", output_root / "week3" / "run_config.json"),
        ("week3_run_log", args.week3_root / "run_log.json", output_root / "week3" / "run_log.json"),
        ("week4_memo", args.week4_root / "week4_lightweight_router.md", output_root / "week4" / "week4_lightweight_router.md"),
        ("week4_predictions_csv", args.week4_root / "predictions.csv", output_root / "week4" / "predictions.csv"),
        ("week4_metrics_json", args.week4_root / "metrics.json", output_root / "week4" / "metrics.json"),
        ("week4_run_config", args.week4_root / "run_config.json", output_root / "week4" / "run_config.json"),
        ("week4_run_log", args.week4_root / "run_log.json", output_root / "week4" / "run_log.json"),
    ]
    for label, source, target in copy_files:
        manifest["artifact_files"].append(copy_file_if_exists(label, source, target))

    manifest["artifact_files"].extend(write_frozen_path_indexes(output_root, args))

    (output_root / "final_frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(output_root / "README.md", manifest)
    print(f"frozen_results={output_root}")


def copy_tree_if_exists(label: str, source: Path, target: Path) -> dict[str, Any]:
    """Copy a directory tree when present and record missing groups explicitly."""

    if not source.exists():
        return {
            "label": label,
            "source": source.as_posix(),
            "target": target.as_posix(),
            "copied": False,
            "reason": "source_missing",
        }
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return {
        "label": label,
        "source": source.as_posix(),
        "target": target.as_posix(),
        "copied": True,
        "reason": None,
    }


def write_frozen_path_indexes(output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Write package-local versions of source manifests for reviewers using only the frozen directory."""

    files: list[dict[str, Any]] = []
    replacements = {
        args.week2_root.as_posix(): (output_root / "week2").as_posix(),
        str(args.week2_root): (output_root / "week2").as_posix(),
        args.week3_root.as_posix(): (output_root / "week3").as_posix(),
        str(args.week3_root): (output_root / "week3").as_posix(),
        args.week4_root.as_posix(): (output_root / "week4").as_posix(),
        str(args.week4_root): (output_root / "week4").as_posix(),
        args.week7_root.as_posix(): (output_root / "week7_statistics").as_posix(),
        str(args.week7_root): (output_root / "week7_statistics").as_posix(),
    }
    for label, source_name, target_name in [
        ("week2_run_manifest_frozen_paths", "run_manifest.csv", "run_manifest_frozen_paths.csv"),
        (
            "week2_embedding_manifest_frozen_paths",
            "embedding_manifest.csv",
            "embedding_manifest_frozen_paths.csv",
        ),
    ]:
        source = output_root / "week2" / source_name
        target = output_root / "week2" / target_name
        if not source.exists():
            files.append(
                {
                    "label": label,
                    "source": source.as_posix(),
                    "target": target.as_posix(),
                    "copied": False,
                    "reason": "source_missing",
                }
            )
            continue
        text = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        target.write_text(text, encoding="utf-8")
        files.append(
            {
                "label": label,
                "source": source.as_posix(),
                "target": target.as_posix(),
                "copied": True,
                "reason": None,
            }
        )
    return files


def copy_file_if_exists(label: str, source: Path, target: Path) -> dict[str, Any]:
    """Copy a single artifact file when present and record missing files explicitly."""

    if not source.exists():
        return {
            "label": label,
            "source": source.as_posix(),
            "target": target.as_posix(),
            "copied": False,
            "reason": "source_missing",
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "label": label,
        "source": source.as_posix(),
        "target": target.as_posix(),
        "copied": True,
        "reason": None,
    }


def reproduction_commands(args: argparse.Namespace) -> list[str]:
    """Return final package reproduction commands."""

    return [
        "python -m src.utils.build_week2_artifacts --search-root experiments/week2_galaxyppg_corrected_2026-05-01/runs --output-root experiments/week2_galaxyppg_corrected_2026-05-01 --tag-name week2-galaxyppg-corrected-2026-05-01",
        "python -m src.utils.build_week3_artifacts --week2-root experiments/week2_galaxyppg_corrected_2026-05-01 --output-root experiments/week3_galaxyppg_regime_oracle_2026-05-13",
        "python -m src.utils.build_week4_artifacts --week3-root experiments/week3_galaxyppg_regime_oracle_2026-05-13 --output-root experiments/week4_galaxyppg_lightweight_router_2026-05-13",
        "python -m src.utils.build_week7_statistics --week4-root experiments/week4_galaxyppg_lightweight_router_2026-05-13 --output-root experiments/week7_final_statistics",
        f"python -m src.utils.freeze_final_results --week2-root {args.week2_root.as_posix()} --week3-root {args.week3_root.as_posix()} --week4-root {args.week4_root.as_posix()} --week7-root {args.week7_root.as_posix()}",
    ]


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    """Write a short README for the frozen result package."""

    lines = [
        "# Final Frozen Results",
        "",
        "This directory contains copied, immutable experiment artifacts for paper drafting and reproducibility review.",
        "",
        "## Sources",
        "",
    ]
    for key, value in manifest["sources"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Reproduction Commands", ""])
    for command in manifest["commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["## Artifact Groups", ""])
    for artifact in manifest["artifacts"]:
        lines.append(f"- {artifact['label']}: `{artifact['target']}` copied={artifact['copied']}")
    lines.extend(["", "## Artifact Files", ""])
    for artifact in manifest["artifact_files"]:
        lines.append(f"- {artifact['label']}: `{artifact['target']}` copied={artifact['copied']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
