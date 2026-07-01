"""Build an honest external-validation status table without fabricating metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RUN_SPECS = [
    ("PPG-DaLiA", "baseline_peak", "week5_ppgdalia_external_validation/runs/baseline_peak/metrics.json"),
    ("PPG-DaLiA", "baseline_spectral", "week5_ppgdalia_external_validation/runs/baseline_spectral/metrics.json"),
    ("PPG-DaLiA", "pulseppg_linear", "week5_ppgdalia_external_validation/runs/pulseppg_linear/metrics.json"),
    ("PPG-DaLiA", "pulseppg_ridge", "week5_ppgdalia_external_validation/runs/pulseppg_ridge/metrics.json"),
    ("PPG-DaLiA", "papagei_linear", "week5_ppgdalia_external_validation/runs/papagei_linear/metrics.json"),
    ("PPG-DaLiA", "papagei_ridge", "week5_ppgdalia_external_validation/runs/papagei_ridge/metrics.json"),
    ("WildPPG wrist", "baseline_peak", "week6_wildppg_wrist_external_validation/runs/baseline_peak/metrics.json"),
    ("WildPPG wrist", "baseline_spectral", "week6_wildppg_wrist_external_validation/runs/baseline_spectral/metrics.json"),
    ("WildPPG wrist", "pulseppg_linear", "week6_wildppg_wrist_external_validation/runs/pulseppg_linear/metrics.json"),
    ("WildPPG wrist", "pulseppg_ridge", "week6_wildppg_wrist_external_validation/runs/pulseppg_ridge/metrics.json"),
    ("WildPPG wrist", "papagei_linear", "week6_wildppg_wrist_external_validation/runs/papagei_linear/metrics.json"),
    ("WildPPG wrist", "papagei_ridge", "week6_wildppg_wrist_external_validation/runs/papagei_ridge/metrics.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/external_validation_result_table.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/external_validation_result_table.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.experiments_root)
    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    write_markdown(args.output_md, table)
    print(f"csv={args.output_csv}")
    print(f"markdown={args.output_md}")


def build_rows(experiments_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, run_name, relative_metrics_path in RUN_SPECS:
        metrics_path = experiments_root / relative_metrics_path
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "run": run_name,
                    "metrics_path": metrics_path.as_posix(),
                    "status": "complete",
                    "mae": metrics.get("mae", metrics.get("MAE", math.nan)),
                    "rmse": metrics.get("rmse", metrics.get("RMSE", math.nan)),
                    "p95_absolute_error": metrics.get("p95_absolute_error", math.nan),
                    "catastrophic_error_rate_20bpm": metrics.get("catastrophic_error_rate_20bpm", math.nan),
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "run": run_name,
                    "metrics_path": metrics_path.as_posix(),
                    "status": "requires_real_data",
                    "mae": math.nan,
                    "rmse": math.nan,
                    "p95_absolute_error": math.nan,
                    "catastrophic_error_rate_20bpm": math.nan,
                }
            )
    return rows


def write_markdown(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# External Validation Result Table",
        "",
        "This table is generated from local Week 5/6 metrics when real external-dataset runs exist.",
        "Rows marked `requires_real_data` are placeholders, not fabricated results.",
        "",
        table.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
