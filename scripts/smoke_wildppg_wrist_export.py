"""Smoke-test WildPPG wrist manifest export and baseline reuse.

This script writes only to a temporary directory. It does not create or commit
synthetic repository artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wildppg_smoke_") as tmp:
        tmp_root = Path(tmp)
        dataset_root = tmp_root / "WildPPG-wrist"
        output_root = tmp_root / "processed"
        baseline_root = tmp_root / "baseline_peak"
        dataset_root.mkdir(parents=True)

        manifest_rows = []
        for participant_index in range(1, 3):
            participant_id = f"W{participant_index:02d}"
            write_participant_csvs(dataset_root, participant_id, phase=participant_index * 0.2)
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": "synthetic",
                    "activity": "walking",
                    "ppg_path": f"{participant_id}/wrist_ppg.csv",
                    "ppg_col": "ppg",
                    "ppg_sampling_hz": 64,
                    "acc_path": f"{participant_id}/wrist_acc.csv",
                    "acc_x_col": "x",
                    "acc_y_col": "y",
                    "acc_z_col": "z",
                    "acc_sampling_hz": 32,
                    "reference_path": f"{participant_id}/rr.csv",
                    "reference_time_col": "timestamp_ms",
                    "rr_col": "rr_interval_ms",
                    "time_unit": "ms",
                }
            )
        pd.DataFrame(manifest_rows).to_csv(dataset_root / "wildppg_wrist_manifest.csv", index=False)

        run(
            [
                sys.executable,
                "-m",
                "src.data.export_wildppg_wrist",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--test-size",
                "0.5",
            ]
        )

        manifest_path = output_root / "wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json"
        assert manifest_path.exists(), manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataset"] == "WildPPG-wrist"
        assert manifest["window_seconds"] == 10.0
        assert manifest["stride_seconds"] == 2.0
        assert manifest["label_method"] == "beat_interval_instant_hr"
        assert manifest["label_aggregation"] == "median"
        assert manifest["split_config_path"] is None
        assert manifest["num_windows"] > 0

        run(
            [
                sys.executable,
                "-m",
                "src.baseline.run_baseline",
                "--processed-manifest",
                str(manifest_path),
                "--method",
                "peak",
                "--output-dir",
                str(baseline_root),
            ]
        )

        metrics = json.loads((baseline_root / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["split_config_path"] is None
        assert metrics["label_generation"]["instant_hr_formula"] == "instant_hr_bpm = 60000 / rr_interval_ms"
        assert metrics["num_windows"] > 0

    print("WildPPG wrist synthetic export smoke test passed")


def write_participant_csvs(root: Path, participant_id: str, phase: float) -> None:
    duration_s = 80
    participant_root = root / participant_id
    participant_root.mkdir(parents=True)

    t_ppg = np.arange(duration_s * 64) / 64
    ppg = np.sin(2.0 * np.pi * 1.0 * t_ppg + phase) + 0.05 * np.sin(2.0 * np.pi * 2.0 * t_ppg)
    pd.DataFrame({"ppg": ppg}).to_csv(participant_root / "wrist_ppg.csv", index=False)

    t_acc = np.arange(duration_s * 32) / 32
    acc = pd.DataFrame(
        {
            "x": 0.01 * np.sin(2.0 * np.pi * 0.5 * t_acc),
            "y": 0.01 * np.cos(2.0 * np.pi * 0.5 * t_acc),
            "z": np.ones_like(t_acc),
        }
    )
    acc.to_csv(participant_root / "wrist_acc.csv", index=False)

    beat_times_ms = np.arange(1000, duration_s * 1000, 1000, dtype=int)
    pd.DataFrame({"timestamp_ms": beat_times_ms, "rr_interval_ms": 1000.0}).to_csv(
        participant_root / "rr.csv",
        index=False,
    )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
