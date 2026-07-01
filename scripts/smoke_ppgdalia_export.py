"""Smoke-test PPG-DaLiA export and baseline reuse with synthetic data.

This script writes only to a temporary directory. It does not create or commit
synthetic repository artifacts.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ppgdalia_smoke_") as tmp:
        tmp_root = Path(tmp)
        dataset_root = tmp_root / "PPG-DaLiA"
        output_root = tmp_root / "processed"
        baseline_root = tmp_root / "baseline_peak"
        dataset_root.mkdir(parents=True)

        for participant_index in range(1, 3):
            write_participant_pickle(dataset_root / f"S{participant_index}.pkl", phase=participant_index * 0.2)

        run(
            [
                sys.executable,
                "-m",
                "src.data.export_ppgdalia",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--test-size",
                "0.5",
            ]
        )

        manifest_path = output_root / "ppg_dalia_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json"
        assert manifest_path.exists(), manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataset"] == "PPG-DaLiA"
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

    print("PPG-DaLiA synthetic export smoke test passed")


def write_participant_pickle(path: Path, phase: float) -> None:
    duration_s = 80
    bvp_hz = 64
    acc_hz = 32
    ecg_hz = 700
    t_bvp = np.arange(duration_s * bvp_hz) / bvp_hz
    t_acc = np.arange(duration_s * acc_hz) / acc_hz
    bvp = np.sin(2.0 * np.pi * 1.0 * t_bvp + phase) + 0.05 * np.sin(2.0 * np.pi * 2.0 * t_bvp)
    acc = np.column_stack(
        [
            0.01 * np.sin(2.0 * np.pi * 0.5 * t_acc),
            0.01 * np.cos(2.0 * np.pi * 0.5 * t_acc),
            np.ones_like(t_acc),
        ]
    )
    ecg = np.zeros(duration_s * ecg_hz, dtype=float)
    rpeaks = np.arange(ecg_hz, duration_s * ecg_hz, ecg_hz, dtype=int)
    ecg[rpeaks] = 1.0
    payload = {
        "signal": {"wrist": {"BVP": bvp, "ACC": acc}, "chest": {"ECG": ecg}},
        "rpeaks": rpeaks,
        "activity": np.full(duration_s * 4, 7, dtype=int),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
