"""Run baseline HR estimation methods on GalaxyPPG windows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.baseline import apply_peak_detection_baseline, apply_spectral_baseline
from src.data import build_window_dataset, default_stride_rationale, split_by_participant
from src.regression.evaluate import evaluate_prediction_frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["peak", "spectral"], default="peak")
    parser.add_argument("--reference-source", choices=["hr", "ibi"], default="hr")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "baseline_results" / datetime.now().strftime("%Y-%m-%d"),
    )
    return parser.parse_args()


def main() -> None:
    """Build windows, run one baseline, evaluate, and save the outputs."""

    args = parse_args()
    windows = build_window_dataset(
        reference_source=args.reference_source,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    _, test_windows, train_participants, test_participants = split_by_participant(
        windows,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.method == "peak":
        predictions = apply_peak_detection_baseline(test_windows)
    else:
        predictions = apply_spectral_baseline(test_windows)

    summary = evaluate_prediction_frame(predictions)
    metadata = {
        "method": args.method,
        "reference_source": args.reference_source,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "stride_rationale": default_stride_rationale(),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_participants": train_participants,
        "test_participants": test_participants,
        **summary.to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.method}_predictions.csv"
    metrics_path = args.output_dir / f"{args.method}_metrics.json"
    run_log_path = args.output_dir / f"{args.method}_run_log.json"
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    run_log_path.write_text(
        json.dumps(
            {
                "module": "src.baseline.run_baseline",
                "argv": sys.argv,
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
                "metrics": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(f"run_log={run_log_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
