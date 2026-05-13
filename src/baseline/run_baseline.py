"""Run baseline HR estimation methods on GalaxyPPG windows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.baseline import apply_peak_detection_baseline, apply_spectral_baseline
from src.data import (
    LabelGenerationConfig,
    build_window_dataset,
    configured_participant_ids,
    default_stride_rationale,
    load_fixed_split_config,
    load_processed_manifest,
    load_processed_windows,
    resolve_split_config_path,
    split_by_participant,
)
from src.regression.evaluate import evaluate_prediction_frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["peak", "spectral"], default="peak")
    parser.add_argument("--reference-source", choices=["auto", "hr", "ibi", "ecg"], default="ibi")
    parser.add_argument(
        "--label-method",
        choices=["beat_interval_instant_hr", "provided_hr_samples"],
        default="beat_interval_instant_hr",
    )
    parser.add_argument("--label-aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--min-valid-beats", type=int, default=2)
    parser.add_argument("--min-reference-samples", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--processed-manifest", type=Path, default=None)
    parser.add_argument(
        "--ppg-source",
        choices=["canonical", "raw"],
        default="canonical",
        help="Use canonical inverted PPG or raw non-inverted PPG from processed windows.",
    )
    parser.add_argument("--split-config", type=Path, default=None)
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
    split_config_path = resolve_split_config_path(args.split_config)
    fixed_split = load_fixed_split_config(split_config_path) if split_config_path is not None else None

    if args.processed_manifest is not None:
        processed_manifest = load_processed_manifest(args.processed_manifest)
        windows = load_processed_windows(processed_manifest["windows_path"])
        if windows.empty:
            raise RuntimeError("The processed manifest exists, but its window table is empty.")
        if "split" not in windows.columns:
            raise KeyError("Processed windows must contain a `split` column.")
        train_windows = windows.loc[windows["split"] == "train"].reset_index(drop=True)
        test_windows = windows.loc[windows["split"] == "test"].reset_index(drop=True)
        train_participants = sorted(train_windows["participant_id"].unique().tolist())
        test_participants = sorted(test_windows["participant_id"].unique().tolist())
        if fixed_split is not None:
            if train_participants != sorted(fixed_split.train_participants):
                raise ValueError(
                    "Processed manifest train participants do not match the fixed split config. "
                    f"Observed {train_participants}, expected {sorted(fixed_split.train_participants)}."
                )
            if test_participants != sorted(fixed_split.test_participants):
                raise ValueError(
                    "Processed manifest test participants do not match the fixed split config. "
                    f"Observed {test_participants}, expected {sorted(fixed_split.test_participants)}."
                )
        input_source = {
            "mode": "processed_manifest",
            "processed_manifest_path": str(Path(processed_manifest["manifest_path"]).resolve()),
            "split_config_path": None if fixed_split is None else str(fixed_split.split_config_path),
            "ppg_source": args.ppg_source,
        }
    else:
        processed_manifest = {}
        participant_ids = configured_participant_ids(fixed_split) if fixed_split is not None else None
        windows = build_window_dataset(
            dataset_root=args.dataset_root,
            participant_ids=participant_ids,
            reference_source=args.reference_source,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            label_method=args.label_method,
            label_aggregation=args.label_aggregation,
            min_valid_beats=args.min_valid_beats,
            min_reference_samples=args.min_reference_samples,
        )
        if windows.empty:
            raise RuntimeError(
                "Baseline preprocessing produced zero windows. Check the dataset path, participant folders, "
                "and preprocessing parameters."
            )
        _, test_windows, train_participants, test_participants = split_by_participant(
            windows,
            test_size=args.test_size,
            random_state=args.random_state,
            test_participants=None if fixed_split is None else fixed_split.test_participants,
        )
        input_source = {
            "mode": "raw_dataset",
            "dataset_root": str(args.dataset_root) if args.dataset_root is not None else "",
            "split_config_path": None if fixed_split is None else str(fixed_split.split_config_path),
            "ppg_source": args.ppg_source,
        }

    if test_windows.empty:
        raise RuntimeError("The selected baseline run has zero test windows, so evaluation cannot proceed.")

    ppg_inverted_for_run = bool(processed_manifest.get("ppg_inverted")) if args.ppg_source == "canonical" else False
    if args.ppg_source == "raw":
        if "ppg_raw_values" not in test_windows.columns:
            raise KeyError("`--ppg-source raw` requires processed windows with `ppg_raw_values`.")
        test_windows = test_windows.copy()
        test_windows["ppg_values"] = test_windows["ppg_raw_values"]
        if "ppg_inverted" in test_windows.columns:
            test_windows["ppg_inverted"] = False

    if args.method == "peak":
        predictions = apply_peak_detection_baseline(test_windows)
    else:
        predictions = apply_spectral_baseline(test_windows)
    predictions["ppg_source"] = args.ppg_source
    predictions["inversion"] = ppg_inverted_for_run

    summary = evaluate_prediction_frame(predictions)
    metadata = {
        "method": args.method,
        "reference_source": processed_manifest.get("reference_source", args.reference_source),
        "window_seconds": processed_manifest.get("window_seconds", args.window_seconds),
        "stride_seconds": processed_manifest.get("stride_seconds", args.stride_seconds),
        "stride_rationale": default_stride_rationale(),
        "label_method": processed_manifest.get("label_method", args.label_method),
        "label_aggregation": processed_manifest.get("label_aggregation", args.label_aggregation),
        "min_valid_beats": processed_manifest.get("min_valid_beats", args.min_valid_beats),
        "min_reference_samples": processed_manifest.get("min_reference_samples", args.min_reference_samples),
        "label_generation": processed_manifest.get(
            "label_generation",
            LabelGenerationConfig(
                method=args.label_method,
                aggregation=args.label_aggregation,
                min_valid_beats=args.min_valid_beats,
                min_reference_samples=args.min_reference_samples,
            ).to_dict(),
        ),
        "ppg_inverted": ppg_inverted_for_run,
        "ppg_source": args.ppg_source,
        "ppg_canonical_source": processed_manifest.get("ppg_canonical_source"),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_participants": train_participants,
        "test_participants": test_participants,
        "split_config_path": None if fixed_split is None else str(fixed_split.split_config_path),
        "split_name": None if fixed_split is None else fixed_split.split_name,
        "validation_strategy": None if fixed_split is None else fixed_split.validation_strategy,
        "validation_folds": None if fixed_split is None else [fold.to_dict() for fold in fixed_split.validation_folds],
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
                "input_source": input_source,
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
