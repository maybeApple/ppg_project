"""Build and persist processed GalaxyPPG windows and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import (
    LabelGenerationConfig,
    build_window_dataset,
    canonical_schema_description,
    list_participants,
    resolve_dataset_root,
    split_by_participant,
)
from src.data.cache import annotate_window_dataset, save_processed_dataset
from src.data.splits import configured_participant_ids, load_fixed_split_config, resolve_split_config_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--reference-source", choices=["auto", "hr", "ibi", "ecg"], default="ibi")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument(
        "--label-method",
        choices=["beat_interval_instant_hr", "provided_hr_samples"],
        default="beat_interval_instant_hr",
    )
    parser.add_argument("--label-aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--min-valid-beats", type=int, default=2)
    parser.add_argument("--min-reference-samples", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Build processed windows, add a participant split, and save the artifacts."""

    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    split_config_path = resolve_split_config_path(args.split_config)
    fixed_split = load_fixed_split_config(split_config_path) if split_config_path is not None else None
    participant_ids = configured_participant_ids(fixed_split) if fixed_split is not None else None
    if participant_ids is not None:
        available_participants = set(list_participants(dataset_root))
        missing_participants = sorted(set(participant_ids) - available_participants)
        if missing_participants:
            raise FileNotFoundError(
                "The split config references participant folders that are missing from the raw dataset: "
                f"{missing_participants}"
            )

    windows = build_window_dataset(
        dataset_root=dataset_root,
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
            "Preprocessing produced zero windows. Check the raw dataset placement, expected folder structure, "
            "timestamp alignment inputs, and filtering thresholds."
        )

    _, _, train_participants, test_participants = split_by_participant(
        windows,
        test_size=args.test_size,
        random_state=args.random_state,
        test_participants=None if fixed_split is None else fixed_split.test_participants,
    )
    annotated_windows = annotate_window_dataset(
        windows=windows,
        train_participants=train_participants,
        test_participants=test_participants,
    )
    manifest = save_processed_dataset(
        windows=annotated_windows,
        dataset_root=dataset_root,
        canonical_schema=canonical_schema_description().to_dict(),
        reference_source=args.reference_source,
        ppg_inverted=bool(annotated_windows["ppg_inverted"].all()),
        ppg_canonical_source=str(annotated_windows["ppg_canonical_source"].iloc[0]),
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_method=args.label_method,
        label_aggregation=args.label_aggregation,
        min_valid_beats=args.min_valid_beats,
        min_reference_samples=args.min_reference_samples,
        label_generation=LabelGenerationConfig(
            method=args.label_method,
            aggregation=args.label_aggregation,
            min_valid_beats=args.min_valid_beats,
            min_reference_samples=args.min_reference_samples,
        ).to_dict(),
        test_size=args.test_size,
        random_state=args.random_state,
        train_participants=train_participants,
        test_participants=test_participants,
        split_config_path=None if fixed_split is None else fixed_split.split_config_path,
        split_name=None if fixed_split is None else fixed_split.split_name,
        validation_strategy=None if fixed_split is None else fixed_split.validation_strategy,
        validation_folds=None if fixed_split is None else [fold.to_dict() for fold in fixed_split.validation_folds],
        output_root=args.output_root,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
