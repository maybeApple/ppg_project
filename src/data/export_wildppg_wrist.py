"""Build processed WildPPG wrist windows and ECG/RR-derived labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.cache import annotate_window_dataset, save_processed_dataset
from src.data.canonical import canonical_schema_description
from src.data.labels import LabelGenerationConfig
from src.data.preprocessing import align_participant_sessions
from src.data.wildppg_loader import (
    DATASET_NAME,
    list_wildppg_participants,
    load_wildppg_participant_data,
    resolve_wildppg_root,
)
from src.data.windowing import default_stride_rationale, generate_windows_from_sessions, split_by_participant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
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
    parser.add_argument("--min-ppg-coverage", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--participants", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_wildppg_root(args.dataset_root)
    participant_ids = args.participants or list_wildppg_participants(dataset_root)
    windows = build_wildppg_window_dataset(
        dataset_root=dataset_root,
        participant_ids=participant_ids,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_method=args.label_method,
        label_aggregation=args.label_aggregation,
        min_valid_beats=args.min_valid_beats,
        min_reference_samples=args.min_reference_samples,
        min_ppg_coverage=args.min_ppg_coverage,
    )
    if windows.empty:
        raise RuntimeError(
            "WildPPG wrist preprocessing produced zero windows. Check the manifest paths, timestamp columns, "
            "sampling rates, and ECG/RR reference columns."
        )

    _, _, train_participants, test_participants = split_by_participant(
        windows=windows,
        test_size=args.test_size,
        random_state=args.random_state,
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
        reference_source="ecg",
        ppg_inverted=False,
        ppg_canonical_source="manifest_configured_wrist_ppg",
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_method=args.label_method,
        label_aggregation=args.label_aggregation,
        min_valid_beats=args.min_valid_beats,
        min_reference_samples=args.min_reference_samples,
        label_generation={
            **LabelGenerationConfig(
                method=args.label_method,
                aggregation=args.label_aggregation,
                min_valid_beats=args.min_valid_beats,
                min_reference_samples=args.min_reference_samples,
            ).to_dict(),
            "dataset_note": "WildPPG wrist labels are generated from the manifest-configured ECG or RR-reference columns.",
            "stride_rationale": default_stride_rationale(),
        },
        test_size=args.test_size,
        random_state=args.random_state,
        train_participants=train_participants,
        test_participants=test_participants,
        split_config_path=None,
        split_name="wildppg_wrist_random_subject_holdout",
        validation_strategy={
            "type": "participant_grouped_validation_pending",
            "note": "Week 6 starts with a deterministic subject-independent holdout. Add fixed validation folds before final reporting.",
        },
        validation_folds=None,
        output_root=args.output_root,
        dataset_name=DATASET_NAME,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))


def build_wildppg_window_dataset(
    dataset_root: str | Path | None = None,
    participant_ids: list[str] | None = None,
    window_seconds: float = 10.0,
    stride_seconds: float = 2.0,
    label_method: str = "beat_interval_instant_hr",
    label_aggregation: str = "median",
    min_valid_beats: int = 2,
    min_reference_samples: int = 1,
    min_ppg_coverage: float = 0.8,
) -> pd.DataFrame:
    """Load WildPPG wrist records and generate one window table."""

    selected_participants = participant_ids or list_wildppg_participants(dataset_root)
    label_config = LabelGenerationConfig(
        method=label_method,
        aggregation=label_aggregation,
        min_valid_beats=min_valid_beats,
        min_reference_samples=min_reference_samples,
    )
    all_windows: list[pd.DataFrame] = []
    for participant_id in selected_participants:
        participant = load_wildppg_participant_data(
            participant_id=participant_id,
            dataset_root=dataset_root,
            reference_source="ecg",
        )
        aligned_sessions = align_participant_sessions(participant=participant, min_duration_seconds=window_seconds)
        participant_windows = generate_windows_from_sessions(
            aligned_sessions=aligned_sessions,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            label_config=label_config,
            min_ppg_coverage=min_ppg_coverage,
        )
        if not participant_windows.empty:
            all_windows.append(participant_windows)
    if not all_windows:
        return pd.DataFrame()
    return pd.concat(all_windows, ignore_index=True)


if __name__ == "__main__":
    main()
