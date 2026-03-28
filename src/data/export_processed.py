"""Build and persist processed GalaxyPPG windows and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import build_window_dataset, resolve_dataset_root, split_by_participant
from src.data.cache import annotate_window_dataset, save_processed_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--reference-source", choices=["hr", "ibi"], default="hr")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--label-aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Build processed windows, add a participant split, and save the artifacts."""

    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    windows = build_window_dataset(
        dataset_root=dataset_root,
        reference_source=args.reference_source,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_aggregation=args.label_aggregation,
    )
    split_config = load_split_config(args.split_config)
    _, _, train_participants, test_participants = split_by_participant(
        windows,
        test_size=args.test_size,
        random_state=args.random_state,
        test_participants=split_config.get("test_participants"),
    )
    annotated_windows = annotate_window_dataset(
        windows=windows,
        train_participants=train_participants,
        test_participants=test_participants,
    )
    manifest = save_processed_dataset(
        windows=annotated_windows,
        dataset_root=dataset_root,
        reference_source=args.reference_source,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_aggregation=args.label_aggregation,
        test_size=args.test_size,
        random_state=args.random_state,
        train_participants=train_participants,
        test_participants=test_participants,
        output_root=args.output_root,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))


def load_split_config(split_config_path: Path | None) -> dict[str, object]:
    """Load a participant split config when one is provided."""

    if split_config_path is None:
        return {}
    payload = json.loads(split_config_path.read_text(encoding="utf-8"))
    if "test_participants" not in payload:
        raise KeyError("Split config must contain a `test_participants` field.")
    return payload


if __name__ == "__main__":
    main()
