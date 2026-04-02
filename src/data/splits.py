"""Helpers for fixed participant splits and validation folds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ValidationFold:
    """One fixed validation fold defined at the participant level."""

    fold: int
    train_participants: list[str]
    validation_participants: list[str]
    num_train_windows: int | None = None
    num_validation_windows: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the fold definition into a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class FixedSplitConfig:
    """Repository-level reproducibility split definition."""

    split_name: str
    random_state: int
    train_participants: list[str]
    test_participants: list[str]
    validation_strategy: dict[str, object]
    validation_folds: list[ValidationFold]
    split_config_path: Path

    def to_dict(self) -> dict[str, object]:
        """Convert the split config into a JSON-serializable dictionary."""

        payload = asdict(self)
        payload["split_config_path"] = str(self.split_config_path)
        return payload


def default_split_config_path() -> Path:
    """Return the repository-default fixed split config path."""

    return Path(__file__).resolve().parents[2] / "configs" / "galaxyppg_submission_split.json"


def resolve_split_config_path(split_config_path: str | Path | None = None) -> Path | None:
    """Resolve a split config path, falling back to the repository default when present."""

    candidate = Path(split_config_path) if split_config_path is not None else default_split_config_path()
    if not candidate.exists():
        if split_config_path is None:
            return None
        raise FileNotFoundError(f"Split config does not exist: {candidate}")
    return candidate.resolve()


def load_fixed_split_config(split_config_path: str | Path | None = None) -> FixedSplitConfig:
    """Load and validate the fixed split config used by this repository."""

    resolved_path = resolve_split_config_path(split_config_path)
    if resolved_path is None:
        raise FileNotFoundError(
            "No split config was found. Pass --split-config or place "
            "`configs/galaxyppg_submission_split.json` in the repository."
        )

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    required_keys = {
        "split_name",
        "random_state",
        "train_participants",
        "test_participants",
        "validation_strategy",
        "validation_folds",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise KeyError(f"Split config is missing keys: {missing_keys}")

    validation_folds = [
        ValidationFold(
            fold=int(item["fold"]),
            train_participants=list(item["train_participants"]),
            validation_participants=list(item["validation_participants"]),
            num_train_windows=_as_optional_int(item.get("num_train_windows")),
            num_validation_windows=_as_optional_int(item.get("num_validation_windows")),
        )
        for item in payload["validation_folds"]
    ]
    return FixedSplitConfig(
        split_name=str(payload["split_name"]),
        random_state=int(payload["random_state"]),
        train_participants=list(payload["train_participants"]),
        test_participants=list(payload["test_participants"]),
        validation_strategy=dict(payload["validation_strategy"]),
        validation_folds=validation_folds,
        split_config_path=resolved_path,
    )


def configured_participant_ids(split_config: FixedSplitConfig) -> list[str]:
    """Return the configured participants in a stable train-then-test order."""

    participant_ids: list[str] = []
    for participant_id in [*split_config.train_participants, *split_config.test_participants]:
        if participant_id not in participant_ids:
            participant_ids.append(participant_id)
    return participant_ids


def describe_validation_folds(split_config: FixedSplitConfig) -> list[dict[str, object]]:
    """Return the saved validation fold definitions as dictionaries."""

    return [fold.to_dict() for fold in split_config.validation_folds]


def build_fixed_validation_splits(
    train_metadata: pd.DataFrame,
    split_config: FixedSplitConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build explicit row-index CV splits from the saved participant fold definitions."""

    if "participant_id" not in train_metadata.columns:
        raise KeyError("Training metadata must contain a `participant_id` column.")

    observed_participants = sorted(train_metadata["participant_id"].dropna().unique().tolist())
    expected_participants = sorted(split_config.train_participants)
    if observed_participants != expected_participants:
        raise ValueError(
            "Training participants do not match the fixed split config. "
            f"Expected {expected_participants}, observed {observed_participants}."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    participant_series = train_metadata["participant_id"]
    for fold in split_config.validation_folds:
        train_mask = participant_series.isin(fold.train_participants).to_numpy()
        validation_mask = participant_series.isin(fold.validation_participants).to_numpy()
        if np.any(train_mask & validation_mask):
            raise ValueError(f"Fold {fold.fold} assigns at least one row to both train and validation.")
        if np.any(~(train_mask | validation_mask)):
            uncovered = sorted(participant_series.loc[~(train_mask | validation_mask)].dropna().unique().tolist())
            raise ValueError(f"Fold {fold.fold} leaves participants uncovered: {uncovered}")

        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"Fold {fold.fold} is empty after applying the fixed participant assignments.")
        splits.append((train_indices, validation_indices))

    return splits


def _as_optional_int(value: object) -> int | None:
    """Convert optional numeric JSON values to int."""

    if value is None:
        return None
    return int(value)
