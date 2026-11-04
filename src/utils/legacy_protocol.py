"""Fail-closed boundaries around retrospective, historically biased pipelines."""

from __future__ import annotations

import argparse


CORRECTED_COMMAND = "python -m src.utils.run_submission"
TRAIN_SELECTION_SOURCE = "train_oof"
TRAIN_SELECTION_PROTOCOL = "participant_grouped_cross_fitting"


def add_legacy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-legacy-retrospective", action="store_true",
        help="Reproduce historical biased analysis only; outputs are not independent test results.",
    )


def guard_legacy_execution(args: argparse.Namespace, module: str) -> dict[str, object]:
    """Require explicit historical-reproduction intent before any output is written."""

    if not getattr(args, "allow_legacy_retrospective", False):
        raise RuntimeError(
            f"{module} is a retrospective legacy pipeline with test-selection/training risk. "
            f"Use `{CORRECTED_COMMAND}` for corrected independent evaluation. "
            "Historical reproduction requires --allow-legacy-retrospective."
        )
    return {
        "status": "legacy_retrospective_not_independent",
        "independent_test_valid": False,
        "selection_source": "legacy_test_retrospective",
        "selection_protocol": "historical_reproduction_only",
        "corrected_command": CORRECTED_COMMAND,
    }


def require_training_selection(
    selection_source: str | None, selection_protocol: str | None,
    *, allow_legacy_retrospective: bool = False,
) -> None:
    """Prevent selection helpers from silently treating test MAE as validation MAE."""

    if allow_legacy_retrospective:
        return
    if (selection_source, selection_protocol) != (TRAIN_SELECTION_SOURCE, TRAIN_SELECTION_PROTOCOL):
        raise ValueError(
            "Expert selection requires selection_source='train_oof' and "
            "selection_protocol='participant_grouped_cross_fitting'; final test error "
            "must never determine expert, preprocessing or router configuration."
        )
