"""PulsePPG feature extraction entry point."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.common import (
    build_window_signal_matrix,
    default_external_root,
    load_windows_from_manifest,
    save_embedding_artifacts,
    select_window_subset,
)

DEFAULT_TARGET_HZ = 50


@dataclass(slots=True)
class PulsePPGCheckpointInfo:
    """Resolved external repository and checkpoint paths for PulsePPG."""

    repo_root: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments") / "pulseppg_results" / datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-bandpass", action="store_true", default=False)
    parser.add_argument("--no-zscore", action="store_true", default=False)
    return parser.parse_args()


def resolve_pulseppg_paths(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> PulsePPGCheckpointInfo:
    """Resolve the cloned PulsePPG repository and a usable checkpoint path."""

    base_external_root = Path(external_root) if external_root is not None else default_external_root()
    repo_root = base_external_root / "pulseppg"
    if not repo_root.exists():
        raise FileNotFoundError(f"PulsePPG repository was not found: {repo_root}")

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"PulsePPG checkpoint does not exist: {resolved_checkpoint}")
        return PulsePPGCheckpointInfo(repo_root=repo_root, checkpoint_path=resolved_checkpoint)

    candidate_paths = [
        repo_root / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl",
        repo_root / "checkpoint_best.pkl",
    ]
    candidate_paths.extend(sorted(repo_root.rglob("checkpoint_best.pkl")))
    for candidate in candidate_paths:
        if candidate.exists():
            return PulsePPGCheckpointInfo(repo_root=repo_root, checkpoint_path=candidate)

    raise FileNotFoundError(
        "Could not find a PulsePPG checkpoint. Download the official weights and provide --checkpoint-path if needed."
    )


def load_pulseppg_encoder(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
):
    """Load the PulsePPG encoder from the cloned official repository."""

    checkpoint_info = resolve_pulseppg_paths(external_root=external_root, checkpoint_path=checkpoint_path)
    if str(checkpoint_info.repo_root) not in sys.path:
        sys.path.insert(0, str(checkpoint_info.repo_root))

    import torch
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Net(
        in_channels=1,
        base_filters=128,
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=12,
        finalpool="max",
    )
    model.instnorm = torch.nn.InstanceNorm1d(1, affine=False, track_running_stats=False)
    state = torch.load(checkpoint_info.checkpoint_path, map_location=resolved_device)
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device, checkpoint_info


def extract_pulseppg_embeddings(
    windows: pd.DataFrame,
    checkpoint_path: str | Path | None = None,
    external_root: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 256,
    apply_bandpass: bool = True,
    apply_zscore: bool = True,
) -> tuple[np.ndarray, PulsePPGCheckpointInfo]:
    """Extract PulsePPG embeddings for a processed window table."""

    import torch

    model, resolved_device, checkpoint_info = load_pulseppg_encoder(
        external_root=external_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    matrix = build_window_signal_matrix(
        windows=windows,
        target_sampling_hz=DEFAULT_TARGET_HZ,
        apply_zscore=apply_zscore,
        apply_bandpass=apply_bandpass,
    )
    tensor = torch.as_tensor(matrix, dtype=torch.float32).unsqueeze(1)

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start_idx in range(0, len(tensor), batch_size):
            batch = tensor[start_idx : start_idx + batch_size].to(resolved_device)
            batch_outputs = model(batch).detach().cpu().numpy()
            outputs.append(batch_outputs)
    embeddings = np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)
    return embeddings, checkpoint_info


def main() -> None:
    """Load processed windows, extract embeddings, and save the artifacts."""

    args = parse_args()
    windows, processed_manifest = load_windows_from_manifest(args.manifest_path)
    windows = select_window_subset(windows, max_windows=args.max_windows)

    embeddings, checkpoint_info = extract_pulseppg_embeddings(
        windows=windows,
        checkpoint_path=args.checkpoint_path,
        external_root=args.external_root,
        device=args.device,
        batch_size=args.batch_size,
        apply_bandpass=not args.no_bandpass,
        apply_zscore=not args.no_zscore,
    )
    summary = save_embedding_artifacts(
        model_name="pulseppg",
        embeddings=embeddings,
        windows=windows,
        target_sampling_hz=DEFAULT_TARGET_HZ,
        output_dir=args.output_dir,
        source_windows_path=processed_manifest["windows_path"],
        checkpoint_path=str(checkpoint_info.checkpoint_path),
        external_repo_root=checkpoint_info.repo_root,
        extra_manifest_fields={
            "processed_manifest_path": str(args.manifest_path) if args.manifest_path is not None else "",
            "apply_bandpass": not args.no_bandpass,
            "apply_zscore": not args.no_zscore,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
