"""PaPaGei feature extraction entry point."""

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

DEFAULT_TARGET_HZ = 125


@dataclass(slots=True)
class PaPaGeiCheckpointInfo:
    """Resolved external repository and checkpoint paths for PaPaGei."""

    repo_root: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments") / "papagei_results" / datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-bandpass", action="store_true", default=False)
    parser.add_argument("--no-zscore", action="store_true", default=False)
    return parser.parse_args()


def resolve_papagei_paths(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> PaPaGeiCheckpointInfo:
    """Resolve the cloned PaPaGei repository and a usable checkpoint path."""

    base_external_root = Path(external_root) if external_root is not None else default_external_root()
    repo_root = base_external_root / "papagei-foundation-model"
    if not repo_root.exists():
        raise FileNotFoundError(f"PaPaGei repository was not found: {repo_root}")

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"PaPaGei checkpoint does not exist: {resolved_checkpoint}")
        return PaPaGeiCheckpointInfo(repo_root=repo_root, checkpoint_path=resolved_checkpoint)

    candidate_paths = [
        repo_root / "weights" / "papagei_s.pt",
        repo_root / "weights" / "papagei_p.pt",
    ]
    candidate_paths.extend(sorted(repo_root.rglob("papagei*.pt")))
    for candidate in candidate_paths:
        if candidate.exists():
            return PaPaGeiCheckpointInfo(repo_root=repo_root, checkpoint_path=candidate)

    raise FileNotFoundError(
        "Could not find a PaPaGei checkpoint. Download the official weights and provide --checkpoint-path if needed."
    )


def load_papagei_encoder(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
):
    """Load the PaPaGei encoder from the cloned official repository."""

    checkpoint_info = resolve_papagei_paths(external_root=external_root, checkpoint_path=checkpoint_path)
    if str(checkpoint_info.repo_root) not in sys.path:
        sys.path.insert(0, str(checkpoint_info.repo_root))

    import torch
    from models.resnet import ResNet1DMoE

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=18,
        n_classes=512,
        n_experts=3,
    )
    state_dict = torch.load(checkpoint_info.checkpoint_path, map_location=resolved_device)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    model.load_state_dict(cleaned_state_dict)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device, checkpoint_info


def extract_papagei_embeddings(
    windows: pd.DataFrame,
    checkpoint_path: str | Path | None = None,
    external_root: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 256,
    apply_bandpass: bool = True,
    apply_zscore: bool = True,
) -> tuple[np.ndarray, PaPaGeiCheckpointInfo]:
    """Extract PaPaGei embeddings for a processed window table."""

    import torch

    model, resolved_device, checkpoint_info = load_papagei_encoder(
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
            batch_outputs = model(batch)
            embeddings = batch_outputs[0] if isinstance(batch_outputs, (tuple, list)) else batch_outputs
            outputs.append(embeddings.detach().cpu().numpy())
    stacked = np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)
    return stacked, checkpoint_info


def main() -> None:
    """Load processed windows, extract embeddings, and save the artifacts."""

    args = parse_args()
    windows, processed_manifest = load_windows_from_manifest(args.manifest_path)
    windows = select_window_subset(windows, max_windows=args.max_windows)

    embeddings, checkpoint_info = extract_papagei_embeddings(
        windows=windows,
        checkpoint_path=args.checkpoint_path,
        external_root=args.external_root,
        device=args.device,
        batch_size=args.batch_size,
        apply_bandpass=not args.no_bandpass,
        apply_zscore=not args.no_zscore,
    )
    summary = save_embedding_artifacts(
        model_name="papagei",
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
