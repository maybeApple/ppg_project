"""PulsePPG feature extraction entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.common import (
    SignalPreprocessingConfig,
    build_signal_preprocessing_config,
    build_window_signal_matrix,
    default_external_root,
    load_signal_preprocessing_config,
    load_windows_from_manifest,
    save_embedding_artifacts,
    select_window_subset,
)
from src.vendor import PulsePPGNet

DEFAULT_TARGET_HZ = 50
UPSTREAM_REPOSITORY = "https://github.com/maxxu05/pulseppg"
UPSTREAM_COMMIT = "716eaf9cf966e8f76436f2263872ef38b1f90166"
DEFAULT_CHECKPOINT_DIR = "pulseppg"


@dataclass(slots=True)
class PulsePPGCheckpointInfo:
    """Resolved vendored model-code path and checkpoint path for PulsePPG."""

    model_code_path: Path
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
    parser.add_argument(
        "--ppg-source",
        choices=["canonical", "raw"],
        default="canonical",
        help="Use canonical inverted PPG or raw non-inverted PPG from processed windows.",
    )
    parser.add_argument("--experiment-config", type=Path, default=Path("configs") / "experiment_modes.json")
    parser.add_argument("--experiment-mode", choices=["harmonized", "model_faithful"], default=None)
    parser.add_argument("--preprocessing-mode", choices=["harmonized", "model_faithful"], default="harmonized")
    parser.add_argument(
        "--normalization",
        choices=["none", "per_window_zscore", "person_specific_zscore", "causal_running_zscore"],
        default=None,
    )
    parser.add_argument("--no-bandpass", action="store_true", default=False)
    parser.add_argument("--no-zscore", action="store_true", default=False)
    return parser.parse_args()


def resolve_pulseppg_paths(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> PulsePPGCheckpointInfo:
    """Resolve the vendored model-code path and a usable checkpoint path."""

    base_external_root = Path(external_root) if external_root is not None else default_external_root()
    weights_root = base_external_root / DEFAULT_CHECKPOINT_DIR
    model_code_path = Path(__file__).resolve().parents[1] / "vendor" / "pulseppg_resnet1d.py"

    if checkpoint_path is not None:
        resolved_checkpoint = Path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"PulsePPG checkpoint does not exist: {resolved_checkpoint}")
        return PulsePPGCheckpointInfo(model_code_path=model_code_path, checkpoint_path=resolved_checkpoint)

    candidate_paths = [
        weights_root / "checkpoint_best.pkl",
        weights_root / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl",
    ]
    if weights_root.exists():
        candidate_paths.extend(sorted(weights_root.rglob("checkpoint_best.pkl")))
    for candidate in candidate_paths:
        if candidate.exists():
            return PulsePPGCheckpointInfo(model_code_path=model_code_path, checkpoint_path=candidate)

    raise FileNotFoundError(
        "Could not find a PulsePPG checkpoint. Expected either "
        f"`{weights_root / 'checkpoint_best.pkl'}` or the legacy upstream path under "
        f"`{weights_root / 'pulseppg' / 'experiments' / 'out' / 'pulseppg'}`."
    )


def load_pulseppg_encoder(
    external_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
):
    """Load the PulsePPG encoder from the vendored runtime model definition."""

    checkpoint_info = resolve_pulseppg_paths(external_root=external_root, checkpoint_path=checkpoint_path)

    import torch

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PulsePPGNet(
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
    preprocessing_config: SignalPreprocessingConfig | None = None,
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
        preprocessing_config=preprocessing_config,
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
    if args.ppg_source == "raw":
        if "ppg_raw_values" not in windows.columns:
            raise KeyError("`--ppg-source raw` requires processed windows with `ppg_raw_values`.")
        windows = windows.copy()
        windows["ppg_values"] = windows["ppg_raw_values"]
        if "ppg_inverted" in windows.columns:
            windows["ppg_inverted"] = False
    normalization = "none" if args.no_zscore else args.normalization
    if args.experiment_config is not None and args.experiment_config.exists():
        preprocessing_config = load_signal_preprocessing_config(
            config_path=args.experiment_config,
            model_name="pulseppg",
            mode_name=args.experiment_mode or args.preprocessing_mode,
            target_sampling_hz=DEFAULT_TARGET_HZ,
            normalization_override=normalization,
            apply_bandpass_override=False if args.no_bandpass else None,
        )
    else:
        preprocessing_config = build_signal_preprocessing_config(
            model_name="pulseppg",
            target_sampling_hz=DEFAULT_TARGET_HZ,
            mode=args.preprocessing_mode,
            normalization=normalization,
            apply_bandpass=not args.no_bandpass,
        )

    embeddings, checkpoint_info = extract_pulseppg_embeddings(
        windows=windows,
        checkpoint_path=args.checkpoint_path,
        external_root=args.external_root,
        device=args.device,
        batch_size=args.batch_size,
        preprocessing_config=preprocessing_config,
    )
    summary = save_embedding_artifacts(
        model_name="pulseppg",
        embeddings=embeddings,
        windows=windows,
        target_sampling_hz=DEFAULT_TARGET_HZ,
        output_dir=args.output_dir,
        source_windows_path=processed_manifest["windows_path"],
        checkpoint_path=str(checkpoint_info.checkpoint_path),
        external_repo_root=checkpoint_info.model_code_path,
        extra_manifest_fields={
            "processed_manifest_path": str(args.manifest_path) if args.manifest_path is not None else "",
            "experiment_config_path": str(args.experiment_config) if args.experiment_config is not None else "",
            "experiment_mode": args.experiment_mode or args.preprocessing_mode or preprocessing_config.mode,
            "preprocessing": preprocessing_config.to_dict(),
            "preprocessing_mode": preprocessing_config.mode,
            "apply_bandpass": preprocessing_config.apply_bandpass,
            "normalization": preprocessing_config.normalization,
            "apply_zscore": preprocessing_config.normalization != "none",
            "batch_size": args.batch_size,
            "device": args.device,
            "ppg_source": args.ppg_source,
            "ppg_inverted": args.ppg_source == "canonical" and bool(processed_manifest.get("ppg_inverted", True)),
            "model_code_strategy": "vendored_minimal_source",
            "model_code_path": str(checkpoint_info.model_code_path),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        },
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
