# Vendored model code provenance

The repository vendors the minimal model-definition files needed to load the official pretrained checkpoints during feature extraction. No extra model-code checkout is required at runtime.

## PulsePPG

- Upstream repository: `https://github.com/maxxu05/pulseppg`
- Exact upstream commit: `716eaf9cf966e8f76436f2263872ef38b1f90166`
- Upstream source file used: `pulseppg/nets/ResNet1D/ResNet1D_Net.py`
- Vendored runtime files:
  - `src/vendor/resnet1d_shared.py`
  - `src/vendor/pulseppg_resnet1d.py`

## PaPaGei

- Upstream repository: `https://github.com/Nokia-Bell-Labs/papagei-foundation-model`
- Exact upstream commit: `0c537dad4d2850e15b724260de820dd68d77f0b0`
- Upstream source file used: `models/resnet.py`
- Vendored runtime files:
  - `src/vendor/resnet1d_shared.py`
  - `src/vendor/papagei_resnet.py`

## Scope

Only the model architecture code required to instantiate the checkpoint-compatible encoders is vendored here. Training scripts, dataset loaders, and the rest of the upstream repositories are intentionally not copied because this project only needs deterministic feature extraction.
