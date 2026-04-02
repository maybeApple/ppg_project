"""Minimal PulsePPG encoder vendored for reproducible feature extraction.

Adapted from:
- Repository: https://github.com/maxxu05/pulseppg
- Commit: 716eaf9cf966e8f76436f2263872ef38b1f90166
- Source file: pulseppg/nets/ResNet1D/ResNet1D_Net.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .resnet1d_shared import Conv1dSame, ResidualBlock


class PulsePPGNet(nn.Module):
    """PulsePPG ResNet1D encoder used for embedding extraction."""

    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        kernel_size: int,
        stride: int,
        groups: int,
        n_block: int,
        finalpool: str | None = None,
        downsample_gap: int = 2,
        increasefilter_gap: int = 4,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.n_block = n_block
        self.stride = stride
        self.finalpool = finalpool
        self.use_batch_norm = use_batch_norm

        self.first_block_conv = Conv1dSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        self.basicblock_list = nn.ModuleList()
        out_channels = base_filters
        for block_index in range(n_block):
            is_first_block = block_index == 0
            downsample = block_index % downsample_gap == 1
            if is_first_block:
                block_in_channels = base_filters
                out_channels = block_in_channels
            else:
                block_in_channels = int(base_filters * 2 ** ((block_index - 1) // increasefilter_gap))
                out_channels = (
                    block_in_channels * 2
                    if block_index % increasefilter_gap == 0 and block_index != 0
                    else block_in_channels
                )
            self.basicblock_list.append(
                ResidualBlock(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    downsample=downsample,
                    use_batch_norm=use_batch_norm,
                    use_dropout=use_dropout,
                    is_first_block=is_first_block,
                )
            )

        self.instnorm = nn.InstanceNorm1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of PPG windows into fixed-size embeddings."""

        out = self.instnorm(x)
        out = self.first_block_conv(out)
        if self.use_batch_norm:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        for block in self.basicblock_list:
            out = block(out)

        if self.finalpool == "avg":
            return torch.mean(out, dim=-1)
        if self.finalpool == "max":
            return torch.max(out, dim=-1)[0]
        return out
