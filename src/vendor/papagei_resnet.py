"""Minimal PaPaGei encoder vendored for reproducible feature extraction.

Adapted from:
- Repository: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
- Commit: 0c537dad4d2850e15b724260de820dd68d77f0b0
- Source file: models/resnet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .resnet1d_shared import Conv1dSame, ResidualBlock


class ResNet1DMoE(nn.Module):
    """PaPaGei ResNet1D encoder with the mixture-of-experts heads kept for weight compatibility."""

    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        kernel_size: int,
        stride: int,
        groups: int,
        n_block: int,
        n_classes: int,
        n_experts: int = 2,
        downsample_gap: int = 2,
        increasefilter_gap: int = 4,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        use_projection: bool = False,
    ):
        super().__init__()
        self.n_block = n_block
        self.use_batch_norm = use_batch_norm
        self.use_projection = use_projection
        self.n_experts = n_experts

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
            block_in_channels = (
                base_filters
                if is_first_block
                else int(base_filters * 2 ** ((block_index - 1) // increasefilter_gap))
            )
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

        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        self.expert_layers_1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_channels, out_channels // 2),
                    nn.ReLU(),
                    nn.Linear(out_channels // 2, 1),
                )
                for _ in range(n_experts)
            ]
        )
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, n_experts),
            nn.Softmax(dim=1),
        )

        self.expert_layers_2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_channels, out_channels // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(out_channels // 2, 1),
                )
                for _ in range(n_experts)
            ]
        )
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, n_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return class logits, two MoE regression heads, and the pooled embedding."""

        out = self.first_block_conv(x)
        if self.use_batch_norm:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        for block in self.basicblock_list:
            out = block(out)

        if self.use_batch_norm:
            out = self.final_bn(out)
        out = self.final_relu(out)
        pooled = out.mean(-1)
        out_class = self.projector(pooled) if self.use_projection else self.dense(pooled)

        expert_outputs_1 = torch.stack([expert(pooled) for expert in self.expert_layers_1], dim=1)
        gate_weights_1 = self.gating_network_1(pooled)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        expert_outputs_2 = torch.stack([expert(pooled) for expert in self.expert_layers_2], dim=1)
        gate_weights_2 = self.gating_network_2(pooled)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, pooled
