"""Shared 1D ResNet building blocks vendored from upstream model repositories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSame(nn.Module):
    """Conv1d layer with TensorFlow-style SAME padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the input to preserve SAME semantics before convolution."""

        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        total_padding = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right), mode="constant", value=0))


class MaxPool1dSame(nn.Module):
    """MaxPool1d layer with SAME padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SAME padding before max pooling."""

        stride = 1
        in_dim = x.shape[-1]
        out_dim = (in_dim + stride - 1) // stride
        total_padding = max(0, (out_dim - 1) * stride + self.kernel_size - in_dim)
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right), mode="constant", value=0))


class ResidualBlock(nn.Module):
    """Minimal residual block shared by the vendored PulsePPG and PaPaGei encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        use_batch_norm: bool,
        use_dropout: bool,
        is_first_block: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.is_first_block = is_first_block

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=groups,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=groups,
        )
        self.identity_pool = MaxPool1dSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block and identity shortcut."""

        identity = x
        out = x
        if not self.is_first_block:
            if self.use_batch_norm:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_dropout:
                out = self.dropout1(out)
        out = self.conv1(out)

        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_dropout:
            out = self.dropout2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.identity_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            channels_left = (self.out_channels - self.in_channels) // 2
            channels_right = self.out_channels - self.in_channels - channels_left
            identity = F.pad(identity, (channels_left, channels_right), mode="constant", value=0)
            identity = identity.transpose(-1, -2)

        return out + identity
