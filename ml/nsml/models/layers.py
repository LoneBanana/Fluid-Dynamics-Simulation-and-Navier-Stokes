"""Dimension-generic building blocks (1D or 2D) for periodic domains."""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn


def conv_cls(dim: int) -> type[nn.Module]:
    return {1: nn.Conv1d, 2: nn.Conv2d}[dim]


def periodic_conv(dim: int, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1) -> nn.Module:
    """'Same'-size convolution that wraps around the domain edges."""
    pad = dilation * (kernel - 1) // 2
    return conv_cls(dim)(in_ch, out_ch, kernel, padding=pad, dilation=dilation, padding_mode="circular")


def avg_pool(x, dim: int):
    return F.avg_pool1d(x, 2) if dim == 1 else F.avg_pool2d(x, 2)


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")
