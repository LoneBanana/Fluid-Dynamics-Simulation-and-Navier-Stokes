"""Convolutional surrogates: a periodic U-Net and a dilated ResNet.

Both only use small convolutions, so they port directly to GLSL compute/fragment
shaders - the main reason they are in the line-up next to the FNO.
"""
from __future__ import annotations

import torch
from torch import nn

from nsml.models.layers import avg_pool, conv_cls, periodic_conv, upsample


class _DoubleConv(nn.Sequential):
    def __init__(self, dim: int, in_ch: int, out_ch: int):
        super().__init__(
            periodic_conv(dim, in_ch, out_ch), nn.GroupNorm(8, out_ch), nn.GELU(),
            periodic_conv(dim, out_ch, out_ch), nn.GroupNorm(8, out_ch), nn.GELU(),
        )


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) with circular padding for periodic domains."""

    def __init__(self, in_ch: int, out_ch: int, dim: int, base: int = 32, levels: int = 3):
        super().__init__()
        self.dim = dim
        ch = [base * 2**i for i in range(levels + 1)]
        self.inc = _DoubleConv(dim, in_ch, ch[0])
        self.down = nn.ModuleList(_DoubleConv(dim, ch[i], ch[i + 1]) for i in range(levels))
        self.up = nn.ModuleList(_DoubleConv(dim, ch[i + 1] + ch[i], ch[i]) for i in reversed(range(levels)))
        self.out = conv_cls(dim)(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for block in self.down:
            skips.append(block(avg_pool(skips[-1], self.dim)))
        h = skips.pop()
        for block in self.up:
            h = block(torch.cat([upsample(h), skips.pop()], dim=1))
        return self.out(h)


class DilResNet(nn.Module):
    """Dilated residual CNN (Stachenfeld et al., 2022, "Learned coarse models for efficient
    turbulence simulation"): encoder, residual blocks of dilated convs with rates
    1,2,4,8,4,2,1 that grow the receptive field cheaply, decoder."""

    def __init__(self, in_ch: int, out_ch: int, dim: int, width: int = 48, blocks: int = 4,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2, 1)):
        super().__init__()
        self.encode = periodic_conv(dim, in_ch, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(*[layer for d in dilations
                            for layer in (nn.ReLU(), periodic_conv(dim, width, width, dilation=d))])
            for _ in range(blocks)
        )
        self.decode = periodic_conv(dim, width, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        for block in self.blocks:
            h = h + block(h)
        return self.decode(h)
