"""Fourier Neural Operator (Li et al., 2021) for 1D and 2D periodic fields.

Each layer mixes channels pointwise and, in parallel, applies a learned linear map to
the lowest `modes` Fourier coefficients of every channel:

    v <- gelu( W v + iFFT( R * FFT(v)[low modes] ) )

so a single layer sees the entire domain. Periodic domains need no padding. The
complex weights R are stored as separate real and imaginary tensors, which keeps the
module portable across devices and makes the weights easy to export to C++/GLSL.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nsml.models.layers import conv_cls


class SpectralConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int, dim: int):
        super().__init__()
        self.dim, self.modes, self.out_ch = dim, modes, out_ch
        blocks = 1 if dim == 1 else 2  # in 2D: one block for k_y >= 0, one for k_y < 0
        shape = (blocks, in_ch, out_ch) + (modes,) * dim
        scale = 1.0 / (in_ch * out_ch)
        self.weight_re = nn.Parameter(scale * torch.rand(shape))
        self.weight_im = nn.Parameter(scale * torch.rand(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-self.dim, 0))
        x_hat = torch.fft.rfftn(x, dim=dims)
        w = torch.complex(self.weight_re, self.weight_im)
        out = torch.zeros(x.shape[0], self.out_ch, *x_hat.shape[2:], dtype=x_hat.dtype, device=x.device)
        m = self.modes
        if self.dim == 1:
            out[..., :m] = torch.einsum("bix,iox->box", x_hat[..., :m], w[0])
        else:
            out[..., :m, :m] = torch.einsum("bixy,ioxy->boxy", x_hat[..., :m, :m], w[0])
            out[..., -m:, :m] = torch.einsum("bixy,ioxy->boxy", x_hat[..., -m:, :m], w[1])
        return torch.fft.irfftn(out, s=x.shape[-self.dim:], dim=dims)


class FNO(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int, width: int = 32, modes: int = 12,
                 layers: int = 4):
        super().__init__()
        conv = conv_cls(dim)
        self.lift = conv(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, width, modes, dim) for _ in range(layers))
        self.pointwise = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(conv(width, 4 * width, 1), nn.GELU(), conv(4 * width, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        last = len(self.spectral) - 1
        for i, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            x = spectral(x) + pointwise(x)
            if i < last:
                x = F.gelu(x)
        return self.project(x)
