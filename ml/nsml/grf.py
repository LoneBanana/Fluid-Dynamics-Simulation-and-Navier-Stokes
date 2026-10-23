"""Gaussian random fields on periodic domains, used as random initial conditions.

Fourier coefficients are drawn as independent complex Gaussians with a power-law
envelope (|k|^2 + tau^2)^(-alpha/2): larger alpha gives smoother fields, tau sets
the wavenumber where the spectrum rolls over. Every sample is normalised to zero
mean and unit RMS so that amplitudes are comparable across a dataset.
"""
from __future__ import annotations

import torch


def _normalise(u: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    u = u - u.mean(dim=dims, keepdim=True)
    return u / u.pow(2).mean(dim=dims, keepdim=True).sqrt()


def grf_1d(batch: int, n: int, alpha: float, tau: float, kmax: int | None = None,
           generator: torch.Generator | None = None,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sample `batch` smooth periodic functions on n points of [0, 2pi). Shape (batch, n)."""
    k = torch.fft.rfftfreq(n, d=1.0 / n, dtype=dtype)
    envelope = (k**2 + tau**2) ** (-alpha / 2)
    envelope[0] = 0.0
    if kmax is not None:
        envelope[k > kmax] = 0.0
    re = torch.randn(batch, k.numel(), generator=generator, dtype=dtype)
    im = torch.randn(batch, k.numel(), generator=generator, dtype=dtype)
    u = torch.fft.irfft(torch.complex(re, im) * envelope, n=n)
    return _normalise(u, dims=(-1,))


def grf_2d(batch: int, n: int, alpha: float, tau: float, kmax: int | None = None,
           generator: torch.Generator | None = None,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sample `batch` periodic fields on an n x n grid of [0, 2pi)^2. Shape (batch, n, n)."""
    ky = torch.fft.fftfreq(n, d=1.0 / n, dtype=dtype)[:, None]
    kx = torch.fft.rfftfreq(n, d=1.0 / n, dtype=dtype)[None, :]
    k2 = kx**2 + ky**2
    envelope = (k2 + tau**2) ** (-alpha / 2)
    envelope[0, 0] = 0.0
    if kmax is not None:
        envelope[k2 > kmax**2] = 0.0
    shape = (batch, n, n // 2 + 1)
    re = torch.randn(shape, generator=generator, dtype=dtype)
    im = torch.randn(shape, generator=generator, dtype=dtype)
    w = torch.fft.irfft2(torch.complex(re, im) * envelope, s=(n, n))
    return _normalise(w, dims=(-2, -1))
