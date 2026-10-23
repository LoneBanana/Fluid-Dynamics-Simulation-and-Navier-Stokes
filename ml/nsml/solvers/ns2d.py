"""Pseudo-spectral solver for 2D incompressible Navier-Stokes in vorticity form.

    w_t + u . grad(w) = nu lap(w) - drag * w + f,    u = (psi_y, -psi_x),  lap(psi) = -w

on the 2pi-periodic torus. Arrays are laid out as (..., y, x), so
`imshow(w, origin="lower")` shows x horizontally. The default forcing is Kolmogorov
flow, f = -k_f cos(k_f y): the curl of the body force sin(k_f y) in x, and the
standard benchmark for learned turbulence models (Kochkov et al. 2021).

2D flows cannot blow up (vorticity is only transported and diffused, so max|w| stays
controlled), which makes this a well-posed playground for comparing surrogates; the
blowup question lives in 3D, where vortex stretching can amplify w.
"""
from __future__ import annotations

import math

import torch

from nsml.solvers.common import ifrk4_step


def kolmogorov_forcing(n: int, k_f: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Vorticity forcing -k_f cos(k_f y) on an n x n grid, shape (n, n)."""
    y = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)
    return (-k_f * torch.cos(k_f * y))[:, None].expand(n, n).to(dtype).contiguous()


def resample_spectral(w: torch.Tensor, n_out: int) -> torch.Tensor:
    """Resample a periodic (..., n, n) field onto an n_out x n_out grid by Fourier truncation
    (downsampling) or zero-padding (upsampling). Nyquist modes are dropped."""
    n = w.shape[-1]
    if n == n_out:
        return w
    w_hat = torch.fft.rfft2(w)
    h = min(n, n_out) // 2
    out = torch.zeros(*w.shape[:-2], n_out, n_out // 2 + 1, dtype=w_hat.dtype, device=w.device)
    out[..., :h, :h] = w_hat[..., :h, :h]
    out[..., n_out - h + 1:, :h] = w_hat[..., n - h + 1:, :h]
    return torch.fft.irfft2(out, s=(n_out, n_out)) * (n_out / n) ** 2


class NS2DSpectral:
    """Pseudo-spectral 2D Navier-Stokes solver (2/3-rule dealiasing, integrating-factor RK4)."""

    def __init__(self, n: int, nu: float, dt: float, drag: float = 0.0,
                 forcing: torch.Tensor | None = None, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.n, self.nu, self.dt, self.drag = n, nu, dt, drag
        ky = torch.fft.fftfreq(n, d=1.0 / n).to(device=device, dtype=dtype)[:, None]
        kx = torch.fft.rfftfreq(n, d=1.0 / n).to(device=device, dtype=dtype)[None, :]
        k2 = kx**2 + ky**2
        self.inv_k2 = torch.where(k2 > 0, 1.0 / k2.clamp(min=1.0), torch.zeros_like(k2))
        self.ikx, self.iky = 1j * kx, 1j * ky
        self.dealias = ((kx.abs() < n / 3) & (ky.abs() < n / 3)).to(dtype)
        decay = -nu * k2 - drag
        self.e_half = torch.exp(decay * dt / 2)
        self.e_full = torch.exp(decay * dt)
        self.f_hat = None
        if forcing is not None:
            self.f_hat = torch.fft.rfft2(forcing.to(device=device, dtype=dtype)) * self.dealias

    def to_spectral(self, w: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(w.to(self.dealias)) * self.dealias

    def to_physical(self, w_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(w_hat, s=(self.n, self.n))

    def velocity(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Physical velocity components (u, v) from spectral vorticity."""
        psi_hat = w_hat * self.inv_k2
        return self.to_physical(self.iky * psi_hat), self.to_physical(-self.ikx * psi_hat)

    def _nonlinear(self, w_hat: torch.Tensor) -> torch.Tensor:
        w_hat = w_hat * self.dealias
        psi_hat = w_hat * self.inv_k2
        u, v, w = torch.fft.irfft2(torch.stack([self.iky * psi_hat, -self.ikx * psi_hat, w_hat]),
                                   s=(self.n, self.n))
        flux = torch.fft.rfft2(torch.stack([u * w, v * w]))
        out = -(self.ikx * flux[0] + self.iky * flux[1]) * self.dealias
        return out if self.f_hat is None else out + self.f_hat

    def step(self, w_hat: torch.Tensor) -> torch.Tensor:
        return ifrk4_step(w_hat, self._nonlinear, self.e_half, self.e_full, self.dt)

    def advance(self, w_hat: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            w_hat = self.step(w_hat)
        return w_hat

    @torch.no_grad()
    def simulate(self, w0: torch.Tensor, t_final: float, save_every: float,
                 n_out: int | None = None) -> torch.Tensor:
        """Integrate from w0 (batch, n, n); returns snapshots (batch, n_saves + 1, n_out, n_out)
        including t=0, each band-limited to an n_out grid (default: the solver grid)."""
        steps = round(save_every / self.dt)
        if abs(steps * self.dt - save_every) > 1e-9 * max(1.0, save_every):
            raise ValueError(f"save interval {save_every} is not a multiple of dt {self.dt}")
        n_out = n_out or self.n
        w_hat = self.to_spectral(resample_spectral(w0, self.n))
        frames = [resample_spectral(self.to_physical(w_hat), n_out)]
        for _ in range(round(t_final / save_every)):
            w_hat = self.advance(w_hat, steps)
            frames.append(resample_spectral(self.to_physical(w_hat), n_out))
        return torch.stack(frames, dim=1)


def max_velocity(solver: NS2DSpectral, w_hat: torch.Tensor) -> float:
    u, v = solver.velocity(w_hat)
    return torch.sqrt(u**2 + v**2).max().item()
