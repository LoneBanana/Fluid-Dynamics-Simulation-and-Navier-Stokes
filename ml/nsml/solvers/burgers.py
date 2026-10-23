"""Solvers for the 1D viscous Burgers equation on the periodic domain [0, 2pi).

    u_t + (u^2 / 2)_x = nu u_xx

Burgers is the classic 1D cartoon of Navier-Stokes blowup. Without viscosity, smooth
data steepens until the slope becomes infinite at the finite time t* = 1 / max(-u0');
small viscosity regularises that singularity into a thin shock of width ~nu.

* `BurgersSpectral`: Fourier pseudo-spectral + integrating-factor RK4. Accurate to
  near machine precision while the solution is resolved, so run on a fine grid it
  produces the ground truth.
* `BurgersFV`: second-order finite volumes (MUSCL reconstruction, exact Godunov flux,
  SSP-RK2). The robust, shock-capturing scheme you would write by hand in C++; it is
  the classical baseline the neural surrogates have to beat on a coarse grid.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nsml.solvers.common import ifrk4_step


class BurgersSpectral:
    """Pseudo-spectral Burgers solver with 2/3-rule dealiasing. Fields have shape (..., n)."""

    def __init__(self, n: int, nu: float, dt: float, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.n, self.nu, self.dt = n, nu, dt
        k = torch.fft.rfftfreq(n, d=1.0 / n).to(device=device, dtype=dtype)
        self.ik = 1j * k
        self.dealias = (k < n / 3).to(dtype)
        decay = -nu * k**2
        self.e_half = torch.exp(decay * dt / 2)
        self.e_full = torch.exp(decay * dt)

    def to_spectral(self, u: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(u.to(self.dealias)) * self.dealias

    def to_physical(self, u_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(u_hat, n=self.n)

    def _nonlinear(self, u_hat: torch.Tensor) -> torch.Tensor:
        u = torch.fft.irfft(u_hat * self.dealias, n=self.n)
        return -0.5 * self.ik * self.dealias * torch.fft.rfft(u * u)

    def step(self, u_hat: torch.Tensor) -> torch.Tensor:
        return ifrk4_step(u_hat, self._nonlinear, self.e_half, self.e_full, self.dt)

    @torch.no_grad()
    def simulate(self, u0: torch.Tensor, t_final: float, save_every: float) -> torch.Tensor:
        """Integrate from u0 (batch, n); returns snapshots (batch, n_saves + 1, n) incl. t=0."""
        steps_per_save = _steps_per_interval(save_every, self.dt)
        n_saves = round(t_final / save_every)
        u_hat = self.to_spectral(u0)
        frames = [self.to_physical(u_hat)]
        for _ in range(n_saves):
            for _ in range(steps_per_save):
                u_hat = self.step(u_hat)
            frames.append(self.to_physical(u_hat))
        return torch.stack(frames, dim=1)


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def _godunov_flux(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    """Exact Riemann-solver flux for the convex flux f(u) = u^2 / 2."""
    return torch.maximum(0.5 * u_left.clamp(min=0) ** 2, 0.5 * u_right.clamp(max=0) ** 2)


class BurgersFV:
    """MUSCL-Godunov finite-volume Burgers solver on cell averages of shape (..., n)."""

    def __init__(self, n: int, nu: float, cfl: float = 0.4):
        self.n, self.nu, self.cfl = n, nu, cfl
        self.dx = 2 * math.pi / n

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        du_minus = u - torch.roll(u, 1, -1)
        du_plus = torch.roll(u, -1, -1) - u
        slope = _minmod(du_minus, du_plus)
        u_left = u + 0.5 * slope                       # state just left of interface i+1/2
        u_right = torch.roll(u - 0.5 * slope, -1, -1)  # state just right of interface i+1/2
        flux = _godunov_flux(u_left, u_right)
        advection = -(flux - torch.roll(flux, 1, -1)) / self.dx
        diffusion = self.nu * (du_plus - du_minus) / self.dx**2
        return advection + diffusion

    def step(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        u1 = u + dt * self._rhs(u)
        return 0.5 * (u + u1 + dt * self._rhs(u1))

    @torch.no_grad()
    def advance(self, u: torch.Tensor, interval: float) -> torch.Tensor:
        """Advance by `interval` using as many CFL-limited substeps as needed."""
        speed = u.abs().max().item() + 1e-6
        dt_max = min(self.cfl * self.dx / speed, 0.25 * self.dx**2 / max(self.nu, 1e-12))
        substeps = max(1, math.ceil(interval / dt_max))
        for _ in range(substeps):
            u = self.step(u, interval / substeps)
        return u


def coarse_grain_1d(u_fine: torch.Tensor, factor: int) -> torch.Tensor:
    """Average a fine periodic grid function over coarse cells (..., n) -> (..., n // factor).

    Coarse cell i is centred on fine point i * factor and spans `factor` fine cells; the
    trapezoid weights make the average exactly centred. This is what a finite-volume
    scheme on the coarse grid evolves, so it is the natural coarse "ground truth".
    """
    n = u_fine.shape[-1]
    if n % factor or factor % 2:
        raise ValueError("factor must be even and divide the grid size")
    weights = torch.ones(factor + 1, dtype=u_fine.dtype, device=u_fine.device)
    weights[0] = weights[-1] = 0.5
    weights /= factor
    half = factor // 2
    x = F.pad(u_fine.reshape(-1, 1, n), (half, half), mode="circular")
    out = F.conv1d(x, weights.view(1, 1, -1), stride=factor)
    return out.reshape(*u_fine.shape[:-1], n // factor)


def _steps_per_interval(interval: float, dt: float) -> int:
    steps = round(interval / dt)
    if steps < 1 or abs(steps * dt - interval) > 1e-9 * max(1.0, interval):
        raise ValueError(f"save interval {interval} is not a multiple of dt {dt}")
    return steps
