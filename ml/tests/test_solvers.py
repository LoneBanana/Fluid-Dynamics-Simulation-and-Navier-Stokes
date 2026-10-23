import math

import torch

from nsml.grf import grf_2d
from nsml.solvers import (BurgersFV, BurgersSpectral, NS2DSpectral, coarse_grain_1d,
                          kolmogorov_forcing, resample_spectral)

F64 = torch.float64


def grid(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=F64) * (2 * math.pi / n)


def burgers_exact_inviscid(x: torch.Tensor, t: float) -> torch.Tensor:
    """Solve u = sin(x - u t) by Newton iteration (valid before the shock at t = 1)."""
    u = torch.sin(x)
    for _ in range(50):
        g = u - torch.sin(x - u * t)
        u = u - g / (1 + t * torch.cos(x - u * t))
    return u


def test_burgers_spectral_matches_characteristics():
    n = 256
    x = grid(n)
    solver = BurgersSpectral(n, nu=0.0, dt=1e-3, dtype=F64)
    u = solver.simulate(torch.sin(x)[None], t_final=0.5, save_every=0.5)[0, -1]
    assert (u - burgers_exact_inviscid(x, 0.5)).abs().max() < 1e-8


def test_burgers_fv_conservative_and_second_order():
    t = 0.5
    errors = []
    for n in (128, 256):
        x = grid(n)
        solver = BurgersFV(n, nu=0.0)
        # cell averages of sin over each cell
        u0 = torch.sin(x) * math.sin(solver.dx / 2) / (solver.dx / 2)
        u = solver.advance(u0, t)
        assert abs(u.sum() - u0.sum()) < 1e-10
        fine = burgers_exact_inviscid(grid(n * 16), t)
        errors.append((u - coarse_grain_1d(fine, 16)).abs().mean().item())
    assert errors[1] < errors[0] / 3  # roughly second order on smooth data


def test_coarse_grain_is_centred_cell_average():
    n_fine, factor = 1024, 8
    coarse = coarse_grain_1d(torch.sin(grid(n_fine)), factor)
    dx = 2 * math.pi / (n_fine // factor)
    exact = torch.sin(grid(n_fine // factor)) * math.sin(dx / 2) / (dx / 2)
    assert (coarse - exact).abs().max() < 1e-4


def test_ns2d_single_mode_decays_exactly():
    n, nu, drag, t = 32, 0.05, 0.1, 1.0
    x = grid(n)
    w0 = torch.sin(x)[None, :] * torch.sin(x)[:, None]
    solver = NS2DSpectral(n, nu=nu, dt=0.01, drag=drag, dtype=F64)
    w = solver.simulate(w0[None], t_final=t, save_every=t)[0, -1]
    assert (w - w0 * math.exp(-(2 * nu + drag) * t)).abs().max() < 1e-10


def test_ns2d_kolmogorov_laminar_state_is_steady():
    n, nu, drag, k_f = 32, 0.05, 0.1, 4
    y = grid(n)[:, None].expand(n, n)
    w_lam = -k_f * torch.cos(k_f * y) / (nu * k_f**2 + drag)
    solver = NS2DSpectral(n, nu=nu, dt=0.01, drag=drag,
                          forcing=kolmogorov_forcing(n, k_f, dtype=F64), dtype=F64)
    w = solver.simulate(w_lam[None], t_final=1.0, save_every=1.0)[0, -1]
    assert (w - w_lam).abs().max() < 1e-10


def test_ns2d_inviscid_conserves_energy_and_enstrophy():
    n = 64
    solver = NS2DSpectral(n, nu=0.0, dt=2e-3, dtype=F64)
    w_hat = solver.to_spectral(grf_2d(2, n, alpha=4, tau=3, generator=torch.Generator().manual_seed(0), dtype=F64))

    def invariants(w_hat):
        power = w_hat.abs() ** 2
        power[..., 1:-1] *= 2  # rfft half-spectrum: interior columns appear twice
        return (power * solver.inv_k2).sum(dim=(-2, -1)), power.sum(dim=(-2, -1))

    e0, z0 = invariants(w_hat)
    e1, z1 = invariants(solver.advance(w_hat, 250))
    assert ((e1 - e0) / e0).abs().max() < 1e-7
    assert ((z1 - z0) / z0).abs().max() < 1e-7


def test_resample_spectral_round_trip():
    w = grf_2d(3, 32, alpha=3, tau=2, kmax=10, generator=torch.Generator().manual_seed(1), dtype=F64)
    up = resample_spectral(w, 64)
    assert (up[..., ::2, ::2] - w).abs().max() < 1e-12
    assert (resample_spectral(up, 32) - w).abs().max() < 1e-12
