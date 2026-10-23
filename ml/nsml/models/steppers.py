"""Time steppers: anything that maps the state at t to the state one save-interval later.

Every stepper acts on physical fields shaped (batch, *space) and has the same call
signature, so training, rollouts and benchmarks treat neural, hybrid and classical
models identically.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from nsml.solvers import BurgersFV, BurgersSpectral, NS2DSpectral, coarse_grain_1d, kolmogorov_forcing, resample_spectral

Step = Callable[[torch.Tensor], torch.Tensor]


class NeuralStepper(nn.Module):
    """x_next = x + out_scale * net([x / in_scale, static fields]).

    Predicting the increment instead of the next state, with O(1) normalised inputs,
    makes training far better conditioned. `static` holds fixed input channels such
    as the forcing pattern, which convolutions cannot otherwise know about since they
    are translation-equivariant.
    """

    def __init__(self, net: nn.Module, static: torch.Tensor | None = None):
        super().__init__()
        self.net = net
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("out_scale", torch.tensor(1.0))
        self.register_buffer("static", static if static is not None else torch.zeros(0))

    def set_scales(self, in_scale: float, out_scale: float) -> None:
        self.in_scale.fill_(in_scale)
        self.out_scale.fill_(out_scale)

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        h = (x / self.in_scale).unsqueeze(1)
        if self.static.numel():
            h = torch.cat([h, self.static.expand(x.shape[0], *self.static.shape)], dim=1)
        return self.out_scale * self.net(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.correction(x)


class HybridStepper(NeuralStepper):
    """A coarse numerical step followed by a learned correction ("solver in the loop",
    Um et al. 2020; cf. Kochkov et al. 2021). The solver supplies the physics it can
    resolve; the network only learns what the coarse grid gets wrong."""

    def __init__(self, solver_step: Step, net: nn.Module, static: torch.Tensor | None = None):
        super().__init__(net, static)
        self.solver_step = solver_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.solver_step(x)
        return y + self.correction(y)


class NativeSolver:
    """A classical solver that evolves its own native-resolution state.

    `lift` maps a data-grid field to solver state, `advance` moves the state one save
    interval, and `project` maps it back to the data grid. Calling the solver does all
    three (one step from a data-grid field, as hybrids need), but rollouts lift once and
    then only project for output: re-sampling every step would throw away the fine
    scales a higher-resolution solver exists to keep.
    """

    def lift(self, x: torch.Tensor):
        return x

    def advance(self, state):
        raise NotImplementedError

    def project(self, state) -> torch.Tensor:
        return state

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.advance(self.lift(x)))


class SolverStepper(nn.Module):
    """A classical solver behind the stepper interface (no parameters, no training)."""

    def __init__(self, native: NativeSolver):
        super().__init__()
        self.native = native

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.native(x)


class NS2DSolver(NativeSolver):
    """Spectral 2D Navier-Stokes at resolution `n`, observed on an `n_data` grid."""

    def __init__(self, n: int, n_data: int, nu: float, drag: float, k_f: int, interval: float,
                 device: torch.device):
        self.steps = math.ceil(interval / (0.64 / n))  # CFL-safe dt ~ 0.64 / n for this flow
        self.n_data = n_data
        self.solver = NS2DSpectral(n, nu, interval / self.steps, drag=drag,
                                   forcing=kolmogorov_forcing(n, k_f), device=device)

    def lift(self, w: torch.Tensor) -> torch.Tensor:
        return self.solver.to_spectral(resample_spectral(w, self.solver.n))

    def advance(self, w_hat: torch.Tensor) -> torch.Tensor:
        return self.solver.advance(w_hat, self.steps)

    def project(self, w_hat: torch.Tensor) -> torch.Tensor:
        return resample_spectral(self.solver.to_physical(w_hat), self.n_data)


def upsample_linear_periodic(u: torch.Tensor, factor: int) -> torch.Tensor:
    n = u.shape[-1]
    j = torch.arange(n * factor, device=u.device)
    i0, frac = j // factor, (j % factor).to(u.dtype) / factor
    return u[..., i0] * (1 - frac) + u[..., (i0 + 1) % n] * frac


class BurgersFVSolver(NativeSolver):
    """Finite-volume Burgers on a grid `factor` times finer than the data grid."""

    def __init__(self, n_data: int, factor: int, nu: float, interval: float):
        self.factor, self.interval = factor, interval
        self.solver = BurgersFV(n_data * factor, nu)

    def lift(self, u: torch.Tensor) -> torch.Tensor:
        return u if self.factor == 1 else upsample_linear_periodic(u, self.factor)

    def advance(self, u: torch.Tensor) -> torch.Tensor:
        return self.solver.advance(u, self.interval)

    def project(self, u: torch.Tensor) -> torch.Tensor:
        return u if self.factor == 1 else coarse_grain_1d(u, self.factor)


class BurgersSpectralSolver(NativeSolver):
    """Pseudo-spectral Burgers directly on the data grid (no shock capturing)."""

    def __init__(self, n: int, nu: float, interval: float, device: torch.device):
        self.steps = math.ceil(interval / 2.5e-3)
        self.solver = BurgersSpectral(n, nu, interval / self.steps, device=device)

    def lift(self, u: torch.Tensor) -> torch.Tensor:
        return self.solver.to_spectral(u)

    def advance(self, u_hat: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            u_hat = self.solver.step(u_hat)
        return u_hat

    def project(self, u_hat: torch.Tensor) -> torch.Tensor:
        return self.solver.to_physical(u_hat)
