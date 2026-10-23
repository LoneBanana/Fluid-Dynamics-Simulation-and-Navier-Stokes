"""Time integration shared by the pseudo-spectral solvers."""
from __future__ import annotations

from typing import Callable

import torch


def ifrk4_step(v_hat: torch.Tensor, nonlinear: Callable[[torch.Tensor], torch.Tensor],
               e_half: torch.Tensor, e_full: torch.Tensor, dt: float) -> torch.Tensor:
    """One integrating-factor RK4 step of v_t = L v + N(v) for a diagonal linear operator L.

    `e_half = exp(L dt / 2)` and `e_full = exp(L dt)`. The stiff linear (viscous) part is
    integrated exactly, so the step size is limited only by the advective CFL condition.
    """
    a = dt * nonlinear(v_hat)
    b = dt * nonlinear(e_half * (v_hat + a / 2))
    c = dt * nonlinear(e_half * v_hat + b / 2)
    d = dt * nonlinear(e_full * v_hat + e_half * c)
    return e_full * v_hat + (e_full * a + 2 * e_half * (b + c) + d) / 6
