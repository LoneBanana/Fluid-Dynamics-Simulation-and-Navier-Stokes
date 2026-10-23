"""Physics-informed neural networks that discover self-similar blowup profiles.

For inviscid Burgers, u_t + u u_x = 0, a singularity forming at x = 0, t = T looks like

    u(x, t) = (T - t)^lam U(y),    y = x / (T - t)^(1 + lam),

where the profile U and the exponent lam must solve

    R := -lam U + ((1 + lam) y + U) U' = 0.

Every lam has (non-trivial) solutions, but they are smooth only for lam = 1/2, 1/4,
1/6, ... (implicitly y = -U - C U^(2i+3)). lam = 1/2 is the stable, generic blowup;
the others are *unstable*: they only form from finely tuned initial data.

A PINN represents U with a smooth network and minimises the equation residual. A smooth
network cannot reproduce the non-smooth solutions, so its residual collapses only at
the admissible exponents. This is a miniature of the strategy DeepMind and collaborators
used in 2025 to discover unstable singularities of the Boussinesq and IPM equations,
where no closed form is known. Two ingredients matter, as they did there:

* Smoothness has to be probed where it can fail, at the origin and across scales. The
  residual is weighted by 1/y^5 on log-spaced points down to y = 1e-2, which forces the
  profile to be Taylor-consistent through the y^5 term.
* Precision. First-order optimisers stall around 1e-4, too coarse to separate nearby
  exponents, so the fit finishes with Levenberg-Marquardt (Gauss-Newton).

Parameterisation: U(y) = y (-1 + y^2 G(y)) with G even, so U is odd with U'(0) = -1, and
G(1) = 1/2 so that U(1) = -1/2. The latter fixes the scaling symmetry U(y) -> a U(y / a)
and excludes the trivial solution U = -y, which solves the equation for every lam.
Substituting, R / y^3 = (2 lam - 1) G + lam y G' + y^2 G (3 G + y G') with no cancellation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call, grad, jacrev, vmap

DTYPE = torch.float64
Y_MAX = 5.0


class ProfileNet(nn.Module):
    """The even function G(y) = 1/2 + M(y) + M(-y) - M(1) - M(-1), M a small tanh MLP."""

    def __init__(self, width: int = 24, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers).to(DTYPE)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(y[..., :1, :])
        return 0.5 + self.mlp(y) + self.mlp(-y) - self.mlp(one) - self.mlp(-one)

    def profile(self, y: torch.Tensor) -> torch.Tensor:
        """U(y) for y of shape (N, 1)."""
        return y * (-1 + y**2 * self(y))


def collocation() -> torch.Tensor:
    """Log-spaced points near the origin (where smoothness is decided) plus a uniform grid.
    The residual is odd in y, so y > 0 suffices."""
    near = torch.logspace(-2, 0, 160, dtype=DTYPE)
    far = torch.linspace(0, Y_MAX, 241, dtype=DTYPE)[1:]
    return torch.cat([near, far]).unsqueeze(-1)


def _weighted_residual(g, dg, y, lam):
    """R / y^3 from G and G', then weighted by (1 + 1/y^2): ~R/y^5 near 0, ~R/y^3 far out."""
    rho = (2 * lam - 1) * g + lam * y * dg + y**2 * g * (3 * g + y * dg)
    return rho * (1 + 1 / y**2)


@dataclass
class FitResult:
    net: ProfileNet
    lam: float
    loss: float
    lam_history: list[float]
    loss_history: list[float]


def _residual_vector_fn(net: ProfileNet, y: torch.Tensor):
    """f(theta, lam) -> weighted residual vector, with theta the flattened network weights."""
    names = [name for name, _ in net.named_parameters()]
    shapes = [p.shape for p in net.parameters()]
    sizes = [p.numel() for p in net.parameters()]
    points = y.squeeze(-1)

    def g_scalar(params, yi):
        return functional_call(net, params, (yi.view(1, 1),)).squeeze()

    g_all = vmap(g_scalar, in_dims=(None, 0))
    dg_all = vmap(grad(g_scalar, argnums=1), in_dims=(None, 0))

    def f(theta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        params = {n: c.view(s) for n, c, s in zip(names, theta.split(sizes), shapes)}
        r = _weighted_residual(g_all(params, points), dg_all(params, points), points, lam)
        return r / math.sqrt(points.numel())

    return f, torch.cat([p.detach().flatten() for p in net.parameters()])


def fit_profile(lam0: float, learn_lam: bool, adam_iters: int = 500, lm_iters: int = 300,
                seed: int = 0, tol: float = 1e-24) -> FitResult:
    """Fit a profile network: a short Adam warm start, then Levenberg-Marquardt until the
    loss drops below `tol`, stalls (< 0.1% progress over 50 iterations) or `lm_iters` run
    out. With `learn_lam` the exponent is optimised jointly with the network weights."""
    torch.manual_seed(seed)
    net = ProfileNet()
    y = collocation()
    f, theta = _residual_vector_fn(net, y)
    n_theta = theta.numel()
    lam = torch.tensor(lam0, dtype=DTYPE)
    lam_history, loss_history = [], []

    def pack(theta, lam):
        return torch.cat([theta, lam.view(1)]) if learn_lam else theta

    def g(z):
        return f(z[:n_theta], z[n_theta]) if learn_lam else f(z, lam)

    z = pack(theta, lam).requires_grad_(True)
    adam = torch.optim.Adam([z], lr=2e-3)
    for _ in range(adam_iters):
        adam.zero_grad()
        loss = g(z).pow(2).sum()
        loss.backward()
        adam.step()
        lam_history.append(z[n_theta].item() if learn_lam else lam0)
        loss_history.append(loss.item())

    z = z.detach()
    jac = jacrev(g)
    mu, r = 1e-3, g(z)
    for _ in range(lm_iters):
        J = jac(z)
        A, b = J.T @ J, J.T @ r
        damping = torch.diag(torch.diag(A).clamp_min(1e-12))
        improved = False
        while mu < 1e12:
            step = torch.linalg.solve(A + mu * damping, -b)
            r_new = g(z + step)
            if r_new.pow(2).sum() < r.pow(2).sum():
                z, r, mu, improved = z + step, r_new, max(mu / 3, 1e-12), True
                break
            mu *= 4
        lam_history.append(z[n_theta].item() if learn_lam else lam0)
        loss_history.append(r.pow(2).sum().item())
        stalled = len(loss_history) > adam_iters + 100 and loss_history[-1] > 0.999 * loss_history[-51]
        if not improved or loss_history[-1] < tol or stalled:
            break

    torch.nn.utils.vector_to_parameters(z[:n_theta], net.parameters())
    lam_final = z[n_theta].item() if learn_lam else lam0
    return FitResult(net, lam_final, loss_history[-1], lam_history, loss_history)


def profile_residual(net: ProfileNet, lam: float, n: int = 2001) -> float:
    """Plain (unweighted) RMS residual of the profile equation on a uniform grid of [0, Y_MAX]."""
    y = torch.linspace(0, Y_MAX, n, dtype=DTYPE).unsqueeze(-1).requires_grad_(True)
    u = net.profile(y)
    du = torch.autograd.grad(u.sum(), y)[0]
    return (-lam * u + ((1 + lam) * y + u) * du).pow(2).mean().sqrt().item()


def exact_profile(y: torch.Tensor, i: int) -> torch.Tensor:
    """Exact smooth profile for lam = 1/(2i+2) with U(1) = -1/2: solves y = -U - C U^(2i+3),
    C = 2^(2i+2), by Newton iteration (the map U -> y is monotone, so this converges)."""
    p, c = 2 * i + 3, 2.0 ** (2 * i + 2)
    u = -y / (1 + y.abs()) ** (1 - 1 / p)
    for _ in range(100):
        f = -u - c * u**p - y
        u = u - f / (-1 - c * p * u ** (p - 1))
    return u


# ---------------------------------------------------------------------------------------
# Exact blowup by characteristics: watching an unstable singularity lose to the stable one
# ---------------------------------------------------------------------------------------
#
# For inviscid Burgers with data u0(x) = -sin x + b sin 3x, characteristics give the exact
# solution x = xi + t u0(xi), u = u0(xi) up to the blowup time T = 1 / max(-u0'). With
# b = 1/27 the cubic Taylor term of u0 at the origin vanishes, which is exactly the fine
# tuning the lam = 1/4 singularity needs; any other b gives the generic lam = 1/2 blowup.
#
# Self-similarity predicts the width of the steep front shrinks like (T - t)^(1 + lam), so
# the local slope of log(width) against log(T - t), minus one, measures the exponent lam.

def front_width(b: float, s: torch.Tensor) -> torch.Tensor:
    """Half-width (in x) of the |u_x| peak at the origin, at times t = T - s (s > 0, float64).

    Everything is written in terms of s and of 1 - cos differences so the cancellation in
    1 + t u0'(xi) stays accurate down to s ~ 1e-12.
    """
    a = 1 - 3 * b                                       # -u0'(0); T = 1 / a
    t = 1 / a - s

    def du0_plus_a(xi):                                 # u0'(xi) + a, cancellation-free
        return 2 * torch.sin(xi / 2) ** 2 - 6 * b * torch.sin(3 * xi / 2) ** 2

    def ux(xi):                                         # u_x along the characteristic from xi
        return (du0_plus_a(xi) - a) / (a * s + t * du0_plus_a(xi))

    target = 0.5 * ux(torch.zeros_like(s))              # half of the peak value -1/s
    lo, hi = torch.zeros_like(s), torch.full_like(s, 0.5)
    for _ in range(200):                                # bisection: |u_x| decreases away from 0
        mid = (lo + hi) / 2
        inside = ux(mid) < target
        lo, hi = torch.where(inside, mid, lo), torch.where(inside, hi, mid)
    xi = (lo + hi) / 2
    return xi + t * (-torch.sin(xi) + b * torch.sin(3 * xi))


def effective_exponent(b: float, s: torch.Tensor) -> torch.Tensor:
    """Local blowup exponent lam_eff(s) = d log(width) / d log(s) - 1."""
    log_w = torch.log(front_width(b, s))
    log_s = torch.log(s)
    return torch.diff(log_w) / torch.diff(log_s) - 1
