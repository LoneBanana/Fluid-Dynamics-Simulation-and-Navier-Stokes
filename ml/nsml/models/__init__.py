"""Model zoo and factory.

`build_stepper(name, meta, device)` returns a stepper for the dataset described by
`meta` (the dict stored in each dataset file). Learned models:

    fno        Fourier Neural Operator         (global spectral convolutions)
    unet       U-Net                           (multiscale convolutions)
    resnet     dilated ResNet                  (local convolutions, GLSL-friendly)
    hybrid     coarse solver + small learned corrector
    hybrid-96  (2D only) the same corrector on a 96² spectral solver, which evolves every
               Fourier mode of the 64² data instead of only the 2/3 that survive dealiasing

Classical baselines (no training): `spectral-<n>` for both datasets (a spectral solver
at resolution n), and `fv-<factor>x` for Burgers (finite volumes on a grid `factor`
times finer than the data).
"""
from __future__ import annotations

import torch
from torch import nn

from nsml.models.cnn import DilResNet, UNet
from nsml.models.fno import FNO
from nsml.models.steppers import (BurgersFVSolver, BurgersSpectralSolver, HybridStepper, NeuralStepper,
                                  NS2DSolver, SolverStepper)
from nsml.solvers import kolmogorov_forcing

LEARNED = ("fno", "unet", "resnet", "hybrid", "hybrid-96")

# Per-dataset architecture settings, sized so that every model is cheap enough to be
# a plausible candidate for a real-time C++/GLSL port.
ARCH = {
    "burgers": {
        "fno": dict(width=48, modes=32, layers=4),
        "unet": dict(base=32, levels=3),
        "resnet": dict(width=64, blocks=4),
        "hybrid": dict(width=32, blocks=2),
    },
    "ns2d": {
        "fno": dict(width=32, modes=12, layers=4),
        "unet": dict(base=32, levels=3),
        # Same network as the hybrids' corrector: a controlled "with vs. without solver" test.
        "resnet": dict(width=32, blocks=2),
        "hybrid": dict(width=32, blocks=2),
        "hybrid-96": dict(width=32, blocks=2),
    },
}


def _net(arch: str, dim: int, in_ch: int, cfg: dict) -> nn.Module:
    if arch == "fno":
        return FNO(in_ch, 1, dim, **cfg)
    if arch == "unet":
        return UNet(in_ch, 1, dim, **cfg)
    return DilResNet(in_ch, 1, dim, **cfg)


def _baseline_solver(name: str, meta: dict, device: torch.device):
    kind, _, size = name.partition("-")
    if meta["name"] == "burgers":
        if kind == "fv":
            return BurgersFVSolver(meta["n"], int(size.rstrip("x")), meta["nu"], meta["save_every"])
        if kind == "spectral":
            return BurgersSpectralSolver(int(size), meta["nu"], meta["save_every"], device)
    elif kind == "spectral":
        return NS2DSolver(int(size), meta["n"], meta["nu"], meta["drag"], meta["k_f"], meta["save_every"], device)
    raise ValueError(f"unknown model {name!r} for dataset {meta['name']!r}")


def build_stepper(name: str, meta: dict, device: torch.device) -> nn.Module:
    dataset = meta["name"]
    dim = 1 if dataset == "burgers" else 2
    static = None
    if dataset == "ns2d":
        static = (kolmogorov_forcing(meta["n"], meta["k_f"]) / meta["k_f"])[None]  # (1, n, n)
    in_ch = 1 + (0 if static is None else static.shape[0])
    if name in ("fno", "unet", "resnet"):
        stepper = NeuralStepper(_net(name, dim, in_ch, ARCH[dataset][name]), static)
    elif name in ARCH[dataset] and name.startswith("hybrid"):
        resolution = name.partition("-")[2] or meta["n"]
        coarse = "fv-1x" if dataset == "burgers" else f"spectral-{resolution}"
        corrector = _net("resnet", dim, in_ch, ARCH[dataset][name])
        stepper = HybridStepper(_baseline_solver(coarse, meta, device), corrector, static)
    else:
        stepper = SolverStepper(_baseline_solver(name, meta, device))
    return stepper.to(device)
