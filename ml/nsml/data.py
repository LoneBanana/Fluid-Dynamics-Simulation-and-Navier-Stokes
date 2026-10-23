"""Dataset definitions and generation.

Both datasets follow the same recipe: simulate at high resolution with a spectral
solver, then coarse-grain onto the grid the learned models work on. The ML task is
always "given the coarse state now, predict the coarse state one save-interval later".

Files are dicts saved with torch.save: {"meta": {...}, "train"|"val"|"test": tensor}
with trajectory tensors shaped (n_traj, n_frames, n) for 1D or (n_traj, n_frames, n, n) for 2D.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch

from nsml.grf import grf_1d, grf_2d
from nsml.solvers import BurgersSpectral, NS2DSpectral, coarse_grain_1d, kolmogorov_forcing
from nsml.utils import DATA_DIR


@dataclass(frozen=True)
class BurgersConfig:
    """Viscous Burgers with random smooth data: shocks (near-blowup) form at t ~ 0.1-0.3."""
    nu: float = 0.01
    n_fine: int = 2048
    n: int = 256
    dt: float = 5e-4
    save_every: float = 0.02
    t_final: float = 1.0
    alpha: float = 2.5
    tau: float = 3.0
    kmax: int = 16
    splits: dict = field(default_factory=lambda: {"train": 1024, "val": 128, "test": 128})


@dataclass(frozen=True)
class NS2DConfig:
    """Kolmogorov flow at Re = 1/nu = 1000 with linear drag (Kochkov et al. 2021 setup)."""
    nu: float = 1e-3
    drag: float = 0.1
    k_f: int = 4
    n_fine: int = 256
    n: int = 64
    dt: float = 2.5e-3
    save_every: float = 0.1
    t_spinup: float = 20.0
    t_final: float = 20.0
    batch: int = 48
    splits: dict = field(default_factory=lambda: {"train": 72, "val": 8, "test": 16})


def dataset_path(name: str):
    return DATA_DIR / f"{name}.pt"


def load_dataset(name: str) -> dict:
    return torch.load(dataset_path(name), weights_only=False)


def generate_burgers(cfg: BurgersConfig, device: torch.device, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    solver = BurgersSpectral(cfg.n_fine, cfg.nu, cfg.dt, device=device)
    out = {"meta": {"name": "burgers", **asdict(cfg)}}
    for split, count in cfg.splits.items():
        u0 = grf_1d(count, cfg.n_fine, cfg.alpha, cfg.tau, cfg.kmax, generator=gen)
        fine = solver.simulate(u0.to(device), cfg.t_final, cfg.save_every)
        out[split] = coarse_grain_1d(fine, cfg.n_fine // cfg.n).cpu()
        print(f"burgers/{split}: {tuple(out[split].shape)}", flush=True)
    return out


def generate_ns2d(cfg: NS2DConfig, device: torch.device, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    solver = NS2DSpectral(cfg.n_fine, cfg.nu, cfg.dt, drag=cfg.drag,
                          forcing=kolmogorov_forcing(cfg.n_fine, cfg.k_f), device=device)
    spinup_steps = round(cfg.t_spinup / cfg.dt)
    total = sum(cfg.splits.values())
    w0 = 2.0 * grf_2d(total, cfg.n_fine, alpha=4.0, tau=3.0, kmax=10, generator=gen)
    trajectories = []
    for start in range(0, total, cfg.batch):
        tic = time.time()
        w_hat = solver.to_spectral(w0[start:start + cfg.batch].to(device))
        w_hat = solver.advance(w_hat, spinup_steps)  # forget the random initial condition
        traj = solver.simulate(solver.to_physical(w_hat), cfg.t_final, cfg.save_every, n_out=cfg.n)
        trajectories.append(traj.cpu())
        print(f"ns2d: trajectories {start}-{start + traj.shape[0]} done in {time.time() - tic:.0f}s",
              flush=True)
    trajectories = torch.cat(trajectories)
    out = {"meta": {"name": "ns2d", **asdict(cfg)}}
    offset = 0
    for split, count in cfg.splits.items():
        out[split] = trajectories[offset:offset + count]
        offset += count
    return out
