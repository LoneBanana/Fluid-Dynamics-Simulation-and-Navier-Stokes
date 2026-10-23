"""Evaluate steppers on held-out test trajectories: accuracy, physical fidelity, stability, speed.

Every stepper starts from the same coarse initial state as the ground truth and rolls out
autoregressively over the whole test horizon. Classical baselines go through exactly
the same code path as the learned models, so the comparison is apples to apples.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from nsml.models import LEARNED, build_stepper
from nsml.train import rollout
from nsml.utils import CHECKPOINT_DIR, count_params, time_per_call

# Approximate real FLOPs per output element for the elementwise ops the models and solvers use.
_ELEMENTWISE_COST = {
    "add": 1, "sub": 1, "rsub": 1, "mul": 1, "div": 1, "neg": 1, "abs": 1, "sign": 1, "maximum": 1,
    "minimum": 1, "clamp": 1, "clamp_min": 1, "clamp_max": 1, "where": 1, "relu": 1, "pow": 2, "sqrt": 4,
    "exp": 8, "tanh": 8, "gelu": 12, "native_group_norm": 8, "avg_pool1d": 2, "avg_pool2d": 4,
    "sum": 1, "mean": 1, "amax": 1,
}


class _ArithmeticCounter(TorchDispatchMode):
    """Counts FFT and elementwise FLOPs (convolutions and matmuls are left to FlopCounterMode).
    A real FFT of length L costs ~2.5 L log2 L per transform, a complex one ~5 L log2 L; complex
    elementwise ops are counted at 4x their real cost."""

    def __init__(self):
        super().__init__()
        self.flops = 0.0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name.startswith("_fft_"):
            dims = args[1]
            real = args[0] if name == "_fft_r2c" else out
            length = math.prod(real.shape[d] for d in dims)
            factor = 5.0 if name == "_fft_c2c" else 2.5
            self.flops += factor * real.numel() * math.log2(max(length, 2))
        elif name in _ELEMENTWISE_COST:
            source = args[0] if name in ("sum", "mean", "amax") else out
            if isinstance(source, torch.Tensor):
                self.flops += _ELEMENTWISE_COST[name] * source.numel() * (4 if source.is_complex() else 1)
        return out


def mflops_per_step(name: str, meta: dict) -> float:
    """Estimated MFLOPs for one step of a single simulation: an implementation-independent
    cost that predicts what a hand-written C++/GLSL port would pay, unlike PyTorch timings
    which are dominated by per-call overhead at these tiny sizes."""
    stepper = load_stepper(name, meta, torch.device("cpu"))
    shape = (1, meta["n"]) if meta["name"] == "burgers" else (1, meta["n"], meta["n"])
    x = torch.randn(shape)
    native = getattr(stepper, "native", None)
    state = native.lift(x) if native is not None else None
    with torch.no_grad(), FlopCounterMode(display=False) as dense, _ArithmeticCounter() as arith:
        native.advance(state) if native is not None else stepper(x)
    return (dense.get_total_flops() + arith.flops) / 1e6


def load_stepper(name: str, meta: dict, device: torch.device) -> nn.Module:
    stepper = build_stepper(name, meta, device)
    if name in LEARNED:
        ckpt = torch.load(CHECKPOINT_DIR / f"{meta['name']}_{name}.pt", map_location=device, weights_only=False)
        stepper.load_state_dict(ckpt["state_dict"])
    return stepper.eval()


def _flat(x: torch.Tensor) -> torch.Tensor:
    """(batch, time, *space) -> (batch, time, points)."""
    return x.reshape(*x.shape[:2], -1)


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """Mean over trajectories of ||pred - truth|| / ||truth|| for every frame."""
    p, t = _flat(pred).double(), _flat(truth).double()
    err = (p - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-12)
    return torch.nan_to_num(err, nan=np.inf).mean(0).numpy()


def correlation(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """Mean over trajectories of the Pearson correlation between fields, for every frame."""
    p, t = _flat(pred).double(), _flat(truth).double()
    p, t = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
    corr = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)
    return torch.nan_to_num(corr, nan=0.0).mean(0).numpy()


def time_above(curve: np.ndarray, times: np.ndarray, threshold: float) -> float:
    """Time until `curve` first drops below `threshold` (the whole horizon if it never does)."""
    below = np.nonzero(curve < threshold)[0]
    return float(times[below[0]]) if below.size else float(times[-1])


def energy_spectrum_2d(w: torch.Tensor) -> np.ndarray:
    """Shell-summed kinetic energy spectrum E(k) of 2D vorticity fields (..., n, n), averaged
    over all leading dims. Uses E = |w_hat|^2 / (2 |k|^2) per mode."""
    n = w.shape[-1]
    w_hat = torch.fft.rfft2(w.double()) / n**2
    ky = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)[:, None]
    kx = torch.fft.rfftfreq(n, d=1.0 / n, dtype=torch.float64)[None, :]
    k2 = kx**2 + ky**2
    energy = w_hat.abs() ** 2 / (2 * k2.clamp_min(1.0))
    energy[..., 1:-1] *= 2  # half-spectrum: interior kx columns stand for +/- kx
    shells = torch.sqrt(k2).round().long().flatten()
    per_mode = energy.reshape(-1, k2.numel()).mean(0)
    spectrum = torch.zeros(n, dtype=torch.float64).index_add_(0, shells, per_mode)
    return spectrum[: n // 2].numpy()


def max_gradient_1d(u: torch.Tensor) -> np.ndarray:
    """Mean over trajectories of max|du/dx| (one-sided differences), per frame."""
    dx = 2 * math.pi / u.shape[-1]
    return ((torch.roll(u, -1, -1) - u).abs().amax(-1) / dx).mean(0).numpy()


def total_variation_1d(u: torch.Tensor) -> np.ndarray:
    """Mean total variation sum|u_{i+1} - u_i| per frame. The true (entropy) solution never
    increases it, so growth signals spurious oscillations such as Gibbs ringing at shocks."""
    return (torch.roll(u, -1, -1) - u).abs().sum(-1).mean(0).numpy()


def speed_ms_per_step(name: str, meta: dict, device: torch.device, batch: int = 1) -> float:
    """Wall-clock milliseconds for one stepper call on a single state (the real-time use case)."""
    stepper = load_stepper(name, meta, device)
    shape = (batch, meta["n"]) if meta["name"] == "burgers" else (batch, meta["n"], meta["n"])
    x = torch.randn(shape, device=device)
    native = getattr(stepper, "native", None)
    with torch.no_grad():
        if native is not None:  # time the solver on its own state, as it runs in a rollout
            state = native.lift(x)
            return 1e3 * time_per_call(lambda: native.advance(state), device, warmup=3, repeats=10)
        return 1e3 * time_per_call(lambda: stepper(x), device, warmup=3, repeats=10)


@torch.no_grad()
def evaluate_models(names: list[str], data: dict, device: torch.device,
                    chunk: int = 32) -> tuple[dict, dict[str, torch.Tensor]]:
    """Roll every model out over the full test set. Returns (metrics, rollouts on the CPU)."""
    meta, test = data["meta"], data["test"]
    steps = test.shape[1] - 1
    times = np.arange(steps + 1) * meta["save_every"]
    metrics: dict = {"times": times.tolist(), "models": {}}
    rollouts = {}
    for name in names:
        stepper = load_stepper(name, meta, device)
        pred = torch.cat([rollout(stepper, test[i:i + chunk, 0].to(device), steps).cpu()
                          for i in range(0, test.shape[0], chunk)])
        rollouts[name] = pred
        err, corr = relative_l2(pred, test), correlation(pred, test)
        entry = {
            "params": count_params(stepper),
            "rel_l2": err.tolist(),
            "corr": corr.tolist(),
            "mean_rel_l2": float(np.mean(err[1:])),
            "time_corr_above_0.9": time_above(corr, times, 0.9),
            "time_corr_above_0.5": time_above(corr, times, 0.5),
            "finite": bool(torch.isfinite(pred).all()),
        }
        if meta["name"] == "burgers":
            entry["max_gradient"] = max_gradient_1d(pred).tolist()
            entry["total_variation"] = total_variation_1d(pred).tolist()
        else:
            late = pred[:, steps // 2:]
            entry["spectrum_late"] = energy_spectrum_2d(late).tolist()
            entry["enstrophy"] = (0.5 * pred.double().pow(2).mean((-2, -1))).mean(0).tolist()
        metrics["models"][name] = entry
        print(f"{name:>14}: mean rel L2 {entry['mean_rel_l2']:.4f}   corr>0.9 until t={entry['time_corr_above_0.9']:.2f}",
              flush=True)

    truth = {"rel_l2": [0.0] * (steps + 1), "corr": [1.0] * (steps + 1)}
    if meta["name"] == "burgers":
        truth["max_gradient"] = max_gradient_1d(test).tolist()
        truth["total_variation"] = total_variation_1d(test).tolist()
    else:
        truth["spectrum_late"] = energy_spectrum_2d(test[:, steps // 2:]).tolist()
        truth["enstrophy"] = (0.5 * test.double().pow(2).mean((-2, -1))).mean(0).tolist()
    metrics["truth"] = truth
    return metrics, rollouts
