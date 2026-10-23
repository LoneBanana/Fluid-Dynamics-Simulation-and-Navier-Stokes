"""Training loop shared by every learned stepper.

Models learn from short windows of ground-truth trajectories. The loss unrolls the
model autoregressively for K steps and backpropagates through the whole rollout,
with K ramped up from 1 as training progresses. Pure one-step training produces
models that look accurate for one step but drift or explode in long rollouts,
because they never see their own (slightly wrong) outputs as inputs.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from nsml.models.steppers import HybridStepper
from nsml.utils import CHECKPOINT_DIR, count_params, seed_everything


@dataclass
class TrainConfig:
    iters: int = 4000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_unroll: int = 4
    ramp_start: float = 0.2   # fraction of training spent on one-step loss
    ramp_end: float = 0.6     # fraction at which the unroll length reaches max_unroll
    grad_clip: float = 1.0
    val_every: int = 250
    val_steps: int = 50
    seed: int = 0


def unroll_length(it: int, cfg: TrainConfig) -> int:
    progress = it / cfg.iters
    if progress < cfg.ramp_start:
        return 1
    ramp = min(1.0, (progress - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start))
    return 1 + round(ramp * (cfg.max_unroll - 1))


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    return ((pred - target).pow(2).sum(dims) / target.pow(2).sum(dims).clamp_min(1e-12)).mean()


@torch.no_grad()
def rollout(stepper: nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
    """Autoregressive rollout; returns (batch, steps + 1, *space) including x0.
    Classical solvers keep their native-resolution state between the output frames."""
    native = getattr(stepper, "native", None)
    if native is not None:
        state, frames = native.lift(x0), [x0]
        for _ in range(steps):
            state = native.advance(state)
            frames.append(native.project(state))
        return torch.stack(frames, dim=1)
    frames = [x0]
    for _ in range(steps):
        frames.append(stepper(frames[-1]))
    return torch.stack(frames, dim=1)


@torch.no_grad()
def rollout_error(stepper: nn.Module, trajectories: torch.Tensor, steps: int) -> float:
    """Mean relative L2 error over a rollout started from each trajectory's first frame."""
    pred = rollout(stepper, trajectories[:, 0], steps)
    dims = tuple(range(2, pred.ndim))
    truth = trajectories[:, : steps + 1]
    err = (pred - truth).pow(2).sum(dims).sqrt() / truth.pow(2).sum(dims).sqrt()
    err = torch.nan_to_num(err[:, 1:], nan=1e3).clamp(max=1e3)
    return err.mean().item()


@torch.no_grad()
def fit_scales(stepper: nn.Module, train: torch.Tensor, max_frames: int = 512) -> None:
    """Normalise network inputs by the field std and outputs by the std of what they must
    predict: the one-step increment, or for hybrids the solver's one-step error."""
    gen = torch.Generator().manual_seed(0)
    n_traj, n_t = train.shape[:2]
    idx = torch.randint(n_traj, (max_frames,), generator=gen)
    t = torch.randint(n_t - 1, (max_frames,), generator=gen)
    x, y = train[idx, t], train[idx, t + 1]
    if isinstance(stepper, HybridStepper):
        target = torch.cat([y[i:i + 64] - stepper.solver_step(x[i:i + 64]) for i in range(0, max_frames, 64)])
    else:
        target = y - x
    # Reduce on the CPU in float64: cheap, and immune to accelerator reduction quirks.
    stepper.set_scales(train.cpu().double().std().item(), target.cpu().double().std().item())


def train(stepper: nn.Module, data: dict, cfg: TrainConfig, device: torch.device, tag: str,
          log=print) -> dict:
    """Train `stepper` on data["train"], keeping the weights with the best validation rollout.
    Saves a checkpoint to checkpoints/<tag>.pt and returns the training history."""
    seed_everything(cfg.seed)
    train_set = data["train"].to(device)
    val_set = data["val"].to(device)
    fit_scales(stepper, train_set)
    n_traj, n_t = train_set.shape[:2]
    val_steps = min(cfg.val_steps, val_set.shape[1] - 1)

    opt = torch.optim.AdamW(stepper.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(1, cfg.iters // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warmup) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / cfg.iters))))

    history = {"iter": [], "train_loss": [], "val_rollout_error": [], "unroll": [], "seconds": []}
    best = (float("inf"), None)
    running, count, tic = 0.0, 0, time.time()
    log(f"[{tag}] {count_params(stepper):,} trainable parameters on {device}")
    for it in range(cfg.iters):
        k = unroll_length(it, cfg)
        idx = torch.randint(n_traj, (cfg.batch_size,), device=device)
        t0 = torch.randint(n_t - k, (cfg.batch_size,), device=device)
        window = train_set[idx[:, None], t0[:, None] + torch.arange(k + 1, device=device)]
        x, loss = window[:, 0], 0.0
        for step in range(1, k + 1):
            x = stepper(x)
            loss = loss + relative_mse(x, window[:, step]) / k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(stepper.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        running, count = running + loss.item(), count + 1

        if (it + 1) % cfg.val_every == 0 or it + 1 == cfg.iters:
            stepper.eval()
            val_err = rollout_error(stepper, val_set, val_steps)
            stepper.train()
            if val_err < best[0]:
                best = (val_err, copy.deepcopy(stepper.state_dict()))
            for key, value in (("iter", it + 1), ("train_loss", running / count), ("val_rollout_error", val_err),
                               ("unroll", k), ("seconds", time.time() - tic)):
                history[key].append(value)
            log(f"[{tag}] iter {it + 1:5d}  unroll {k}  train {running / count:.3e}  "
                f"val rollout err {val_err:.4f}  ({time.time() - tic:.0f}s)")
            running, count = 0.0, 0

    stepper.load_state_dict(best[1])
    stepper.eval()
    reloaded = rollout_error(stepper, val_set, val_steps)
    if abs(reloaded - best[0]) > 0.01 * best[0]:
        log(f"[{tag}] WARNING: restored weights give val error {reloaded:.4f}, expected {best[0]:.4f}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "history": history, "config": asdict(cfg), "meta": data["meta"]},
               CHECKPOINT_DIR / f"{tag}.pt")
    return history
