"""Small shared helpers: device selection, seeding, timing and paths."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ML_ROOT / "data"
CHECKPOINT_DIR = ML_ROOT / "checkpoints"
RESULTS_DIR = ML_ROOT / "results"


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the fastest available device (CUDA > MPS > CPU).

    `prefer` or the NSML_DEVICE environment variable forces a specific device.
    """
    choice = prefer or os.environ.get("NSML_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    """Block until queued GPU work is done, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_per_call(fn, device: torch.device, warmup: int = 3, repeats: int = 20) -> float:
    """Median wall-clock seconds of `fn()` after warmup."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
