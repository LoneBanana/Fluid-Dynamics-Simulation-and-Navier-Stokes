"""Generate the training/validation/test datasets.

    python scripts/generate_data.py burgers
    python scripts/generate_data.py ns2d
"""
from __future__ import annotations

import argparse
import time

import torch

from nsml.data import BurgersConfig, NS2DConfig, dataset_path, generate_burgers, generate_ns2d
from nsml.utils import DATA_DIR, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["burgers", "ns2d"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    tic = time.time()
    if args.dataset == "burgers":
        data = generate_burgers(BurgersConfig(), device, args.seed)
    else:
        data = generate_ns2d(NS2DConfig(), device, args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, dataset_path(args.dataset))
    print(f"wrote {dataset_path(args.dataset)} in {time.time() - tic:.0f}s on {device}")


if __name__ == "__main__":
    main()
