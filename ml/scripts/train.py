"""Train learned steppers on a dataset.

    python scripts/train.py burgers fno unet resnet hybrid
    python scripts/train.py ns2d fno --iters 6000 --batch-size 16
"""
from __future__ import annotations

import argparse

from nsml.data import load_dataset
from nsml.models import LEARNED, build_stepper
from nsml.train import TrainConfig, train
from nsml.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["burgers", "ns2d"])
    parser.add_argument("models", nargs="+", choices=LEARNED)
    defaults = TrainConfig()
    for field_name in ("iters", "batch_size", "lr", "max_unroll", "val_steps", "seed"):
        default = getattr(defaults, field_name)
        parser.add_argument(f"--{field_name.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    data = load_dataset(args.dataset)
    cfg = TrainConfig(iters=args.iters, batch_size=args.batch_size, lr=args.lr, max_unroll=args.max_unroll,
                      val_steps=args.val_steps, seed=args.seed)
    for name in args.models:
        stepper = build_stepper(name, data["meta"], device)
        train(stepper, data, cfg, device, tag=f"{args.dataset}_{name}")


if __name__ == "__main__":
    main()
