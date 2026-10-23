"""Evaluate trained models and classical baselines on the test set; write metrics and figures.

    python scripts/evaluate.py burgers
    python scripts/evaluate.py ns2d

Outputs go to results/<dataset>/: metrics.json, summary.md, PNG figures and a GIF.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from nsml import plotting
from nsml.data import load_dataset
from nsml.evaluate import evaluate_models, mflops_per_step, speed_ms_per_step
from nsml.models import LEARNED
from nsml.utils import CHECKPOINT_DIR, RESULTS_DIR, get_device

BASELINES = {
    "burgers": ["fv-1x", "fv-2x", "fv-4x", "fv-8x", "spectral-256"],
    "ns2d": ["spectral-32", "spectral-48", "spectral-64", "spectral-96", "spectral-128", "spectral-256"],
}
# Baselines drawn in the time-series figures (the rest only appear in the cost/accuracy ladder).
KEY_BASELINES = {"burgers": ["fv-1x", "spectral-256"], "ns2d": ["spectral-64", "spectral-128"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["burgers", "ns2d"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from metrics.json, re-simulating only the sample trajectory")
    args = parser.parse_args()

    out = RESULTS_DIR / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    data = load_dataset(args.dataset)
    meta = data["meta"]
    if args.replot:
        metrics = json.loads((out / "metrics.json").read_text())
        names = list(metrics["models"])
        learned = [m for m in names if m in LEARNED]
        plot_metrics(metrics, names, learned, args.dataset, out)
        sample = {"test": data["test"][:1], "meta": meta}
        shown = learned + KEY_BASELINES[args.dataset]
        plot_rollouts(evaluate_models(shown, sample, device)[1], data, learned, args.dataset, out, args.no_gif)
        write_summary(metrics, names, args.dataset, out / "summary.md")
        return

    learned = [m for m in LEARNED if (CHECKPOINT_DIR / f"{args.dataset}_{m}.pt").exists()]
    names = learned + BASELINES[args.dataset]
    metrics, rollouts = evaluate_models(names, data, device)

    for name in names:
        entry = metrics["models"][name]
        entry["mflops_per_step"] = mflops_per_step(name, meta)
        entry["ms_per_step_cpu"] = speed_ms_per_step(name, meta, torch.device("cpu"))
        entry["ms_per_step_gpu"] = speed_ms_per_step(name, meta, device) if device.type != "cpu" else None
        if name in learned:
            ckpt = torch.load(CHECKPOINT_DIR / f"{args.dataset}_{name}.pt", map_location="cpu", weights_only=False)
            entry["train_seconds"] = ckpt["history"]["seconds"][-1]
        print(f"{name:>14}: {entry['mflops_per_step']:.1f} MFLOP/step, {entry['ms_per_step_cpu']:.2f} ms/step on CPU",
              flush=True)
    metrics["meta"] = meta
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    plot_metrics(metrics, names, learned, args.dataset, out)
    plot_rollouts(rollouts, data, learned, args.dataset, out, args.no_gif)
    write_summary(metrics, names, args.dataset, out / "summary.md")
    print(f"wrote results to {out}")


def plot_rollouts(rollouts: dict, data: dict, learned: list[str], dataset: str, out, no_gif: bool) -> None:
    """Figures and animation of the first test trajectory."""
    meta, dt = data["meta"], data["meta"]["save_every"]
    shown = learned + KEY_BASELINES[dataset]
    truth = data["test"][0].numpy()
    preds = {name: rollouts[name][0].numpy() for name in shown}
    if dataset == "burgers":
        plotting.plot_burgers_snapshots(truth, preds, [0, 5, 10, 25, 50], dt, out / "snapshots.png")
        plotting.plot_spacetime(truth, {n: preds[n] for n in learned + ["fv-1x"]}, meta["t_final"],
                                out / "spacetime.png")
        if not no_gif:
            plotting.animate_burgers(truth, preds, dt, out / "rollout.gif")
    else:
        plotting.plot_vorticity_grid(truth, preds, [0, 20, 50, 100, 200], dt, out / "snapshots.png")
        if not no_gif:  # a few representative panels keep the GIF small
            gif_models = [m for m in ("fno", "unet", "hybrid-96", "spectral-64") if m in preds]
            plotting.animate_vorticity(truth, {m: preds[m] for m in gif_models}, dt, out / "rollout.gif")


def plot_metrics(metrics: dict, names: list[str], learned: list[str], dataset: str, out) -> None:
    """Every figure that only needs the metrics (not the rollouts themselves)."""
    shown = learned + KEY_BASELINES[dataset]
    title = "Burgers" if dataset == "burgers" else "2D turbulence"
    plotting.plot_metric_curves(metrics, shown, "rel_l2", "relative L2 error",
                                f"{title}: rollout error vs. ground truth", out / "error.png", log=True)
    points = []
    for name in names:
        entry = metrics["models"][name]
        score = entry["mean_rel_l2"] if dataset == "burgers" else entry["time_corr_above_0.9"]
        points.append({"name": name, "cost": entry["mflops_per_step"], "score": score,
                       "family": solver_family(name), "short": solver_short_label(name, metrics["meta"])})
    if dataset == "burgers":
        plotting.plot_cost_accuracy(points, "mean relative L2 error (lower is better)",
                                    "Burgers: accuracy vs. cost", out / "cost_accuracy.png", log_y=True)
        plotting.plot_metric_curves(metrics, shown, "max_gradient", "max |du/dx|",
                                    "Shock sharpness: steepest gradient", out / "max_gradient.png")
        plotting.plot_metric_curves(metrics, shown, "total_variation", "total variation",
                                    "Spurious oscillations: total variation (truth never increases it)",
                                    out / "total_variation.png")
    else:
        plotting.plot_metric_curves(metrics, shown, "corr", "vorticity correlation",
                                    f"{title}: correlation with ground truth", out / "correlation.png",
                                    threshold=0.9)
        plotting.plot_cost_accuracy(points, "time until correlation < 0.9 (higher is better)",
                                    "2D turbulence: accuracy vs. cost", out / "cost_accuracy.png")
        plotting.plot_spectra(metrics, shown, out / "spectra.png")
        plotting.plot_metric_curves(metrics, shown, "enstrophy", "mean enstrophy ½⟨ω²⟩",
                                    "Long-run stability: enstrophy stays in the true range?", out / "enstrophy.png")


def solver_family(name: str) -> str | None:
    if name in LEARNED:
        return None
    return "Finite volume" if name.startswith("fv") else "Spectral solver"


def solver_short_label(name: str, meta: dict) -> str:
    if name in LEARNED:
        return name
    kind, _, size = name.partition("-")
    n = meta["n"] * int(size.rstrip("x")) if kind == "fv" else int(size)
    return f"{n}" if meta["name"] == "burgers" else f"{n}²"


def write_summary(metrics: dict, names: list[str], dataset: str, path) -> None:
    times = np.asarray(metrics["times"])
    probe = [t for t in ((0.1, 0.5, 1.0) if dataset == "burgers" else (1.0, 5.0, 10.0, 20.0)) if t <= times[-1]]
    cols = ["model", "params", "MFLOP/step", "ms/step CPU", "ms/step GPU"] + [f"rel. L2 @ t={t:g}" for t in probe]
    cols += ["mean rel. L2"] + ([] if dataset == "burgers" else ["t (corr > 0.9)"])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name in names:
        e = metrics["models"][name]
        gpu = "-" if e["ms_per_step_gpu"] is None else f"{e['ms_per_step_gpu']:.2f}"
        row = [plotting.label(name), f"{e['params']:,}", f"{e['mflops_per_step']:.1f}", f"{e['ms_per_step_cpu']:.2f}", gpu]
        row += [f"{e['rel_l2'][int(round(t / (times[1] - times[0])))]:.3f}" for t in probe]
        row += [f"{e['mean_rel_l2']:.3f}"] + ([] if dataset == "burgers" else [f"{e['time_corr_above_0.9']:.1f}"])
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
