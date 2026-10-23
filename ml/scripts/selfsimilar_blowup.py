"""Discover Burgers' self-similar blowup profiles with a PINN, and watch an unstable one decay.

    python scripts/selfsimilar_blowup.py                    # scan + learn (~1 h on 4 CPU cores)
    python scripts/selfsimilar_blowup.py --stages learn     # redo one stage, reuse the others
    python scripts/selfsimilar_blowup.py --stages verify    # optional, slow (several hours)
    python scripts/selfsimilar_blowup.py --stages none      # only redraw the figures

Stages: (1) coarse funnel scan over a grid of fixed exponents, (2) learn the exponent
jointly with the profile from several starting guesses, (3) verify each learned exponent
with a fine, longer-trained scan of its neighbourhood (a genuine solution shows a sharp
funnel), (4) exact blowup by characteristics. Stage results are cached in
results/selfsimilar/metrics.json and runs.json, next to the figures (funnel.png,
funnel_zoom.png, lambda_convergence.png, profiles.png, instability.png) and summary.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from nsml import plotting
from nsml.selfsimilar import (DTYPE, Y_MAX, effective_exponent, exact_profile, fit_profile,
                              profile_residual)
from nsml.utils import RESULTS_DIR

plt = plotting.plt
ADMISSIBLE = {0: 1 / 2, 1: 1 / 4, 2: 1 / 6}
ADAM_ITERS = 500  # fit_profile's default warm start


def _scan_point(lam: float) -> dict:
    torch.set_num_threads(2)
    tic = time.time()
    fit = fit_profile(lam, learn_lam=False, lm_iters=400)
    print(f"scan lam={lam:.3f}: weighted loss {fit.loss:.2e}  ({time.time() - tic:.0f}s)", flush=True)
    return {"lam": lam, "loss": fit.loss, "plain_rms": profile_residual(fit.net, fit.lam)}


def _verify_point(lam: float) -> dict:
    torch.set_num_threads(2)
    fit = fit_profile(lam, learn_lam=False, lm_iters=2000)
    print(f"verify lam={lam:.6f}: weighted loss {fit.loss:.2e}", flush=True)
    return {"lam": lam, "loss": fit.loss}


def verification_scan(learned: list[dict], workers: int) -> list[dict]:
    """Fine scans around each distinct learned exponent: runs are grouped by λ rounded to two
    decimals and each group is centred on its lowest-loss run (offsets from the *learned* value)."""
    best: dict[float, dict] = {}
    for r in learned:
        key = round(r["lam"], 2)
        if key not in best or r["loss"] < best[key]["loss"]:
            best[key] = r
    centres = sorted(r["lam"] for r in best.values())
    offsets = [-0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02]
    grid = [c + o for c in centres for o in offsets]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        points = list(pool.map(_verify_point, grid))
    return [{**p, "centre": c} for p, c in zip(points, [c for c in centres for _ in offsets])]


def _learn_run(lam0: float) -> dict:
    torch.set_num_threads(2)
    fit = fit_profile(lam0, learn_lam=True, lm_iters=2500)
    nearest = min(ADMISSIBLE, key=lambda i: abs(ADMISSIBLE[i] - fit.lam))
    y = torch.linspace(0, Y_MAX, 1001, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        u = fit.net.profile(y)
        err = (u - exact_profile(y, nearest)).abs().max().item()
    print(f"learn lam from {lam0}: lam={fit.lam:.10f} (exact {ADMISSIBLE[nearest]:.10f}), "
          f"loss {fit.loss:.2e}, max|U - exact| {err:.2e}", flush=True)
    return {"lam0": lam0, "lam": fit.lam, "loss": fit.loss, "nearest_exact": ADMISSIBLE[nearest],
            "max_profile_error": err, "lam_history": fit.lam_history, "profile": u.squeeze(-1).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default="scan,learn",
                        help="comma-separated subset of scan,learn,verify to (re)compute, or 'none'")
    parser.add_argument("--quick", action="store_true", help="coarser scan grid")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stages = set(args.stages.split(",")) - {"none"}
    out = RESULTS_DIR / "selfsimilar"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, runs_path = out / "metrics.json", out / "runs.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
    if "learn" in stages:  # 2. learn lam jointly with the profile, from several guesses
        learn_jobs = [pool.submit(_learn_run, lam0) for lam0 in (0.42, 0.58, 0.22, 0.29, 0.15, 0.19)]
    if "scan" in stages:  # 1. coarse funnel scan at fixed exponents
        grid = [float(v) for v in np.round(np.arange(0.12, 0.6001, 0.04 if args.quick else 0.02), 4)]
        metrics["scan"] = list(pool.map(_scan_point, grid))
    if "learn" in stages:
        runs = [job.result() for job in learn_jobs]
        runs_path.write_text(json.dumps(runs, separators=(",", ":")))
        metrics["learned"] = [{k: v for k, v in r.items() if k not in ("lam_history", "profile")} for r in runs]
    pool.shutdown()
    if "verify" in stages:  # 3. fine, longer-trained scans around each learned exponent
        metrics["verification"] = verification_scan(metrics["learned"], args.workers)

    # 4. Exact blowup by characteristics: the unstable lam = 1/4 singularity loses to lam = 1/2.
    s = torch.logspace(-2, -12, 81, dtype=DTYPE)
    s_mid = torch.sqrt(s[1:] * s[:-1])
    cases = [("generic data (b = 0)", 0.0), ("tuned data (b = 1/27)", 1 / 27),
             ("tuned, perturbed by 1e-3", 1 / 27 - 1e-3), ("tuned, perturbed by 1e-5", 1 / 27 - 1e-5)]
    characteristics = {name: effective_exponent(b, s).tolist() for name, b in cases}
    metrics["characteristics"] = {"T_minus_t": s_mid.tolist(), **characteristics}
    metrics_path.write_text(json.dumps(metrics, indent=1))

    plot_instability(s_mid.numpy(), characteristics, out / "instability.png")
    if "scan" in metrics and "learned" in metrics:
        plot_funnel(metrics["scan"], metrics["learned"], out / "funnel.png")
    if "verification" in metrics:
        plot_funnel_zoom(metrics["verification"], metrics["learned"], out / "funnel_zoom.png")
    if runs:
        plot_lambda_convergence(runs, out / "lambda_convergence.png")
        plot_profiles(runs, out / "profiles.png")
        write_summary(runs, out / "summary.md")
    print(f"wrote results to {out}")


def _admissible_guides(ax):
    for i, lam in ADMISSIBLE.items():
        ax.axvline(lam, color=plotting.AXIS, lw=1.0, zorder=0)
        ax.annotate(f"λ = 1/{2 * i + 2}", (lam, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=plotting.MUTED)


def plot_funnel(scan, runs, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lam = [p["lam"] for p in scan]
    loss = [np.sqrt(p["loss"]) for p in scan]
    ax.semilogy(lam, loss, color=plotting.MODEL_COLORS["fno"], marker="o", ms=4.5, mec=plotting.SURFACE,
                mew=1.2, label="best residual at fixed λ (grid scan)")
    ax.scatter([r["lam"] for r in runs], [np.sqrt(max(r["loss"], 1e-30)) for r in runs], s=64,
               color=plotting.MODEL_COLORS["unet"], edgecolor=plotting.SURFACE, linewidth=2, zorder=3,
               label="λ learned jointly with the profile")
    _admissible_guides(ax)
    ax.set_xlabel("blowup exponent λ")
    ax.set_ylabel("weighted residual (RMS)")
    ax.set_title("Funnel inference: smooth profiles exist only at special λ", loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_funnel_zoom(points, learned, path):
    """One panel per learned exponent: residual vs. offset from it, after longer training."""
    centres = sorted({p["centre"] for p in points})
    fig, axes = plt.subplots(1, len(centres), figsize=(3.8 * len(centres), 3.8), sharey=True, squeeze=False)
    for ax, centre in zip(axes[0], centres):
        sel = sorted((p for p in points if p["centre"] == centre), key=lambda p: p["lam"])
        ax.semilogy([p["lam"] for p in sel], [np.sqrt(max(p["loss"], 1e-30)) for p in sel],
                    color=plotting.MODEL_COLORS["fno"], marker="o", ms=5, mec=plotting.SURFACE, mew=1.2)
        exact = min(ADMISSIBLE.values(), key=lambda v: abs(v - centre))
        ax.axvline(exact, color=plotting.AXIS, lw=1.0, zorder=0)
        ax.set_title(f"around learned λ = {centre:.6f}", loc="left", fontsize=10)
        ax.set_xlabel("fixed exponent λ")
    axes[0][0].set_ylabel("weighted residual (RMS)")
    fig.suptitle("Verification: each learned exponent sits at the bottom of a funnel", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lambda_convergence(runs, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for r in runs:
        target = min(ADMISSIBLE.values(), key=lambda v: abs(v - r["lam"]))
        rank = sorted(ADMISSIBLE.values(), reverse=True).index(target)
        color = list(plotting.MODEL_COLORS.values())[rank]
        ax.plot(np.arange(len(r["lam_history"])), r["lam_history"], color=color, lw=1.6)
        ax.annotate(f"start {r['lam0']}", (0, r["lam0"]), xytext=(4, 2), textcoords="offset points",
                    fontsize=7.5, color=plotting.MUTED)
    for i, lam in ADMISSIBLE.items():
        ax.axhline(lam, color=plotting.AXIS, lw=1.0, zorder=0)
        ax.annotate(f"1/{2 * i + 2}", (1, lam), xycoords=("axes fraction", "data"), xytext=(-18, 3),
                    textcoords="offset points", fontsize=8, color=plotting.MUTED)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, max(len(r["lam_history"]) for r in runs))
    ax.axvline(ADAM_ITERS, color=plotting.AXIS, lw=1.0, zorder=0)
    ax.annotate("Levenberg-Marquardt starts", (ADAM_ITERS, 1), xycoords=("data", "axes fraction"),
                xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8, color=plotting.MUTED)
    ax.set_xlabel("optimiser iteration (Adam, then Levenberg-Marquardt)")
    ax.set_ylabel("λ")
    ax.set_title("Learning the blowup exponent", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(runs, path):
    y = np.linspace(0, Y_MAX, 1001)
    yy = torch.from_numpy(y).unsqueeze(-1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    shown = set()
    for r in runs:
        target = min(ADMISSIBLE, key=lambda i: abs(ADMISSIBLE[i] - r["lam"]))
        if target in shown or abs(ADMISSIBLE[target] - r["lam"]) > 0.01:
            continue
        shown.add(target)
        color = list(plotting.MODEL_COLORS.values())[target]
        exact = exact_profile(yy, target).squeeze(-1).numpy()
        ax.plot(np.r_[-y[::-1], y], np.r_[-exact[::-1], exact], color=plotting.INK, lw=2.4)
        u = np.asarray(r["profile"])
        ax.plot(np.r_[-y[::-1], y], np.r_[-u[::-1], u], color=color, lw=1.6, dashes=(4, 2),
                label=f"PINN, λ = {r['lam']:.6f} (exact 1/{2 * target + 2})")
    ax.plot([], [], color=plotting.INK, lw=2.4, label="exact profile")
    ax.set_xlabel("self-similar coordinate y = x / (T - t)^(1+λ)")
    ax.set_ylabel("U(y)")
    ax.set_title("Blowup profiles found by the network vs. exact solutions", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_instability(s, characteristics, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [plotting.INK, plotting.MODEL_COLORS["fno"], plotting.MODEL_COLORS["unet"],
              plotting.MODEL_COLORS["resnet"]]
    for (name, lam), color in zip(characteristics.items(), colors):
        ax.semilogx(s, lam, color=color, lw=1.8, label=name)
    ax.invert_xaxis()
    for lam in (0.5, 0.25):
        ax.axhline(lam, color=plotting.AXIS, lw=1.0, zorder=0)
    ax.set_ylim(0.15, 0.6)
    ax.set_xlabel("time to blowup T - t  (approaching the singularity →)")
    ax.set_ylabel("measured exponent λ")
    ax.set_title("Exact solutions: the λ = 1/4 blowup is unstable", loc="left")
    ax.legend(fontsize=8.5, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(runs, path):
    lines = ["| start λ | learned λ | exact λ | abs. error in λ | max abs. profile error |", "|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['lam0']} | {r['lam']:.10f} | {r['nearest_exact']:.10f} | "
                     f"{abs(r['lam'] - r['nearest_exact']):.1e} | {r['max_profile_error']:.1e} |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
