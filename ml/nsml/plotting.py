"""Figures and animations for the experiment reports.

Colours follow the entity, never its rank: each model keeps one hue in every figure.
Learned models take the first four categorical slots of the validated reference
palette; classical solvers are neutral greys with dashed lines; ground truth is ink.
Vorticity uses a blue <-> red diverging map with a neutral grey midpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_COLORS = {"fno": "#2a78d6", "unet": "#eb6834", "resnet": "#1baf7a", "hybrid": "#eda100",
                "hybrid-96": "#e87ba4"}
LABELS = {"truth": "Ground truth", "fno": "FNO", "unet": "U-Net", "resnet": "Dilated ResNet",
          "hybrid": "Hybrid (solver + CNN)", "hybrid-96": "Hybrid on 96² solver"}
VORTICITY_CMAP = LinearSegmentedColormap.from_list(
    "vorticity", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#a12a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.labelcolor": INK_2, "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    kind, _, size = name.partition("-")
    if kind == "spectral":
        return f"Spectral solver, N={size}"
    factor = size.rstrip("x")
    return "Finite volume" if factor == "1" else f"Finite volume, {factor}x finer grid"


SHORT_LABELS = {"truth": "Truth", "resnet": "ResNet", "hybrid": "Hybrid", "hybrid-96": "Hybrid-96"}


def short_label(name: str) -> str:
    """Compact label for narrow panels."""
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    if name in LABELS:
        return LABELS[name]
    kind, _, size = name.partition("-")
    return f"Spectral {size}" if kind == "spectral" else f"FV {size}"


def style(name: str, baseline_rank: int = 0) -> dict:
    """Line style for a series: colour for learned models, greys + dashes for solvers."""
    if name == "truth":
        return dict(color=INK, lw=2.0)
    if name in MODEL_COLORS:
        return dict(color=MODEL_COLORS[name], lw=1.8)
    greys = [INK_2, MUTED, AXIS]
    dashes = [(4, 2), (1.5, 1.5), (6, 2, 1.5, 2)]
    return dict(color=greys[baseline_rank % 3], lw=1.6, dashes=dashes[baseline_rank % 3])


def plot_metric_curves(metrics: dict, names: list[str], key: str, ylabel: str, title: str, path: Path,
                       log: bool = False, threshold: float | None = None):
    times = np.asarray(metrics["times"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if key not in ("rel_l2", "corr") and key in metrics.get("truth", {}):
        ax.plot(times[1:], np.asarray(metrics["truth"][key], dtype=float)[1:], label=label("truth"), **style("truth"))
    rank = 0
    for name in names:
        values = np.asarray(metrics["models"][name][key], dtype=float)
        st = style(name, rank)
        if name not in MODEL_COLORS:
            rank += 1
        ax.plot(times[1:], values[1:], label=label(name), **st)
    if threshold is not None:
        ax.axhline(threshold, color=AXIS, lw=1.0)
        ax.annotate(f"{threshold}", (times[1], threshold), xytext=(0, -11), textcoords="offset points",
                    fontsize=8, color=MUTED)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(ncol=2, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_accuracy(points: list[dict], ylabel: str, title: str, path: Path, log_y: bool = False):
    """Cost vs accuracy. `points`: dicts with name, cost (MFLOP per step), score, short label and,
    for classical solvers, a `family`; each family is drawn as a ladder of resolutions."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    families = sorted({p["family"] for p in points if p.get("family")})
    for rank, family in enumerate(families):
        ladder = sorted([p for p in points if p.get("family") == family], key=lambda p: p["cost"])
        st = style(f"solver-{rank}", rank)
        ax.plot([p["cost"] for p in ladder], [p["score"] for p in ladder], color=st["color"], lw=1.4,
                dashes=st["dashes"], marker="o", ms=6, mfc=SURFACE, mec=st["color"], mew=1.6, zorder=2,
                label=family)
        for p in ladder:
            ax.annotate(p["short"], (p["cost"], p["score"]), xytext=(5, -11), textcoords="offset points",
                        fontsize=8, color=MUTED)
    for p in points:
        if p["name"] in MODEL_COLORS:
            ax.scatter(p["cost"], p["score"], s=70, color=MODEL_COLORS[p["name"]], edgecolor=SURFACE,
                       linewidth=2, zorder=3, label=label(p["name"]))
            ax.annotate(label(p["name"]), (p["cost"], p["score"]), xytext=(7, 4), textcoords="offset points",
                        fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    for axis in ([ax.xaxis, ax.yaxis] if log_y else [ax.xaxis]):
        if axis is ax.yaxis:
            axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("estimated MFLOP per step (one simulation)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_burgers_snapshots(truth: np.ndarray, preds: dict[str, np.ndarray], frames: list[int], dt: float,
                           path: Path, zoom_frames: tuple[int, int] = (25, 50), half_width: float = 0.35):
    """Left: the true solution steepening into shocks. Right: zooms on the steepest shock at two
    times, where the models differ (at full scale every curve overlaps)."""
    from matplotlib.colors import LinearSegmentedColormap as _Ramp

    n = truth.shape[-1]
    x = np.arange(n) * 2 * np.pi / n
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ramp = _Ramp.from_list("time", ["#86b6ef", "#104281"])
    for i, f in enumerate(frames):
        axes[0].plot(x, truth[f], color=ramp(i / max(1, len(frames) - 1)), lw=1.6, label=f"t = {f * dt:.2f}")
    axes[0].set_title("Ground truth: smooth data steepens into shocks", loc="left", fontsize=10)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8, loc="lower left")
    for ax, f in zip(axes[1:], zoom_frames):
        centre = x[np.argmin(np.diff(truth[f], append=truth[f][:1]))]
        window = np.abs((x - centre + np.pi) % (2 * np.pi) - np.pi) < half_width
        order = np.argsort((x[window] - centre + np.pi) % (2 * np.pi) - np.pi)
        xs = ((x[window] - centre + np.pi) % (2 * np.pi) - np.pi)[order]
        ax.plot(xs, truth[f][window][order], label=label("truth"), **style("truth"))
        rank = 0
        for name, u in preds.items():
            st = style(name, rank)
            if name not in MODEL_COLORS:
                rank += 1
            ax.plot(xs, u[f][window][order], label=label(name), **{**st, "lw": st["lw"] * 0.85})
        ax.set_title(f"Zoom on the steepest shock, t = {f * dt:.2f}", loc="left", fontsize=10)
        ax.set_xlabel("x - shock position")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spacetime(truth: np.ndarray, preds: dict[str, np.ndarray], t_final: float, path: Path):
    """x-t diagrams (one trajectory): shocks show up as sharp diagonal fronts."""
    panels = {"truth": truth, **preds}
    vmax = np.abs(truth).max()
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 3.4), sharey=True)
    for ax, (name, u) in zip(axes, panels.items()):
        im = ax.imshow(u, origin="lower", aspect="auto", cmap=VORTICITY_CMAP, vmin=-vmax, vmax=vmax,
                       extent=(0, 2 * np.pi, 0, t_final), interpolation="nearest")
        ax.set_title(short_label(name), loc="left", fontsize=9)
        ax.set_xlabel("x")
        ax.grid(False)
    axes[0].set_ylabel("t")
    fig.colorbar(im, ax=axes, shrink=0.8, label="u")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vorticity_grid(truth: np.ndarray, preds: dict[str, np.ndarray], frames: list[int], dt: float, path: Path):
    """Rows: ground truth then each model; columns: times."""
    rows = {"truth": truth, **preds}
    vmax = np.percentile(np.abs(truth), 99.5)
    fig, axes = plt.subplots(len(rows), len(frames), figsize=(1.9 * len(frames), 1.9 * len(rows)))
    for r, (name, w) in enumerate(rows.items()):
        for c, f in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(w[f], origin="lower", cmap=VORTICITY_CMAP, vmin=-vmax, vmax=vmax, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"t = {f * dt:.1f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label(name), fontsize=8.5, color=INK_2)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_spectra(metrics: dict, names: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    truth = np.asarray(metrics["truth"]["spectrum_late"])
    k = np.arange(len(truth))
    shells = slice(1, len(truth) - 1)  # skip the mean and the half-empty Nyquist shell
    rank = 0
    for name in names:
        st = style(name, rank)
        if name not in MODEL_COLORS:
            rank += 1
        ax.loglog(k[shells], np.asarray(metrics["models"][name]["spectrum_late"])[shells], label=label(name), **st)
    ax.loglog(k[shells], truth[shells], label=label("truth"), **style("truth"))
    ax.set_ylim(truth[shells].min() / 30, truth[shells].max() * 30)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("energy E(k)")
    ax.set_title("Energy spectrum, second half of the rollout", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate_vorticity(truth: np.ndarray, preds: dict[str, np.ndarray], dt: float, path: Path, stride: int = 4):
    panels = {"truth": truth, **preds}
    vmax = np.percentile(np.abs(truth), 99.5)
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(2.3 * cols, 2.75))
    images = []
    for ax, (name, w) in zip(axes, panels.items()):
        images.append(ax.imshow(w[0], origin="lower", cmap=VORTICITY_CMAP, vmin=-vmax, vmax=vmax,
                                interpolation="nearest"))  # flat cells compress far better in a GIF
        ax.set_title(label(name), fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    stamp = fig.text(0.01, 0.02, "", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    frames = list(range(0, truth.shape[0], stride))

    def update(i):
        for im, w in zip(images, panels.values()):
            im.set_data(w[frames[i]])
        stamp.set_text(f"t = {frames[i] * dt:.1f}")
        return images + [stamp]

    FuncAnimation(fig, update, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=8), dpi=56)
    plt.close(fig)


def animate_burgers(truth: np.ndarray, preds: dict[str, np.ndarray], dt: float, path: Path):
    n = truth.shape[-1]
    x = np.arange(n) * 2 * np.pi / n
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lines = {"truth": ax.plot(x, truth[0], label=label("truth"), **style("truth"))[0]}
    rank = 0
    for name, u in preds.items():
        st = style(name, rank)
        if name not in MODEL_COLORS:
            rank += 1
        lines[name] = ax.plot(x, u[0], label=label(name), **{**st, "lw": st["lw"] * 0.85})[0]
    lim = 1.1 * np.abs(truth).max()
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    stamp = ax.set_title("", loc="left")
    fig.tight_layout()
    series = {"truth": truth, **preds}

    def update(f):
        for name, line in lines.items():
            line.set_ydata(series[name][f])
        stamp.set_text(f"Burgers rollout, t = {f * dt:.2f}")
        return list(lines.values())

    FuncAnimation(fig, update, frames=truth.shape[0], blit=False).save(path, writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)
