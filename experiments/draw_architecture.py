#!/usr/bin/env python3
"""Draw the network used for each dataset, as PDF and SVG.

One hidden layer, sigmoid throughout, MSE loss -- the architecture of Karakurt,
Saygili & Karci (2025), kept identical across every dataset so that the only
thing varying between runs is the optimizer.  Only the input and output widths
change, and those are fixed by the data.

Widths are read from the loaders and epochs from run_analysis.DATASETS, so a new
dataset or a changed budget shows up here without anyone editing this file.

    python3 experiments/draw_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann.plotting import save, setup  # noqa: E402

FIGURES = Path("figures")

# What each dataset's outputs mean; everything else is measured, not typed.
CLASS_NAMES = {
    "vehicle": "4 vehicle silhouettes",
    "satimage": "6 soil types",
    "dry_bean": "7 bean cultivars",
    "segment": "7 image segments",
    "optdigits": "10 digits",
    "pendigits": "10 digits",
    "mnist": "10 digits",
    "letter": "26 letters",
}

TITLES = {
    "vehicle": "Vehicle",
    "satimage": "Satimage",
    "dry_bean": "Dry Bean",
    "segment": "Segment",
    "optdigits": "Optdigits",
    "pendigits": "Pendigits",
    "mnist": "MNIST",
    "letter": "Letter Recognition",
}

#: Display order, matching experiments/update_readme.py.
ORDER = ["vehicle", "satimage", "dry_bean", "segment", "optdigits", "pendigits",
         "mnist", "letter"]


def networks() -> dict:
    """(input, hidden, output, epochs, title, classes) per dataset.

    Derived rather than transcribed: the widths come from the loaders and the
    epochs from the single DATASETS table the experiments themselves use, so this
    cannot drift out of step with what was actually trained.
    """
    from karcifann import load
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_analysis import DATASETS as CONFIG

    # Ordered by chance MSE, the same order the README's tables use.
    order = [n for n in ORDER if n in CONFIG] + [n for n in CONFIG if n not in ORDER]

    specs = {}
    for name in order:
        cfg = CONFIG[name]
        x, t = load(name)
        specs[name] = (x.shape[1], cfg["hidden"], t.shape[1], cfg["epochs"],
                       TITLES.get(name, name), CLASS_NAMES.get(name, f"{t.shape[1]} classes"))
    return specs

SHOWN = 6          # neurons drawn per layer before eliding
NODE = 0.040       # neuron radius, in axis units (the axes are equal-aspect,
                   # so this is a true circle rather than a squashed ellipse)
TOP, BOTTOM = 0.60, 0.10


def _column(ax, x, count, colour, label, sublabel):
    """Draw one layer: a few real neurons, an ellipsis, then the count."""
    drawn = min(count, SHOWN)
    elided = count > SHOWN
    ys = np.linspace(TOP, BOTTOM, drawn) if drawn > 1 else np.array([(TOP + BOTTOM) / 2])
    if elided:                       # leave a gap in the middle for the dots
        gap = drawn // 2
        ys = np.concatenate([ys[:gap] + 0.030, ys[gap:] - 0.030])
        ax.text(x, (ys[gap - 1] + ys[gap]) / 2, "⋮",
                ha="center", va="center", fontsize=11, color="#555555")

    for y in ys:
        ax.add_patch(__import__("matplotlib").patches.Circle(
            (x, y), NODE, facecolor=colour, edgecolor="white", linewidth=1.0, zorder=3))
    ax.text(x, -0.01, label, ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(x, -0.075, sublabel, ha="center", va="center", fontsize=7.5, color="#444444")
    return ys


def _connect(ax, x0, ys0, x1, ys1):
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0 + NODE, x1 - NODE], [y0, y1],
                    color="#9aa5b1", linewidth=0.35, alpha=0.55, zorder=1)


def draw(ax, spec) -> None:
    n_in, n_hidden, n_out, epochs, title, classes = spec
    xs = (0.14, 0.5, 0.86)
    ys_in = _column(ax, xs[0], n_in, "#4C72B0", "input", f"{n_in} features")
    ys_hidden = _column(ax, xs[1], n_hidden, "#DD8452", "hidden", f"{n_hidden} units")
    ys_out = _column(ax, xs[2], n_out, "#55A868", "output", f"{n_out} units")
    _connect(ax, xs[0], ys_in, xs[1], ys_hidden)
    _connect(ax, xs[1], ys_hidden, xs[2], ys_out)

    # the bias, which the papers fold in as one more randomly initialised weight
    for x, ys in ((xs[0], ys_hidden), (xs[1], ys_out)):
        ax.add_patch(__import__("matplotlib").patches.Circle(
            (x + 0.11, 0.70), NODE * 0.75, facecolor="white",
            edgecolor="#888888", linewidth=0.9, zorder=3))
        ax.text(x + 0.11, 0.70, "b", ha="center", va="center", fontsize=6.5, color="#555555")
        for y in ys:
            ax.plot([x + 0.11, xs[xs.index(x) + 1] - NODE], [0.70 - NODE * 0.75, y],
                    color="#c3c9d1", linewidth=0.3, alpha=0.5, zorder=1)

    ax.set_title(f"{title}   {n_in}–{n_hidden}–{n_out}", fontsize=10, pad=4)
    ax.text(0.5, -0.155, f"{classes}  ·  sigmoid throughout  ·  MSE loss",
            ha="center", va="center", fontsize=7.5, color="#444444")
    ax.text(0.5, -0.215, f"batch 64  ·  {epochs} epochs  ·  Glorot-uniform init",
            ha="center", va="center", fontsize=7.5, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.26, 0.78)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    plt = setup()
    FIGURES.mkdir(exist_ok=True)

    nets = networks()
    for name, spec in nets.items():
        fig, ax = plt.subplots(figsize=(4.4, 3.0), layout="constrained")
        draw(ax, spec)
        save(fig, f"architecture_{name}", FIGURES)
        print(f"   figures/architecture_{name}.[pdf|svg]")

    ncol = 4
    nrow = -(-len(nets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.3 * nrow),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(nets):]:
        ax.axis("off")
    for ax, spec in zip(axes, nets.values()):
        draw(ax, spec)
    fig.suptitle(
        "Networks used in every experiment — identical except for the widths the data fix;\n"
        "biases are drawn from the same distribution as the weights, and only the optimizer varies",
        fontsize=9)
    save(fig, "architectures", FIGURES)
    print("   figures/architectures.[pdf|svg]")


if __name__ == "__main__":
    main()
