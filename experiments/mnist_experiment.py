#!/usr/bin/env python3
"""MNIST with KarciFANN, Caputo-Fabrizio, Caputo and classical gradient descent.

784-50-10 sigmoid network, MSE loss, 55000 train / 5000 validation / 10000
test -- the model of Karakurt, Saygili & Karci (2025).  Every method at a given
order starts from the *same* random weights, as the papers require.

All four rules have the same shape, ``W <- W - Phi . dJ/dW``, and differ only in
the prefactor:

    gradient descent  Phi = lr                                       (constant)
    KarciFANN         Phi = (J / W)^(a-1)                            (error-driven)
    Caputo-Fabrizio   Phi = (1/a)(1 - exp(-a|W - W0|/(1 - a)))       (distance, bounded)
    Caputo            Phi = |W - W0|^(1-a) / Gamma(2 - a)            (distance, unbounded)

Two comparisons are run.  Section 1 sweeps the order with a unit step
multiplier -- what a naive comparison does.  Section 2 (``--tune``) gives each
rule a step multiplier and reports best against best, which is the comparison
that separates the *shape* of the prefactor from its *magnitude*.

MNIST is downloaded once through ``sklearn.datasets.fetch_openml`` (~15 MB) and
cached in ``~/scikit_learn_data``.

    python3 experiments/mnist_experiment.py --epochs 10
    python3 experiments/mnist_experiment.py --epochs 10 --tune
    python3 experiments/mnist_experiment.py --epochs 1000 --full-batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    MLP,
    SGD,
    CaputoFabrizioGD,
    CaputoGD,
    KarciFANN,
    Table,
    accuracy,
    mnist,
    train,
)

# Caputo and Caputo-Fabrizio are defined for 0 < alpha <= 1 only.
METHODS = {
    "sgd": ("gradient descent", lambda a: SGD(lr=a), None),
    "karci": ("KarciFANN", lambda a: KarciFANN(alpha=a), None),
    "cf": ("Caputo-Fabrizio", lambda a: CaputoFabrizioGD(alpha=a), 1.0),
    "caputo": ("Caputo", lambda a: CaputoGD(alpha=a), 1.0),
}


def _fit_kwargs(args) -> dict:
    fit = dict(epochs=args.epochs)
    if not args.full_batch:
        fit.update(batch_size=args.batch_size, shuffle=True, seed=0)
    return fit


def _run(args, data, optimizer):
    (xtr, ttr), (xva, tva), (xte, tte) = data
    net = MLP([784, args.hidden, 10], activation="sigmoid", loss="mse",
              weight_init="glorot", seed=args.seed)
    start = time.time()
    history = train(net, optimizer, xtr, ttr, validation_data=(xva, tva), **_fit_kwargs(args))
    return {
        "train": history.accuracy[-1],
        "val": history.val_accuracy[-1],
        "test": accuracy(net.predict(xte), tte),
        "mse": history.loss[-1],
        "phi": history.factor_mean[-1] if history.factor_mean else float("nan"),
        "seconds": time.time() - start,
        "diverged": history.diverged_at is not None,
    }


def sweep(args, data, methods, table: Table) -> None:
    print("\n1. Order sweep at unit step multiplier (lr = 1, scale = 1)\n")
    width = 30
    print("   " + f"{'a/lr':>6} | " + " | ".join(f"{METHODS[m][0]:^{width}}" for m in methods))
    print("   " + f"{'':>6} | " + " | ".join(
        f"{'train%':>7}{'val%':>7}{'test%':>7}{'<Phi>':>9}" for _ in methods
    ))
    print("   " + "-" * (9 + len(methods) * (width + 3)))

    for alpha in args.alphas:
        cells = []
        for key in methods:
            label, build, ceiling = METHODS[key]
            if ceiling is not None and alpha > ceiling:
                cells.append(f"{'-':>7}{'-':>7}{'-':>7}{'-':>9}")
                continue
            r = _run(args, data, build(alpha))
            table.add(stage="sweep", coefficient=alpha, method=label, **r)
            flag = "*" if r["diverged"] else ""
            cells.append(
                f"{r['train'] * 100:7.2f}{r['val'] * 100:7.2f}{r['test'] * 100:7.2f}"
                f"{r['phi']:8.3f}{flag:>1}"
            )
        print("   " + f"{alpha:6.2f} | " + " | ".join(cells), flush=True)

    print("\n   '-' = the order is outside the definition's range.  '*' = diverged.")
    print("   alpha = 1.0 must give one identical row across all four methods.")
    print("   Watch <Phi>: at matched alpha the ranking tends to follow it, which")
    print("   means this table compares step sizes more than it compares mechanisms.")


def tune(args, data, methods, table: Table) -> None:
    print("\n2. Scale-controlled: each rule gets a tuned step multiplier\n")

    # The multiplier grids run well past the point of collapse on purpose: a
    # winner sitting on the edge of its grid means the method was under-tuned,
    # and comparing an under-tuned baseline against a tuned method is how a
    # spurious result gets manufactured.  _edge() below flags exactly that.
    steps = (0.5, 1, 2, 4, 8, 16, 32, 64)
    orders = (0.4, 0.6, 0.8, 0.95)
    grids = {
        "sgd": [(SGD(lr=lr), f"lr={lr}", (lr,)) for lr in steps],
        "karci": [
            (KarciFANN(alpha=a, scale=s), f"alpha={a}, scale={s}", (a, s))
            for a in orders + (1.0,)
            for s in steps
        ],
        "cf": [
            (CaputoFabrizioGD(alpha=a, lr=lr), f"alpha={a}, lr={lr}", (a, lr))
            for a in orders
            for lr in steps
        ],
        "caputo": [
            (CaputoGD(alpha=a, lr=lr), f"alpha={a}, lr={lr}", (a, lr))
            for a in orders
            for lr in steps
        ],
    }

    def _edge(grid, coordinates) -> bool:
        """True when the winning setting sits on a boundary of the grid."""
        axes = list(zip(*[c for _, _, c in grid]))
        return any(v in (min(axis), max(axis)) for v, axis in zip(coordinates, axes))

    print(f"   {'method':>17} | {'val%':>7} {'train%':>7} {'test%':>7} {'MSE':>9} | setting")
    print("   " + "-" * 76)
    warnings = []
    for key in methods:
        label = METHODS[key][0]
        best = None
        for optimizer, tag, coordinates in grids[key]:
            r = _run(args, data, optimizer)
            table.add(stage="tune", method=label, setting=tag, **r)
            if best is None or r["val"] > best[0]["val"]:
                best = (r, tag, coordinates)
        r, tag, coordinates = best
        edge = _edge(grids[key], coordinates)
        table.add(stage="best", method=label, setting=tag, grid_edge=int(edge), **r)
        print(
            f"   {label:>17} | {r['val'] * 100:7.2f} {r['train'] * 100:7.2f} "
            f"{r['test'] * 100:7.2f} {r['mse']:9.6f} | {tag}{'  (grid edge!)' if edge else ''}",
            flush=True,
        )
        if edge:
            warnings.append(label)

    if warnings:
        print(
            f"\n   WARNING: the best setting for {', '.join(warnings)} sits on the edge of\n"
            "   the search grid, so that method may be under-tuned and its row is not\n"
            "   comparable.  Widen the grid before drawing any conclusion from it."
        )
    print("\n   If these land together, the fractional prefactors are not buying a")
    print("   better search direction -- only a different default step size.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--train-size", type=int, default=55000)
    parser.add_argument("--tune", action="store_true", help="also run the scale-controlled comparison")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8, 0.95, 1.0, 1.4])
    args = parser.parse_args()

    print("loading MNIST ...", flush=True)
    x, t = mnist()
    n_train = min(args.train_size, 55000)
    data = (
        (x[:n_train], t[:n_train]),
        (x[55000:60000], t[55000:60000]),
        (x[60000:], t[60000:]),
    )

    print(
        f"\nMNIST 784-{args.hidden}-10 sigmoid / MSE, "
        f"{n_train} train / 5000 val / 10000 test, {args.epochs} epochs, "
        f"{'full batch' if args.full_batch else f'batch size {args.batch_size}'}, "
        f"identical initial weights (seed {args.seed})"
    )
    table = Table(f"mnist_methods_{args.epochs}ep")
    sweep(args, data, args.methods, table)
    if args.tune:
        tune(args, data, args.methods, table)
    table.write()


if __name__ == "__main__":
    main()
