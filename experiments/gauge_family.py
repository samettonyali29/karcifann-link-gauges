#!/usr/bin/env python3
"""Is the power gauge the best member of the family, or just the convenient one?

A link gauge collapses a backpropagation chain iff it has the form
``Phi(a,b) = g(a)/g(b)``, giving the learning-rate-free family

    W  <-  W  -  [ g(J) / g(|W|) ] . dJ/dW

KarciFANN is the member with ``g(x) = x^(alpha-1)`` -- the scale-invariant one.
This script asks whether any other generating function does better, on three
datasets spanning 7, 10 and 26 classes (a 3.3x sweep of the MSE loss scale,
which is the quantity the prefactor is most sensitive to).

Two stages, because a comparison at natural magnitude only measures step size:

  A. tune  -- for each family, sweep the shape parameter and a step multiplier
              on one seed, and keep the best by validation accuracy.
  B. confirm -- re-run each winner over several seeds and report mean +- 95% CI.

    python3 experiments/gauge_family.py --dataset dry_bean
    python3 experiments/gauge_family.py --dataset mnist --seeds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    GAUGE_FAMILIES,
    MLP,
    SGD,
    GaugedDescent,
    Table,
    accuracy,
    dry_bean,
    letter,
    mnist,
    train,
    train_val_test_split,
)
from karcifann.datasets import load as load_dataset  # noqa: E402

# epochs chosen so every dataset sees a comparable number of weight updates
DEFAULTS = {
    "mnist": dict(epochs=10, batch_size=64),
    "dry_bean": dict(epochs=60, batch_size=64),
    "letter": dict(epochs=60, batch_size=64),
    "vehicle": dict(epochs=950, batch_size=64),
    "satimage": dict(epochs=125, batch_size=64),
    "segment": dict(epochs=350, batch_size=64),
    "optdigits": dict(epochs=145, batch_size=64),
    "pendigits": dict(epochs=75, batch_size=64),
}
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def load(name: str):
    if name == "mnist":
        x, t = mnist()
        return ((x[:55000], t[:55000]), (x[55000:60000], t[55000:60000]), (x[60000:], t[60000:]))
    x, t = load_dataset(name, scale=False)
    return train_val_test_split(x, t, 0.15, 0.15, seed=0,
                                standardize_features=True)


class Bench:
    def __init__(self, data, hidden: int, fit: dict) -> None:
        (self.xtr, self.ttr), (self.xva, self.tva), (self.xte, self.tte) = data
        self.layers = [self.xtr.shape[1], hidden, self.ttr.shape[1]]
        self.fit = fit

    def run(self, make_optimizer, seed: int) -> dict:
        net = MLP(self.layers, activation="sigmoid", loss="mse",
                  weight_init="glorot", seed=seed)
        history = train(net, make_optimizer(), self.xtr, self.ttr,
                        validation_data=(self.xva, self.tva), **{**self.fit, "seed": seed})
        return {
            "val": history.val_accuracy[-1],
            "test": accuracy(net.predict(self.xte), self.tte),
            "mse": history.loss[-1],
            "diverged": history.diverged_at is not None,
        }


def tune(bench: Bench, seed: int, table: Table, scales=None):
    scales = SCALES if scales is None else scales
    """Stage A: best (shape, scale) per family, by validation accuracy."""
    print("\nA. Tuning (one seed)\n")
    print(f"   {'method':>16} | {'val%':>7} {'test%':>7} {'MSE':>9} | best setting")
    print("   " + "-" * 72)

    winners: dict[str, tuple] = {}
    scored = []
    for lr in scales:
        result = bench.run(lambda lr=lr: SGD(lr=lr), seed)
        table.add(stage="tune", method="gradient descent", shape="", scale=lr, **result)
        scored.append((result, f"lr={lr:g}", (lr,)))
    baseline = max(scored, key=lambda r: r[0]["val"])
    winners["gradient descent"] = (baseline[0], baseline[1], baseline[2], (scales,))
    table.add(stage="best", method="gradient descent", shape="", scale=baseline[2][0],
              grid_edge=int(baseline[2][0] in (min(scales), max(scales))), **baseline[0])
    print(f"   {'gradient descent':>16} | {baseline[0]['val'] * 100:7.2f} "
          f"{baseline[0]['test'] * 100:7.2f} {baseline[0]['mse']:9.5f} | {baseline[1]}", flush=True)

    for family, (cls, shapes) in GAUGE_FAMILIES.items():
        best = None
        for shape in shapes:
            for scale in scales:
                result = bench.run(
                    lambda c=cls, s=shape, k=scale: GaugedDescent(c(s), scale=k), seed
                )
                table.add(stage="tune", method=family, shape=shape, scale=scale, **result)
                if best is None or result["val"] > best[0]["val"]:
                    best = (result, f"shape={shape:g}, scale={scale:g}", (shape, scale))
        winners[family] = (*best, (shapes, scales))
        edge = best[2][0] in (min(shapes), max(shapes)) or best[2][1] in (min(scales), max(scales))
        table.add(stage="best", method=family, shape=best[2][0], scale=best[2][1],
                  grid_edge=int(edge), **best[0])
        label = f"{family} (KarciFANN)" if family == "power" else family
        print(f"   {label:>16} | {best[0]['val'] * 100:7.2f} {best[0]['test'] * 100:7.2f} "
              f"{best[0]['mse']:9.5f} | {best[1]}", flush=True)

    flagged = [
        name for name, (_, _, coords, axes) in winners.items()
        if any(c in (min(a), max(a)) for c, a in zip(coords, axes))
    ]
    if flagged:
        print(f"\n   grid edge: {', '.join(flagged)} -- widen before trusting those rows.")
    return winners


def confirm(bench: Bench, winners: dict, seeds: list[int], table: Table) -> None:
    """Stage B: re-run each winner across seeds, with confidence intervals."""
    print(f"\nB. Confirmation over {len(seeds)} seeds (mean +- 95% CI)\n")
    print(f"   {'method':>16} | {'val%':>16} {'test%':>16} | setting")
    print("   " + "-" * 74)

    def builder(name, coords):
        if name == "gradient descent":
            return lambda: SGD(lr=coords[0])
        cls = GAUGE_FAMILIES[name][0]
        return lambda: GaugedDescent(cls(coords[0]), scale=coords[1])

    for name, (_, tag, coords, _) in winners.items():
        runs = []
        for seed in seeds:
            result = bench.run(builder(name, coords), seed)
            table.add(stage="seed", method=name, setting=tag, seed=seed, **result)
            runs.append(result)
        cells = []
        for key in ("val", "test"):
            values = np.array([r[key] for r in runs]) * 100
            half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            cells.append(f"{values.mean():7.2f} +-{half:5.2f}")
        label = f"{name} (KarciFANN)" if name == "power" else name
        summary = {}
        for key in ("val", "test"):
            values = np.array([r[key] for r in runs])
            half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            summary[f"{key}_mean"], summary[f"{key}_ci95"] = values.mean(), half
        table.add(stage="summary", method=name, setting=tag, n_seeds=len(seeds), **summary)
        print(f"   {label:>16} | {cells[0]:>16} {cells[1]:>16} | {tag}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--tune-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES,
                        help="step-multiplier grid; widen it if a winner lands on an edge")
    args = parser.parse_args()

    started = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    data = load(args.dataset)
    fit = dict(DEFAULTS[args.dataset], shuffle=True)
    if args.epochs:
        fit["epochs"] = args.epochs
    bench = Bench(data, args.hidden, fit)

    print(f"\n{args.dataset}: {'-'.join(map(str, bench.layers))} sigmoid / MSE, "
          f"{len(bench.xtr)} train / {len(bench.xva)} val / {len(bench.xte)} test, "
          f"{fit['epochs']} epochs, batch {fit['batch_size']}")

    table = Table(f"gauge_family_{args.dataset}")
    winners = tune(bench, args.tune_seed, table, args.scales)
    confirm(bench, winners, list(range(100, 100 + args.seeds)), table)
    table.write()
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
