#!/usr/bin/env python3
"""Does the family's behaviour survive outside the configuration it was judged in?

Every gauge comparison in this study -- the average ranks, the Nemenyi result,
the ablations -- is computed inside one configuration: a single hidden layer of
50 sigmoid units trained on mean squared error with a non-adaptive rule.  That
is the configuration the source papers use, which makes comparisons *within* it
fair, and `headroom.py` shows it is also a weak configuration.  So the central
negative finding has an obvious alternative explanation that nothing in the
study rules out: that the gauges neither help nor hurt *here*, in a setting
where little is available to win.

This script removes that explanation, or confirms it.  It repeats the family
comparison in a configuration a reader would actually deploy --

    two hidden layers of 128 ReLU units, softmax output, cross-entropy loss,
    He initialisation

-- against two baselines rather than one: minibatch SGD, as before, and Adam,
which `headroom.py` found to be worth real accuracy on several of these
datasets.  The gauge arm is the power gauge (KarciFANN itself) and `exp`, which
is the best-ranked of the three non-power generators in the main comparison.

Everything else is held at the values the rest of the study uses: the same
splits, the same epoch and batch budgets, tuning on validation with one seed,
and 16 confirmation seeds disjoint from the tuning seed.  The point is that the
*configuration* is the only thing that changes, so a difference in conclusion
between this table and the main one can be attributed to it.

Run under single-threaded BLAS, as every other experiment here is:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 experiments/depth_gauges.py --dataset vehicle

Results land in results/depth_gauges_<dataset>.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from karcifann import (  # noqa: E402
    GAUGE_FAMILIES,
    MLP,
    SGD,
    Adam,
    GaugedDescent,
    Table,
    accuracy,
    train,
)  # noqa: E402

# The depth study deliberately reuses the main comparison's splits and budgets
# rather than restating them: if these drifted apart the two tables would stop
# being comparable, which is the only thing this experiment is for.
from gauge_family import DEFAULTS, load  # noqa: E402

# The gauges this run compares.  `power` is KarciFANN; `exp` is the best-ranked
# of the other three generators in the main comparison, so between them they
# cover "the rule as published" and "the strongest thing the family offers
# instead".  Running all four would quadruple the cost for two arms the main
# study already separates from the power gauge.
GAUGES = ("power", "exp")

# A gauge carries its own magnitude, so it is compared at a tuned multiplier
# rather than at its natural size -- the same two-stage design as the main
# comparison.  The grid is wider here than the main study's [0.5, 32] because
# the loss changes scale with the objective: cross-entropy at ten classes does
# not sit where an output-averaged MSE sits, and a multiplier grid inherited
# from the other configuration would put every winner on an edge.
SCALES = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Baseline grids.  SGD's reaches lower than the main study's step multipliers
# for the same reason; Adam's is the grid headroom.py already uses, so the two
# Adam columns are directly comparable.
SGD_LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ADAM_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


class DeepBench:
    """The same benchmark object as the main comparison, deeper and modernised."""

    def __init__(self, data, hidden: list[int], fit: dict) -> None:
        (self.xtr, self.ttr), (self.xva, self.tva), (self.xte, self.tte) = data
        self.layers = [self.xtr.shape[1], *hidden, self.ttr.shape[1]]
        self.fit = fit

    def run(self, make_optimizer, seed: int) -> dict:
        net = MLP(self.layers, activation="relu", output_activation="softmax",
                  loss="categorical_crossentropy", weight_init="he", seed=seed)
        history = train(net, make_optimizer(), self.xtr, self.ttr,
                        validation_data=(self.xva, self.tva),
                        **{**self.fit, "seed": seed})
        return {
            "val": history.val_accuracy[-1],
            "test": accuracy(net.predict(self.xte), self.tte),
            "loss": history.loss[-1],
            "diverged": history.diverged_at is not None,
        }


def tune(bench: DeepBench, seed: int, table: Table, scales, sgd_lrs, adam_lrs):
    """Stage A: one seed, pick each method's setting by validation accuracy."""
    print("\nA. Tuning (one seed)\n")
    print(f"   {'method':>18} | {'val%':>7} {'test%':>7} {'loss':>9} | best setting")
    print("   " + "-" * 74)

    winners: dict[str, tuple] = {}

    for name, grid, build in (
        ("gradient descent", sgd_lrs, lambda v: SGD(lr=v)),
        ("adam", adam_lrs, lambda v: Adam(lr=v)),
    ):
        best = None
        for value in grid:
            result = bench.run(lambda v=value: build(v), seed)
            table.add(stage="tune", method=name, shape="", scale=value, **result)
            if best is None or result["val"] > best[0]["val"]:
                best = (result, f"lr={value:g}", (value,))
        winners[name] = (*best, (grid,))
        table.add(stage="best", method=name, shape="", scale=best[2][0],
                  grid_edge=int(best[2][0] in (min(grid), max(grid))), **best[0])
        print(f"   {name:>18} | {best[0]['val'] * 100:7.2f} {best[0]['test'] * 100:7.2f} "
              f"{best[0]['loss']:9.5f} | {best[1]}", flush=True)

    for family in GAUGES:
        cls, shapes = GAUGE_FAMILIES[family]
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
        edge = (best[2][0] in (min(shapes), max(shapes))
                or best[2][1] in (min(scales), max(scales)))
        table.add(stage="best", method=family, shape=best[2][0], scale=best[2][1],
                  grid_edge=int(edge), **best[0])
        label = f"{family} (KarciFANN)" if family == "power" else family
        print(f"   {label:>18} | {best[0]['val'] * 100:7.2f} {best[0]['test'] * 100:7.2f} "
              f"{best[0]['loss']:9.5f} | {best[1]}", flush=True)

    flagged = [
        name for name, (_, _, coords, axes) in winners.items()
        if any(c in (min(a), max(a)) for c, a in zip(coords, axes))
    ]
    if flagged:
        print(f"\n   grid edge: {', '.join(flagged)} -- widen before trusting those rows.")
    return winners


def confirm(bench: DeepBench, winners: dict, seeds: list[int], table: Table) -> None:
    """Stage B: re-run each winner over seeds the tuning stage never saw."""
    print(f"\nB. Confirmation over {len(seeds)} seeds (mean +- 95% CI)\n")
    print(f"   {'method':>18} | {'val%':>16} {'test%':>16} | setting")
    print("   " + "-" * 76)

    def builder(name, coords):
        if name == "gradient descent":
            return lambda: SGD(lr=coords[0])
        if name == "adam":
            return lambda: Adam(lr=coords[0])
        cls = GAUGE_FAMILIES[name][0]
        return lambda: GaugedDescent(cls(coords[0]), scale=coords[1])

    for name, (_, tag, coords, _) in winners.items():
        runs = []
        for seed in seeds:
            result = bench.run(builder(name, coords), seed)
            table.add(stage="seed", method=name, setting=tag, seed=seed, **result)
            runs.append(result)
        summary, cells = {}, []
        for key in ("val", "test"):
            values = np.array([r[key] for r in runs])
            half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            summary[f"{key}_mean"], summary[f"{key}_ci95"] = values.mean(), half
            cells.append(f"{values.mean() * 100:7.2f} +-{half * 100:5.2f}")
        table.add(stage="summary", method=name, setting=tag, n_seeds=len(seeds), **summary)
        label = f"{name} (KarciFANN)" if name == "power" else name
        print(f"   {label:>18} | {cells[0]:>16} {cells[1]:>16} | {tag}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--tune-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scales", type=float, nargs="+", default=SCALES)
    parser.add_argument("--sgd-lrs", type=float, nargs="+", default=SGD_LRS)
    parser.add_argument("--adam-lrs", type=float, nargs="+", default=ADAM_LRS)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the run count and exit without training")
    args = parser.parse_args()

    n_tune = (len(args.sgd_lrs) + len(args.adam_lrs)
              + sum(len(GAUGE_FAMILIES[f][1]) for f in GAUGES) * len(args.scales))
    n_confirm = (2 + len(GAUGES)) * args.seeds
    if args.dry_run:
        print(f"{args.dataset}: {n_tune} tuning runs + {n_confirm} confirmation "
              f"runs = {n_tune + n_confirm}")
        return

    started = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    data = load(args.dataset)
    fit = dict(DEFAULTS[args.dataset], shuffle=True)
    if args.epochs:
        fit["epochs"] = args.epochs
    bench = DeepBench(data, list(args.hidden), fit)

    print(f"\n{args.dataset}: {'-'.join(map(str, bench.layers))} relu / softmax / "
          f"cross-entropy, {len(bench.xtr)} train / {len(bench.xva)} val / "
          f"{len(bench.xte)} test, {fit['epochs']} epochs, batch {fit['batch_size']}")
    print(f"{n_tune} tuning runs + {n_confirm} confirmation runs", flush=True)

    table = Table(f"depth_gauges_{args.dataset}")
    winners = tune(bench, args.tune_seed, table, args.scales, args.sgd_lrs, args.adam_lrs)
    confirm(bench, winners, list(range(100, 100 + args.seeds)), table)
    table.write()
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
