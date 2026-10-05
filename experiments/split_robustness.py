#!/usr/bin/env python3
"""How much of the power-versus-SGD difference is the method, and how much is
the split?

Every headline comparison in this repository rests on one stratified 70/15/15
split at seed 0.  Section 8 of the manuscript already reports that this matters
more than it should: under *unstratified* splitting gradient descent led the
rank table at 1.56 against the power gauge's 3.19, and stratifying reversed the
ordering.  That is a confession, not a measurement.  This script measures it.

For each dataset and each of several stratified splits:

  A. tune  -- sweep the power gauge's shape and multiplier, and SGD's rate,
              on *that split's* validation set.  Tuning is nested inside the
              split on purpose: tuning once and then re-splitting would leak
              the original split back in through the hyperparameters.
  B. confirm -- re-run each winner over several seeds on *that split's* test
              set.

That yields, per split, a mean paired difference.  The spread of those means
across splits is split uncertainty; the spread within a split is the seed
uncertainty the rest of the study already reports.  Reporting only the second
understates the first, and the point of this experiment is to find out by how
much.

    python3 experiments/split_robustness.py --dataset vehicle
    python3 experiments/split_robustness.py --dataset letter --splits 10 --seeds 8
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
    train,
    train_val_test_split,
)
from karcifann.datasets import load as load_dataset  # noqa: E402

#: Same budgets as every other experiment here, so the numbers are comparable.
DEFAULTS = {
    "vehicle": dict(epochs=950, batch_size=64),
    "satimage": dict(epochs=125, batch_size=64),
    "segment": dict(epochs=350, batch_size=64),
    "letter": dict(epochs=60, batch_size=64),
    "dry_bean": dict(epochs=60, batch_size=64),
    "optdigits": dict(epochs=145, batch_size=64),
    "pendigits": dict(epochs=75, batch_size=64),
}

#: The multiplier grid of run_all.sh, including Vehicle's wider one -- its
#: optima run below the common floor.
SCALES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
SCALES_VEHICLE = [0.03, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]

TUNE_SEED = 1                       # outside the confirmation set, as elsewhere
CONFIRM_SEEDS = list(range(100, 116))


class Bench:
    """One split, held fixed while its own tuning and confirmation run."""

    def __init__(self, data, fit: dict) -> None:
        (self.xtr, self.ttr), (self.xva, self.tva), (self.xte, self.tte) = data
        self.layers = [self.xtr.shape[1], 50, self.ttr.shape[1]]
        self.fit = fit

    def run(self, make_optimizer, seed: int) -> dict:
        net = MLP(self.layers, activation="sigmoid", loss="mse",
                  weight_init="glorot", seed=seed)
        history = train(net, make_optimizer(), self.xtr, self.ttr,
                        validation_data=(self.xva, self.tva),
                        **{**self.fit, "seed": seed})
        return {
            "val": history.val_accuracy[-1],
            "test": accuracy(net.predict(self.xte), self.tte),
            "diverged": history.diverged_at is not None,
        }


def tune_one_split(bench: Bench, scales, table: Table, dataset: str, split: int):
    """Stage A, inside a single split.  Only power and SGD: the question is
    about the headline pair, and sweeping all four families would quadruple the
    cost without touching it."""
    best_sgd = None
    for lr in scales:
        result = bench.run(lambda lr=lr: SGD(lr=lr), TUNE_SEED)
        table.add(stage="tune", dataset=dataset, split=split,
                  method="gradient descent", shape="", scale=lr, **result)
        if best_sgd is None or result["val"] > best_sgd[0]["val"]:
            best_sgd = (result, (lr,))

    cls, shapes = GAUGE_FAMILIES["power"]
    best_power = None
    for shape in shapes:
        for scale in scales:
            result = bench.run(
                lambda c=cls, s=shape, k=scale: GaugedDescent(c(s), scale=k),
                TUNE_SEED)
            table.add(stage="tune", dataset=dataset, split=split, method="power",
                      shape=shape, scale=scale, **result)
            if best_power is None or result["val"] > best_power[0]["val"]:
                best_power = (result, (shape, scale))

    for name, (result, coords) in (("gradient descent", best_sgd),
                                   ("power", best_power)):
        axes = (scales,) if name == "gradient descent" else (shapes, scales)
        edge = any(c in (min(a), max(a)) for c, a in zip(coords, axes))
        table.add(stage="best", dataset=dataset, split=split, method=name,
                  shape=coords[0] if name == "power" else "",
                  scale=coords[-1], grid_edge=int(edge), **result)
    return best_sgd[1], best_power[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    # shuffle=True is not a default of train(); gauge_family.py sets it
    # explicitly and every result in the study was produced with it.  Omitting
    # it here silently trained a different protocol and made the first run of
    # this experiment worthless -- caught only by checking that split 0
    # reproduces the main study's tuned winner.
    fit = dict(DEFAULTS[args.dataset], shuffle=True)
    scales = SCALES_VEHICLE if args.dataset == "vehicle" else SCALES
    seeds = CONFIRM_SEEDS[: args.seeds]
    x, t = load_dataset(args.dataset, scale=False)

    print(f"{args.dataset}: {args.splits} stratified splits x "
          f"({len(scales)} + 10x{len(scales)} tuning runs + 2x{len(seeds)} seeds), "
          f"{fit['epochs']} epochs", flush=True)

    table = Table(f"split_robustness_{args.dataset}")
    per_split = []
    started = time.perf_counter()

    for split in range(args.splits):
        data = train_val_test_split(x, t, 0.15, 0.15, seed=split,
                                    standardize_features=True)
        bench = Bench(data, fit)
        sgd_coords, power_coords = tune_one_split(bench, scales, table,
                                                  args.dataset, split)

        diffs = []
        for seed in seeds:
            a = bench.run(lambda c=power_coords: GaugedDescent(
                GAUGE_FAMILIES["power"][0](c[0]), scale=c[1]), seed)
            b = bench.run(lambda c=sgd_coords: SGD(lr=c[0]), seed)
            table.add(stage="seed", dataset=args.dataset, split=split,
                      method="power", seed=seed, shape=power_coords[0],
                      scale=power_coords[1], **a)
            table.add(stage="seed", dataset=args.dataset, split=split,
                      method="gradient descent", seed=seed, shape="",
                      scale=sgd_coords[0], **b)
            diffs.append(100.0 * (a["test"] - b["test"]))

        diffs = np.array(diffs)
        mean = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(diffs.size))
        per_split.append(mean)
        table.add(stage="split_summary", dataset=args.dataset, split=split,
                  alpha=power_coords[0] + 1.0, power_scale=power_coords[1],
                  sgd_lr=sgd_coords[0], mean_difference=mean, seed_se=se,
                  n_seeds=diffs.size,
                  n_favouring_power=int((diffs > 0).sum()),
                  n_tied=int((diffs == 0).sum()))
        print(f"   split {split}: power-SGD = {mean:+6.2f} pp (seed SE {se:4.2f}), "
              f"alpha={power_coords[0] + 1.0:.1f}, s={power_coords[1]:g}, "
              f"lr={sgd_coords[0]:g}   [{time.perf_counter() - started:5.0f}s]",
              flush=True)

    means = np.array(per_split)
    between = float(means.std(ddof=1))
    print(f"\n   across {len(means)} splits: mean {means.mean():+.2f} pp, "
          f"between-split SD {between:.2f}, "
          f"range [{means.min():+.2f}, {means.max():+.2f}], "
          f"sign positive on {(means > 0).sum()}/{len(means)}")
    table.add(stage="across_splits", dataset=args.dataset, n_splits=len(means),
              mean_difference=float(means.mean()), between_split_sd=between,
              min_difference=float(means.min()), max_difference=float(means.max()),
              n_splits_favouring_power=int((means > 0).sum()))
    table.write()


if __name__ == "__main__":
    main()
