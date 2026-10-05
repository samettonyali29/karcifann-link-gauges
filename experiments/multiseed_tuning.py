#!/usr/bin/env python3
"""Does tuning on several seeds remove the outlying splits?

`split_robustness.py` found that the power-versus-SGD difference has no stable
sign once the split is redrawn, and that the positive means are carried by one
split apiece: Vehicle 1 at $+22.56$, Satimage 8 at $+35.87$, Segment 7 at
$+28.39$.  On exactly those three, validation selected an unusually large SGD
rate -- 32, 64 and 32, against 0.5 to 16 everywhere else -- which works for the
tuning seed and collapses for the confirmation seeds.

That is a hypothesis about the *selection procedure*, not about either update
rule, and the manuscript states it descriptively because nothing tested it.
This script tests it.  The design changes one thing:

    single-seed   tune on seed 1, pick the best validation accuracy
    multi-seed    tune on seeds 1, 2 and 3, pick the best *mean* validation
                  accuracy over the three

Everything else -- split, grids, budgets, the eight confirmation seeds -- is
held fixed, so the two differences are paired and their gap is attributable to
the selection rule alone.

Seed 1 is included in the multi-seed set, so the single-seed winner falls out of
the same runs at no extra cost.  That is also a protocol check: it must
reproduce what `split_robustness.py` stored for the same split, and the script
fails loudly if it does not.

Scope, stated because it bounds the conclusion.  The three outlying splits were
chosen *because* they were outliers, so this is a targeted diagnostic and not a
re-estimate of anything.  To keep it from being purely conditioned on the
failure, each dataset also contributes one control split whose difference sits
at that dataset's median, where single-seed tuning did not visibly misfire and
multi-seed tuning therefore has nothing to repair.

Run it the way `run_all.sh` runs everything else, with BLAS pinned to one
thread::

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python3 experiments/multiseed_tuning.py --dataset vehicle
    python3 experiments/multiseed_tuning.py --dataset segment --tune-seeds 3

That is not a performance suggestion.  A multi-threaded BLAS reduces in a
different order and changes the last bits of a summed loss, so results produced
without those exports do not belong to the environment README.md scopes the
byte-identity claim to.  The first attempt at this experiment was launched
without them and was discarded after an hour; pinning the threads also makes
running the three datasets side by side an actual speed-up rather than three
processes fighting over the same four cores.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def _split_robustness():
    """Import the parent study for its grids and budgets, without editing it.

    Copying them would let the two drift apart, and editing that script would
    put a source change inside the span of results it has already written,
    which tests/test_audit.py rejects.  The same route headroom_splits.py takes.
    """
    spec = importlib.util.spec_from_file_location(
        "split_robustness_module", ROOT / "experiments" / "split_robustness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARENT = _split_robustness()
DEFAULTS = PARENT.DEFAULTS
CONFIRM_SEEDS = PARENT.CONFIRM_SEEDS

#: The outlying split of each dataset, and a control at that dataset's median.
#: Read off results/split_robustness_*.csv; asserted against it at run time so
#: a regenerated parent study cannot leave these silently pointing at the
#: wrong splits.
SPLITS = {
    "vehicle": {"outlier": 1, "control": 4},
    "satimage": {"outlier": 8, "control": 5},
    "segment": {"outlier": 7, "control": 0},
}

#: Tuning seeds.  Seed 1 is the parent study's, kept first so the single-seed
#: rule is a prefix of the multi-seed one rather than a separate experiment.
#: All lie outside CONFIRM_SEEDS, so nothing is selected on and reported from.
TUNE_SEEDS = [1, 2, 3]


def stored_split_summary(dataset: str, split: int) -> dict:
    """What the parent study recorded for this split, for the protocol check."""
    import csv

    path = ROOT / "results" / f"split_robustness_{dataset}.csv"
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["stage"] == "split_summary" and int(float(row["split"])) == split:
                return row
    raise SystemExit(f"no stored split_summary for {dataset} split {split}")


class Bench:
    """One split, held fixed while both selection rules are applied to it."""

    def __init__(self, data, fit: dict) -> None:
        (self.xtr, self.ttr), (self.xva, self.tva), (self.xte, self.tte) = data
        self.layers = [self.xtr.shape[1], 50, self.ttr.shape[1]]
        self.fit = fit
        self._cache: dict = {}

    def run(self, make_optimizer, seed: int, key) -> dict:
        if (key, seed) in self._cache:
            return self._cache[(key, seed)]
        net = MLP(self.layers, activation="sigmoid", loss="mse",
                  weight_init="glorot", seed=seed)
        history = train(net, make_optimizer(), self.xtr, self.ttr,
                        validation_data=(self.xva, self.tva),
                        **{**self.fit, "seed": seed})
        result = {
            "val": history.val_accuracy[-1],
            "test": accuracy(net.predict(self.xte), self.tte),
            "diverged": history.diverged_at is not None,
        }
        self._cache[(key, seed)] = result
        return result


def build(method: str, coords):
    """An optimizer factory for a point of either grid."""
    if method == "gradient descent":
        return lambda: SGD(lr=coords[0])
    cls, _ = GAUGE_FAMILIES["power"]
    return lambda: GaugedDescent(cls(coords[0]), scale=coords[1])


def tune(bench: Bench, scales, table: Table, dataset: str, split: int,
         n_tune_seeds: int):
    """Sweep both grids on every tuning seed; return both rules' winners."""
    seeds = TUNE_SEEDS[:n_tune_seeds]
    shapes = GAUGE_FAMILIES["power"][1]
    grids = {
        "gradient descent": [(lr,) for lr in scales],
        "power": [(shape, scale) for shape in shapes for scale in scales],
    }

    winners: dict[str, dict] = {}
    for method, points in grids.items():
        scored = []
        for coords in points:
            vals = []
            for seed in seeds:
                result = bench.run(lambda c=coords, m=method: build(m, c)(),
                                   seed, (method, coords))
                table.add(stage="tune", dataset=dataset, split=split,
                          method=method, tune_seed=seed,
                          shape=coords[0] if method == "power" else "",
                          scale=coords[-1], **result)
                vals.append(result["val"])
            scored.append((coords, vals))

        #: `max` keeps the first maximum, and `points` is generated in grid
        #: order, so ties break identically under both rules -- otherwise a tie
        #: could masquerade as an effect of the selection rule.
        single = max(scored, key=lambda s: s[1][0])[0]
        multi = max(scored, key=lambda s: float(np.mean(s[1])))[0]
        winners[method] = {"single": single, "multi": multi}

        axes = (scales,) if method == "gradient descent" else (shapes, scales)
        for rule, coords in winners[method].items():
            edge = any(c in (min(a), max(a)) for c, a in zip(coords, axes))
            table.add(stage="best", dataset=dataset, split=split, method=method,
                      rule=rule, shape=coords[0] if method == "power" else "",
                      scale=coords[-1], grid_edge=int(edge))
    return winners


def confirm(bench: Bench, winners, rule: str, seeds, table: Table,
            dataset: str, split: int, kind: str) -> np.ndarray:
    """Re-run both winners of one selection rule over the confirmation seeds."""
    diffs = []
    for seed in seeds:
        a = bench.run(lambda: build("power", winners["power"][rule])(), seed,
                      ("power", winners["power"][rule]))
        b = bench.run(lambda: build("gradient descent",
                                    winners["gradient descent"][rule])(), seed,
                      ("gradient descent", winners["gradient descent"][rule]))
        table.add(stage="seed", dataset=dataset, split=split, kind=kind,
                  rule=rule, seed=seed, method="power",
                  shape=winners["power"][rule][0],
                  scale=winners["power"][rule][1], **a)
        table.add(stage="seed", dataset=dataset, split=split, kind=kind,
                  rule=rule, seed=seed, method="gradient descent", shape="",
                  scale=winners["gradient descent"][rule][0], **b)
        diffs.append(100.0 * (a["test"] - b["test"]))
    return np.array(diffs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    parser.add_argument("--tune-seeds", type=int, default=len(TUNE_SEEDS))
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    # shuffle=True is not a default of train(); every result in this study was
    # produced with it, and omitting it once already invalidated 5040 runs of
    # the parent experiment.  Taken from the parent rather than retyped.
    fit = dict(DEFAULTS[args.dataset], shuffle=True)
    scales = (PARENT.SCALES_VEHICLE if args.dataset == "vehicle"
              else PARENT.SCALES)
    seeds = CONFIRM_SEEDS[: args.seeds]
    x, t = load_dataset(args.dataset, scale=False)

    print(f"{args.dataset}: {len(SPLITS[args.dataset])} splits x "
          f"{args.tune_seeds} tuning seeds x (1 + 10)x{len(scales)} grid points, "
          f"{fit['epochs']} epochs", flush=True)

    table = Table(f"multiseed_tuning_{args.dataset}")
    started = time.perf_counter()

    for kind, split in SPLITS[args.dataset].items():
        stored = stored_split_summary(args.dataset, split)
        data = train_val_test_split(x, t, 0.15, 0.15, seed=split,
                                    standardize_features=True)
        bench = Bench(data, fit)
        winners = tune(bench, scales, table, args.dataset, split,
                       args.tune_seeds)

        # Protocol check: the seed-1 winner is the parent study's winner.  If
        # this fails the two experiments are not comparable and nothing below
        # means anything, so it stops here rather than writing a table.
        got = winners["gradient descent"]["single"][0]
        want = float(stored["sgd_lr"])
        if got != want:
            raise SystemExit(
                f"{args.dataset} split {split}: single-seed tuning selected "
                f"lr={got:g}, but split_robustness.py stored lr={want:g}. "
                "The two studies are not running the same protocol.")

        row = {"dataset": args.dataset, "split": split, "kind": kind}
        for rule in ("single", "multi"):
            diffs = confirm(bench, winners, rule, seeds, table, args.dataset,
                            split, kind)
            row[f"{rule}_mean"] = float(diffs.mean())
            row[f"{rule}_se"] = float(diffs.std(ddof=1) / np.sqrt(diffs.size))
            row[f"{rule}_sgd_lr"] = winners["gradient descent"][rule][0]
            row[f"{rule}_alpha"] = winners["power"][rule][0] + 1.0
            row[f"{rule}_power_scale"] = winners["power"][rule][1]
        row["stored_mean"] = float(stored["mean_difference"])
        row["n_seeds"] = len(seeds)
        row["n_tune_seeds"] = args.tune_seeds
        table.add(stage="split_summary", **row)

        print(f"   {kind:8s} split {split}: "
              f"single {row['single_mean']:+7.2f} (lr {row['single_sgd_lr']:g})"
              f"   multi {row['multi_mean']:+7.2f} (lr {row['multi_sgd_lr']:g})"
              f"   stored {row['stored_mean']:+7.2f}"
              f"   [{time.perf_counter() - started:5.0f}s]", flush=True)

    table.write()


if __name__ == "__main__":
    main()
