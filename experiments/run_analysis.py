#!/usr/bin/env python3
"""Run the full analysis suite for one dataset and write tidy CSVs.

Five experiments, all sharing one seed list so every comparison is paired:

  E1 convergence   per-epoch curves and final metrics for each tuned method
  E2 gauge ablation  Phi = g(J)/g(W) split into its error half and weight half
  E3 sensitivity   one-dimensional basins: the learning rate of gradient
                   descent against the fractional order of KarciFANN
  E4 config ablation  activation x loss x optimizer, separated
  E5 factor spread  the distribution of Phi across parameters

Everything lands in ``results/<dataset>_<experiment>.csv``; figures are drawn
separately by ``make_figures.py`` so they can be redrawn without recomputing.

    python3 experiments/run_analysis.py --dataset dry_bean
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    MLP,
    SGD,
    Adam,
    ExpGauge,
    GaugedDescent,
    LogGauge,
    PowerGauge,
    TanhGauge,
    accuracy,
    ExpGauge,
    LogGauge,
    PowerGauge,
    TanhGauge,
    mnist,
    precision_recall_f1,
    train,
    train_val_test_split,
)
from karcifann.datasets import load as load_dataset  # noqa: E402

RESULTS = Path("results")
SEEDS = list(range(100, 108))
#: Tuning uses a seed outside SEEDS so no run is both selected on and
#: reported from.
TUNE_SEED = 1

DATASETS = {
    "mnist": dict(epochs=10, batch_size=64, hidden=50),
    "dry_bean": dict(epochs=60, batch_size=64, hidden=50),
    "letter": dict(epochs=60, batch_size=64, hidden=50),
    "vehicle": dict(epochs=950, batch_size=64, hidden=50),
    "satimage": dict(epochs=125, batch_size=64, hidden=50),
    "segment": dict(epochs=350, batch_size=64, hidden=50),
    "optdigits": dict(epochs=145, batch_size=64, hidden=50),
    "pendigits": dict(epochs=75, batch_size=64, hidden=50),
}

GAUGE_CLASSES = {"power": PowerGauge, "log": LogGauge, "tanh": TanhGauge, "exp": ExpGauge}


def tuned_optimizers(dataset: str) -> dict:
    """The tuned winner for each method, read from the gauge-family sweep.

    ``gauge_family.py`` records its best (shape, scale) per family to
    ``results/gauge_family_<dataset>.csv`` and this reads them back, so the
    sweep is the single source of truth.  There is deliberately no hard-coded
    fallback: an earlier version kept one, it drifted out of step with the
    sweeps after the grids were widened, and a literal that is only reachable
    when the real data is missing is a literal that is wrong exactly when it
    gets used.
    """
    path = RESULTS / f"gauge_family_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no tuning for {dataset!r}: run gauge_family.py --dataset {dataset} first"
        )

    with path.open() as handle:
        best = [r for r in csv.DictReader(handle) if r["stage"] == "best"]
    if not best:
        raise ValueError(f"{path} records no winners; re-run the sweep")

    chosen: dict = {}
    for row in best:
        scale = float(row["scale"])
        if row["method"] == "gradient descent":
            chosen["gradient descent"] = lambda s=scale: SGD(lr=s)
        else:
            cls = GAUGE_CLASSES[row["method"]]
            shape = float(row["shape"])
            label = "KarciFANN" if row["method"] == "power" else row["method"]
            chosen[label] = lambda c=cls, h=shape, s=scale: GaugedDescent(c(h), scale=s)
    order = ["gradient descent", "KarciFANN", "exp", "tanh", "log"]
    return {k: chosen[k] for k in order if k in chosen}



# Step multipliers used whenever something has to be re-tuned here.
SCALES = {"dry_bean": [0.5, 1, 2, 4, 8, 16, 32],
          "mnist": [1, 2, 4, 8, 16, 32, 64],
          "letter": [4, 8, 16, 32, 64, 128, 256]}
#: Extended below 0.5 after Vehicle's E2 `full` winner landed on the old
#: floor with the score still climbing -- a truncated grid understates the
#: method it truncates.  tests/test_audit.py enforces interiority.
WIDE_SCALES = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]

#: E4 changes the activation, the loss *and* the rule together, and each
#: combination has its own natural step size.  Re-using the sigmoid/MSE grid
#: pegged every cross-entropy cell at its floor: eleven edge winners across
#: three datasets, none of them flagged, because E4 was not audited at all.
#: Both grids are deliberately wider than any cell needs.
E4_GD_GRID = [0.001953125, 0.00390625, 0.0078125, 0.015625,
              0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
E4_ADAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def load(name: str):
    if name == "mnist":
        x, t = mnist()
        return ((x[:55000], t[:55000]), (x[55000:60000], t[55000:60000]), (x[60000:], t[60000:]))
    x, t = load_dataset(name, scale=False)
    return train_val_test_split(x, t, 0.15, 0.15, seed=0,
                                standardize_features=True)


def write(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    if not rows:
        return path
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"   wrote {path} ({len(rows)} rows)", flush=True)
    return path


class Bench:
    def __init__(self, dataset: str, data=None, config=None) -> None:
        """``data`` and ``config`` may be injected, which is how the tests run
        the whole suite on a few hundred synthetic rows in a second."""
        (self.xtr, self.ttr), (self.xva, self.tva), (self.xte, self.tte) = (
            data if data is not None else load(dataset)
        )
        cfg = dict(DATASETS.get(dataset, DATASETS["dry_bean"]))
        cfg.update(config or {})
        self.dataset = dataset
        self.layers = [self.xtr.shape[1], cfg["hidden"], self.ttr.shape[1]]
        self.fit = dict(epochs=cfg["epochs"], batch_size=cfg["batch_size"], shuffle=True)

    def net(self, seed: int, **kwargs) -> MLP:
        options = {"activation": "sigmoid", "output_activation": None,
                   "loss": "mse", "weight_init": "glorot"}
        options.update(kwargs)          # callers may override any default
        layers = options.pop("layers", self.layers)
        return MLP(layers, seed=seed, **options)

    def run(self, make_optimizer, seed: int, keep_history: bool = False, **net_kwargs):
        net = self.net(seed, **net_kwargs)
        history = train(net, make_optimizer(), self.xtr, self.ttr,
                        validation_data=(self.xva, self.tva), **{**self.fit, "seed": seed})
        prediction = net.predict(self.xte)
        precision, recall, f1 = precision_recall_f1(prediction, self.tte)
        record = {
            "val_acc": history.val_accuracy[-1],
            "test_acc": accuracy(prediction, self.tte),
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "train_mse": history.loss[-1],
            "diverged": int(history.diverged_at is not None),
        }
        return (record, history, net) if keep_history else (record, None, None)


# --------------------------------------------------------------------- E1


def e1_convergence(bench: Bench) -> None:
    print("\nE1  convergence and final metrics", flush=True)
    curves, finals = [], []
    for method, make in tuned_optimizers(bench.dataset).items():
        for seed in SEEDS:
            record, history, _ = bench.run(make, seed, keep_history=True)
            finals.append(dict(dataset=bench.dataset, method=method, seed=seed, **record))
            for epoch in range(history.epochs):
                curves.append({
                    "dataset": bench.dataset, "method": method, "seed": seed,
                    "epoch": epoch + 1,
                    "train_mse": history.loss[epoch],
                    "train_acc": history.accuracy[epoch],
                    "val_mse": history.val_loss[epoch],
                    "val_acc": history.val_accuracy[epoch],
                    "factor_mean": history.factor_mean[epoch] if history.factor_mean else "",
                })
        print(f"   {method}: done", flush=True)
    write(f"{bench.dataset}_e1_curves", curves)
    write(f"{bench.dataset}_e1_final", finals)


# --------------------------------------------------------------------- E2


def e2_gauge_ablation(bench: Bench) -> None:
    """Does the error half or the weight half of Phi do the work?"""
    print("\nE2  gauge decomposition (full / error / weight)", flush=True)
    tuning, finals = [], []
    for mode in ("full", "error", "weight"):
        best = None
        for scale in SCALES.get(bench.dataset, WIDE_SCALES):
            record, _, _ = bench.run(
                lambda s=scale, m=mode: GaugedDescent(PowerGauge(-0.2), scale=s, mode=m), TUNE_SEED
            )
            tuning.append(dict(dataset=bench.dataset, mode=mode, scale=scale, **record))
            if best is None or record["val_acc"] > best[0]:
                best = (record["val_acc"], scale)
        scale = best[1]
        for seed in SEEDS:
            record, _, _ = bench.run(
                lambda s=scale, m=mode: GaugedDescent(PowerGauge(-0.2), scale=s, mode=m), seed
            )
            finals.append(dict(dataset=bench.dataset, mode=mode, scale=scale, seed=seed, **record))
        grid = SCALES.get(bench.dataset, WIDE_SCALES)
        edge = scale in (min(grid), max(grid))
        print(f"   {mode}: best scale={scale}{'  (grid edge!)' if edge else ''}", flush=True)
    write(f"{bench.dataset}_e2_tuning", tuning)
    write(f"{bench.dataset}_e2_final", finals)


# --------------------------------------------------------------------- E3


def e3_sensitivity(bench: Bench, seeds: int = 3) -> None:
    """How wide is the basin of the learning rate, versus that of alpha?"""
    print("\nE3  sensitivity basins", flush=True)
    rows = []
    centre = {"dry_bean": 4.0, "mnist": 16.0, "letter": 64.0}.get(bench.dataset, 4.0)
    lr_grid = [centre * 2.0**k for k in range(-5, 6)]
    alpha_grid = [round(0.1 * k, 2) for k in range(1, 21, 2)] + [2.5, 3.0]

    for seed in SEEDS[:seeds]:
        for lr in lr_grid:
            record, _, _ = bench.run(lambda v=lr: SGD(lr=v), seed)
            rows.append(dict(dataset=bench.dataset, sweep="gradient descent: lr",
                             value=lr, seed=seed, **record))
        for alpha in alpha_grid:
            record, _, _ = bench.run(
                lambda a=alpha: GaugedDescent(PowerGauge(a - 1.0), scale=centre), seed)
            rows.append(dict(dataset=bench.dataset, sweep="KarciFANN: alpha",
                             value=alpha, seed=seed, **record))
        print(f"   seed {seed}: done", flush=True)
    write(f"{bench.dataset}_e3_sensitivity", rows)


# --------------------------------------------------------------------- E4


def e4_config_ablation(bench: Bench) -> None:
    """Separate the three things changed together in the headroom comparison."""
    print("\nE4  configuration ablation (activation x loss x optimizer)", flush=True)
    tuning, finals = [], []
    grids = {
        "gd": ("lr", E4_GD_GRID, lambda v: SGD(lr=v)),
        "adam": ("lr", E4_ADAM_GRID, lambda v: Adam(lr=v)),
    }
    for activation in ("sigmoid", "relu"):
        for loss in ("mse", "ce"):
            for rule in ("gd", "adam"):
                kwargs = dict(activation=activation)
                kwargs["weight_init"] = "he" if activation == "relu" else "glorot"
                if loss == "ce":
                    kwargs.update(output_activation="softmax",
                                  loss="categorical_crossentropy")
                label = f"{activation}/{loss}/{rule}"
                _, values, build = grids[rule]
                best = None
                for value in values:
                    record, _, _ = bench.run(lambda v=value: build(v), TUNE_SEED, **kwargs)
                    tuning.append(dict(dataset=bench.dataset, config=label,
                                       activation=activation, loss=loss, rule=rule,
                                       lr=value, **record))
                    if best is None or record["val_acc"] > best[0]:
                        best = (record["val_acc"], value)
                value = best[1]
                for seed in SEEDS:
                    record, _, _ = bench.run(lambda v=value: build(v), seed, **kwargs)
                    finals.append(dict(dataset=bench.dataset, config=label,
                                       activation=activation, loss=loss, rule=rule,
                                       lr=value, seed=seed, **record))
                print(f"   {label}: best lr={value:g}", flush=True)
    write(f"{bench.dataset}_e4_tuning", tuning)
    write(f"{bench.dataset}_e4_final", finals)


# --------------------------------------------------------------------- E5


def e5_factor_spread(bench: Bench) -> None:
    """The distribution of Phi across parameters -- a rescaling or a preconditioner?

    Also records the magnitudes of the *trained* weights.  The audit decides
    whether a gauge is indistinguishable from gradient descent by probing Phi
    over a range of weights, and that range has to be known rather than
    assumed: a probe floor above the smallest weight a network actually holds
    reports a gauge as degenerate when it is not.
    """
    print("\nE5  prefactor distribution", flush=True)
    rows = []
    for method, make in tuned_optimizers(bench.dataset).items():
        _, _, trained = bench.run(make, SEEDS[0], keep_history=True)
        weights = np.abs(np.concatenate([np.ravel(p) for p in trained.parameters]))
        weights = weights[np.isfinite(weights)]
        loss, _ = trained.gradients(bench.xtr[:512], bench.ttr[:512])
        optimizer = make()
        if not hasattr(optimizer, "factors"):
            continue
        factors = np.concatenate([np.ravel(f)
                                  for f in optimizer.factors(trained.parameters, loss)])
        factors = factors[np.isfinite(factors)]
        rows.append({
            "weight_min": weights.min(),
            "weight_p01": np.percentile(weights, 1),
            "weight_median": np.median(weights),
            "weight_max": weights.max(),
            "dataset": bench.dataset, "method": method, "loss": loss,
            "n": factors.size,
            "min": factors.min(), "p05": np.percentile(factors, 5),
            "median": np.median(factors), "p95": np.percentile(factors, 95),
            "max": factors.max(), "mean": factors.mean(),
            "spread_ratio": float(np.percentile(factors, 95) / max(np.percentile(factors, 5), 1e-30)),
        })
    write(f"{bench.dataset}_e5_factor_spread", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--only", nargs="+", choices=["e1", "e2", "e3", "e4", "e5"])
    parser.add_argument("--seeds", type=int, default=len(SEEDS),
                        help="paired runs per method; see analysis.wilcoxon_power_floor "
                             "-- 8 is too few for a Holm-corrected test over 5 methods")
    args = parser.parse_args()
    SEEDS[:] = list(range(100, 100 + args.seeds))

    started = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    bench = Bench(args.dataset)
    print(f"{args.dataset}: {'-'.join(map(str, bench.layers))}, {len(bench.xtr)} train / "
          f"{len(bench.xva)} val / {len(bench.xte)} test, {bench.fit['epochs']} epochs")

    steps = {"e1": e1_convergence, "e2": e2_gauge_ablation, "e3": e3_sensitivity,
             "e4": e4_config_ablation, "e5": e5_factor_spread}
    for key in (args.only or list(steps)):
        steps[key](bench)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
