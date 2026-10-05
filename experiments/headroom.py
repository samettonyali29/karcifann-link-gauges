#!/usr/bin/env python3
"""How much headroom does the KarciFANN evaluation setting leave on the table?

Every method compared in this repository lives in one narrow cell: a single
hidden layer of 50 sigmoid units, MSE loss, and a non-adaptive descent rule.
That is the cell the KarciFANN papers use, so comparisons inside it are fair.
The question this script asks is different -- how good is the cell itself?

Three reference points on the same data and the same budget:

  A. paper cell        50 sigmoid units, MSE, tuned plain gradient descent
  B. modern, same size 50 ReLU units, softmax + cross-entropy, Adam
  C. modern, wider     256 ReLU units, softmax + cross-entropy, Adam

If B and C sit far above A, then the differences being argued over inside A --
fractional or otherwise -- are small next to the cost of the cell itself, and
that belongs in any honest write-up.

    python3 experiments/headroom.py --dataset mnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    MLP,
    SGD,
    Adam,
    Table,
    accuracy,
    dry_bean,
    letter,
    mnist,
    train,
    train_val_test_split,
)
from karcifann.datasets import load as load_dataset  # noqa: E402

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


def load(name: str):
    if name == "mnist":
        x, t = mnist()
        return ((x[:55000], t[:55000]), (x[55000:60000], t[55000:60000]), (x[60000:], t[60000:]))
    x, t = load_dataset(name, scale=False)
    return train_val_test_split(x, t, 0.15, 0.15, seed=0,
                                standardize_features=True)


# Grids deliberately wider than any cell needs.  The originals were narrow
# enough that Vehicle's paper cell peaked at the floor and Segment's wide
# modern cell at the ceiling -- both truncated, and both invisible because the
# headroom entry in EDGE_SPECS was looking for a stage label this script never
# writes, so it matched the files and reported nothing.
CONFIGS = {
    "A  paper cell (sigmoid/MSE, 50)": dict(
        hidden=[50], activation="sigmoid", output_activation=None,
        loss="mse", init="glorot",
        grid=[("lr=%g", lambda lr: SGD(lr=lr), [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])],
    ),
    # Configuration A differs from B and C in five things at once, so a gain
    # from A to C cannot be attributed to any of them.  This arm changes only
    # the optimiser, isolating that one factor inside the paper's own cell.
    "A+ paper cell with Adam (sigmoid/MSE, 50)": dict(
        hidden=[50], activation="sigmoid", output_activation=None,
        loss="mse", init="glorot",
        grid=[("lr=%g", lambda lr: Adam(lr=lr), [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])],
    ),
    "B  modern, same size (ReLU/CE, 50)": dict(
        hidden=[50], activation="relu", output_activation="softmax",
        loss="categorical_crossentropy", init="he",
        grid=[("lr=%g", lambda lr: Adam(lr=lr), [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])],
    ),
    "C  modern, wider (ReLU/CE, 256)": dict(
        hidden=[256], activation="relu", output_activation="softmax",
        loss="categorical_crossentropy", init="he",
        grid=[("lr=%g", lambda lr: Adam(lr=lr), [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    print(f"loading {args.dataset} ...", flush=True)
    (xtr, ttr), (xva, tva), (xte, tte) = load(args.dataset)
    fit = dict(DEFAULTS[args.dataset], shuffle=True)

    def run(cfg, optimizer, seed):
        net = MLP([xtr.shape[1], *cfg["hidden"], ttr.shape[1]],
                  activation=cfg["activation"], output_activation=cfg["output_activation"],
                  loss=cfg["loss"], weight_init=cfg["init"], seed=seed)
        h = train(net, optimizer(), xtr, ttr, validation_data=(xva, tva),
                  **{**fit, "seed": seed})
        return h.val_accuracy[-1], accuracy(net.predict(xte), tte)

    print(f"\n{args.dataset}: {len(xtr)} train / {len(xva)} val / {len(xte)} test, "
          f"{fit['epochs']} epochs, batch {fit['batch_size']}, {args.seeds} seeds\n")
    print(f"   {'configuration':>34} | {'val%':>16} {'test%':>16} | setting")
    print("   " + "-" * 92)

    ci = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    table = Table(f"headroom_{args.dataset}")
    for label, cfg in CONFIGS.items():
        template, build, values = cfg["grid"][0]
        scored = []
        for v in values:
            result = run(cfg, lambda v=v: build(v), 1)
            table.add(stage="tune", configuration=label, hidden=cfg["hidden"][0],
                      activation=cfg["activation"], loss=cfg["loss"], lr=v,
                      val_acc=result[0], test_acc=result[1])
            scored.append((result, v))
        best = max(scored, key=lambda r: r[0][0])
        value = best[1]
        runs = np.array([run(cfg, lambda v=value: build(v), s)
                         for s in range(100, 100 + args.seeds)]) * 100
        for seed, (val, test) in zip(range(100, 100 + args.seeds), runs / 100):
            table.add(stage="seed", configuration=label, hidden=cfg["hidden"][0],
                      activation=cfg["activation"], loss=cfg["loss"], lr=value,
                      seed=seed, val_acc=val, test_acc=test)
        cells = [f"{runs[:, i].mean():7.2f} +-{ci(runs[:, i]):5.2f}" for i in (0, 1)]
        edge = "  (grid edge!)" if value in (min(values), max(values)) else ""
        table.add(stage="summary", configuration=label, hidden=cfg["hidden"][0],
                  activation=cfg["activation"], loss=cfg["loss"], lr=value,
                  val_mean=runs[:, 0].mean() / 100, val_ci95=ci(runs[:, 0]) / 100,
                  test_mean=runs[:, 1].mean() / 100, test_ci95=ci(runs[:, 1]) / 100,
                  n_seeds=args.seeds, grid_edge=int(bool(edge)))
        print(f"   {label:>34} | {cells[0]:>16} {cells[1]:>16} | "
              f"{template % value}{edge}", flush=True)
    table.write()


if __name__ == "__main__":
    main()
