#!/usr/bin/env python3
"""XOR with KarciFANN and with classical ANN, as in Karakurt et al. (2022).

A 2-4-1 sigmoid network, MSE loss, identical random initial weights for both
methods, and the same numeric value used for alpha and for the learning rate.

    python3 experiments/xor_experiment.py [--epochs 20000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import MLP, KarciFANN, SGD, Table, train, xor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
    )
    args = parser.parse_args()

    x, t = xor()
    table = Table("xor_experiment")
    print(f"XOR, 2-4-1 sigmoid network, MSE, {args.epochs} full-batch epochs, seed {args.seed}\n")
    header = f"{'alpha / lr':>10} | {'KarciFANN MSE':>14} {'outputs':>30} | {'classical MSE':>14} {'outputs':>30}"
    print(header)
    print("-" * len(header))

    for alpha in args.alphas:
        base = MLP([2, 4, 1], activation="sigmoid", loss="mse", weight_init="uniform01",
                   seed=args.seed)
        net_k, net_s = base.copy(), base.copy()
        hk = train(net_k, KarciFANN(alpha=alpha), x, t, epochs=args.epochs)
        hs = train(net_s, SGD(lr=alpha), x, t, epochs=args.epochs)
        out_k = np.round(net_k.predict(x).ravel(), 3)
        out_s = np.round(net_s.predict(x).ravel(), 3)
        print(
            f"{alpha:10.1f} | {hk.loss[-1]:14.8f} {str(out_k):>30} |"
            f" {hs.loss[-1]:14.8f} {str(out_s):>30}"
        )
        for method, history, outputs in (("KarciFANN", hk, out_k), ("classical ANN", hs, out_s)):
            table.add(coefficient=alpha, method=method, mse=history.loss[-1],
                      accuracy=history.accuracy[-1],
                      diverged=int(history.diverged_at is not None),
                      **{f"output_{i}": float(v) for i, v in enumerate(outputs)})

    table.write()
    print(
        "\nExpected targets: [0. 1. 1. 0.]\n"
        "KarciFANN uses no learning rate at all -- the step size at every\n"
        "iteration is (J/W)^(alpha-1), computed from the network's own error."
    )


if __name__ == "__main__":
    main()
