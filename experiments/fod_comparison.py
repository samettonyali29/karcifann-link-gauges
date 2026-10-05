#!/usr/bin/env python3
"""Karci vs Caputo-Fabrizio vs Caputo, as optimizers for the same network.

All three replace the Newton gradient in the weight update by a fractional
derivative of the loss with respect to that weight, and all three collapse to a
scalar prefactor on the ordinary gradient:

    Karci             Phi = (J / W)^(alpha-1)                         error-driven
    Caputo-Fabrizio   Phi = (1/a)(1 - exp(-a|W - W0| / (1 - a)))      distance-driven, bounded
    Caputo            Phi = |W - W0|^(1-a) / Gamma(2 - a)             distance-driven, unbounded

all equal to 1 at alpha = 1.  The script runs three comparisons:

  1. the shape of the prefactors themselves;
  2. a raw sweep at unit step multiplier, which is what a naive comparison does;
  3. a scale-controlled comparison, where each rule gets a tuned multiplier, so
     the shape of the prefactor is compared rather than its magnitude.

    python3 experiments/fod_comparison.py [--epochs 60]
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
    CaputoFabrizioGD,
    CaputoGD,
    KarciFANN,
    Table,
    caputo_factor,
    cf_factor,
    digits,
    fod_factor,
    train,
    train_val_test_split,
)


def show_prefactors(loss: float, table: Table) -> None:
    print(f"\n1. The three prefactors, at a loss of J = {loss} and W0 = 0\n")
    weights = [0.01, 0.05, 0.2, 0.5, 1.0, 3.0]
    for alpha in (0.2, 0.5, 0.8, 1.0):
        print(f"   alpha = {alpha}")
        print(f"   {'|W|':>8} {'Karci':>12} {'Caputo-Fabrizio':>17} {'Caputo':>12}")
        for w in weights:
            karci = float(fod_factor(loss, np.array([w]), alpha)[0])
            cf = float(cf_factor(w, alpha))
            cap = float(caputo_factor(w, alpha))
            for name, value in (("Karci", karci), ("Caputo-Fabrizio", cf), ("Caputo", cap)):
                table.add(stage="prefactor", coefficient=alpha, weight=w, loss=loss,
                          method=name, prefactor=value)
            print(f"   {w:8.2f} {karci:12.4f} {cf:17.4f} {cap:12.4f}")
        print()
    print("   Karci's prefactor also moves with J; the other two never see it.")


def raw_sweep(data, fit, table: Table) -> None:
    (xtr, ttr), (xva, tva), _ = data
    print("\n2. Raw sweep, unit step multiplier (lr = 1, scale = 1)\n")
    header = (
        f"   {'alpha':>6} | {'KarciFANN':^24} | {'Caputo-Fabrizio':^24} | {'Caputo':^24}\n"
        f"   {'':>6} | {'val%':>7} {'MSE':>8} {'<Phi>':>7} | {'val%':>7} {'MSE':>8} {'<Phi>':>7} |"
        f" {'val%':>7} {'MSE':>8} {'<Phi>':>7}"
    )
    print(header)
    print("   " + "-" * 84)
    for alpha in (0.2, 0.4, 0.6, 0.8, 0.95, 1.0):
        cells = [f"   {alpha:6.2f} |"]
        for label, optimizer in (
            ("KarciFANN", KarciFANN(alpha=alpha)),
            ("Caputo-Fabrizio", CaputoFabrizioGD(alpha=alpha)),
            ("Caputo", CaputoGD(alpha=alpha)),
        ):
            net = MLP([64, 50, 10], weight_init="glorot", seed=11)
            h = train(net, optimizer, xtr, ttr, validation_data=(xva, tva), **fit)
            table.add(stage="sweep", coefficient=alpha, method=label,
                      val_acc=h.val_accuracy[-1], train_acc=h.accuracy[-1],
                      train_mse=h.loss[-1], mean_prefactor=h.factor_mean[-1],
                      diverged=int(h.diverged_at is not None))
            cells.append(
                f" {h.val_accuracy[-1] * 100:7.2f} {h.loss[-1]:8.5f} {h.factor_mean[-1]:7.3f} |"
            )
        print("".join(cells).rstrip("|"))
    print(
        "\n   The ranking follows <Phi> almost exactly: at the same nominal alpha this\n"
        "   is a comparison of step sizes, not of mechanisms."
    )


def scale_controlled(data, fit, table: Table) -> None:
    (xtr, ttr), (xva, tva), _ = data
    print("\n3. Scale-controlled: each rule gets a tuned step multiplier\n")

    grids = {
        "gradient descent": [(SGD(lr=lr), f"lr={lr}") for lr in (0.5, 1, 2, 4, 8, 16, 32)],
        "KarciFANN": [
            (KarciFANN(alpha=a, scale=s), f"alpha={a}, scale={s}")
            for a in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
            for s in (0.25, 0.5, 1, 2, 4)
        ],
        "Caputo-Fabrizio": [
            (CaputoFabrizioGD(alpha=a, lr=lr), f"alpha={a}, lr={lr}")
            for a in (0.2, 0.4, 0.6, 0.8, 0.95)
            for lr in (1, 2, 4, 8, 16, 32)
        ],
        "Caputo": [
            (CaputoGD(alpha=a, lr=lr), f"alpha={a}, lr={lr}")
            for a in (0.2, 0.4, 0.6, 0.8, 0.95)
            for lr in (1, 2, 4, 8, 16, 32)
        ],
    }

    print(f"   {'method':>17} | {'best val%':>9} {'train%':>8} {'MSE':>9} | setting")
    print("   " + "-" * 72)
    for label, grid in grids.items():
        best = None
        for optimizer, tag in grid:
            net = MLP([64, 50, 10], weight_init="glorot", seed=11)
            h = train(net, optimizer, xtr, ttr, validation_data=(xva, tva), **fit)
            table.add(stage="tune", method=label, setting=tag,
                      val_acc=h.val_accuracy[-1], train_acc=h.accuracy[-1],
                      train_mse=h.loss[-1], diverged=int(h.diverged_at is not None))
            score = (h.val_accuracy[-1], h.accuracy[-1], h.loss[-1], tag)
            if best is None or score[0] > best[0]:
                best = score
        table.add(stage="best", method=label, setting=best[3],
                  val_acc=best[0], train_acc=best[1], train_mse=best[2])
        print(
            f"   {label:>17} | {best[0] * 100:9.2f} {best[1] * 100:8.2f} {best[2]:9.5f} | {best[3]}"
        )
    print(
        "\n   Once the step size is free to be tuned, the fractional prefactors stop\n"
        "   separating from plain gradient descent."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    x, t = digits()
    data = train_val_test_split(x, t, 0.15, 0.15, seed=0)
    fit = dict(epochs=args.epochs, batch_size=args.batch_size, shuffle=True, seed=0)

    print(
        f"UCI digits 64-50-10 sigmoid / MSE, {len(data[0][0])} train / {len(data[1][0])} val, "
        f"{args.epochs} epochs, batch size {args.batch_size}"
    )
    table = Table("fod_comparison_digits")
    show_prefactors(0.09, table)
    raw_sweep(data, fit, table)
    if not args.skip_tuning:
        scale_controlled(data, fit, table)
    table.write()


if __name__ == "__main__":
    main()
