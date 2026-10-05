#!/usr/bin/env python3
"""Alpha / learning-rate sweep on the 8x8 UCI digits, KarciFANN vs classical ANN.

Reproduces the shape of Tables 1 and 2 of Karakurt, Saygili & Karci (2025) on a
dataset small enough to run in under a minute.  Both networks start from the
same random weights and use the same numeric value for alpha and for the
learning rate, as the paper requires.

    python3 experiments/digits_experiment.py [--full-batch] [--epochs 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    MLP,
    KarciFANN,
    SGD,
    digits,
    Table,
    precision_recall_f1,
    train,
    train_val_test_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="one update per epoch over the whole training set, as in the papers "
             "(needs a few thousand epochs to converge)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    )
    args = parser.parse_args()

    x, t = digits()
    (xtr, ttr), (xva, tva), (xte, tte) = train_val_test_split(x, t, 0.15, 0.15, seed=0)
    fit = dict(epochs=args.epochs)
    if not args.full_batch:
        fit.update(batch_size=args.batch_size, shuffle=True, seed=0)

    print(
        f"UCI digits 64-{args.hidden}-10 sigmoid / MSE, "
        f"{len(xtr)} train / {len(xva)} val / {len(xte)} test, "
        f"{args.epochs} epochs, "
        f"{'full batch' if args.full_batch else f'batch size {args.batch_size}'}\n"
    )
    header = (
        f"{'alpha/lr':>8} | {'KarciFANN':^33} | {'classical ANN':^33}\n"
        f"{'':>8} | {'train%':>7} {'val%':>7} {'test%':>7} {'MSE':>9} |"
        f" {'train%':>7} {'val%':>7} {'test%':>7} {'MSE':>9}"
    )
    print(header)
    print("-" * 86)
    table = Table("digits_experiment")

    for alpha in args.alphas:
        base = MLP([64, args.hidden, 10], activation="sigmoid", loss="mse",
                   weight_init="glorot", seed=args.seed)
        row = [f"{alpha:8.2f} |"]
        for label, (net, optimizer) in zip(
            ("KarciFANN", "classical ANN"),
            ((base.copy(), KarciFANN(alpha=alpha)), (base.copy(), SGD(lr=alpha))),
        ):
            history = train(net, optimizer, xtr, ttr, validation_data=(xva, tva), **fit)
            from karcifann import accuracy

            test_acc = accuracy(net.predict(xte), tte)
            flag = "*" if history.diverged_at is not None else " "
            table.add(coefficient=alpha, method=label,
                      train_acc=history.accuracy[-1], val_acc=history.val_accuracy[-1],
                      test_acc=test_acc, train_mse=history.loss[-1],
                      diverged=int(history.diverged_at is not None))
            row.append(
                f" {history.accuracy[-1] * 100:7.2f} {history.val_accuracy[-1] * 100:7.2f}"
                f" {test_acc * 100:7.2f} {history.loss[-1]:9.6f}{flag}|"
            )
        print("".join(row).rstrip("|"))

    table.write()
    print("\n* = training stopped early on non-finite values.")
    print("alpha = 1.0 must reproduce the classical ANN column exactly.")

    # A closer look at one good setting.
    best = MLP([64, args.hidden, 10], activation="sigmoid", loss="mse",
               weight_init="glorot", seed=args.seed)
    train(best, KarciFANN(alpha=0.4), xtr, ttr, **fit)
    precision, recall, f1 = precision_recall_f1(best.predict(xte), tte)
    print(
        f"\nalpha = 0.4 on the test set: "
        f"precision {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}"
    )


if __name__ == "__main__":
    main()
