#!/usr/bin/env python3
"""Three ways to put a fractional derivative into a network, compared.

  1. KarciFANN            Karci's operator at every link.  Because the
                          prefactors telescope this *is* the single-factor
                          optimizer, and the script checks that identity
                          numerically before using it.
  2. CF terminal          Caputo-Fabrizio applied once, to J along the weight
                          axis; backpropagation stays Newtonian.
  3. CF chained           Caputo-Fabrizio applied at every link, with the
                          prefactors compounding instead of collapsing.

Plain gradient descent is the reference.  All four start from the same random
weights, and each also gets a tuned step multiplier, because a comparison at
matched nominal alpha only measures step size.

    python3 experiments/chained_vs_terminal.py
    python3 experiments/chained_vs_terminal.py --dataset mnist --epochs 10
    python3 experiments/chained_vs_terminal.py --skip-tuning
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import (  # noqa: E402
    SGD,
    CaputoFabrizioGD,
    KarciFANN,
    Table,
    cf_factor,
    digits,
    fod_factor,
    mnist,
    train,
    train_val_test_split,
)
from karcifann.chained import ChainedGaugeMLP, cf_gauge, karci_gauge  # noqa: E402

METHODS = ("gradient descent", "KarciFANN", "CF terminal", "CF chained")


@dataclass
class Setup:
    layers: list[int]
    seed: int
    fit: dict
    train: tuple
    val: tuple

    def net(self, gauge=None) -> ChainedGaugeMLP:
        return ChainedGaugeMLP(
            self.layers, activation="sigmoid", loss="mse",
            weight_init="glorot", seed=self.seed, gauge=gauge,
        )

    def runs(self, alpha: float, scale: float):
        """The four methods at one order, each from identical initial weights."""
        yield METHODS[0], self.net(), SGD(lr=alpha * scale)
        yield METHODS[1], self.net(), KarciFANN(alpha=alpha, scale=scale)
        yield METHODS[2], self.net(), CaputoFabrizioGD(alpha=alpha, lr=scale)
        yield METHODS[3], self.net(gauge=cf_gauge(alpha)), SGD(lr=scale)

    def evaluate(self, net, optimizer):
        return train(net, optimizer, *self.train, validation_data=self.val, **self.fit)


def verify_telescoping(setup: Setup, table: Table) -> None:
    """Chained Karci must equal the collapsed (J/W)^(a-1) prefactor."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (8, setup.layers[0]))
    t = np.eye(setup.layers[-1])[rng.integers(0, setup.layers[-1], 8)]

    print("\n0a. Sanity check: does the Karci gauge telescope in this network?\n")
    print(f"   {'alpha':>6} | {'max relative error':>20}")
    print("   " + "-" * 31)
    for alpha in (0.4, 0.8, 1.3):
        net = setup.net()
        loss, plain = net.gradients(x, t)
        net.gauge = karci_gauge(alpha, eps=0.0)
        _, chained = net.gradients(x, t)
        error = max(
            float(np.max(np.abs(c - fod_factor(loss, p, alpha) * n)
                         / (np.abs(fod_factor(loss, p, alpha) * n) + 1e-300)))
            for p, n, c in zip(net.parameters, plain, chained)
        )
        table.add(stage="telescoping", coefficient=alpha, max_relative_error=error)
        print(f"   {alpha:6.2f} | {error:20.3e}", flush=True)
    print("\n   So run 1 below uses the single-factor optimizer; it is the same thing.")


def cf_residue(setup: Setup, alphas, table: Table) -> None:
    """The same check for Caputo-Fabrizio, which must fail.

    ``verify_telescoping`` shows the Karci gauge chained link by link equals
    its collapsed endpoint factor to round-off.  A denominator-only gauge
    cannot do that -- no constant-free ``h(b)`` satisfies Sincov's equation --
    so applying CF at every link and applying it once at the endpoint are
    different methods, not two routes to one.  Asserting that is easy; the
    useful thing is the size of the gap, because "does not telescope exactly"
    and "differs enough to matter" are separate claims.

    The comparison is deliberately the same one, in the same network and on
    the same batch: chained gradients against ``Phi_CF(W) . plain``.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (8, setup.layers[0]))
    t = np.eye(setup.layers[-1])[rng.integers(0, setup.layers[-1], 8)]

    print("\n0b. And the same check for Caputo-Fabrizio, which cannot pass\n")
    print(f"   {'alpha':>6} | {'median rel. residue':>20} | {'max rel. residue':>18}")
    print("   " + "-" * 51)
    for alpha in alphas:
        net = setup.net()
        _, plain = net.gradients(x, t)
        net.gauge = cf_gauge(alpha, eps=0.0)
        _, chained = net.gradients(x, t)
        ratios = np.concatenate([
            (np.abs(c - cf_factor(p, alpha) * n)
             / (np.abs(cf_factor(p, alpha) * n) + 1e-300)).ravel()
            for p, n, c in zip(net.parameters, plain, chained)
        ])
        median, worst = float(np.median(ratios)), float(np.max(ratios))
        table.add(stage="cf_residue", coefficient=alpha,
                  median_relative_error=median, max_relative_error=worst)
        print(f"   {alpha:6.3f} | {median:20.3e} | {worst:18.3e}", flush=True)
    print("\n   Compare 0a: the gauge that telescopes agrees to 1e-12.")


def effective_gain(setup: Setup, alphas, table: Table) -> None:
    """Mean |modified step| / |Newton step|, putting all four on one scale."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (32, setup.layers[0]))
    t = np.eye(setup.layers[-1])[rng.integers(0, setup.layers[-1], 32)]

    print("\n0c. Step magnitude relative to plain gradient descent\n")
    print(f"   {'alpha':>6} | {'KarciFANN':>12} {'CF terminal':>12} {'CF chained':>12}")
    print("   " + "-" * 48)
    for alpha in alphas:
        base = setup.net()
        loss, newton = base.gradients(x, t)
        norm = np.sqrt(sum(float(np.sum(g**2)) for g in newton))

        gains = []
        for optimizer in (KarciFANN(alpha=alpha), CaputoFabrizioGD(alpha=alpha, lr=1.0)):
            factors = optimizer.factors(base.parameters, loss)
            gains.append(
                np.sqrt(sum(float(np.sum((f * g) ** 2)) for f, g in zip(factors, newton))) / norm
            )
        _, chained = setup.net(gauge=cf_gauge(alpha)).gradients(x, t)
        gains.append(np.sqrt(sum(float(np.sum(g**2)) for g in chained)) / norm)
        for name, gain in zip(("KarciFANN", "CF terminal", "CF chained"), gains):
            table.add(stage="step_gain", coefficient=alpha, method=name, relative_step=gain)
        print(f"   {alpha:6.2f} | {gains[0]:12.4f} {gains[1]:12.4f} {gains[2]:12.4f}", flush=True)


def sweep(setup: Setup, alphas, table: Table) -> None:
    print("\n1. Order sweep at unit step multiplier\n")
    print("   " + f"{'alpha':>6} | " + " | ".join(f"{n:^22}" for n in METHODS))
    print("   " + f"{'':>6} | " + " | ".join(f"{'val%':>8}{'MSE':>9}{'<Phi>':>5}" for _ in METHODS))
    print("   " + "-" * 111)
    for alpha in alphas:
        cells = []
        for label, net, optimizer in setup.runs(alpha, 1.0):
            h = setup.evaluate(net, optimizer)
            phi = h.factor_mean[-1] if h.factor_mean else float("nan")
            table.add(stage="sweep", coefficient=alpha, method=label,
                      val_acc=h.val_accuracy[-1], train_acc=h.accuracy[-1],
                      train_mse=h.loss[-1], mean_prefactor=phi,
                      diverged=int(h.diverged_at is not None))
            flag = "*" if h.diverged_at is not None else ""
            cells.append(
                f"{h.val_accuracy[-1] * 100:8.2f}{h.loss[-1]:9.5f}"
                + (f"{phi:5.2f}" if np.isfinite(phi) else "    -")
                + flag
            )
        print("   " + f"{alpha:6.2f} | " + " | ".join(cells), flush=True)
    print("\n   <Phi> is the optimizer's mean prefactor; 'CF chained' keeps its gauge")
    print("   in the derivative, so the optimizer reports none -- see 0b instead.")


def tuned(setup: Setup, alphas, scales, table: Table) -> None:
    print("\n2. Scale-controlled: best of a tuned step multiplier\n")
    print(f"   {'method':>17} | {'val%':>7} {'train%':>7} {'MSE':>9} | setting")
    print("   " + "-" * 70)
    best: dict[str, tuple] = {}
    for alpha in alphas:
        for scale in scales:
            for label, net, optimizer in setup.runs(alpha, scale):
                h = setup.evaluate(net, optimizer)
                table.add(stage="tune", coefficient=alpha, scale=scale, method=label,
                          val_acc=h.val_accuracy[-1], train_acc=h.accuracy[-1],
                          train_mse=h.loss[-1], diverged=int(h.diverged_at is not None))
                score = (h.val_accuracy[-1], h.accuracy[-1], h.loss[-1], (alpha, scale))
                if label not in best or score[0] > best[label][0]:
                    best[label] = score

    edges = ({min(alphas), max(alphas)}, {min(scales), max(scales)})
    flagged = []
    for label in METHODS:
        val, tr, mse, (alpha, scale) = best[label]
        edge = alpha in edges[0] or scale in edges[1]
        table.add(stage="best", method=label, coefficient=alpha, scale=scale,
                  val_acc=val, train_acc=tr, train_mse=mse, grid_edge=int(edge))
        print(f"   {label:>17} | {val * 100:7.2f} {tr * 100:7.2f} {mse:9.5f} | "
              f"alpha={alpha}, scale={scale}{'  (grid edge!)' if edge else ''}", flush=True)
        if edge:
            flagged.append(label)
    if flagged:
        print(f"\n   WARNING: {', '.join(flagged)} peaked on the edge of the grid and may be\n"
              "   under-tuned.  Widen it before comparing those rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "mnist"), default="digits")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
    # 0.999 is included deliberately: it is the degenerate end where the CF
    # gauge collapses to the identity, and on MNIST both CF variants turn out
    # to peak there.  Leaving it out makes them look under-tuned.
    parser.add_argument("--tune-alphas", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8, 0.95, 0.999])
    parser.add_argument("--tune-scales", type=float, nargs="+",
                        default=[0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()

    if args.dataset == "mnist":
        print("loading MNIST ...", flush=True)
        x, t = mnist()
        train_set, val_set = (x[:55000], t[:55000]), (x[55000:60000], t[55000:60000])
    else:
        x, t = digits()
        (xtr, ttr), (xva, tva), _ = train_val_test_split(x, t, 0.15, 0.15, seed=0)
        train_set, val_set = (xtr, ttr), (xva, tva)

    layers = [train_set[0].shape[1], args.hidden, train_set[1].shape[1]]
    setup = Setup(
        layers=layers,
        seed=args.seed,
        fit=dict(epochs=args.epochs, batch_size=args.batch_size, shuffle=True, seed=0),
        train=train_set,
        val=val_set,
    )

    print(f"\n{args.dataset} {'-'.join(map(str, layers))} sigmoid / MSE, "
          f"{len(train_set[0])} train / {len(val_set[0])} val, "
          f"{args.epochs} epochs, batch size {args.batch_size}, "
          f"identical initial weights (seed {args.seed})")
    table = Table(f"chained_vs_terminal_{args.dataset}")
    verify_telescoping(setup, table)
    cf_residue(setup, [a for a in args.alphas if a < 1.0] + [0.999], table)
    effective_gain(setup, [a for a in args.alphas if a < 1.0], table)
    sweep(setup, args.alphas, table)
    if not args.skip_tuning:
        tuned(setup, args.tune_alphas, args.tune_scales, table)
    table.write()


if __name__ == "__main__":
    main()
