#!/usr/bin/env python3
"""Test the loss-rescaling identity the manuscript derives.

Scaling the objective to ``c*J`` multiplies the gradient by ``c`` and the
KarciFANN prefactor by ``c**(alpha-1)``, so the whole update should scale by
``c**alpha``::

    dW(c*J)  =  c**alpha  .  dW(J)

Three sections are inspected, in increasing order of how much they can go
wrong:

    A. the identity on a single update, elementwise, away from the epsilon
       floor -- this should hold to floating-point precision;
    B. what happens when the floor *is* active, which is where the identity is
       expected to fail and it is worth knowing by how much;
    C. the practical consequence: rescaling the loss by ``c`` while dividing
       the step multiplier by ``c**alpha`` should leave the trajectory
       unchanged, which is the claim that lets a tuned multiplier absorb the
       aggregation convention.

    python3 experiments/loss_scaling.py [--dataset digits]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann import MLP, KarciFANN, one_hot, train  # noqa: E402
from karcifann.datasets import digits, load, train_val_test_split  # noqa: E402
from karcifann.losses import Loss, get_loss  # noqa: E402
from karcifann.records import Table  # noqa: E402

RESULTS = Path("results")

#: Objective multipliers.  Spread over four orders of magnitude so that a
#: discrepancy growing with ``c`` would be visible.
SCALES = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]

#: Orders either side of one, since the sign of ``alpha - 1`` flips which end
#: of the prefactor diverges.
ORDERS = [0.4, 0.8, 1.0, 1.3]


class ScaledLoss(Loss):
    """``c * base``.  Both the value and the gradient carry the factor."""

    def __init__(self, base: Loss, c: float) -> None:
        self.base = get_loss(base)
        self.c = float(c)
        self.name = f"{self.base.name}x{c:g}"
        self.fuses_with = self.base.fuses_with

    def value(self, y, t):
        return self.c * self.base.value(y, t)

    def gradient(self, y, t):
        return self.c * self.base.gradient(y, t)


def one_update(x, t, alpha, c, seed, eps=1e-12):
    """The first update a fresh network takes, for objective ``c*J``.

    Two versions of the same increment come back.  ``direct`` is
    ``factor * grad``, the quantity the optimiser forms before it touches the
    parameters.  ``reconstructed`` is ``before - after``, which is what an
    outside observer can recover and what this script used to measure.

    They are not equally precise.  A weight is of order ``0.1`` and an
    increment here can be of order ``1e-12``, so the subtraction cancels away
    about eleven significant digits and the residue it leaves is a property of
    float64, not of the identity being tested.  Reporting the reconstruction
    as "the relative error of the identity" overstated it by orders of
    magnitude at the small scales, which is exactly where the identity was
    supposed to be under the most strain.  Both are returned so the table can
    show the difference instead of asserting it.
    """
    net = MLP([x.shape[1], 20, t.shape[1]], activation="sigmoid",
              loss=ScaledLoss("mse", c), weight_init="glorot", seed=seed)
    params = net.parameters
    before = [p.copy() for p in params]
    cache = net.forward(x)
    loss_value = net.loss.value(cache.prediction, t)
    grads_w, grads_b = net.backward(cache, t)
    grads = list(grads_w) + (list(grads_b) if grads_b is not None else [])

    optimizer = KarciFANN(alpha=alpha, eps=eps)
    # The same arithmetic step() performs, read off before the subtraction.
    # scale is 1 and there is no decay or clipping here, so this is exact.
    direct = [f * g for f, g in zip(optimizer.factors(params, loss_value), grads)]
    optimizer.step(params, grads, loss_value)
    reconstructed = [b - a for b, a in zip(before, params)]
    return direct, reconstructed, loss_value


#: Coordinates whose predicted increment is this far below the largest one in
#: the same update carry no significant digits of their own, and a ratio taken
#: there measures rounding rather than the identity.  They are counted and
#: excluded rather than silently dominating a maximum.
RELATIVE_FLOOR = 1e-12


def compare(scaled, base, predicted):
    """Coordinatewise and normwise agreement of ``scaled`` with ``predicted * base``.

    The coordinatewise maximum is the sharper statistic and the one the
    manuscript tabulates; the normwise figure is reported beside it because a
    single badly conditioned coordinate cannot dominate it.
    """
    s = np.concatenate([a.ravel() for a in scaled])
    b = np.concatenate([a.ravel() for a in base])
    target = predicted * b

    keep = np.abs(target) > RELATIVE_FLOOR * np.max(np.abs(target))
    coordinatewise = (float(np.max(np.abs(s[keep] / target[keep] - 1.0)))
                      if keep.any() else float("nan"))
    denominator = float(np.linalg.norm(target))
    normwise = float(np.linalg.norm(s - target) / denominator) if denominator else float("nan")
    nonzero = np.abs(b) > 0
    median_ratio = float(np.median(s[nonzero] / b[nonzero])) if nonzero.any() else float("nan")
    return {
        "observed": median_ratio,
        "max_relative_error": coordinatewise,
        "normwise_relative_error": normwise,
        "n_coordinates": int(b.size),
        "n_below_floor": int((~keep).sum()),
    }


def section_a(x, t, table: Table) -> None:
    """The identity away from the floor."""
    print("\nA. dW(cJ) / dW(J) against c^alpha  (eps = 1e-12)\n")
    print(f"   {'alpha':>6} {'c':>8} {'predicted':>12} {'observed':>12} "
          f"{'max rel err':>12} {'normwise':>12} {'by subtraction':>14}")
    for alpha in ORDERS:
        base, base_sub, _ = one_update(x, t, alpha, 1.0, seed=7)
        for c in SCALES:
            scaled, scaled_sub, _ = one_update(x, t, alpha, c, seed=7)
            predicted = c ** alpha
            direct = compare(scaled, base, predicted)
            subtracted = compare(scaled_sub, base_sub, predicted)
            print(f"   {alpha:6.2f} {c:8g} {predicted:12.6g} "
                  f"{direct['observed']:12.6g} {direct['max_relative_error']:12.3e} "
                  f"{direct['normwise_relative_error']:12.3e} "
                  f"{subtracted['max_relative_error']:14.3e}")
            table.add(section="A", alpha=alpha, scale=c, predicted=predicted,
                      eps=1e-12,
                      max_relative_error_by_subtraction=subtracted["max_relative_error"],
                      **direct)


def section_b(x, t, table: Table) -> None:
    """The identity where the epsilon floor bites.

    The floor clamps |J| before the division, so once ``c*J`` falls below it
    the prefactor stops responding to ``c`` and the identity must fail.  A
    deliberately large floor makes that regime reachable.
    """
    print("\nB. the same identity with a floor large enough to bite "
          "(eps = 1e-2)\n")
    print(f"   {'alpha':>6} {'c':>8} {'predicted':>12} {'observed':>12} "
          f"{'max rel err':>12}")
    for alpha in (0.4, 0.8):
        base, _, loss_one = one_update(x, t, alpha, 1.0, seed=7, eps=1e-2)
        for c in SCALES:
            scaled, _, _ = one_update(x, t, alpha, c, seed=7, eps=1e-2)
            predicted = c ** alpha
            direct = compare(scaled, base, predicted)
            floored = "yes" if c * loss_one / c < 1e-2 or c * loss_one < 1e-2 else "no"
            print(f"   {alpha:6.2f} {c:8g} {predicted:12.6g} "
                  f"{direct['observed']:12.6g} {direct['max_relative_error']:12.3e}"
                  f"   loss below floor: {floored}")
            table.add(section="B", alpha=alpha, scale=c, predicted=predicted,
                      eps=1e-2, **direct)


#: Step multiplier for the equivalence runs.  Chosen so the baseline actually
#: trains: an equivalence between two runs that both fail to learn would say
#: nothing.
BASE_SCALE = 8.0

#: Epoch counts at which the two trajectories are compared.
CHECKPOINTS = [1, 2, 5, 10, 25]


def _trained_weights(xtr, ttr, alpha, loss, scale, epochs):
    net = MLP([xtr.shape[1], 20, ttr.shape[1]], activation="sigmoid",
              loss=loss, weight_init="glorot", seed=11)
    train(net, KarciFANN(alpha=alpha, scale=scale), xtr, ttr,
          epochs=epochs, batch_size=32, shuffle=True, seed=11)
    return net, np.concatenate([p.ravel() for p in net.parameters])


def section_c(data, table: Table) -> None:
    """Does a compensating step multiplier restore the trajectory?

    Section A shows the per-step identity is exact.  Compensating with
    ``s * c**-alpha`` should therefore reproduce the base trajectory exactly --
    and it does, to floating point, for as long as floating point holds.  The
    interesting quantity is how long that is.

    ``multiplier_roundtrip_error`` was logged to test the obvious explanation,
    that the compensating multiplier is itself inexact.  It refutes it: the
    error is exactly zero in 20 of these 30 runs, including the two that
    separate the most.  Every other operation on the trajectory rounds too, and
    this design does not tell them apart -- so the column is a ruled-out
    candidate, not the cause.
    """
    (xtr, ttr), (xva, tva) = data
    print(f"\nC. loss x c with the multiplier divided by c^alpha "
          f"(base multiplier {BASE_SCALE:g})\n")
    print("   weight divergence max|w_base - w_scaled| against epochs trained\n")
    header = "   " + f"{'alpha':>6} {'c':>7} " + "".join(f"{e:>12}" for e in CHECKPOINTS)
    print(header)
    for alpha in (0.4, 0.8, 1.3):
        for c in (0.1, 10.0):
            gaps = []
            for epochs in CHECKPOINTS:
                _, w0 = _trained_weights(xtr, ttr, alpha, "mse", BASE_SCALE, epochs)
                net1, w1 = _trained_weights(xtr, ttr, alpha, ScaledLoss("mse", c),
                                            BASE_SCALE * c ** -alpha, epochs)
                gaps.append(float(np.max(np.abs(w0 - w1))))
                table.add(section="C", alpha=alpha, scale=c, epochs=epochs,
                          weight_divergence=gaps[-1],
                          multiplier_roundtrip_error=abs(c ** alpha * c ** -alpha - 1.0))
            print("   " + f"{alpha:6.2f} {c:7g} "
                  + "".join(f"{g:12.2e}" for g in gaps))
    print("\n   The compensation is exact at the first step.  What grows is not\n"
          "   the multiplier's own rounding error -- that is exactly zero in\n"
          "   most of these runs -- and this design does not identify what is.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="digits")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.dataset == "digits":
        x, t = digits()
    else:
        x, t = load(args.dataset)
    (xtr, ttr), (xva, tva), _ = train_val_test_split(x, t, 0.15, 0.15, seed=0)
    print(f"{args.dataset}: {xtr.shape[1]}-20-{ttr.shape[1]}, "
          f"{len(xtr)} train / {len(xva)} val")

    table = Table("loss_scaling", RESULTS)
    probe_x, probe_t = xtr[:64], ttr[:64]
    section_a(probe_x, probe_t, table)
    section_b(probe_x, probe_t, table)
    section_c(((xtr, ttr), (xva, tva)), table)
    table.write()


if __name__ == "__main__":
    main()
