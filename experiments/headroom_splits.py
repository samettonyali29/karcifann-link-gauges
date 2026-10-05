#!/usr/bin/env python3
"""Is the configuration gain split-dependent, the way the gauge difference is?

`split_robustness.py` found that the power-versus-SGD difference has no stable
sign once the split is redrawn, and that the instability comes from a specific
place: validation sometimes selects a learning rate that works for the tuning
seed and collapses for the others, which costs the SGD baseline far more than
it costs the damped gauge.

Section 6.10's headroom result is a difference of the same shape -- best
configuration minus the paper's configuration A, at one fixed split -- and
configuration A *is* tuned SGD on sigmoid/MSE.  So the same failure would
inflate the gain on exactly the splits where A collapses.  Common split
difficulty cancels in a within-split difference, so the gain might well be
steadier than either term; this script measures which.

It reuses `headroom.CONFIGS` unchanged rather than copying them, so the two
studies cannot drift apart, and re-tunes every configuration inside every
split.

    python3 experiments/headroom_splits.py --dataset vehicle
    python3 experiments/headroom_splits.py --dataset letter --splits 10 --seeds 8
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

from karcifann import MLP, Table, accuracy, train, train_val_test_split  # noqa: E402
from karcifann.datasets import load as load_dataset  # noqa: E402


def _headroom():
    """Import headroom.py for its CONFIGS and budgets, without editing it.

    Editing that script would put a source change inside the span of the
    results it has already produced, which tests/test_audit.py rejects -- and
    rightly, since those results would no longer be from one generation.
    """
    spec = importlib.util.spec_from_file_location(
        "headroom_module", ROOT / "experiments" / "headroom.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADROOM = _headroom()
CONFIGS = HEADROOM.CONFIGS
DEFAULTS = HEADROOM.DEFAULTS

TUNE_SEED = 1
CONFIRM_SEEDS = list(range(100, 116))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    fit = dict(DEFAULTS[args.dataset], shuffle=True)
    seeds = CONFIRM_SEEDS[: args.seeds]
    x, t = load_dataset(args.dataset, scale=False)
    labels = list(CONFIGS)

    print(f"{args.dataset}: {args.splits} splits x {len(labels)} configurations, "
          f"{fit['epochs']} epochs", flush=True)

    table = Table(f"headroom_splits_{args.dataset}")
    gains, optimiser_only = [], []
    started = time.perf_counter()

    for split in range(args.splits):
        (xtr, ttr), (xva, tva), (xte, tte) = train_val_test_split(
            x, t, 0.15, 0.15, seed=split, standardize_features=True)

        def run(cfg, optimizer, seed):
            net = MLP([xtr.shape[1], *cfg["hidden"], ttr.shape[1]],
                      activation=cfg["activation"],
                      output_activation=cfg["output_activation"],
                      loss=cfg["loss"], weight_init=cfg["init"], seed=seed)
            h = train(net, optimizer(), xtr, ttr, validation_data=(xva, tva),
                      **{**fit, "seed": seed})
            return h.val_accuracy[-1], accuracy(net.predict(xte), tte)

        means = {}
        for label in labels:
            cfg = CONFIGS[label]
            template, build, values = cfg["grid"][0]
            best = None
            for v in values:
                val, _ = run(cfg, lambda v=v: build(v), TUNE_SEED)
                table.add(stage="tune", dataset=args.dataset, split=split,
                          configuration=label, setting=template % v, val=val)
                if best is None or val > best[0]:
                    best = (val, v)
            scores = []
            for seed in seeds:
                val, test = run(cfg, lambda v=best[1]: build(v), seed)
                table.add(stage="seed", dataset=args.dataset, split=split,
                          configuration=label, setting=template % best[1],
                          seed=seed, val=val, test=test)
                scores.append(100.0 * test)
            means[label] = float(np.mean(scores))
            table.add(stage="summary", dataset=args.dataset, split=split,
                      configuration=label, setting=template % best[1],
                      test_mean=means[label],
                      grid_edge=int(best[1] in (min(values), max(values))))

        base = means[labels[0]]
        gain = max(means.values()) - base
        plus = next((means[l] - base for l in labels if l.startswith("A+")), float("nan"))
        gains.append(gain)
        optimiser_only.append(plus)
        table.add(stage="split_summary", dataset=args.dataset, split=split,
                  paper_cell=base, best=max(means.values()), gain=gain,
                  optimiser_only=plus,
                  best_configuration=max(means, key=means.get))
        print(f"   split {split}: A={base:6.2f}  gain={gain:+6.2f}  "
              f"A+ - A={plus:+6.2f}  best={max(means, key=means.get)[:2]}"
              f"   [{time.perf_counter() - started:5.0f}s]", flush=True)

    g = np.array(gains)
    sd = float(g.std(ddof=1)) if g.size > 1 else float("nan")
    print(f"\n   across {len(g)} splits: gain mean {g.mean():+.2f}, "
          f"median {np.median(g):+.2f}, SD {sd:.2f}, "
          f"range [{g.min():+.2f}, {g.max():+.2f}]")
    table.add(stage="across_splits", dataset=args.dataset, n_splits=len(g),
              gain_mean=float(g.mean()), gain_median=float(np.median(g)),
              gain_sd=sd, gain_min=float(g.min()), gain_max=float(g.max()))
    table.write()


if __name__ == "__main__":
    main()
