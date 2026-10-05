#!/usr/bin/env python3
"""Regenerate every numeric table in README.md from results/.

Three times in this project a table was typed into prose from a run, the
experiment was later re-run, and the prose silently stopped matching the data.
Hand-copying is the defect; this script removes the opportunity.

Each generated table sits between marker comments::

    <!-- table:name -->
    ... generated ...
    <!-- /table:name -->

Anything outside the markers is left untouched, so the surrounding argument
stays hand-written while the numbers come from the CSVs.

    python3 experiments/update_readme.py [--check]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("results")
README = Path("README.md")

PRETTY = {"vehicle": "Vehicle", "satimage": "Satimage", "dry_bean": "Dry Bean",
          "segment": "Segment", "optdigits": "Optdigits", "pendigits": "Pendigits",
          "mnist": "MNIST", "letter": "Letter"}
ORDER = ["vehicle", "satimage", "dry_bean", "segment", "optdigits", "pendigits",
         "mnist", "letter"]
METHODS = ["gradient descent", "KarciFANN", "exp", "tanh", "log"]


def read(name: str) -> list[dict]:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _final(dataset: str, method: str, metric: str = "test_acc"):
    for row in read("summary_final"):
        if (row["dataset"] == dataset and row["method"] == method
                and row["metric"] == metric):
            return float(row["mean"]) * 100, float(row["ci95"]) * 100
    return None, None


def table_accuracy() -> str:
    """Test accuracy for every method on every dataset."""
    head = "| dataset | " + " | ".join(f"`{m}`" if m != "gradient descent" else m
                                       for m in METHODS) + " |"
    rule = "|---|" + "---:|" * len(METHODS)
    lines = [head, rule]
    for dataset in ORDER:
        cells = []
        best = max((_final(dataset, m)[0] or -1) for m in METHODS)
        for method in METHODS:
            mean, ci = _final(dataset, method)
            if mean is None:
                cells.append("—")
            else:
                text = f"{mean:.2f} ± {ci:.2f}"
                cells.append(f"**{text}**" if abs(mean - best) < 1e-9 else text)
        lines.append(f"| {PRETTY[dataset]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_ranks() -> str:
    """Friedman average ranks with the Nemenyi verdict."""
    rows = sorted(read("stats_nemenyi"), key=lambda r: float(r["average_rank"]))
    if not rows:
        return "_(no results)_"
    friedman = read("stats_friedman")[0]
    lines = [
        f"Friedman over {friedman['blocks_datasets']} datasets: "
        f"χ² = {float(friedman['statistic']):.3g}, **p = {float(friedman['pvalue']):.2g}** "
        f"(power floor {float(friedman['power_floor_best_case_p']):.1g}); "
        f"Nemenyi critical difference {float(friedman['nemenyi_critical_difference']):.2f}.",
        "",
        "| method | average rank | gap to best | separated from best? |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        name = row["method"] if row["method"] == "gradient descent" else f"`{row['method']}`"
        rank, gap = float(row["average_rank"]), float(row["gap_to_best"])
        mark = "**yes**" if row["separated_from_best"] == "1" else "no"
        first = row is rows[0]
        lines.append(
            f"| {'**' + name + '**' if first else name} | "
            f"{'**' + format(rank, '.2f') + '**' if first else format(rank, '.2f')} | "
            f"{'—' if first else format(gap, '.2f')} | {'—' if first else mark} |"
        )
    return "\n".join(lines)


def table_headroom() -> str:
    """The cost of the evaluation cell, per dataset.

    Datasets are rows: the suite is eight wide now, and the previous layout put
    them in columns over a hardcoded three.  The gap is taken over whichever
    configuration actually wins, not over the last one listed.
    """
    labels, data = [], {}
    for dataset in ORDER:
        rows = [r for r in read(f"headroom_{dataset}") if r["stage"] == "summary"]
        if not rows:
            continue
        rows.sort(key=lambda r: r["configuration"])
        labels = [r["configuration"].strip() for r in rows]
        data[dataset] = [(float(r["test_mean"]) * 100, float(r["test_ci95"]) * 100)
                         for r in rows]
    if not labels or not data:
        return "_(no results)_"

    # The A+ arm differs from A only in the optimiser, so the difference between
    # them isolates that one factor out of the five that separate A from C.
    adam_only = next((i for i, lab in enumerate(labels) if lab.startswith("A+")), None)
    extra = " optimiser alone (A+ − A) |" if adam_only is not None else ""

    lines = ["| dataset | " + " | ".join(labels) + " | gap, best − paper cell |"
             + extra,
             "|---|" + "---:|" * (len(labels) + 1 + (1 if extra else 0))]
    for dataset, cells in data.items():
        best = max(range(len(cells)), key=lambda i: cells[i][0])
        rendered = [
            (f"**{m:.2f} ± {c:.2f}**" if i == best else f"{m:.2f} ± {c:.2f}")
            for i, (m, c) in enumerate(cells)
        ]
        gap = cells[best][0] - cells[0][0]
        row = (f"| {PRETTY[dataset]} | " + " | ".join(rendered)
               + f" | **{gap:+.2f}** |")
        if adam_only is not None:
            row += f" {cells[adam_only][0] - cells[0][0]:+.2f} |"
        lines.append(row)
    return "\n".join(lines)


def table_headroom_optimiser() -> str:
    """The A+ - A difference, paired over seeds, judged by the study's own rule.

    The gap column of table_headroom is a difference of two means, which says
    nothing about how consistent it is.  A+ and A share their seeds, so the
    difference is paired and an interval over the per-seed differences is
    available.

    The interval is descriptive and pointwise.  Detection uses the standard
    declared everywhere else in this repository -- the exact sign-flip
    permutation test, Holm-corrected -- over these eight comparisons as their
    own family.  An earlier version marked a row detected when its 95 %
    interval excluded zero, which is a different and more permissive rule and
    reported five detections where this one reports four.
    """
    import math

    import numpy as np

    from karcifann.analysis import holm, paired_permutation_test

    measured = []
    for dataset in ORDER:
        paired = {}
        for row in read(f"headroom_{dataset}"):
            if row["stage"] != "seed":
                continue
            arm = row["configuration"].strip()[:2].strip()
            if arm in ("A", "A+"):
                paired.setdefault(row["seed"], {})[arm] = float(row["test_acc"])
        seeds = [k for k in sorted(paired) if {"A", "A+"} <= set(paired[k])]
        if len(seeds) < 2:
            continue
        plus = np.array([paired[k]["A+"] for k in seeds])
        base = np.array([paired[k]["A"] for k in seeds])
        diffs = 100.0 * (plus - base)
        measured.append((dataset, diffs,
                         float(np.ravel(paired_permutation_test(plus, base))[0])))
    if not measured:
        return "_(no results)_"

    adjusted = np.ravel(holm([p for _, _, p in measured]))
    lines = ["| dataset | A+ − A (pp) | 95% CI | seeds favouring A+ | "
             "sign-flip Holm (8) | detected |",
             "|---|---:|---:|---:|---:|---|"]
    for (dataset, diffs, _), p_adj in zip(measured, adjusted):
        n = diffs.size
        mean = float(diffs.mean())
        half = 1.96 * float(diffs.std(ddof=1)) / math.sqrt(n)
        detected = p_adj <= 0.05
        lines.append(f"| {PRETTY[dataset]} | {mean:+.2f} | "
                     f"[{mean - half:+.2f}, {mean + half:+.2f}] | "
                     f"{int((diffs > 0).sum())}/{n} | {float(p_adj):.4f} | "
                     f"{'**yes**' if detected else 'no'} |")
    return "\n".join(lines)


def table_cf_residue() -> str:
    """How far Caputo-Fabrizio is from telescoping, in the trained network.

    The counterpart of table_telescoping.  That one shows a gauge of the ratio
    form agreeing with its collapsed factor to round-off; this one shows what
    a denominator-only gauge does instead, measured the same way on the same
    network and batch.
    """
    lines = ["| dataset | order | median relative residue | largest relative residue |",
             "|---|---:|---:|---:|"]
    for dataset in ("digits", "mnist"):
        for row in read(f"chained_vs_terminal_{dataset}"):
            if row["stage"] != "cf_residue":
                continue
            lines.append(f"| {dataset} | {float(row['coefficient']):g} | "
                         f"{float(row['median_relative_error']):.3g} | "
                         f"{float(row['max_relative_error']):.2e} |")
    return "\n".join(lines) if len(lines) > 2 else "_(no results)_"


def table_split_robustness() -> str:
    """Split uncertainty beside seed uncertainty, for the headline pair.

    Every other table in this study fixes one stratified split at seed 0 and
    reports variation over seeds.  This one re-tunes and re-runs inside ten
    splits, so the two sources can be compared directly.  The median column is
    there because one outlying split per dataset carries the mean.
    """
    import statistics

    rows = ["| dataset | splits | mean | median | between-split SD | "
            "mean seed SE | SD/SE | splits favouring power |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    any_row = False
    for dataset in ORDER:
        summaries = [r for r in read(f"split_robustness_{dataset}")
                     if r["stage"] == "split_summary"]
        if len(summaries) < 2:
            continue
        any_row = True
        diffs = [float(r["mean_difference"]) for r in summaries]
        ses = [float(r["seed_se"]) for r in summaries]
        sd = statistics.stdev(diffs)
        se = statistics.fmean(ses)
        rows.append(
            f"| {PRETTY[dataset]} | {len(diffs)} | {statistics.fmean(diffs):+.2f} | "
            f"{statistics.median(diffs):+.2f} | {sd:.2f} | {se:.2f} | "
            f"{sd / se:.1f}× | {sum(d > 0 for d in diffs)}/{len(diffs)} |")
    return "\n".join(rows) if any_row else "_(no results)_"


def table_multiseed_tuning() -> str:
    """Does tuning on three seeds instead of one remove the outlying splits?

    Section 6.3 says, descriptively, that the positive per-split means come
    from validation picking an SGD rate that works for the tuning seed and
    collapses for the others.  This is the paired test of that: same split,
    same grids, same eight confirmation seeds, one thing changed.
    """
    rows = ["| dataset | split | selected on 1 seed | on 3 seeds | "
            "difference, 1 seed | 3 seeds | change | Δ power | Δ SGD | "
            "seed SE, 1 seed | 3 seeds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    any_row = False
    for dataset in ORDER:
        seeds = [r for r in read(f"multiseed_tuning_{dataset}") if r["stage"] == "seed"]

        def shift(split, method):
            """Change in that method's own mean confirmation accuracy."""
            def mean(rule):
                v = [float(r["test"]) for r in seeds
                     if r["rule"] == rule and r["method"] == method
                     and float(r["split"]) == split]
                return 100.0 * statistics.fmean(v)
            return mean("multi") - mean("single")

        for r in sorted((r for r in read(f"multiseed_tuning_{dataset}")
                         if r["stage"] == "split_summary"),
                        key=lambda r: (r["kind"] != "outlier", float(r["split"]))):
            any_row = True
            d_power = shift(float(r["split"]), "power")
            d_sgd = shift(float(r["split"]), "gradient descent")
            single, multi = float(r["single_mean"]), float(r["multi_mean"])
            rows.append(
                f"| {PRETTY[dataset]} | {int(float(r['split']))} "
                f"({r['kind']}) | `lr={float(r['single_sgd_lr']):g}` | "
                f"`lr={float(r['multi_sgd_lr']):g}` | {single:+.2f} | "
                f"{multi:+.2f} | {multi - single:+.2f} | "
                f"{d_power:+.2f} | {d_sgd:+.2f} | "
                f"{float(r['single_se']):.2f} | {float(r['multi_se']):.2f} |")
    if not any_row:
        return "_(no results)_"
    rows.append("")
    rows.append("The `single` column reproduces `split_robustness.py`'s stored "
                "result for the same split; the script refuses to write a table "
                "if it does not. `Δ power` and `Δ SGD` decompose the change: on "
                "the outliers almost all of it is SGD recovering, and on the "
                "controls SGD does not move at all, because its selected rate "
                "is unchanged there.")
    return "\n".join(rows)


def table_split_robustness_detail() -> str:
    """Per split: the tuned winners and the difference they produced."""
    rows = ["| dataset | split | power − SGD (pp) | seed SE | α | multiplier | "
            "SGD rate | seeds W/T/L |", "|---|---:|---:|---:|---:|---:|---:|:-:|"]
    any_row = False
    for dataset in ORDER:
        for r in read(f"split_robustness_{dataset}"):
            if r["stage"] != "split_summary":
                continue
            any_row = True
            n = int(r["n_seeds"])
            won = int(r["n_favouring_power"])
            tied = int(r["n_tied"])
            rows.append(
                f"| {PRETTY[dataset]} | {r['split']} | "
                f"{float(r['mean_difference']):+.2f} | {float(r['seed_se']):.2f} | "
                f"{float(r['alpha']):.1f} | {float(r['power_scale']):g} | "
                f"{float(r['sgd_lr']):g} | {won}/{tied}/{n - won - tied} |")
    return "\n".join(rows) if any_row else "_(no results)_"


def table_headroom_selection() -> str:
    """The configuration gain under two selection rules.

    headroom_splits.py picks each configuration's *rate* by validation, then
    picks the winning *configuration* by the largest test mean.  That is a
    retrospective maximum: it cannot be negative, because configuration A is
    among the candidates, and it is not what a practitioner choosing without
    the test set would obtain.  This recomputes the same stored runs with the
    configuration also chosen by validation, script order breaking ties.
    """
    import collections
    import statistics as st

    lines = ["| dataset | gain, best test mean | gain, validation-selected | "
             "median, validation-selected |", "|---|---:|---:|---:|"]
    any_row = False
    for dataset in ORDER:
        rows = read(f"headroom_splits_{dataset}")
        if not rows:
            continue
        order, best_val, test_mean = [], collections.defaultdict(dict), \
            collections.defaultdict(dict)
        for row in rows:
            if row["stage"] == "tune":
                label = row["configuration"]
                if label not in order:
                    order.append(label)
                split = int(row["split"])
                value = float(row["val"])
                if label not in best_val[split] or value > best_val[split][label]:
                    best_val[split][label] = value
            elif row["stage"] == "summary":
                test_mean[int(row["split"])][row["configuration"]] = \
                    float(row["test_mean"])
        if len(test_mean) < 2:
            continue
        any_row = True
        reported, selected = [], []
        for split in sorted(test_mean):
            base = test_mean[split][order[0]]
            reported.append(max(test_mean[split].values()) - base)
            chosen = max(order, key=lambda c: (best_val[split][c], -order.index(c)))
            selected.append(test_mean[split][chosen] - base)
        lines.append(f"| {PRETTY[dataset]} | {st.fmean(reported):+.2f} | "
                     f"{st.fmean(selected):+.2f} | {st.median(selected):+.2f} |")
    return "\n".join(lines) if any_row else "_(no results)_"


def table_split_stability() -> str:
    """Each method's own variability across splits -- gauge, SGD and Adam.

    table_split_robustness reports the *difference* between power and SGD and
    finds it unstable.  Reporting methods separately shows the gauge is far
    steadier than tuned constant-rate SGD -- and that Adam on the same cell is
    just as steady or steadier, so the steadiness is not distinctive to the
    gauge.  The A+ column is what keeps the first observation from being
    over-read; it comes from headroom_splits, over the same ten splits.
    """
    import collections
    import statistics as st

    lines = ["| dataset | power | tuned SGD | Adam, same cell (A+) | "
             "power: seed SD | SGD: seed SD |", "|---|---:|---:|---:|---:|---:|"]
    any_row = False
    for dataset in ORDER:
        rows = [r for r in read(f"split_robustness_{dataset}")
                if r["stage"] == "seed"]
        if not rows:
            continue
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in rows:
            by[row["method"]][int(row["split"])].append(100 * float(row["test"]))
        if len(by.get("power", {})) < 2:
            continue
        cell = {}
        for method in ("power", "gradient descent"):
            means = [st.fmean(v) for _, v in sorted(by[method].items())]
            cell[method] = (st.stdev(means),
                            st.fmean([st.stdev(v) for v in by[method].values()]))
        adam = [float(r["test_mean"])
                for r in read(f"headroom_splits_{dataset}")
                if r["stage"] == "summary"
                and r["configuration"].strip().startswith("A+")]
        if len(adam) < 2:
            continue
        any_row = True
        lines.append(
            f"| {PRETTY[dataset]} | {cell['power'][0]:.2f} | "
            f"{cell['gradient descent'][0]:.2f} | {st.stdev(adam):.2f} | "
            f"{cell['power'][1]:.2f} | {cell['gradient descent'][1]:.2f} |")
    return "\n".join(lines) if any_row else "_(no results)_"


def table_headroom_splits() -> str:
    """Is the configuration gain stable when the split is redrawn?

    The counterpart of table_split_robustness for the headroom comparison.
    Both are within-split differences, so common split difficulty cancels in
    both; the question is whether the same collapsing-baseline artifact
    survives that cancellation, and on three of four datasets it does.
    """
    import statistics as st

    lines = ["| dataset | splits | gain mean | gain median | gain SD | "
             "gain range | A's own range | winning configuration |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    any_row = False
    for dataset in ORDER:
        rows = [r for r in read(f"headroom_splits_{dataset}")
                if r["stage"] == "split_summary"]
        if len(rows) < 2:
            continue
        any_row = True
        gains = [float(r["gain"]) for r in rows]
        cells = [float(r["paper_cell"]) for r in rows]
        wins = {}
        for r in rows:
            key = r["best_configuration"].strip()[:2].strip()
            wins[key] = wins.get(key, 0) + 1
        won = ", ".join(f"{k} {v}" for k, v in
                        sorted(wins.items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {PRETTY[dataset]} | {len(gains)} | {st.fmean(gains):+.2f} | "
            f"{st.median(gains):+.2f} | {st.stdev(gains):.2f} | "
            f"[{min(gains):+.2f}, {max(gains):+.2f}] | "
            f"{max(cells) - min(cells):.2f} | {won} |")
    return "\n".join(lines) if any_row else "_(no results)_"


def table_tolerance() -> str:
    """Hyperparameter basin widths, in decades."""
    rows = read("stats_tolerance")
    if not rows:
        return "_(no results)_"
    by = {}
    for row in rows:
        by.setdefault(row["dataset"], {})[row["sweep"]] = float(row["decades"])
    sweeps = sorted({s for v in by.values() for s in v})
    lines = ["| dataset | " + " | ".join(s.split(": ")[0] + ": `" + s.split(": ")[1] + "`"
                                         for s in sweeps) + " |",
             "|---|" + "---:|" * len(sweeps)]
    for dataset in ORDER:
        if dataset not in by:
            continue
        lines.append(f"| {PRETTY[dataset]} | "
                     + " | ".join(f"{by[dataset].get(s, float('nan')):.2f} decades"
                                  for s in sweeps) + " |")
    return "\n".join(lines)


def table_gauge_ablation() -> str:
    """Which half of the prefactor does the work."""
    rows = read("summary_gauge_ablation")
    if not rows:
        return "_(no results)_"
    by = {}
    for row in rows:
        by.setdefault(row["dataset"], {})[row["mode"]] = (
            float(row["mean_test_acc"]) * 100, float(row["ci95"]) * 100)
    modes = ["full", "error", "weight"]
    lines = ["| dataset | " + " | ".join(f"`{m}`" for m in modes) + " |",
             "|---|" + "---:|" * len(modes)]
    for dataset in ORDER:
        if dataset not in by:
            continue
        best = max(by[dataset][m][0] for m in modes if m in by[dataset])
        cells = []
        for mode in modes:
            mean, ci = by[dataset][mode]
            text = f"{mean:.2f} ± {ci:.2f}"
            cells.append(f"**{text}**" if abs(mean - best) < 1e-9 else text)
        lines.append(f"| {PRETTY[dataset]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_headline() -> str:
    """KarciFANN against gradient descent, dataset by dataset.

    The sign-test column is Holm-corrected over the same 80 primary
    comparisons as the permutation column beside it.  It printed raw p-values
    under a caption promising corrected ones until 2026-09-23, which made the
    table disagree with the paragraph below it -- and is the kind of semantic
    error that checking numerical containment alone cannot detect.
    """
    import numpy as np

    from karcifann.analysis import holm

    rows = read("stats_wilcoxon")
    if not rows:
        return "_(no results)_"
    sign_holm = dict(zip(
        ((r["dataset"], r["method_a"], r["method_b"]) for r in rows),
        np.ravel(holm([float(r["p_sign"]) for r in rows]))))
    lines = ["| dataset | KarcıFANN − GD (pp) | 95% CI | seeds W/T/L | "
             "perm. Holm (within) | perm. Holm (study-wide) | sign Holm (80) | verdict |",
             "|---|---:|---:|:-:|---:|---:|---:|---|"]
    wins = not_detected = losses = 0
    for dataset in ORDER:
        match = [r for r in rows if r["dataset"] == dataset
                 and {r["method_a"], r["method_b"]} == {"KarciFANN", "gradient descent"}]
        if not match:
            continue
        row = match[0]
        diff = float(row["difference"]) * 100
        seeds = int(row["n_seeds"])
        # Wins, ties and losses are read off separately.  Reversing the pair by
        # subtracting the win count from the seed count counts ties as wins.
        won = int(row["n_favouring_a"])
        lost = int(row["n_favouring_b"])
        tied = int(row["n_tied"])
        if row["method_a"] == "gradient descent":
            diff = -diff
            won, lost = lost, won
        half = float(row["diff_ci95"]) * 100
        within = float(row["p_permutation_holm"])
        study = float(row["p_permutation_study_wide"])
        sign = float(sign_holm[(row["dataset"], row["method_a"], row["method_b"])])
        # The declared standard: permutation test, corrected across the study.
        significant = int(row["robust"]) == 1
        if significant and diff > 0:
            verdict, wins = "**KarcıFANN**", wins + 1
        elif significant and diff < 0:
            verdict, losses = "**gradient descent**", losses + 1
        else:
            verdict, not_detected = "not detected", not_detected + 1
        lines.append(
            f"| {PRETTY[dataset]} | {diff:+.2f} | "
            f"[{diff - half:+.2f}, {diff + half:+.2f}] | {won}/{tied}/{lost} | "
            f"{within:.4f} | {study:.4f} | {sign:.4f} | {verdict} |")
    lines.append("")
    lines.append(f"**{wins} win{'s' if wins != 1 else ''} for KarcıFANN, "
                 f"{losses} for gradient descent, {not_detected} non-detections** "
                 "under the study-wide permutation standard. "
                 "Non-detection does not establish equivalence.")
    return "\n".join(lines)


def table_evidence() -> str:
    """How many comparisons clear each standard, with a sample-skew diagnostic.

    These counts used to be typed into the prose; one of them ("27 of 80")
    survived a re-run that changed it to 38.  Generating them removes the only
    way that can happen.
    """
    rows = read("stats_wilcoxon")
    if not rows:
        return "_(no results)_"
    total = len(rows)
    count = lambda key: sum(int(r[key]) for r in rows)
    skewed = sum(1 for r in rows if abs(float(r["difference_skew"])) > 1)
    lines = [
        f"| standard | significant of {total} |",
        "|---|---:|",
        f"| Wilcoxon, Holm within dataset | {count('significant_holm_0.05')} |",
        f"| Wilcoxon, Holm across the study | {count('significant_study_wide')} |",
        f"| permutation, Holm within dataset | {count('significant_permutation')} |",
        f"| **permutation, Holm across the study** (the standard used here) "
        f"| **{count('robust')}** |",
        "",
        f"Paired-difference samples with `|skew| > 1`: **{skewed} of {total}**. "
        "This is a diagnostic warning about symmetry, not a formal assumption test.",
    ]
    return "\n".join(lines)


def _pivot(rows, stage, row_key, col_key, value_key, order=None, fmt="{:.2f}",
           scale=1.0, highlight=None):
    """A stage of a results file as a markdown table, rows x methods."""
    subset = [r for r in rows if r["stage"] == stage]
    if not subset:
        return "_(no results)_"
    columns = order or sorted({r[col_key] for r in subset})
    keys = sorted({float(r[row_key]) for r in subset})
    lines = ["| " + row_key + " | " + " | ".join(columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for key in keys:
        cells = []
        values = {}
        for column in columns:
            match = [r for r in subset
                     if float(r[row_key]) == key and r[col_key] == column and r.get(value_key)]
            values[column] = float(match[0][value_key]) * scale if match else None
        best = max((v for v in values.values() if v is not None), default=None)
        for column in columns:
            value = values[column]
            if value is None:
                cells.append("—")
            elif highlight and best is not None and abs(value - best) < 1e-9:
                cells.append("**" + fmt.format(value) + "**")
            else:
                cells.append(fmt.format(value))
        lines.append(f"| {key:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_fod_sweep() -> str:
    """Karci vs Caputo-Fabrizio vs Caputo at unit multiplier, on digits."""
    return _pivot(read("fod_comparison_digits"), "sweep", "coefficient", "method",
                  "val_acc", order=["KarciFANN", "Caputo-Fabrizio", "Caputo"],
                  scale=100.0, highlight=True)


def table_fod_tuned() -> str:
    """Best of a tuned step multiplier, on digits."""
    rows = [r for r in read("fod_comparison_digits") if r["stage"] == "best"]
    if not rows:
        return "_(no results)_"
    lines = ["| method | val% | train% | training MSE | setting |", "|---|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: -float(r["val_acc"])):
        lines.append(f"| {row['method']} | {float(row['val_acc']) * 100:.2f} | "
                     f"{float(row['train_acc']) * 100:.2f} | {float(row['train_mse']):.5f} | "
                     f"`{row['setting']}` |")
    return "\n".join(lines)


def table_mnist_methods() -> str:
    """The four rules on MNIST at unit multiplier (test accuracy)."""
    return _pivot(read("mnist_methods_10ep"), "sweep", "coefficient", "method", "test",
                  order=["gradient descent", "KarciFANN", "Caputo-Fabrizio", "Caputo"],
                  scale=100.0, highlight=True)


def table_mnist_methods_tuned() -> str:
    rows = [r for r in read("mnist_methods_10ep") if r["stage"] == "best"]
    if not rows:
        return "_(no results)_"
    lines = ["| method | val% | train% | test% | MSE | setting |",
             "|---|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: -float(r["val"])):
        edge = "  (grid edge)" if row.get("grid_edge") == "1" else ""
        lines.append(f"| {row['method']} | {float(row['val']) * 100:.2f} | "
                     f"{float(row['train']) * 100:.2f} | {float(row['test']) * 100:.2f} | "
                     f"{float(row['mse']):.5f} | `{row['setting']}`{edge} |")
    return "\n".join(lines)


def _chained(dataset: str, stage: str, value: str, scale: float, fmt: str) -> str:
    return _pivot(read(f"chained_vs_terminal_{dataset}"), stage, "coefficient", "method",
                  value, order=["gradient descent", "KarciFANN", "CF terminal", "CF chained"],
                  scale=scale, fmt=fmt, highlight=(stage == "sweep"))


def table_chained_digits_gain() -> str:
    return _pivot(read("chained_vs_terminal_digits"), "step_gain", "coefficient", "method",
                  "relative_step", order=["KarciFANN", "CF terminal", "CF chained"],
                  fmt="{:.3f}")


def table_chained_digits() -> str:
    return _chained("digits", "sweep", "val_acc", 100.0, "{:.2f}")


def table_chained_mnist_gain() -> str:
    return _pivot(read("chained_vs_terminal_mnist"), "step_gain", "coefficient", "method",
                  "relative_step", order=["KarciFANN", "CF terminal", "CF chained"],
                  fmt="{:.3f}")


def table_chained_mnist() -> str:
    return _chained("mnist", "sweep", "val_acc", 100.0, "{:.2f}")


def _chained_tuned(dataset: str) -> str:
    rows = [r for r in read(f"chained_vs_terminal_{dataset}") if r["stage"] == "best"]
    if not rows:
        return "_(no results)_"
    lines = ["| method | val% | train% | training MSE | setting |", "|---|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: -float(r["val_acc"])):
        edge = "  (grid edge)" if row.get("grid_edge") == "1" else ""
        lines.append(f"| {row['method']} | {float(row['val_acc']) * 100:.2f} | "
                     f"{float(row['train_acc']) * 100:.2f} | {float(row['train_mse']):.5f} | "
                     f"`α = {float(row['coefficient']):g}, scale = {float(row['scale']):g}`"
                     f"{edge} |")
    return "\n".join(lines)


def table_chained_digits_tuned() -> str:
    return _chained_tuned("digits")


def table_chained_mnist_tuned() -> str:
    return _chained_tuned("mnist")


def table_stability() -> str:
    """What each rule does when the step multiplier is pushed to 64 on MNIST.

    The point of the bounded Caputo-Fabrizio kernel: it cannot blow up, so it
    survives a multiplier that collapses the others.
    """
    rows = [r for r in read("mnist_methods_10ep")
            if r["stage"] == "tune" and re.search(r"(?:lr|scale)=64\b", r.get("setting", ""))]
    if not rows:
        return "_(no results)_"
    # The whole order grid, not the best of it: a rule that holds up at one
    # order and collapses at the next is not robust, and taking the maximum
    # would hide exactly that.
    methods = ["KarciFANN", "Caputo-Fabrizio", "Caputo"]
    orders = sorted({float(re.search(r"alpha=([\d.]+)", r["setting"]).group(1))
                     for r in rows if "alpha=" in r.get("setting", "")})
    baseline = [r for r in rows if r["method"] == "gradient descent"]
    lines = ["| α | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for order in orders:
        cells = []
        for method in methods:
            match = [r for r in rows if r["method"] == method
                     and f"alpha={order:g}," in r.get("setting", "")]
            cells.append(f"{float(match[0]['test']) * 100:.2f}" if match else "—")
        if all(c == "—" for c in cells):
            continue                      # an order no rule recorded at this multiplier
        lines.append(f"| {order:g} | " + " | ".join(cells) + " |")
    if baseline:
        lines.append(f"| *gradient descent (no α)* | "
                     + " | ".join([f"*{float(baseline[0]['test']) * 100:.2f}*"] + [""] *
                                  (len(methods) - 1)) + " |")
    return "\n".join(lines)


def table_datasets() -> str:
    """The benchmark suite, measured from the loaders rather than transcribed.

    The update budget is computed the same way the training loop computes it --
    ceil(n_train / batch) * epochs -- because a hand-written range went stale:
    the README claimed 8550-9000 for every dataset while four were outside it
    and Letter was half as long again as MNIST.
    """
    import math
    import sys

    from karcifann import load
    from karcifann.datasets import train_val_test_split

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_analysis import DATASETS as CONFIG

    lines = ["| dataset | n | features | classes | uniform-pred. MSE | epochs | updates |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        try:
            x, t = load(name)
        except Exception:
            return "_(datasets unavailable)_"
        k = t.shape[1]
        cfg = CONFIG[name]
        if name == "mnist":                       # fixed slices, not a split
            n_train = 55000
        else:
            (xtr, _), _, _ = train_val_test_split(x, t, 0.15, 0.15, seed=0)
            n_train = len(xtr)
        updates = math.ceil(n_train / cfg["batch_size"]) * cfg["epochs"]
        lines.append(f"| {PRETTY[name]} | {x.shape[0]} | {x.shape[1]} | {k} | "
                     f"{(k - 1) / k ** 2:.4f} | {cfg['epochs']} | {updates} |")
    return "\n".join(lines)


def table_power_floors() -> str:
    """The smallest p each design can return, before any data is read.

    This is a *resolution* bound, not statistical power: power needs a stated
    alternative, a variance and a sampling design, none of which appears here.
    The verdicts say only what the design can and cannot resolve.
    """
    from karcifann.analysis import friedman_power_floor, wilcoxon_power_floor

    rows = [
        ("Friedman, datasets as blocks (asymptotic χ²)", "3 datasets × 5 methods",
         friedman_power_floor(3, 5), "cannot resolve below 0.017"),
        ("Friedman, datasets as blocks (asymptotic χ²)", "**8 datasets × 5 methods**",
         friedman_power_floor(8, 5), "resolution not the binding constraint"),
        ("Wilcoxon + Holm, seeds as blocks (exact)", "8 seeds, 10 pairs",
         wilcoxon_power_floor(8, 10), "**cannot reach 0.05 at all**"),
        ("Wilcoxon + Holm, seeds as blocks (exact)", "16 seeds, 10 pairs",
         wilcoxon_power_floor(16, 10), "can reach 0.05"),
    ]
    lines = ["| test | design | best attainable p | verdict |", "|---|---|---:|---|"]
    for test, design, floor, verdict in rows:
        shown = f"{floor:.4f}" if floor >= 1e-4 else f"{floor:.1e}"
        lines.append(f"| {test} | {design} | {shown} | {verdict} |")
    return "\n".join(lines)


def table_telescoping() -> str:
    """How exactly the Karci gauge collapses, per dataset."""
    lines = ["| dataset | order | max relative error |", "|---|---:|---:|"]
    for dataset in ("digits", "mnist"):
        for row in read(f"chained_vs_terminal_{dataset}"):
            if row["stage"] == "telescoping":
                lines.append(f"| {dataset} | {float(row['coefficient']):g} | "
                             f"{float(row['max_relative_error']):.1e} |")
    return "\n".join(lines) if len(lines) > 2 else "_(no results)_"


def table_gauge_ablation_stats() -> str:
    """Whether the full/error/weight split makes any difference at all.

    The numbers here were prose once, hand-typed from a three-dataset run, and
    they said the opposite of what eight datasets say.  Generated now, and
    tested the way the rest of the study is: exact paired permutation over the
    seeds, Holm across every comparison, Friedman across datasets.
    """
    import collections

    import numpy as np

    from karcifann.analysis import (holm, nemenyi_critical_difference,
                                    paired_permutation_test)

    modes = ["full", "error", "weight"]
    means, pairs = {}, []
    for dataset in ORDER:
        rows = read(f"{dataset}_e2_final")
        if not rows:
            continue
        by = collections.defaultdict(dict)
        for row in rows:
            by[row["mode"]][int(row["seed"])] = float(row["test_acc"]) * 100
        if not all(by.get(m) for m in modes):
            continue
        means[dataset] = [float(np.mean(list(by[m].values()))) for m in modes]
        for i, a in enumerate(modes):
            for b in modes[i + 1:]:
                seeds = sorted(set(by[a]) & set(by[b]))
                x = np.array([by[a][s] for s in seeds])
                y = np.array([by[b][s] for s in seeds])
                pairs.append([dataset, a, b, float(np.mean(x - y)),
                              float(paired_permutation_test(x, y))])
    if len(means) < 3:
        return "_(not enough datasets)_"

    from scipy.stats import friedmanchisquare, rankdata

    matrix = np.array([means[d] for d in means])
    chi2, pvalue = friedmanchisquare(*[matrix[:, i] for i in range(len(modes))])
    ranks = np.array([len(modes) + 1 - rankdata(row) for row in matrix])
    critical = nemenyi_critical_difference(len(modes), len(means))

    adjusted, rejected = holm([row[4] for row in pairs])
    lines = [
        f"Friedman over {len(means)} datasets: χ² = {chi2:.2f}, "
        f"**p = {pvalue:.2f}**; Nemenyi critical difference {critical:.2f}.",
        "",
        "| mode | average rank |",
        "|---|---:|",
    ]
    for i, mode in enumerate(modes):
        lines.append(f"| `{mode}` | {ranks[:, i].mean():.2f} |")

    survivors = [(row, adj) for row, adj, rej in zip(pairs, adjusted, rejected) if rej]
    lines += ["", f"Pairwise, {len(survivors)} of {len(pairs)} comparisons survive Holm:", ""]
    if survivors:
        lines += ["| dataset | comparison | difference | Holm p |", "|---|---|---:|---:|"]
        for (dataset, a, b, diff, _), adj in sorted(survivors, key=lambda t: t[1]):
            lines.append(f"| {PRETTY[dataset]} | `{a}` − `{b}` | {diff:+.2f} pp | {adj:.4f} |")
    else:
        lines.append("_(none)_")
    return "\n".join(lines)


def table_factor_spread() -> str:
    """How much the prefactor varies across parameters within a single step.

    A gauge whose 5th and 95th percentiles nearly coincide is a rescaled
    learning rate wearing a different name; a wide spread is a genuine diagonal
    preconditioner.  This is the measurement that separates the two, and it
    explains why the saturating members track tuned gradient descent so
    closely.
    """
    methods = ["KarciFANN", "exp", "tanh", "log"]
    # The percentile ratio alone describes only the central 90% of parameters.
    # A gauge can read 1.00 there and still put a factor of 15 on some single
    # weight, so the extremes are reported beside it.
    lines = ["| dataset | " + " | ".join(f"`{m}`" for m in methods)
             + " | `tanh` min | `tanh` max |",
             "|---|" + "---:|" * (len(methods) + 2)]
    any_row = False
    for dataset in ORDER:
        rows = {r["method"]: r for r in read(f"{dataset}_e5_factor_spread")}
        if not rows:
            continue
        cells = []
        for method in methods:
            row = rows.get(method)
            cells.append(f"{float(row['spread_ratio']):.2f}" if row else "--")
        tanh = rows.get("tanh")
        cells += ([f"{float(tanh['min']):.2f}", f"{float(tanh['max']):.2f}"]
                  if tanh else ["--", "--"])
        lines.append(f"| {PRETTY[dataset]} | " + " | ".join(cells) + " |")
        any_row = True
    return "\n".join(lines) if any_row else "_(no results)_"


def table_splits() -> str:
    """Exact split sizes, so the protocol can be checked rather than trusted."""
    import sys

    from karcifann import load
    from karcifann.datasets import train_val_test_split

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_analysis import DATASETS as CONFIG

    lines = ["| dataset | train | validation | test | split |",
             "|---|---:|---:|---:|---|"]
    for name in ORDER:
        if name not in CONFIG:
            continue
        if name == "mnist":
            lines.append(f"| {PRETTY[name]} | 55000 | 5000 | 10000 | "
                         "fixed slices, not stratified |")
            continue
        try:
            x, t = load(name)
        except Exception:
            return "_(datasets unavailable)_"
        (xtr, _), (xva, _), (xte, _) = train_val_test_split(x, t, 0.15, 0.15, seed=0)
        lines.append(f"| {PRETTY[name]} | {len(xtr)} | {len(xva)} | {len(xte)} | "
                     "stratified 70/15/15, seed 0 |")
    return "\n".join(lines)


def table_grids() -> str:
    """Every search grid, its size, and the resulting tuning cost.

    The cost column is the point: gradient descent has one axis to search and
    the gauge families have two, so a like-for-like comparison of tuned winners
    is not a like-for-like comparison of tuning effort.
    """
    import collections

    from karcifann.gauges import GAUGE_FAMILIES

    # The multiplier count is measured from the stored sweeps, not assumed.  It
    # was hard-coded at 9, which is wrong for Vehicle: its optima run below the
    # common floor, so run_all.sh gives it a 13-point grid.
    counts = {}
    for dataset in ORDER:
        seen = {row["scale"] for row in read(f"gauge_family_{dataset}")
                if row.get("scale")}
        if seen:
            counts[dataset] = len(seen)
    if not counts:
        return "_(no results)_"
    common = collections.Counter(counts.values()).most_common(1)[0][0]
    odd = {PRETTY[d]: c for d, c in counts.items() if c != common}

    def cell(per_shape: int) -> str:
        base = f"{per_shape * common}"
        if not odd:
            return base
        extra = ", ".join(f"{per_shape * c} on {name}" for name, c in odd.items())
        return f"{base} ({extra})"

    scales = f"{common}" + (
        " (" + ", ".join(f"{c} on {n}" for n, c in odd.items()) + ")" if odd else "")
    lines = ["| method | shape grid | shapes | step multipliers | tuning runs |",
             "|---|---|---:|---:|---:|"]
    lines.append(f"| gradient descent | — (rate only) | — | {scales} | {cell(1)} |")
    for name in ("power", "log", "tanh", "exp"):
        family = GAUGE_FAMILIES.get(name)
        if not family:
            continue
        grid = list(family[1])
        shown = f"{grid[0]:g} … {grid[-1]:g}"
        lines.append(f"| `{name}` | {shown} | {len(grid)} | {scales} | "
                     f"{cell(len(grid))} |")
    return "\n".join(lines)


def table_loss_scaling() -> str:
    """The rescaling identity, measured rather than derived."""
    rows = read("loss_scaling")
    if not rows:
        return "_(no results)_"

    a = [r for r in rows if r["section"] == "A"]
    b = [r for r in rows if r["section"] == "B"]
    lines = ["| order α | objective ×c | predicted c^α | observed | max rel. error "
             "| normwise | by subtraction |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in a:
        lines.append(f"| {float(r['alpha']):.1f} | {float(r['scale']):g} | "
                     f"{float(r['predicted']):.6g} | {float(r['observed']):.6g} | "
                     f"{float(r['max_relative_error']):.1e} | "
                     f"{float(r['normwise_relative_error']):.1e} | "
                     f"{float(r['max_relative_error_by_subtraction']):.1e} |")
    worst_a = max(float(r["max_relative_error"]) for r in a)
    worst_norm = max(float(r["normwise_relative_error"]) for r in a)
    worst_sub = max(float(r["max_relative_error_by_subtraction"]) for r in a)
    lines.append("")
    lines.append(f"Worst relative error away from the floor: **{worst_a:.1e}** "
                 f"coordinatewise, **{worst_norm:.1e}** normwise.")
    lines.append("")
    lines.append(
        f"The last column recovers each increment as `before - after` instead of "
        f"reading it off the optimiser. A weight is of order 1e-1 and an increment "
        f"can be of order 1e-12, so that subtraction cancels away about eleven "
        f"digits and reports up to **{worst_sub:.1e}** -- an artefact of the "
        f"reconstruction, not of the identity. Its growth with α is the same "
        f"artefact: a larger α makes the increment smaller at `c < 1`.")
    lines.append("")
    below = {int(float(r["n_below_floor"])) for r in a}
    lines.append(
        f"Coordinates whose predicted increment falls below 1e-12 of the largest "
        f"in the same update are counted, not ratioed: {min(below)}--{max(below)} "
        f"of {int(float(a[0]['n_coordinates']))} per setting.")
    if b:
        broken = [r for r in b if float(r["max_relative_error"]) > 1e-6]
        lines.append(f"With the floor raised to `eps = 1e-2` the identity fails on "
                     f"{len(broken)} of {len(b)} settings, and only where `c·J` has "
                     f"fallen below the floor:")
        lines.append("")
        lines.append("| order α | objective ×c | predicted | observed | rel. error |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in broken:
            lines.append(
                f"| {float(r['alpha']):.1f} | {float(r['scale']):g} | "
                f"{float(r['predicted']):.6g} | {float(r['observed']):.6g} | "
                f"**{float(r['max_relative_error']):.2f}** |")
        intact = len(b) - len(broken)
        lines.append("")
        worst_intact = max(float(r["max_relative_error"]) for r in b
                           if float(r["max_relative_error"]) <= 1e-6)
        lines.append(f"The other {intact} settings are unaffected, agreeing to "
                     f"{worst_intact:.0e} as above.")
    return "\n".join(lines)


def table_loss_scaling_drift() -> str:
    """How long the compensated trajectory stays identical."""
    rows = [r for r in read("loss_scaling") if r["section"] == "C"]
    if not rows:
        return "_(no results)_"
    epochs = sorted({int(float(r["epochs"])) for r in rows})
    keys = sorted({(float(r["alpha"]), float(r["scale"])) for r in rows})
    lines = ["| order α | objective ×c | " + " | ".join(f"{e} ep." for e in epochs) + " |",
             "|---:|---:|" + "---:|" * len(epochs)]
    for alpha, scale in keys:
        cells = []
        for e in epochs:
            match = [r for r in rows if float(r["alpha"]) == alpha
                     and float(r["scale"]) == scale and int(float(r["epochs"])) == e]
            cells.append(f"{float(match[0]['weight_divergence']):.1e}" if match else "--")
        lines.append(f"| {alpha:.1f} | {scale:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLES = {
    "headline": table_headline,
    "fod-sweep": table_fod_sweep,
    "fod-tuned": table_fod_tuned,
    "mnist-methods": table_mnist_methods,
    "mnist-methods-tuned": table_mnist_methods_tuned,
    "chained-digits-gain": table_chained_digits_gain,
    "chained-digits": table_chained_digits,
    "chained-mnist-gain": table_chained_mnist_gain,
    "chained-mnist": table_chained_mnist,
    "chained-digits-tuned": table_chained_digits_tuned,
    "chained-mnist-tuned": table_chained_mnist_tuned,
    "telescoping": table_telescoping,
    "cf-residue": table_cf_residue,
    "stability": table_stability,
    "datasets": table_datasets,
    "power-floors": table_power_floors,
    "evidence": table_evidence,
    "accuracy": table_accuracy,
    "ranks": table_ranks,
    "headroom": table_headroom,
    "headroom-optimiser": table_headroom_optimiser,
    "split-robustness": table_split_robustness,
    "multiseed-tuning": table_multiseed_tuning,
    "headroom-splits": table_headroom_splits,
    "headroom-selection": table_headroom_selection,
    "split-stability": table_split_stability,
    "split-robustness-detail": table_split_robustness_detail,
    "tolerance": table_tolerance,
    "gauge-ablation": table_gauge_ablation,
    "gauge-ablation-stats": table_gauge_ablation_stats,
    "factor-spread": table_factor_spread,
    "splits": table_splits,
    "grids": table_grids,
    "loss-scaling": table_loss_scaling,
    "loss-scaling-drift": table_loss_scaling_drift,
}


def render(text: str) -> tuple[str, list[str]]:
    """Replace every marked block; returns the new text and the names seen."""
    seen = []

    def substitute(match):
        name = match.group(1)
        seen.append(name)
        body = TABLES[name]() if name in TABLES else match.group(2).strip("\n")
        return f"<!-- table:{name} -->\n{body}\n<!-- /table:{name} -->"

    # The content group may be empty, so an unmarked placeholder still matches.
    pattern = re.compile(r"<!-- table:([a-z-]+) -->\n(.*?)<!-- /table:\1 -->", re.S)
    return pattern.sub(substitute, text), seen


# Prose numbers that are legitimately not copied from a generated table.
# Every entry needs a reason: this list is the audit trail for the one defect
# class that kept recurring in this project -- narrative written for an earlier
# run left sitting beside a regenerated table.
DERIVED_NUMBERS = {
    "3.10": "minimum Python version",
    "0.30": "spread across three rows of mnist-methods-tuned",
    "0.27": "spread across three rows of mnist-methods-tuned",
    "0.28": "gradient descent minus CF chained, chained-mnist-tuned",
    "0.22": "binomial standard error at n=5000",
    "0.18": "binomial standard error at n=10000",
    "1.05": "M/alpha at alpha=0.95, an analytic constant",
    "0.25": "lower bound of a search grid",
    "3.29": "across-split SD over mean paired seed SD, Satimage, split_robustness",
    "4.04": "across-split SD over mean paired seed SD, Segment, split_robustness",
    "0.03": "floor of Vehicle's 13-point multiplier grid, run_all.sh",
    "0.003": "the shape-grid floor that section 6.2 reports as too high",
    "56.54": "configuration A on Vehicle's collapsing split, "
             "results/headroom_splits_vehicle.csv",
    "53.98": "configuration A on Satimage's collapsing split, "
             "results/headroom_splits_satimage.csv",
    "68.32": "configuration A on Segment's collapsing split, "
             "results/headroom_splits_segment.csv",
    "77.15": "configuration A at split 0 on Vehicle, "
             "results/headroom_splits_vehicle.csv",
    "0.01": "threshold in the exp(-999|d|) argument",
    "1.72": "mean CI half-width of the tanh column, section 6.1",
    "0.026": "lower end of the CF/Newton ratio range",
    "1.79": "upper end of the CF/Newton ratio range",
    "0.411": "mean CI half-width of the exp column, section 6.1",
    "0.245": "mean CI half-width of the KarciFANN column, section 6.1",
    "0.173": "mean CI half-width of the weight column, section 8.4",
    # Figures from superseded runs, retained deliberately to record what each
    # flaw cost. Section 6.2 and 8.1b name them as superseded.
    "95.93": "superseded: GD on Letter before the grid was widened",
    "85.17": "superseded: GD on Letter under the --scales bug",
    "87.25": "superseded: GD on Letter after that bug was fixed",
    "1.56": "superseded: GD average rank under unstratified splits",
    "3.19": "superseded: KarciFANN average rank under unstratified splits",
    "3.49": "superseded: Vehicle head-to-head under unstratified splits",
}


def check_prose_numbers(text: str) -> list[str]:
    """Find decimal numbers in prose that no generated table backs.

    Catches narrative left over from an earlier run. Anything genuinely
    computed rather than quoted belongs in DERIVED_NUMBERS with a reason.
    """
    table_numbers: set[str] = set()
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"<!-- table:([a-z-]+) -->\n(.*?)\n<!-- /table:\1 -->",
                             text, re.S):
        spans.append(match.span())
        table_numbers |= set(re.findall(r"\d+\.\d+", match.group(2)))

    chunks, last = [], 0
    for start, end in spans:
        chunks.append((last, text[last:start]))
        last = end
    chunks.append((last, text[last:]))

    problems = []
    for offset, chunk in chunks:
        chunk = re.sub(r"```.*?```", lambda m: " " * len(m.group(0)), chunk, flags=re.S)
        for match in re.finditer(r"(?<![\w.])(\d{1,3}\.\d{2,3})(?![\w.])", chunk):
            value = match.group(1)
            if value in table_numbers or value in DERIVED_NUMBERS:
                continue
            line = text[:offset + match.start()].count("\n") + 1
            context = " ".join(chunk[max(0, match.start() - 60):match.start() + 30].split())
            problems.append(f"  README.md:{line}: {value} is in no table -- ...{context}...")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the README is out of date")
    args = parser.parse_args()

    original = README.read_text()
    updated, seen = render(original)
    missing = sorted(set(TABLES) - set(seen))

    if args.check:
        if updated != original:
            print("README tables are out of date; run experiments/update_readme.py")
            raise SystemExit(1)
        stale = check_prose_numbers(original)
        if stale:
            print("prose quotes numbers that no generated table contains:")
            print("\n".join(stale))
            raise SystemExit(1)
        print(f"README is in step with results/ ({len(seen)} generated tables, "
              f"prose numbers all table-backed or declared)")
        return

    README.write_text(updated)
    print(f"regenerated {len(seen)} table(s): {', '.join(seen) or 'none'}")
    if missing:
        print(f"  not present in README: {', '.join(missing)}")


if __name__ == "__main__":
    main()
