#!/usr/bin/env python3
"""Turn the CSVs from run_analysis.py into statistics tables and figures.

Reads ``results/*.csv``, writes derived statistics back to ``results/`` and
every figure to ``figures/`` as both PDF and SVG.  Runs in seconds, so figures
can be restyled without recomputing anything.

    python3 experiments/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karcifann.analysis import (  # noqa: E402
    friedman_over_datasets,
    pairwise_wilcoxon,
    tolerance_region,
    wilcoxon_power_floor,
)
from karcifann.plotting import colour, marker, save, setup  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("figures")
# Ordered by class count, which is the axis the loss scale varies along.
DATASETS = ["vehicle", "satimage", "dry_bean", "segment", "optdigits",
            "pendigits", "mnist", "letter"]
PRETTY = {
    "vehicle": "Vehicle (4 classes)",
    "satimage": "Satimage (6 classes)",
    "dry_bean": "Dry Bean (7 classes)",
    "segment": "Segment (7 classes)",
    "optdigits": "Optdigits (10 classes)",
    "pendigits": "Pendigits (10 classes)",
    "mnist": "MNIST (10 classes)",
    "letter": "Letter (26 classes)",
}
#: Every dataset, in the order the README uses.  The per-dataset plotters skip
#: any dataset whose CSV is absent, so coverage follows the data rather than a
#: hand-kept list -- this was three names for a while after the suite grew to
#: eight, which silently held the convergence and annealing figures at three
#: even though E1 had been run for all of them.
DEEP = DATASETS
METHOD_ORDER = ["gradient descent", "KarciFANN", "exp", "tanh", "log"]

#: How a method is *written*, as opposed to how it is keyed.  The CSVs and the
#: statistics use ASCII names, so the data path must keep them; only labels
#: drawn for a reader get the Turkish dotless i.  Keeping the two separate is
#: what stops a rename from silently failing to match a results column.
DISPLAY_NAMES = {"KarciFANN": "Karc\u0131FANN"}


def display(name: str) -> str:
    """The reader-facing spelling of a method or configuration name."""
    return DISPLAY_NAMES.get(name, name)


def read(name: str) -> list[dict]:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out = {}
            for key, value in row.items():
                try:
                    out[key] = float(value) if value not in ("", None) else np.nan
                except ValueError:
                    out[key] = value
            rows.append(out)
    return rows


def write(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"   wrote results/{name}.csv ({len(rows)} rows)")


def sd(values, axis=None):
    """Sample SD that degrades to zero rather than warning on a single run."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis] if axis is not None else values.size
    if n < 2:
        return np.zeros(np.delete(values.shape, axis)) if axis is not None else 0.0
    return values.std(axis=axis, ddof=1)


def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def group(rows, key, value):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row[value])
    return {k: np.array(v, dtype=float) for k, v in out.items()}


# ------------------------------------------------------------------ summary


def summary_and_tests(plt) -> None:
    print("\nsummary, Wilcoxon and Friedman")
    summary, wilcoxon_rows = [], []
    accuracy_table, present = [], []

    for dataset in DATASETS:
        rows = read(f"{dataset}_e1_final")
        if not rows:
            continue
        methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
        scores = {}
        for metric in ("test_acc", "test_f1", "val_acc"):
            for method in methods:
                values = np.array([r[metric] for r in rows
                                   if r["method"] == method and not np.isnan(r[metric])])
                if metric == "test_acc":
                    scores[method] = values
                summary.append({
                    "dataset": dataset, "method": method, "metric": metric,
                    "mean": values.mean(), "ci95": ci95(values),
                    "std": values.std(ddof=1), "n": values.size,
                    "min": values.min(), "max": values.max(),
                })

        result = pairwise_wilcoxon(scores)
        n_pairs = len(result.names) * (len(result.names) - 1) // 2
        flat_perm, index = [], []
        for a_i in range(len(result.names)):
            for b_i in range(a_i + 1, len(result.names)):
                flat_perm.append(result.permutation[a_i, b_i])
                index.append((a_i, b_i))
        from karcifann.analysis import holm as _holm

        perm_adjusted, _ = _holm(flat_perm)
        perm_lookup = {pair: value for pair, value in zip(index, perm_adjusted)}
        for i, a in enumerate(result.names):
            for j, b in enumerate(result.names):
                if i >= j:
                    continue
                wilcoxon_rows.append({
                    "dataset": dataset, "method_a": a, "method_b": b,
                    "mean_a": scores[a].mean(), "mean_b": scores[b].mean(),
                    "difference": scores[a].mean() - scores[b].mean(),
                    "p_raw": result.pvalues[i, j], "p_holm": result.adjusted[i, j],
                    "cohens_d": result.effect[i, j], "cliffs_delta": result.delta[i, j],
                    "significant_holm_0.05": int(result.adjusted[i, j] <= 0.05),
                    "n_seeds": scores[a].size,
                    "holm_floor": wilcoxon_power_floor(scores[a].size, n_pairs),
                    "p_permutation": result.permutation[i, j],
                    "p_permutation_holm": perm_lookup[(i, j)],
                    "significant_permutation": int(perm_lookup[(i, j)] <= 0.05),
                    "p_sign": result.sign[i, j],
                    "difference_skew": result.skew[i, j],
                    "sign_agrees_at_0.05": int(
                        (result.pvalues[i, j] <= 0.05) == (result.sign[i, j] <= 0.05)),
                    # A p-value says whether an effect was detected; these say
                    # how large it is and how consistent across seeds.  Reporting
                    # only the former hides that a "significant" difference can
                    # still run the other way on several seeds.
                    **_paired_effect(scores[a], scores[b]),
                })
        accuracy_table.append([scores[m].mean() for m in methods])
        present.append(dataset)
        plot_pvalue_matrix(plt, dataset, result)

    if wilcoxon_rows:
        from karcifann.analysis import holm

        study_wide, _ = holm([r["p_raw"] for r in wilcoxon_rows])
        perm_study, _ = holm([r["p_permutation"] for r in wilcoxon_rows])
        for row, adjusted, perm in zip(wilcoxon_rows, study_wide, perm_study):
            row["p_holm_study_wide"] = adjusted
            row["significant_study_wide"] = int(adjusted <= 0.05)
            row["survives_both_corrections"] = int(
                row["significant_holm_0.05"] and adjusted <= 0.05)
            row["p_permutation_study_wide"] = perm
            # The headline standard: an assumption-free test, corrected over
            # every comparison in the study.
            row["robust"] = int(perm <= 0.05)

    write("summary_final", summary)
    write("stats_wilcoxon", wilcoxon_rows)

    if len(accuracy_table) >= 2:
        table = np.array(accuracy_table)
        outcome = friedman_over_datasets(table)
        methods = [m for m in METHOD_ORDER][: table.shape[1]]
        write("stats_friedman", [{
            "blocks_datasets": outcome["blocks"], "methods": outcome["methods"],
            "statistic": outcome["statistic"], "pvalue": outcome["pvalue"],
            "power_floor_best_case_p": outcome["power_floor"],
            "underpowered": int(outcome["power_floor"] > 0.01),
            "nemenyi_critical_difference": outcome["critical_difference"],
            **{f"rank_{m}": r for m, r in zip(methods, outcome["ranks"])},
        }])
        best = int(np.argmin(outcome["ranks"]))
        write("stats_nemenyi", [{
            "method": m, "average_rank": r,
            "gap_to_best": r - outcome["ranks"][best],
            "critical_difference": outcome["critical_difference"],
            "separated_from_best": int(
                r - outcome["ranks"][best] > outcome["critical_difference"]),
            "best_method": methods[best],
        } for m, r in zip(methods, outcome["ranks"])])
        plot_ranks(plt, methods, outcome, present)


# ------------------------------------------------------------------ figures



def _paired_effect(a, b) -> dict:
    """Effect size of a paired comparison, beside the significance verdict.

    The interval is a normal approximation over seeds sharing one fixed split,
    so it describes variation due to initialisation and shuffling only -- not
    over new splits or new datasets.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    half = 1.96 * sd(d) / np.sqrt(n) if n > 1 else float("nan")
    # All three counts are recorded rather than two and a subtraction.  A
    # consumer that reversed the pair by computing ``n - n_favouring_a`` was
    # silently counting ties as wins for the other method, which turned 13
    # wins and 3 ties on Vehicle into a printed "16/16".
    return {
        "diff_ci95": half,
        "diff_sd": sd(d),
        "diff_min": float(d.min()),
        "diff_max": float(d.max()),
        "n_favouring_a": int((d > 0).sum()),
        "n_favouring_b": int((d < 0).sum()),
        "n_tied": int((d == 0).sum()),
    }


def plot_convergence(plt) -> None:
    for dataset in DEEP:
        rows = read(f"{dataset}_e1_curves")
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
        for method in METHOD_ORDER:
            subset = [r for r in rows if r["method"] == method]
            if not subset:
                continue
            epochs = sorted({int(r["epoch"]) for r in subset})
            for ax, field in zip(axes, ("val_acc", "train_mse")):
                values = np.array([[r[field] for r in subset if int(r["epoch"]) == e]
                                   for e in epochs], dtype=float)
                mean, spread = values.mean(axis=1), sd(values, axis=1)
                ax.plot(epochs, mean, color=colour(method), label=display(method))
                ax.fill_between(epochs, mean - spread, mean + spread, color=colour(method), alpha=0.18,
                                linewidth=0)
        axes[0].set_ylabel("validation accuracy")
        axes[1].set_ylabel("training MSE")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("epoch")
        axes[0].legend(ncol=2, loc="lower right")
        # Count the seeds rather than naming a number: the suptitle said "8"
        # for months after the runs moved to 16.
        n_seeds = len({r["seed"] for r in rows})
        fig.suptitle(f"Convergence — {PRETTY[dataset]}  "
                     f"(mean ± 1 SD over {n_seeds} seeds)")
        save(fig, f"convergence_{dataset}", FIGURES)
        print(f"   figures/convergence_{dataset}.[pdf|svg]")


def plot_annealing(plt) -> None:
    for dataset in DEEP:
        rows = [r for r in read(f"{dataset}_e1_curves") if not np.isnan(r.get("factor_mean", np.nan))]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for method in METHOD_ORDER:
            subset = [r for r in rows if r["method"] == method]
            if not subset:
                continue
            epochs = sorted({int(r["epoch"]) for r in subset})
            mean = [np.mean([r["factor_mean"] for r in subset if int(r["epoch"]) == e])
                    for e in epochs]
            ax.plot(epochs, mean, color=colour(method), label=display(method))
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"mean prefactor  $\langle \Phi \rangle$")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Step-size self-annealing — {PRETTY[dataset]}")
        save(fig, f"annealing_{dataset}", FIGURES)
        print(f"   figures/annealing_{dataset}.[pdf|svg]")


def plot_sensitivity(plt) -> None:
    tolerance_rows = []
    for dataset in DEEP:
        rows = read(f"{dataset}_e3_sensitivity")
        if not rows:
            continue
        sweeps = sorted({r["sweep"] for r in rows})
        fig, axes = plt.subplots(1, len(sweeps), figsize=(3.6 * len(sweeps), 2.6),
                                 layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, sweep in zip(axes, sweeps):
            subset = [r for r in rows if r["sweep"] == sweep]
            values = sorted({r["value"] for r in subset})
            mean = np.array([np.mean([r["test_acc"] for r in subset if r["value"] == v])
                             for v in values])
            spread = np.array([sd([r["test_acc"] for r in subset if r["value"] == v])
                               for v in values])
            name = "KarciFANN" if "alpha" in sweep else "gradient descent"
            ax.errorbar(values, mean * 100, yerr=spread * 100, color=colour(name),
                        marker=marker(name), linewidth=1.2)
            region = tolerance_region(values, mean)
            ax.axhspan(region["best"] * 99, region["best"] * 100, color=colour(name), alpha=0.10)
            ax.axvspan(region["min"], region["max"], color=colour(name), alpha=0.12, linewidth=0)
            ax.set_xscale("log")
            ax.set_xlabel(sweep.split(": ")[1])
            ax.set_ylabel("test accuracy (%)")
            ax.set_title(f"{display(name)}: {region['decades']:.2f} decades within 1%")
            tolerance_rows.append({
                "dataset": dataset, "sweep": sweep, "tolerance": 0.01, **region,
            })
        fig.suptitle(f"Hyperparameter basin — {PRETTY[dataset]}")
        save(fig, f"sensitivity_{dataset}", FIGURES)
        print(f"   figures/sensitivity_{dataset}.[pdf|svg]")
    write("stats_tolerance", tolerance_rows)


def plot_gauge_ablation(plt) -> None:
    rows_all, summary = [], []
    for dataset in DEEP:
        rows = read(f"{dataset}_e2_final")
        if rows:
            rows_all.append((dataset, rows))
    if not rows_all:
        return
    modes = ["full", "error", "weight"]
    # Eight datasets at 5.2in ran the tick labels into one another; widen and
    # slant them so each stays legible at journal column width.
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    width = 0.25
    positions = np.arange(len(rows_all))
    span = []
    for k, mode in enumerate(modes):
        means, errors = [], []
        for dataset, rows in rows_all:
            values = np.array([r["test_acc"] for r in rows if r["mode"] == mode]) * 100
            means.append(values.mean())
            errors.append(ci95(values))
            summary.append({"dataset": dataset, "mode": mode,
                            "mean_test_acc": values.mean() / 100, "ci95": ci95(values) / 100,
                            "n": values.size})
            span += [values.mean() - ci95(values), values.mean() + ci95(values)]
        ax.bar(positions + (k - 1) * width, means, width, yerr=errors,
               color=colour(mode), label=mode, edgecolor="white", linewidth=0.4)
    for k, (dataset, _) in enumerate(rows_all):
        reference = read(f"{dataset}_e1_final")
        values = np.array([r["test_acc"] for r in reference
                           if r["method"] == "gradient descent"]) * 100
        if values.size:
            ax.hlines(values.mean(), k - 1.6 * width, k + 1.6 * width,
                      color="black", linestyle="--", linewidth=1.0,
                      label="gradient descent" if k == 0 else None)
    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[d].split(" (")[0] for d, _ in rows_all],
                       rotation=25, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rows_all) - 0.4)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(r"Which half of $\Phi = g(J)/g(|W|)$ does the work?")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.46))
    # Zoom to the data: with every bar above 80% a zero-based axis hides the
    # very differences the figure exists to show.
    low, high = min(span), max(span)
    pad = max(0.25 * (high - low), 0.3)
    ax.set_ylim(low - pad, high + pad)
    save(fig, "gauge_ablation", FIGURES)
    write("summary_gauge_ablation", summary)
    print("   figures/gauge_ablation.[pdf|svg]")


def plot_config_ablation(plt) -> None:
    summary = []
    for dataset in DEEP:
        rows = read(f"{dataset}_e4_final")
        if not rows:
            continue
        configs = sorted({r["config"] for r in rows})
        means = {c: np.array([r["test_acc"] for r in rows if r["config"] == c]) * 100
                 for c in configs}
        order = sorted(configs, key=lambda c: means[c].mean())
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        ax.barh(np.arange(len(order)), [means[c].mean() for c in order],
                xerr=[ci95(means[c]) for c in order],
                color=["#0072B2" if c.startswith("sigmoid/mse/gd") else "#999999" for c in order],
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("test accuracy (%)")
        ax.set_title(f"Configuration ablation — {PRETTY[dataset]}")
        low = min(means[c].mean() - ci95(means[c]) for c in order)
        high = max(means[c].mean() + ci95(means[c]) for c in order)
        ax.set_xlim(low - 0.25 * (high - low) - 0.3, high + 0.1 * (high - low) + 0.3)
        save(fig, f"config_ablation_{dataset}", FIGURES)
        print(f"   figures/config_ablation_{dataset}.[pdf|svg]")
        for c in configs:
            summary.append({"dataset": dataset, "config": c,
                            "mean_test_acc": means[c].mean() / 100,
                            "ci95": ci95(means[c]) / 100})
    write("summary_config_ablation", summary)


def plot_factor_spread(plt) -> None:
    rows = [r for d in DATASETS for r in read(f"{d}_e5_factor_spread")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.8), layout="constrained")
    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    width = 0.8 / max(len(methods), 1)
    for k, method in enumerate(methods):
        offsets, lows, highs, mids = [], [], [], []
        for i, dataset in enumerate(datasets):
            match = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not match:
                continue
            r = match[0]
            offsets.append(i + (k - len(methods) / 2 + 0.5) * width)
            lows.append(r["p05"]); highs.append(r["p95"]); mids.append(r["median"])
        if not offsets:
            continue
        ax.vlines(offsets, lows, highs, color=colour(method), linewidth=3, alpha=0.8,
                  label=display(method))
        ax.plot(offsets, mids, marker(method), color="white", markeredgecolor=colour(method),
                markersize=4, linestyle="none")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([PRETTY[d].split(" (")[0] for d in datasets])
    ax.set_yscale("log")
    ax.set_ylabel(r"prefactor $\Phi$  (5th–95th pct)")
    ax.set_title(r"Spread of $\Phi$ across parameters: rescaling or preconditioner?")
    ax.margins(y=0.18)
    ax.legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    save(fig, "factor_spread", FIGURES)
    print("   figures/factor_spread.[pdf|svg]")


def plot_pvalue_matrix(plt, dataset: str, result) -> None:
    n = len(result.names)
    fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    # Mask the diagonal rather than colouring it: a method compared with itself
    # is not a p-value of 1, it is not a comparison at all.
    matrix = np.ma.masked_invalid(np.log10(np.clip(result.adjusted, 1e-4, 1.0)))
    cmap = plt.get_cmap("viridis_r").with_extremes(bad="#f2f2f2")
    image = ax.imshow(matrix, cmap=cmap, vmin=-4, vmax=0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#888888")
            else:
                p = result.adjusted[i, j]
                ax.text(j, i, f"{p:.3f}" if p >= 0.001 else "<.001", ha="center", va="center",
                        fontsize=6, color="white" if p < 0.05 else "black")
    names = [display(x) for x in result.names]
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046)
    bar.set_label(r"$\log_{10}$ Holm-adjusted $p$  (brighter = stronger)", fontsize=7)
    bar.ax.axhline(np.log10(0.05), color="red", linewidth=1.0)
    bar.ax.text(1.6, np.log10(0.05), "0.05", color="red", fontsize=6,
                va="center", transform=bar.ax.get_yaxis_transform())
    ax.set_title(f"Paired Wilcoxon — {PRETTY[dataset]}", fontsize=9)
    save(fig, f"pvalue_matrix_{dataset}", FIGURES)
    print(f"   figures/pvalue_matrix_{dataset}.[pdf|svg]")


def plot_ranks(plt, methods, outcome, datasets) -> None:
    """A Demsar critical-difference diagram.

    The bar chart this replaces put average rank on a length scale, which
    invites reading bar lengths as effect sizes.  The conventional diagram
    places each method on a rank axis and joins the groups the post-hoc test
    cannot separate, which is the only comparison the statistic supports.
    """
    ranks = np.asarray(outcome["ranks"], dtype=float)
    cd = float(outcome["critical_difference"])
    order = np.argsort(ranks)                       # best (lowest) rank first
    names = [display(methods[i]) for i in order]
    tints = [colour(methods[i]) for i in order]
    values = ranks[order]
    k = len(order)
    lo, hi = 1.0, float(k)

    # maximal runs whose extremes lie within the critical difference
    cliques, i = [], 0
    while i < k:
        j = i
        while j + 1 < k and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
        i += 1

    clique_top = -0.22
    step = 0.16
    rows_top = clique_top - step * max(len(cliques), 1) - 0.34
    row_step = 0.30
    per_side = (k + 1) // 2
    bottom = rows_top - row_step * (per_side - 1) - 0.34

    fig, ax = plt.subplots(figsize=(5.6, 1.5 + 0.30 * k), layout="constrained")
    span = hi - lo
    ax.set_xlim(lo - 0.42 * span, hi + 0.42 * span)
    ax.set_ylim(bottom, 1.15)
    ax.axis("off")

    ax.plot([lo, hi], [0, 0], color="black", linewidth=1.0)
    for tick in range(int(lo), int(hi) + 1):
        ax.plot([tick, tick], [0, 0.10], color="black", linewidth=1.0)
        ax.text(tick, 0.16, str(tick), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, 0.62, "average rank over datasets (1 = best)",
            ha="center", va="bottom", fontsize=8)

    # cliques sit directly under the axis, above the labels
    for depth, (a, b) in enumerate(cliques):
        y = clique_top - step * depth
        ax.plot([values[a] - 0.04, values[b] + 0.04], [y, y],
                color="black", linewidth=3.4, solid_capstyle="butt")

    # each method: stem down to its row, then out to a label in the margin
    for position, (name, rank, tint) in enumerate(zip(names, values, tints)):
        left = position < per_side
        row = rows_top - row_step * (position if left else k - 1 - position)
        edge = lo - 0.40 * span if left else hi + 0.40 * span
        ax.plot([rank, rank], [0, row], color=tint, linewidth=1.2)
        ax.plot([rank, edge], [row, row], color=tint, linewidth=1.2)
        ax.text(edge + (-0.03 if left else 0.03) * span, row,
                f"{name} ({rank:.2f})",
                ha="right" if left else "left", va="center", fontsize=8)

    # critical-difference scale bar
    ax.plot([lo, lo + cd], [0.44, 0.44], color="black", linewidth=1.8,
            solid_capstyle="butt")
    for x in (lo, lo + cd):
        ax.plot([x, x], [0.39, 0.49], color="black", linewidth=1.0)
    ax.text(lo + cd / 2, 0.50, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Friedman over {outcome['blocks']} datasets, "
                 f"p = {outcome['pvalue']:.2g}; Nemenyi post-hoc", fontsize=8)
    save(fig, "average_ranks", FIGURES)
    print("   figures/average_ranks.[pdf|svg]")


SPLIT_DATASETS = ["vehicle", "satimage", "segment", "letter"]


def plot_split_differences(plt) -> None:
    """The ten per-split differences, one point each, against zero.

    Section 6.3 reports a mean, a median and an SD per dataset, and the review
    of 2026-09-23 was right that prose cannot show the shape those summarise:
    the distribution is skewed by a single split apiece, and a reader given
    only a negative median could take it for evidence of inferiority.  One
    point per split, a zero line, and the tuned SGD rate beside the outliers
    says what the numbers are.

    The y scale is symmetric-log because Satimage reaches +35.87 points while
    Letter stays inside one point; on a linear axis three datasets would be a
    flat line at zero.
    """
    rows = {d: [r for r in read(f"split_robustness_{d}")
                if r.get("stage") == "split_summary"]
            for d in SPLIT_DATASETS}
    rows = {d: r for d, r in rows.items() if r}
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.1), layout="constrained")
    datasets = list(rows)
    rng = np.random.default_rng(0)          # jitter only, so points do not overlap
    labelled_far = False
    for i, dataset in enumerate(datasets):
        diffs = np.array([r["mean_difference"] for r in rows[dataset]])
        # An "outlier" here is purely descriptive: a split whose difference sits
        # more than five points from the dataset's median.  It is not a test.
        far = np.abs(diffs - np.median(diffs)) > 5.0
        x = i + rng.uniform(-0.13, 0.13, diffs.size)
        ax.plot(x[~far], diffs[~far], "o", color=colour("KarciFANN"),
                markersize=4.5, alpha=0.85, linestyle="none",
                label="split" if i == 0 else None)
        ax.plot(x[far], diffs[far], "D", color="white",
                markeredgecolor=colour("gradient descent"), markeredgewidth=1.3,
                markersize=6, linestyle="none",
                label=None if labelled_far or not far.any()
                else "selected SGD rate fails on other seeds")
        labelled_far = labelled_far or bool(far.any())
        for j in np.flatnonzero(far):
            ax.annotate(f"lr {rows[dataset][j]['sgd_lr']:g}",
                        (x[j], diffs[j]), textcoords="offset points",
                        xytext=(7, -2), fontsize=6.5)
        ax.plot([i - 0.28, i + 0.28], [np.median(diffs)] * 2,
                color="0.35", linewidth=1.4, zorder=0,
                label="median" if i == 0 else None)

    ax.axhline(0.0, color="0.15", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.7)
    ax.set_yticks([-1, 0, 1, 10, 30])
    ax.set_yticklabels(["-1", "0", "1", "10", "30"])
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([PRETTY[d].split(" (")[0] for d in datasets])
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylabel("power $-$ SGD (pp)")
    ax.set_title("Per-split differences, tuning nested inside each split")
    ax.set_ylim(top=90)                     # room for the outliers' rate labels
    ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "split_differences", FIGURES)
    print("   figures/split_differences.[pdf|svg]")


def main(argv: list[str] | None = None) -> None:
    # argv is explicit so the tests can call main() without pytest's own flags
    # being parsed as this script's.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args([] if argv is None and "pytest" in sys.modules else argv)
    plt = setup()
    FIGURES.mkdir(exist_ok=True)
    print("building figures and statistics")
    summary_and_tests(plt)
    plot_convergence(plt)
    plot_annealing(plt)
    plot_sensitivity(plt)
    plot_gauge_ablation(plt)
    plot_config_ablation(plt)
    plot_factor_spread(plt)
    plot_split_differences(plt)
    print("\ndone")


if __name__ == "__main__":
    main()
