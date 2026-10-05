"""Do the figures plot what the CSVs say, and what their captions claim?

Eyeballing a figure confirms it is not obviously broken; it does not confirm
that the bars carry the numbers the paper quotes.  These tests rebuild each
figure, read the values back out of the Matplotlib artists, and compare them
against the CSV the figure is supposed to be showing -- so a figure drawn from
the wrong column fails rather than being published.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_figures_module():
    spec = importlib.util.spec_from_file_location(
        "_fig_module", ROOT / "experiments" / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_script(name):
    """Import an experiments/ script by name, as a module."""
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "experiments"))
    return module



def _data_name(figures, label):
    """Map a reader-facing label back to the name the CSVs key on."""
    inverse = {v: k for k, v in figures.DISPLAY_NAMES.items()}
    return inverse.get(label, label)


def read(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def bars_of(ax):
    """The bar rectangles of an axes, in draw order.

    ``ax.patches`` also holds spans and annotations, and ``ax.containers`` may
    lead with an ErrorbarContainer when error bars were drawn -- so select the
    BarContainer explicitly rather than by position.
    """
    container = next(c for c in ax.containers if type(c).__name__ == "BarContainer")
    return list(container.patches)


@pytest.fixture(scope="module")
def figures():
    return load_figures_module()


@pytest.fixture
def capture(figures, monkeypatch):
    """Intercept save() so the figure can be inspected before it is closed.

    The real save() closes each figure; holding them open is the whole point
    here, so they are closed at teardown instead.  Without that they accumulate
    across tests and trip Matplotlib's >20-open-figures warning, which
    pytest.ini turns into an error -- which is what happened the moment the
    per-dataset plots went from three datasets to eight.
    """
    drawn = {}

    def fake_save(fig, name, outdir=None):
        drawn[name] = fig
        return []

    monkeypatch.setattr(figures, "save", fake_save)

    # The plotters also write summary CSVs as a side effect.  Left alone, a
    # test run silently rewrites the committed results/ -- including with
    # half-finished data if experiments are mid-flight.  Capture those too.
    written = {}
    monkeypatch.setattr(figures, "write",
                        lambda name, rows: written.__setitem__(name, rows))
    drawn["_written"] = written

    try:
        yield drawn
    finally:
        import matplotlib.pyplot as plt

        for name, fig in drawn.items():
            if name != "_written":
                plt.close(fig)


requires_results = pytest.mark.skipif(
    not (RESULTS / "summary_final.csv").exists(), reason="no results to check")


# ------------------------------------------------------------------- ranks


@requires_results
@pytest.mark.results
def test_critical_difference_diagram_places_methods_at_their_ranks(figures, capture):
    """The diagram must plot the stored ranks, and join the right groups.

    This replaced a bar chart whose lengths invited reading average rank as an
    effect size.  The checks are on positions and cliques rather than lengths:
    each method's label carries its rank, and the joined groups are exactly the
    maximal runs lying within the critical difference.
    """
    plt = figures.setup()
    rows = read("stats_nemenyi")
    friedman = read("stats_friedman")[0]
    expected = {r["method"]: float(r["average_rank"]) for r in rows}
    cd = float(friedman["nemenyi_critical_difference"])

    outcome = {
        "ranks": np.array([expected[m] for m in expected]),
        "blocks": int(float(friedman["blocks_datasets"])),
        "pvalue": float(friedman["pvalue"]),
        "power_floor": float(friedman["power_floor_best_case_p"]),
        "critical_difference": cd,
    }
    figures.plot_ranks(plt, list(expected), outcome, [])
    ax = capture["average_ranks"].axes[0]

    # every method appears once, labelled with the rank the CSV stores
    labels = [t.get_text() for t in ax.texts]
    for method, rank in expected.items():
        shown = f"{figures.display(method)} ({rank:.2f})"
        assert shown in labels, f"{shown!r} missing from the diagram"

    # the critical difference is stated on the figure
    assert any(f"CD = {cd:.2f}" in t for t in labels)

    # cliques: recompute the maximal within-CD runs and compare against the
    # thick horizontal rules the figure drew
    values = np.sort(np.array(list(expected.values())))
    k = values.size
    wanted, i = [], 0
    while i < k:
        j = i
        while j + 1 < k and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in wanted):
            wanted.append((i, j))
        i += 1
    drawn = sorted(
        (round(min(ln.get_xdata()), 2), round(max(ln.get_xdata()), 2))
        for ln in ax.lines if ln.get_linewidth() > 3.0
    )
    assert len(drawn) == len(wanted), f"{len(drawn)} clique bars, expected {len(wanted)}"
    for (a, b), (x0, x1) in zip(wanted, drawn):
        assert x0 == pytest.approx(values[a], abs=0.06)
        assert x1 == pytest.approx(values[b], abs=0.06)

    title = ax.get_title()
    assert "Friedman" in title and f"{outcome['blocks']} datasets" in title


# --------------------------------------------------------- gauge ablation


@requires_results
@pytest.mark.results
def test_gauge_ablation_bars_match_the_summary_csv(figures, capture):
    plt = figures.setup()
    figures.plot_gauge_ablation(plt)
    if "gauge_ablation" not in capture:
        pytest.skip("no gauge-ablation data")
    ax = capture["gauge_ablation"].axes[0]

    expected = sorted(
        round(float(r["mean_test_acc"]) * 100, 6) for r in read("summary_gauge_ablation"))
    plotted = sorted(round(b.get_height(), 6) for b in
                     [b for c in ax.containers if type(c).__name__ == "BarContainer"
                      for b in c.patches] if b.get_height() > 0)
    assert plotted == pytest.approx(expected), "bar heights are not the recorded accuracies"
    assert "accuracy" in ax.get_ylabel().lower()


# ------------------------------------------------------------ convergence


@requires_results
@pytest.mark.results
def test_convergence_curves_are_the_per_epoch_means(figures, capture):
    plt = figures.setup()
    figures.plot_convergence(plt)
    dataset = figures.DEEP[0]
    key = f"convergence_{dataset}"
    if key not in capture:
        pytest.skip("no convergence data")
    accuracy_axis = capture[key].axes[0]

    rows = read(f"{dataset}_e1_curves")
    for line in accuracy_axis.lines:
        method = line.get_label()
        if method.startswith("_"):
            continue
        method = _data_name(figures, method)
        subset = [r for r in rows if r["method"] == method]
        epochs = sorted({int(float(r["epoch"])) for r in subset})
        expected = [np.mean([float(r["val_acc"]) for r in subset
                             if int(float(r["epoch"])) == e]) for e in epochs]
        assert np.allclose(line.get_xdata(), epochs)
        assert np.allclose(line.get_ydata(), expected), f"{method} curve is not the mean"

    assert "accuracy" in accuracy_axis.get_ylabel().lower()
    assert accuracy_axis.get_xlabel() == "epoch"
    assert capture[key].axes[1].get_yscale() == "log", "MSE panel must be log-scaled"


# ------------------------------------------------------------- sensitivity


@requires_results
@pytest.mark.results
def test_sensitivity_points_match_the_sweep(figures, capture):
    plt = figures.setup()
    figures.plot_sensitivity(plt)
    dataset = figures.DEEP[0]
    key = f"sensitivity_{dataset}"
    if key not in capture:
        pytest.skip("no sensitivity data")

    rows = read(f"{dataset}_e3_sensitivity")
    for axis in capture[key].axes:
        sweep = "alpha" if "alpha" in axis.get_xlabel() else "lr"
        subset = [r for r in rows if r["sweep"].endswith(sweep)]
        values = sorted({float(r["value"]) for r in subset})
        expected = [np.mean([float(r["test_acc"]) for r in subset
                             if float(r["value"]) == v]) * 100 for v in values]
        line = axis.lines[0]
        assert np.allclose(line.get_xdata(), values)
        assert np.allclose(line.get_ydata(), expected, atol=1e-9)
        assert axis.get_xscale() == "log", "basin width is only meaningful on a log axis"


# ----------------------------------------------------------- factor spread


@requires_results
@pytest.mark.results
def test_factor_spread_bars_span_the_recorded_percentiles(figures, capture):
    plt = figures.setup()
    figures.plot_factor_spread(plt)
    if "factor_spread" not in capture:
        pytest.skip("no factor-spread data")
    ax = capture["factor_spread"].axes[0]

    rows = [r for d in figures.DATASETS for r in read(f"{d}_e5_factor_spread")]
    expected = sorted((round(float(r["p05"]), 9), round(float(r["p95"]), 9)) for r in rows)
    drawn = []
    for collection in ax.collections:
        for segment in collection.get_segments():
            drawn.append((round(float(segment[:, 1].min()), 9),
                          round(float(segment[:, 1].max()), 9)))
    assert sorted(drawn) == pytest.approx(expected)
    assert ax.get_yscale() == "log"


# ------------------------------------------------------------ p-value grid


@requires_results
@pytest.mark.results
def test_pvalue_matrix_is_the_holm_adjusted_matrix(figures, capture):
    from karcifann.analysis import pairwise_wilcoxon

    plt = figures.setup()
    dataset = figures.DEEP[0]
    rows = read(f"{dataset}_e1_final")
    scores = {}
    for method in figures.METHOD_ORDER:
        values = [float(r["test_acc"]) for r in rows if r["method"] == method]
        if values:
            scores[method] = np.array(values)
    result = pairwise_wilcoxon(scores)
    figures.plot_pvalue_matrix(plt, dataset, result)

    image = capture[f"pvalue_matrix_{dataset}"].axes[0].images[0]
    # get_array() returns a masked array; np.asarray would discard the mask and
    # make the diagonal check vacuous.
    shown = image.get_array()
    expected = np.log10(np.clip(result.adjusted, 1e-4, 1.0))
    assert np.allclose(shown[~np.isnan(expected)], expected[~np.isnan(expected)])
    assert np.all(np.ma.getmaskarray(shown)[np.isnan(expected)]), "diagonal must be masked"


# -------------------------------------------------------------- annealing


@requires_results
@pytest.mark.results
def test_annealing_curves_are_the_recorded_prefactor_means(figures, capture):
    """<Phi> against epoch must be the factor_mean column, not a proxy."""
    plt = figures.setup()
    figures.plot_annealing(plt)
    dataset = figures.DEEP[0]
    key = f"annealing_{dataset}"
    if key not in capture:
        pytest.skip("no annealing data")
    ax = capture[key].axes[0]

    rows = [r for r in read(f"{dataset}_e1_curves") if r.get("factor_mean")]
    assert ax.lines, "no curves drawn"
    for line in ax.lines:
        method = line.get_label()
        if method.startswith("_"):
            continue
        method = _data_name(figures, method)
        subset = [r for r in rows if r["method"] == method]
        epochs = sorted({int(float(r["epoch"])) for r in subset})
        expected = [np.mean([float(r["factor_mean"]) for r in subset
                             if int(float(r["epoch"])) == e]) for e in epochs]
        assert np.allclose(line.get_xdata(), epochs)
        assert np.allclose(line.get_ydata(), expected), f"{method} is not the recorded mean"

    assert ax.get_yscale() == "log", "a multiplicative factor needs a log axis"
    assert "Phi" in ax.get_ylabel() or r"\Phi" in ax.get_ylabel()

    # gradient descent has no prefactor to report, so it must not appear
    assert "gradient descent" not in {line.get_label() for line in ax.lines}


# -------------------------------------------------------- config ablation


@requires_results
@pytest.mark.results
def test_config_ablation_bars_are_the_recorded_means(figures, capture):
    plt = figures.setup()
    figures.plot_config_ablation(plt)
    dataset = figures.DEEP[0]
    key = f"config_ablation_{dataset}"
    if key not in capture:
        pytest.skip("no configuration-ablation data")
    ax = capture[key].axes[0]

    rows = read(f"{dataset}_e4_final")
    expected = {}
    for config in {r["config"] for r in rows}:
        values = [float(r["test_acc"]) for r in rows if r["config"] == config]
        expected[config] = np.mean(values) * 100

    plotted = {label.get_text(): bar.get_width()
               for label, bar in zip(ax.get_yticklabels(), bars_of(ax))}
    assert set(plotted) == set(expected), "the eight configurations are not all shown"
    for config, width in plotted.items():
        assert width == pytest.approx(expected[config]), f"{config} bar is not its mean"

    assert len(expected) == 8, "2 activations x 2 losses x 2 rules"
    assert "accuracy" in ax.get_xlabel().lower()


# ------------------------------------------------------ architecture sketch


def test_architecture_sketches_draw_the_configured_networks(tmp_path, monkeypatch):
    """The diagram must match the network the experiments actually build."""
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "_arch", ROOT / "experiments" / "draw_architecture.py")
    sketch = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = sketch
    spec.loader.exec_module(sketch)

    spec2 = importlib.util.spec_from_file_location(
        "_ra_arch", ROOT / "experiments" / "run_analysis.py")
    run_analysis = importlib.util.module_from_spec(spec2)
    _sys.modules[spec2.name] = run_analysis
    spec2.loader.exec_module(run_analysis)

    plt = sketch.setup()
    for name, spec_tuple in sketch.networks().items():
        n_in, n_hidden, n_out, epochs, title, classes = spec_tuple
        fig, ax = plt.subplots()
        sketch.draw(ax, spec_tuple)

        labels = {t.get_text() for t in ax.texts}
        assert f"{n_in} features" in labels, f"{name}: input width not shown"
        assert f"{n_hidden} units" in labels and f"{n_out} units" in labels
        assert {"input", "hidden", "output"} <= labels
        assert f"{n_in}–{n_hidden}–{n_out}" in ax.get_title()

        # a column is elided with a vertical ellipsis only when it has to be
        assert ("⋮" in labels) == any(w > sketch.SHOWN for w in (n_in, n_hidden, n_out))

        circles = [p for p in ax.patches if type(p).__name__ == "Circle"]
        drawn = min(n_in, sketch.SHOWN) + min(n_hidden, sketch.SHOWN) + min(n_out, sketch.SHOWN)
        assert len(circles) == drawn + 2, "neurons drawn, plus one bias node per weighted layer"
        plt.close(fig)

        configured = run_analysis.DATASETS[name]
        assert n_hidden == configured["hidden"], f"{name}: sketch and experiment disagree"
        assert epochs == configured["epochs"], f"{name}: epoch count drifted"


@pytest.mark.results
def test_every_dataset_has_an_architecture_sketch():
    """The sketches must cover the suite, not the suite as it was.

    They were drawn for three datasets and stayed at three after the suite grew
    to eight, so coverage is pinned to the DATASETS table rather than trusted.

    This inspects committed artefacts, so it skips on a fresh checkout where
    nothing has been drawn yet -- it failed there until a from-scratch run
    caught it, which made `pytest` red before the user had run anything.
    """
    module = load_script("draw_architecture")
    run = load_script("run_analysis")
    figures = ROOT / "figures"
    if not any(figures.glob("*.pdf")):
        pytest.skip("no figures drawn yet; run experiments/draw_architecture.py")
    for name in run.DATASETS:
        for ext in ("pdf", "svg"):
            path = figures / f"architecture_{name}.{ext}"
            assert path.exists(), f"no sketch for {name}: {path} missing"
    assert set(module.TITLES) >= set(run.DATASETS)
    assert set(module.CLASS_NAMES) >= set(run.DATASETS)


def test_architecture_specs_match_the_loaders_and_the_training_config():
    """Widths and epochs are derived; this checks the derivation, not a copy."""
    module = load_script("draw_architecture")
    run = load_script("run_analysis")
    from karcifann import load

    specs = module.networks()
    assert set(specs) == set(run.DATASETS)
    for name, (n_in, hidden, n_out, epochs, _title, _classes) in specs.items():
        x, t = load(name)
        assert (n_in, n_out) == (x.shape[1], t.shape[1]), name
        assert hidden == run.DATASETS[name]["hidden"], name
        assert epochs == run.DATASETS[name]["epochs"], name


@pytest.mark.results
def test_the_split_difference_figure_plots_every_stored_split():
    """Figure 4 must show all forty points, not a filtered view of them.

    The figure exists because the review of 2026-09-23 observed that a mean and
    a median cannot show a skewed distribution, and that a reader given only a
    negative median could mistake it for evidence of inferiority.  A figure
    that quietly dropped the outliers would defeat exactly that purpose, so
    this counts what the generator would draw against what the producer stored,
    and checks that the highlighted splits are the ones the prose names.
    """
    import csv

    module = load_script("make_figures")
    results = ROOT / "results"
    if not any(results.glob("split_robustness_*.csv")):
        pytest.skip("no split-robustness measurements")

    total, highlighted = 0, []
    for dataset in module.SPLIT_DATASETS:
        path = results / f"split_robustness_{dataset}.csv"
        assert path.exists(), f"the figure names {dataset}, which has no results"
        with path.open() as handle:
            diffs = [float(r["mean_difference"]) for r in csv.DictReader(handle)
                     if r["stage"] == "split_summary"]
        assert len(diffs) == 10, f"{dataset}: {len(diffs)} splits, expected 10"
        total += len(diffs)
        median = sorted(diffs)[len(diffs) // 2 - 1 : len(diffs) // 2 + 1]
        median = sum(median) / 2
        highlighted += [(dataset, d) for d in diffs if abs(d - median) > 5.0]

    assert total == 40
    # Section 6.3 names one outlying split apiece on three datasets and none on
    # Letter; the figure's cut must agree with that sentence.
    assert sorted(d for d, _ in highlighted) == ["satimage", "segment", "vehicle"]
    assert all(value > 20.0 for _, value in highlighted)
