"""End-to-end tests of the experiment scripts.

Both bugs that actually escaped in this project lived here rather than in the
library: a keyword collision that crashed the configuration ablation, and a
flag that silently failed to reach the gradient-descent baseline, leaving it
tuned over a narrower grid than the methods it was being compared against.
Neither was reachable from a unit test of `karcifann/`, so the scripts get
their own tests, with a named regression for each.

The scripts are exercised on injected synthetic data so the whole file runs in
a couple of seconds.
"""

import csv
import re
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"


def load_script(name: str):
    """Import an experiment script without running its ``main``."""
    spec = importlib.util.spec_from_file_location(f"_exp_{name}", EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(n=180, features=6, classes=3, seed=0):
    """A small, learnable classification problem."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (classes, features))
    labels = rng.integers(0, classes, n)
    x = centres[labels] + rng.normal(0, 1, (n, features))
    t = np.eye(classes)[labels]
    cut = int(n * 0.6), int(n * 0.8)
    return ((x[: cut[0]], t[: cut[0]]),
            (x[cut[0] : cut[1]], t[cut[0] : cut[1]]),
            (x[cut[1] :], t[cut[1] :]))


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Redirect every Table and every script's RESULTS at a temp directory."""
    import karcifann.records as records

    target = tmp_path / "results"
    target.mkdir()
    monkeypatch.setattr(records, "RESULTS_DIR", target)
    return target


# =====================================================================
# run_analysis.py
# =====================================================================


@pytest.fixture(scope="module")
def run_analysis():
    return load_script("run_analysis")


@pytest.fixture
def bench(run_analysis, results_dir, monkeypatch):
    monkeypatch.setattr(run_analysis, "RESULTS", results_dir)
    monkeypatch.setattr(run_analysis, "SEEDS", [100, 101, 102])
    # run_analysis reads its tuned winners from the gauge-family sweep, so the
    # fixture provides one rather than a hand-written optimizer table.
    (results_dir / "gauge_family_tiny.csv").write_text(
        "stage,method,shape,scale\n"
        "best,gradient descent,,0.5\n"
        "best,power,-0.2,0.5\n"
        "best,exp,3,0.5\n"
    )
    return run_analysis.Bench(
        "tiny", data=synthetic(), config=dict(epochs=3, batch_size=32, hidden=5)
    )


def test_bench_accepts_injected_data_and_configuration(bench):
    assert bench.layers == [6, 5, 3]
    assert bench.fit["epochs"] == 3


def test_run_returns_every_reported_metric(bench, run_analysis):
    record, history, net = bench.run(lambda: run_analysis.SGD(lr=0.5), 100, keep_history=True)
    assert set(record) == {"val_acc", "test_acc", "test_precision", "test_recall",
                           "test_f1", "train_mse", "diverged"}
    assert all(0.0 <= record[k] <= 1.0 for k in ("val_acc", "test_acc", "test_f1"))
    assert history.epochs == 3 and net is not None


def test_net_accepts_configuration_overrides(bench):
    """Regression: the configuration ablation used to crash here.

    ``Bench.net`` builds a dict of defaults and then applies the caller's
    keywords.  Passing them into the ``dict(...)`` call instead raised
    ``TypeError: dict() got multiple values for keyword argument 'activation'``
    the moment E4 asked for anything other than a sigmoid.
    """
    default = bench.net(100)
    assert default.activations[0].name == "sigmoid"

    overridden = bench.net(100, activation="relu", output_activation="softmax",
                           loss="categorical_crossentropy", weight_init="he")
    assert overridden.activations[0].name == "relu"
    assert overridden.activations[-1].name == "softmax"
    assert overridden.loss.name == "categorical_crossentropy"


@pytest.mark.parametrize(
    "experiment,expected",
    [
        ("e1_convergence", ["tiny_e1_curves", "tiny_e1_final"]),
        ("e2_gauge_ablation", ["tiny_e2_tuning", "tiny_e2_final"]),
        ("e4_config_ablation", ["tiny_e4_tuning", "tiny_e4_final"]),
        ("e5_factor_spread", ["tiny_e5_factor_spread"]),
    ],
)
def test_each_experiment_writes_its_csvs(bench, run_analysis, results_dir, experiment, expected):
    getattr(run_analysis, experiment)(bench)
    for name in expected:
        path = results_dir / f"{name}.csv"
        assert path.exists(), f"{name}.csv was not written"
        rows = read_csv(path)
        assert rows, f"{name}.csv is empty"
        assert all(set(r) == set(rows[0]) for r in rows), "ragged rows"


def test_sensitivity_sweeps_both_hyperparameters(bench, run_analysis, results_dir):
    run_analysis.e3_sensitivity(bench, seeds=1)
    rows = read_csv(results_dir / "tiny_e3_sensitivity.csv")
    sweeps = {r["sweep"] for r in rows}
    assert sweeps == {"gradient descent: lr", "KarciFANN: alpha"}
    assert len({r["value"] for r in rows if "alpha" in r["sweep"]}) > 5


def test_convergence_curves_have_one_row_per_epoch_per_seed(bench, run_analysis, results_dir):
    run_analysis.e1_convergence(bench)
    rows = read_csv(results_dir / "tiny_e1_curves.csv")
    methods = {r["method"] for r in rows}
    assert len(rows) == len(methods) * 3 * 3          # methods x seeds x epochs
    assert {int(r["epoch"]) for r in rows} == {1, 2, 3}


def test_gauge_ablation_covers_all_three_modes(bench, run_analysis, results_dir):
    run_analysis.e2_gauge_ablation(bench)
    rows = read_csv(results_dir / "tiny_e2_final.csv")
    assert {r["mode"] for r in rows} == {"full", "error", "weight"}


def test_config_ablation_covers_the_full_factorial(bench, run_analysis, results_dir):
    run_analysis.e4_config_ablation(bench)
    rows = read_csv(results_dir / "tiny_e4_final.csv")
    assert len({r["config"] for r in rows}) == 8      # 2 activations x 2 losses x 2 rules
    assert {r["activation"] for r in rows} == {"sigmoid", "relu"}
    assert {r["rule"] for r in rows} == {"gd", "adam"}


# =====================================================================
# make_figures.py
# =====================================================================


@pytest.mark.slow
def test_make_figures_produces_statistics_and_both_vector_formats(
    bench, run_analysis, results_dir, tmp_path, monkeypatch
):
    run_analysis.e1_convergence(bench)
    run_analysis.e2_gauge_ablation(bench)
    run_analysis.e3_sensitivity(bench, seeds=1)
    run_analysis.e5_factor_spread(bench)

    figures = load_script("make_figures")
    figure_dir = tmp_path / "figures"
    monkeypatch.setattr(figures, "RESULTS", results_dir)
    monkeypatch.setattr(figures, "FIGURES", figure_dir)
    # DATASETS is every benchmark (E1 only); DEEP is the subset carrying the
    # mechanism studies E2-E4, so both have to point at the fixture.
    monkeypatch.setattr(figures, "DATASETS", ["tiny"])
    monkeypatch.setattr(figures, "DEEP", ["tiny"])
    monkeypatch.setitem(figures.PRETTY, "tiny", "Tiny (3 classes)")
    figures.main()

    for stem in ("convergence_tiny", "sensitivity_tiny", "gauge_ablation",
                 "factor_spread", "pvalue_matrix_tiny"):
        pdf, svg = figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.svg"
        assert pdf.exists() and svg.exists(), f"{stem} missing a format"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert b"<svg" in svg.read_bytes()

    for stem in ("summary_final", "stats_wilcoxon", "stats_tolerance"):
        assert (results_dir / f"{stem}.csv").exists()

    wilcoxon = read_csv(results_dir / "stats_wilcoxon.csv")
    assert wilcoxon and {"p_raw", "p_holm", "cohens_d", "holm_floor"} <= set(wilcoxon[0])
    assert all(float(r["p_holm"]) >= float(r["p_raw"]) - 1e-12 for r in wilcoxon)


def test_mechanism_figures_cover_every_dataset():
    """Every experiment now runs on the whole suite, so every figure should.

    This test used to assert DEEP had exactly three entries -- it was written
    when E2-E4 ran on three datasets, and it went on passing after the suite
    grew to eight, pinning the figures to a subset instead of catching that
    they had fallen behind.  Coverage is per-dataset data now: the plotters
    skip whatever has no CSV.
    """
    figures = load_script("make_figures")
    run_analysis = load_script("run_analysis")
    assert set(figures.DEEP) == set(figures.DATASETS)
    assert set(figures.DATASETS) == set(run_analysis.DATASETS)
    assert set(figures.PRETTY) == set(figures.DATASETS)


def test_make_figures_survives_missing_datasets(tmp_path, monkeypatch):
    """A partial result set must not crash the figure pass."""
    figures = load_script("make_figures")
    monkeypatch.setattr(figures, "RESULTS", tmp_path / "empty")
    monkeypatch.setattr(figures, "FIGURES", tmp_path / "figures")
    (tmp_path / "empty").mkdir()
    figures.main()          # no data at all: should be a no-op, not an error


# =====================================================================
# draw_architecture.py
# =====================================================================


def test_architecture_sketches_match_the_configured_networks(tmp_path, monkeypatch):
    sketch = load_script("draw_architecture")
    monkeypatch.setattr(sketch, "FIGURES", tmp_path)
    sketch.main()

    run_analysis = load_script("run_analysis")
    for name in run_analysis.DATASETS:
        for suffix in ("pdf", "svg"):
            assert (tmp_path / f"architecture_{name}.{suffix}").exists()
    assert (tmp_path / "architectures.pdf").exists()

    for name, (n_in, n_hidden, n_out, epochs, *_ ) in sketch.networks().items():
        configured = run_analysis.DATASETS[name]
        assert n_hidden == configured["hidden"], f"{name}: hidden width drifted"
        assert epochs == configured["epochs"], f"{name}: epoch count drifted"


# =====================================================================
# gauge_family.py  /  the tuning-grid regression
# =====================================================================


@pytest.fixture(scope="module")
def gauge_family():
    return load_script("gauge_family")


def test_scales_flag_reaches_the_gradient_descent_baseline(
    gauge_family, results_dir, monkeypatch
):
    """Regression: a widened grid used to apply to the gauges but not the baseline.

    The ``--scales`` option was threaded into the gauge loops but a failed
    string replacement left the baseline reading the module-level default.  On
    Letter that tuned gradient descent only to 32 while every gauge was tuned
    to 256, and produced an apparent 1.3 pp win for KarciFANN that vanished the
    moment the baseline got the same grid.
    """
    from karcifann import Table

    monkeypatch.setattr(gauge_family, "GAUGE_FAMILIES",
                        {"power": (gauge_family.GaugedDescent.__init__.__globals__["PowerGauge"],
                                   [-0.4])})
    bench = gauge_family.Bench(
        synthetic(), hidden=5, fit=dict(epochs=2, batch_size=32, shuffle=True)
    )
    table = Table("gauge_grid", results_dir)
    custom = [11.0, 13.0, 17.0]
    gauge_family.tune(bench, seed=100, table=table, scales=custom)

    baseline_scales = {row["scale"] for row in table.rows
                       if row["method"] == "gradient descent" and row["stage"] == "tune"}
    gauge_scales = {row["scale"] for row in table.rows
                    if row["method"] == "power" and row["stage"] == "tune"}
    assert baseline_scales == set(custom), "the baseline ignored --scales"
    assert baseline_scales == gauge_scales, "baseline and gauges tuned over different grids"


def test_gauge_family_records_a_grid_edge_warning(gauge_family, results_dir, monkeypatch):
    """A winner on the edge of the grid must be flagged, not silently reported."""
    from karcifann import Table

    bench = gauge_family.Bench(
        synthetic(), hidden=5, fit=dict(epochs=2, batch_size=32, shuffle=True)
    )
    table = Table("gauge_edge", results_dir)
    winners = gauge_family.tune(bench, seed=100, table=table, scales=[1.0, 2.0])
    best = winners["gradient descent"]
    assert best[2][0] in (1.0, 2.0)        # with a two-point grid it must be an edge


# =====================================================================
# the remaining scripts, exercised end to end on a tiny budget
# =====================================================================


@pytest.mark.slow
def test_xor_experiment_runs_and_records_every_cell(results_dir, monkeypatch, capsys):
    script = load_script("xor_experiment")
    monkeypatch.setattr(sys, "argv", ["xor", "--epochs", "40", "--alphas", "0.8", "1.0"])
    script.main()

    rows = read_csv(results_dir / "xor_experiment.csv")
    assert len(rows) == 4                                  # 2 coefficients x 2 methods
    assert {r["method"] for r in rows} == {"KarciFANN", "classical ANN"}
    assert all(f"output_{i}" in rows[0] for i in range(4))

    at_one = {r["method"]: r["mse"] for r in rows if float(r["coefficient"]) == 1.0}
    assert at_one["KarciFANN"] == at_one["classical ANN"], \
        "alpha = 1 must reproduce classical ANN exactly"


@pytest.mark.slow
def test_headroom_records_tuning_seeds_and_summary(results_dir, monkeypatch):
    script = load_script("headroom")
    monkeypatch.setattr(script, "load", lambda name: synthetic(n=240))
    monkeypatch.setattr(script, "DEFAULTS", {"tiny": dict(epochs=2, batch_size=32)})
    for cfg in script.CONFIGS.values():
        cfg["grid"] = [(cfg["grid"][0][0], cfg["grid"][0][1], cfg["grid"][0][2][:2])]
        cfg["hidden"] = [4]
    monkeypatch.setattr(sys, "argv", ["headroom", "--dataset", "tiny", "--seeds", "2"])
    script.main()

    rows = read_csv(results_dir / "headroom_tiny.csv")
    stages = {r["stage"] for r in rows}
    assert stages == {"tune", "seed", "summary"}
    summary = [r for r in rows if r["stage"] == "summary"]
    assert len(summary) == len(script.CONFIGS)             # one row per configuration
    assert all(r["grid_edge"] in ("0", "1") for r in summary)


@pytest.mark.slow
def test_fod_comparison_records_prefactors_and_sweep(results_dir, monkeypatch):
    script = load_script("fod_comparison")
    monkeypatch.setattr(script, "digits", lambda: _flat(synthetic(n=240, features=64, classes=10)))
    monkeypatch.setattr(sys, "argv",
                        ["fod", "--epochs", "2", "--skip-tuning"])
    script.main()

    rows = read_csv(results_dir / "fod_comparison_digits.csv")
    methods = {r["method"] for r in rows if r["stage"] == "prefactor"}
    assert methods == {"Karci", "Caputo-Fabrizio", "Caputo"}

    at_one = [r for r in rows if r["stage"] == "prefactor" and float(r["coefficient"]) == 1.0]
    assert all(float(r["prefactor"]) == pytest.approx(1.0) for r in at_one), \
        "every fractional prefactor must be exactly 1 at alpha = 1"

    sweep = [r for r in rows if r["stage"] == "sweep"]
    assert sweep and {"val_acc", "mean_prefactor"} <= set(sweep[0])


def _flat(split):
    """Re-join a three-way split into the (x, t) pair a loader would return."""
    xs = np.vstack([part[0] for part in split])
    ts = np.vstack([part[1] for part in split])
    return xs, ts


def test_every_experiment_script_is_importable_and_has_a_main():
    """A syntax error or a bad import in any script should fail here, not in a
    three-hour batch run."""
    scripts = sorted(p.stem for p in EXPERIMENTS.glob("*.py"))
    assert len(scripts) >= 9
    for name in scripts:
        module = load_script(name)
        assert callable(getattr(module, "main", None)), f"{name} has no main()"


def test_every_script_writes_into_the_results_directory():
    """Tables must land in results/, not beside whatever the cwd happens to be."""
    from karcifann.records import RESULTS_DIR

    assert RESULTS_DIR == Path("results")
    for name in ("xor_experiment", "digits_experiment", "gauge_family",
                 "headroom", "fod_comparison", "chained_vs_terminal",
                 "mnist_experiment"):
        source = (EXPERIMENTS / f"{name}.py").read_text()
        assert "Table(" in source, f"{name} records no CSV"


def test_every_registered_dataset_has_a_matched_update_budget():
    """Epochs are set so each dataset sees a comparable number of updates.

    Matching *epochs* instead would give the large sets many times the
    training of the small ones and make any cross-dataset ranking meaningless.
    """
    run_analysis = load_script("run_analysis")
    sizes = {"mnist": 55000, "dry_bean": 9527, "letter": 14000, "vehicle": 592,
             "satimage": 4501, "segment": 1617, "optdigits": 3934, "pendigits": 7694}
    budgets = {}
    for name, cfg in run_analysis.DATASETS.items():
        per_epoch = max(1, sizes[name] // cfg["batch_size"])
        budgets[name] = per_epoch * cfg["epochs"]
    # Letter was fixed at 60 epochs before the budget was standardised and
    # carries ~50% more updates than the rest; everything else is within 5%.
    assert max(budgets.values()) / min(budgets.values()) < 1.6, budgets
    without_letter = {k: v for k, v in budgets.items() if k != "letter"}
    assert max(without_letter.values()) / min(without_letter.values()) < 1.12


def test_tuned_settings_are_read_back_from_the_sweep(tmp_path, monkeypatch):
    """run_analysis reads its winners from the gauge-family CSV, not a literal.

    One source of truth: a hand-copied table drifts the moment a sweep is
    re-run with a different grid.
    """
    run_analysis = load_script("run_analysis")
    monkeypatch.setattr(run_analysis, "RESULTS", tmp_path)
    (tmp_path / "gauge_family_fake.csv").write_text(
        "stage,method,shape,scale\n"
        "best,gradient descent,,7\n"
        "best,power,-0.3,5\n"
        "best,exp,4,9\n"
    )
    tuned = run_analysis.tuned_optimizers("fake")
    assert list(tuned) == ["gradient descent", "KarciFANN", "exp"]
    assert tuned["gradient descent"]().lr == 7.0
    karci = tuned["KarciFANN"]()
    assert karci.gauge.shape == pytest.approx(-0.3) and karci.scale == 5.0


def test_a_dataset_with_no_sweep_fails_loudly(tmp_path, monkeypatch):
    run_analysis = load_script("run_analysis")
    monkeypatch.setattr(run_analysis, "RESULTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="run gauge_family"):
        run_analysis.tuned_optimizers("never_tuned")


def test_there_is_no_hard_coded_tuning_fallback():
    """One source of truth: the sweep CSV, never a literal that can drift."""
    run_analysis = load_script("run_analysis")
    assert not hasattr(run_analysis, "TUNED")


@pytest.mark.parametrize("script", ["run_analysis", "gauge_family", "headroom"])
def test_dataset_loading_resolves_every_name_it_uses(script, monkeypatch):
    """A name used only inside a function fails at call time, not import time.

    ``load_dataset`` was once referenced in three scripts but imported in only
    one; every script imported cleanly and then died the moment it was given a
    dataset it had to dispatch on.  Calling ``load`` with the actual fetch
    stubbed out exercises the lookup without downloading anything.
    """
    module = load_script(script)
    monkeypatch.setattr(module, "train_val_test_split",
                        lambda x, t, *a, **k: ((x, t), (x, t), (x, t)))
    import karcifann.datasets as datasets

    fake = (np.zeros((8, 3)), np.eye(2)[[0, 1] * 4])
    monkeypatch.setitem(datasets.TABULAR, "vehicle", lambda scale=True: fake)
    (xtr, _), _, _ = module.load("vehicle")
    assert xtr.shape == (8, 3)


@pytest.mark.results
def test_readme_tables_match_the_results_they_report():
    """Prose must not drift from the data behind it.

    Three separate inconsistencies in this project came from numbers typed into
    the README from one run and left there when the experiment was re-run.  The
    numeric tables are now generated from ``results/`` and this fails if the
    file on disk is not what the generator would produce.
    """
    import subprocess

    if not any((ROOT / "results").glob("*.csv")):
        pytest.skip("no results to check")
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "update_readme.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_generated_table_has_a_home_in_the_readme():
    """A generator with no marker silently produces nothing."""
    module = load_script("update_readme")
    readme = (ROOT / "README.md").read_text()
    for name in module.TABLES:
        assert f"<!-- table:{name} -->" in readme, f"{name} has no marker"


@pytest.mark.slow
def test_training_is_bit_identical_across_blas_thread_counts():
    """Results must not depend on how many threads BLAS happened to use.

    The experiment queues ran at 1, 2 and 4 threads at different times.  If
    float summation order varied with the thread count, a 1e-15 divergence
    could amplify over hundreds of epochs and the runs would not be comparable
    -- which would also make the bit-identical spot-checks of stored results
    meaningless.  Checked in subprocesses because the thread count is fixed
    when the BLAS library loads.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import numpy as np
        from karcifann import MLP, GaugedDescent, PowerGauge, one_hot, train
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (256, 24))
        t = one_hot(rng.integers(0, 4, 256), 4)
        net = MLP([24, 40, 4], weight_init="glorot", seed=7)
        history = train(net, GaugedDescent(PowerGauge(-0.3), scale=2.0), x, t,
                        epochs=120, batch_size=32, shuffle=True, seed=7)
        checksum = sum(float(np.sum(np.abs(p))) for p in net.parameters)
        print(repr(history.loss[-1]), repr(checksum))
    """)

    outputs = []
    for threads in ("1", "4"):
        env = {"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
               "MKL_NUM_THREADS": threads, "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1], (
        f"training is thread-dependent:\n  1 thread : {outputs[0]}\n  4 threads: {outputs[1]}")


@pytest.mark.results
def test_the_reported_verdicts_use_the_standard_the_prose_declares():
    """The verdict column must be the standard the text says it is.

    When the primary test changed from Wilcoxon to a permutation test, the
    prose was updated and the table that computes the verdicts was not, so the
    file declared one standard and applied another.
    """
    import csv

    results = ROOT / "results"
    if not (results / "stats_wilcoxon.csv").exists():
        pytest.skip("no results to check")

    module = load_script("update_readme")
    rendered = module.table_headline()
    with (results / "stats_wilcoxon.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    expected = 0
    for row in rows:
        if {row["method_a"], row["method_b"]} != {"KarciFANN", "gradient descent"}:
            continue
        difference = float(row["difference"]) * (-1 if row["method_a"] == "gradient descent" else 1)
        if int(row["robust"]) and difference > 0:
            expected += 1

    assert f"**{expected} win" in rendered, (
        "the win count is not computed from the `robust` column that §8.2 declares")
    readme = (ROOT / "README.md").read_text()
    assert "study-wide permutation standard" in readme


def test_prose_number_checker_catches_a_stale_quote():
    """The checker must actually fire, not just return an empty list.

    This defect class -- narrative written for an earlier run left beside a
    regenerated table -- recurred repeatedly in this project, so the guard
    against it is itself worth guarding.
    """
    module = load_script("update_readme")
    backed = (
        "Accuracy reached 91.11 %.\n\n"
        "<!-- table:accuracy -->\n"
        "| a | b |\n|---|---|\n| x | 91.11 |\n"
        "<!-- /table:accuracy -->\n"
    )
    assert module.check_prose_numbers(backed) == []

    stale = backed.replace("Accuracy reached 91.11 %", "Accuracy reached 88.42 %")
    problems = module.check_prose_numbers(stale)
    assert len(problems) == 1 and "88.42" in problems[0]


def test_prose_number_checker_ignores_fenced_code_and_declared_numbers():
    module = load_script("update_readme")
    text = (
        "Run it with `--tol 0.55`:\n\n```bash\npython3 run.py --lr 0.99\n```\n\n"
        "<!-- table:accuracy -->\n| a |\n|---|\n| 1.00 |\n<!-- /table:accuracy -->\n"
    )
    assert [p for p in module.check_prose_numbers(text) if "0.99" in p] == []
    assert any("0.55" in p for p in module.check_prose_numbers(text))

    module.DERIVED_NUMBERS["0.55"] = "declared for this test"
    try:
        assert module.check_prose_numbers(text) == []
    finally:
        del module.DERIVED_NUMBERS["0.55"]




# =====================================================================
# run_all.sh  /  the reproduction contract
# =====================================================================


def _run_all_text():
    return (EXPERIMENTS / "run_all.sh").read_text()


def test_run_all_covers_every_dataset_completely():
    """The reproduction script must produce the results that are committed.

    It had grown into a log of repair passes: five datasets were still run with
    `--only e1 e5` and none of them had a headroom line, so following the
    script produced a strictly smaller result set than results/ holds -- with
    nothing to say so.
    """
    text = _run_all_text()
    run_analysis = load_script("run_analysis")

    for dataset in run_analysis.DATASETS:
        assert f"full {dataset} " in text, (
            f"{dataset} has no full pass in run_all.sh")

    # `--only` restricts a run to part of E1-E5; the committed results need all
    # of it, so no dataset may be reproduced through a partial pass.
    assert "--only" not in text, (
        "run_all.sh restricts a run with --only; results/ carries the full E1-E5")


def test_run_all_runs_every_experiment_script():
    """A script with no line here is a result nothing reproduces."""
    text = _run_all_text()
    skip = {"run_all.sh", "update_readme.py", "make_figures.py",
            "draw_architecture.py", "run_analysis.py", "gauge_family.py",
            "headroom.py"}
    for path in sorted(EXPERIMENTS.glob("*.py")):
        if path.name in skip:
            continue
        assert path.name in text, f"{path.name} is never run by run_all.sh"
    # the three driven through the full() helper
    for name in ("gauge_family.py", "run_analysis.py", "headroom.py"):
        assert name in text, f"{name} missing from run_all.sh"


def test_run_all_gauge_grids_match_the_committed_sweeps():
    """The grid in the script must be the grid the results were made on."""
    import csv as _csv

    text = _run_all_text()
    results = ROOT / "results"
    for path in sorted(results.glob("gauge_family_*.csv")):
        dataset = path.name[len("gauge_family_"):-len(".csv")]
        with path.open() as handle:
            scales = sorted({float(r["scale"]) for r in _csv.DictReader(handle)
                             if r.get("stage") == "tune" and r.get("scale")})
        if not scales:
            continue
        variable = "W_VEHICLE" if dataset == "vehicle" else "W"
        declared = re.search(rf'^{variable}="([^"]+)"', text, re.M)
        assert declared, f"{variable} not defined in run_all.sh"
        expected = sorted(float(v) for v in declared.group(1).split())
        assert scales == pytest.approx(expected), (
            f"{dataset}: results swept {scales}, run_all.sh declares {expected}")


def test_the_readme_invokes_every_queue_run_all_defines():
    """The documented command must drive the whole script.

    The README said `for q in 1 2 3 4` long after the script had grown past
    four queues, so the documented reproduction skipped most of the suite.
    """
    import re as _re

    script = _run_all_text()
    defined = {m.group(1) for m in _re.finditer(r"^(\w+)\)\s", script, _re.M)}
    defined -= {"figures"}
    readme = (ROOT / "README.md").read_text()
    invoked = _re.search(r"for q in ([\w ]+); do bash experiments/run_all\.sh", readme)
    assert invoked, "the README no longer shows how to run the queues"
    assert set(invoked.group(1).split()) == defined, (
        f"README runs {sorted(set(invoked.group(1).split()))}, "
        f"run_all.sh defines {sorted(defined)}")


def test_the_install_line_names_every_third_party_import():
    """A missing dependency makes the documented setup fail on first use.

    scipy and matplotlib were imported inside the package -- by the statistics,
    the grid-edge audit and every figure -- while the README's install line
    listed only numpy, scikit-learn and pytest.
    """
    import ast
    import sys

    distributions = {"sklearn": "scikit-learn"}
    stdlib = set(sys.stdlib_module_names)
    local = {"karcifann"} | {p.stem for p in EXPERIMENTS.glob("*.py")}

    imported = set()
    for folder in ("karcifann", "experiments", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                imported |= {n for n in names if n and n not in stdlib and n not in local}

    required = {distributions.get(name, name) for name in imported}
    readme = (ROOT / "README.md").read_text()
    line = re.search(r"^pip install (.+)$", readme, re.M)
    assert line, "the README no longer shows an install command"
    declared = set(line.group(1).split())
    assert required <= declared, (
        f"imported but not in the install line: {sorted(required - declared)}")


def test_section_9_names_every_test_file():
    """The tests section must describe the suite that exists.

    It omitted test_caputo_fabrizio.py -- a whole subject area §5 is built on --
    along with test_figures.py and conftest.py, so the described coverage was
    narrower than the real one and nothing noticed.
    """
    readme = (ROOT / "README.md").read_text()
    section = readme[readme.index("## 9. Tests"):readme.index("## 10.")]
    missing = sorted(path.name for path in (ROOT / "tests").glob("*.py")
                     if path.name not in section)
    assert not missing, f"test files not described in section 9: {missing}"


# =====================================================================
# loss_scaling.py  /  the rescaling identity
# =====================================================================


def test_scaled_loss_scales_both_value_and_gradient():
    """If only the value carried the factor the identity would be wrong."""
    import numpy as np

    module = load_script("loss_scaling")
    base, scaled = module.get_loss("mse"), module.ScaledLoss("mse", 7.0)
    y = np.array([[0.2, 0.8], [0.6, 0.1]])
    t = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert scaled.value(y, t) == pytest.approx(7.0 * base.value(y, t))
    assert scaled.gradient(y, t) == pytest.approx(7.0 * base.gradient(y, t))


@pytest.mark.parametrize("alpha", [0.4, 0.8, 1.0, 1.3])
@pytest.mark.parametrize("c", [0.01, 0.5, 2.0, 100.0])
def test_rescaling_the_loss_scales_the_update_by_c_to_the_alpha(alpha, c):
    """dW(cJ) = c**alpha . dW(J), the identity the manuscript derives.

    The prefactor contributes c**(alpha-1) and the ordinary gradient another
    c, so the exponent is alpha rather than alpha-1.  Getting this wrong was
    the error behind two earlier versions of the loss-scale section, so it is
    checked here rather than argued.

    The increment is taken as the optimiser forms it, not as `before - after`.
    The reconstruction loses about eleven digits to cancellation, which is why
    it needed `rel=1e-8` here and why the published table once reported 1e-8
    agreement for an identity that holds to 1e-12; the direct quantity is
    checked far tighter.
    """
    import numpy as np

    module = load_script("loss_scaling")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6))
    t = np.eye(3)[rng.integers(0, 3, 32)]

    base, base_sub, _ = module.one_update(x, t, alpha, 1.0, seed=3)
    scaled, scaled_sub, _ = module.one_update(x, t, alpha, c, seed=3)
    for b, s in zip(base, scaled):
        assert s == pytest.approx(c ** alpha * b, rel=1e-11, abs=1e-18)

    # And the reconstruction agrees, but only to the looser tolerance -- the
    # gap between the two is the measurement artefact Section 6.9 now reports.
    for b, s in zip(base_sub, scaled_sub):
        assert s == pytest.approx(c ** alpha * b, rel=1e-6, abs=1e-18)


def test_split_robustness_trains_the_same_way_as_the_main_sweep():
    """The nested-split study must not quietly use a different protocol.

    `train()` defaults to shuffle=False; `gauge_family.py` overrides it to True
    and every stored result was produced that way.  The first version of
    `split_robustness.py` forgot the override, so 5040 runs trained without
    shuffling and disagreed with the main study on the same split.  Comparing
    the two DEFAULTS tables and the shuffle override is cheap; rerunning 950
    epochs to find out is not.
    """
    sweep = load_script("gauge_family")
    nested = load_script("split_robustness")

    shared = set(sweep.DEFAULTS) & set(nested.DEFAULTS)
    assert shared, "the two scripts share no datasets"
    for name in shared:
        assert sweep.DEFAULTS[name] == nested.DEFAULTS[name], (
            f"{name}: {sweep.DEFAULTS[name]} vs {nested.DEFAULTS[name]}")

    source = (EXPERIMENTS / "split_robustness.py").read_text()
    assert "shuffle=True" in source, (
        "split_robustness.py must set shuffle=True, as gauge_family.py does")


def test_sync_publishes_the_code_but_not_the_source_articles():
    """sync.sh must never ship the third-party PDFs in the project root.

    Those are the papers this work analyses, under their publishers' copyright,
    and they sit beside the code.  The script therefore publishes a whitelist
    of directories rather than excluding a blacklist, so a newly added file at
    the top level is left behind rather than shipped.  This checks the
    whitelist has not quietly grown a wildcard that would sweep them in.
    """
    script = ROOT / "sync.sh"
    if not script.exists():
        pytest.skip("no sync script")
    source = script.read_text()

    dirs = re.search(r'^DIRS="([^"]*)"', source, re.M)
    files = re.search(r'^FILES="([^"]*)"', source, re.M)
    assert dirs and files, "sync.sh no longer declares DIRS and FILES"

    published_dirs = dirs.group(1).split()
    assert "." not in published_dirs and "*" not in published_dirs, \
        "sync.sh publishes the whole tree; the source PDFs would go with it"

    # Every top-level PDF is a source article and must be outside the whitelist.
    named = set(files.group(1).split())
    strays = [p.name for p in ROOT.glob("*.pdf") if p.name in named]
    assert not strays, f"sync.sh names source articles explicitly: {strays}"

    assert "paper" not in published_dirs, "the manuscript must stay outside the release"


@pytest.mark.results
def test_multiseed_tuning_reproduces_the_parent_study_and_splits_as_claimed():
    """Section 6.4 rests on two facts about the stored runs; check both.

    First, the single-seed arm must reproduce `split_robustness.py` exactly.
    The script asserts this at run time and refuses to write a table otherwise,
    but that guarantee dies with the process -- if either study is ever
    regenerated, the two can drift apart and the paired comparison silently
    stops being paired.

    Second, the paper's claim is specifically that the *rate selection* changed
    on the outlying splits and did not on the controls.  That is the whole
    argument, so it is pinned here rather than left to prose.
    """
    import csv

    results = ROOT / "results"
    files = sorted(results.glob("multiseed_tuning_*.csv"))
    if not files:
        pytest.skip("no multi-seed tuning measurements")

    seen = {"outlier": 0, "control": 0}
    for path in files:
        dataset = path.stem.replace("multiseed_tuning_", "")
        parent = results / f"split_robustness_{dataset}.csv"
        assert parent.exists(), f"{path.name} has no parent study to reproduce"
        with parent.open() as handle:
            stored = {int(float(r["split"])): float(r["mean_difference"])
                      for r in csv.DictReader(handle) if r["stage"] == "split_summary"}
        with path.open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["stage"] == "split_summary"]

        for row in rows:
            split = int(float(row["split"]))
            assert split in stored, f"{dataset} split {split} is not in the parent study"
            assert float(row["single_mean"]) == pytest.approx(stored[split], abs=1e-9), (
                f"{dataset} split {split}: single-seed tuning gives "
                f"{float(row['single_mean']):+.4f}, parent stored "
                f"{stored[split]:+.4f} -- the two studies have drifted apart")
            assert float(row["stored_mean"]) == pytest.approx(stored[split], abs=1e-9)

            kind = row["kind"]
            seen[kind] += 1
            changed = float(row["single_sgd_lr"]) != float(row["multi_sgd_lr"])
            if kind == "outlier":
                assert changed, (
                    f"{dataset} split {split} is an outlier whose rate did not "
                    "change; Section 6.4 claims every outlier's rate changed")
                assert abs(float(row["multi_mean"]) - float(row["single_mean"])) > 10.0
            else:
                assert not changed, (
                    f"{dataset} split {split} is a control whose rate changed "
                    f"from {float(row['single_sgd_lr']):g} to "
                    f"{float(row['multi_sgd_lr']):g}; Section 6.4 claims no "
                    "control's rate moved")
                assert abs(float(row["multi_mean"]) - float(row["single_mean"])) < 10.0

    assert seen["outlier"] == seen["control"] == 3, (
        f"expected three outliers and three controls, found {seen}")


@pytest.mark.results
def test_stored_sweeps_match_the_reproduction_protocol():
    """Check saved confirmation seeds, multiplier grids and MNIST slice boundaries."""
    import csv
    import re

    root = ROOT
    sweeps = sorted((root / "results").glob("gauge_family_*.csv"))
    if not sweeps:
        pytest.skip("no gauge-family sweeps")
    for path in sweeps:
        with path.open() as handle:
            seeds = sorted({int(float(r["seed"])) for r in csv.DictReader(handle)
                            if r.get("stage") == "seed" and r.get("seed")})
        assert seeds == list(range(100, 116)), f"{path.name} confirmed on {seeds[:3]}..."

    # The stored grids have 9 multipliers, 13 on Vehicle. The default is
    # narrower; run_all.sh passes the published grid, so check the entry point
    # rather than the default, which is what a reproduction would use.
    run_all = (root / "experiments" / "run_all.sh").read_text()
    common = re.search(r'^W="([^"]+)"', run_all, re.M)
    vehicle = re.search(r'^W_VEHICLE="([^"]+)"', run_all, re.M)
    assert common and vehicle, "run_all.sh no longer defines the multiplier grids"
    assert len(common.group(1).split()) == 9, "the common grid is no longer 9 points"
    assert len(vehicle.group(1).split()) == 13, "Vehicle's grid is no longer 13 points"
    for path in sweeps:
        with path.open() as handle:
            scales = {float(r["scale"]) for r in csv.DictReader(handle)
                      if r.get("stage") == "tune" and r.get("scale")}
        expected = 13 if "vehicle" in path.name else 9
        assert len(scales) == expected, (
            f"{path.name} swept {len(scales)} multipliers, the reproduction protocol specifies {expected}")

    # MNIST uses fixed slices with 55000 / 5000 / 10000 examples.
    family = (root / "experiments" / "gauge_family.py").read_text()
    assert "x[:55000]" in family and "x[55000:60000]" in family and "x[60000:]" in family, (
        "gauge_family.py no longer takes MNIST's fixed slices")

