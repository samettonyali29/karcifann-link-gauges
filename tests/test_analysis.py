"""The statistical tools, and the guards that stop them being misused."""

import numpy as np
import pytest

from karcifann.analysis import (
    cliffs_delta,
    cohens_d,
    friedman_over_datasets,
    friedman_power_floor,
    holm,
    average_ranks,
    pairwise_wilcoxon,
    tolerance_region,
)


# ------------------------------------------------------------ multiplicity


def test_holm_matches_a_worked_example():
    """Step-down: sorted p times (m - rank), made monotone and clipped."""
    adjusted, rejected = holm([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert list(rejected) == [True, False, False]


def test_holm_is_never_smaller_than_the_raw_value():
    raw = np.array([0.001, 0.02, 0.3, 0.7])
    assert np.all(holm(raw)[0] >= raw)


def test_holm_is_monotone_in_the_sorted_order():
    adjusted, _ = holm([0.001, 0.002, 0.003, 0.9])
    assert np.all(np.diff(np.sort(adjusted)) >= 0)


def test_holm_is_less_conservative_than_bonferroni():
    raw = [0.001, 0.02, 0.3]
    assert np.all(holm(raw)[0] <= np.minimum(np.array(raw) * len(raw), 1.0))


def test_holm_clips_at_one():
    assert np.all(holm([0.6, 0.7, 0.8])[0] <= 1.0)


# ------------------------------------------------------------ effect sizes


def test_cohens_d_sign_follows_the_difference():
    a, b = np.array([2.0, 3.5, 4.0, 5.5]), np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(a, b) > 0
    assert cohens_d(b, a) == pytest.approx(-cohens_d(a, b))


def test_cohens_d_is_zero_for_identical_runs():
    a = np.array([1.0, 2, 3])
    assert cohens_d(a, a) == 0.0


def test_a_perfectly_consistent_difference_is_infinite_not_zero():
    """Zero variance in the differences means maximal reliability, not none."""
    a, b = np.array([2.0, 3, 4, 5]), np.array([1.0, 2, 3, 4])
    assert cohens_d(a, b) == np.inf
    assert cohens_d(b, a) == -np.inf


def test_cliffs_delta_is_one_when_fully_separated():
    assert cliffs_delta([4.0, 5, 6], [1.0, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1.0, 2, 3], [4.0, 5, 6]) == pytest.approx(-1.0)


def test_cliffs_delta_is_zero_for_identical_distributions():
    a = [1.0, 2, 3, 4]
    assert cliffs_delta(a, a) == pytest.approx(0.0)


# ------------------------------------------------------------- Wilcoxon


def test_pairwise_wilcoxon_shapes_and_symmetry():
    rng = np.random.default_rng(0)
    scores = {name: rng.normal(loc, 1, 8) for name, loc in
              (("a", 0.0), ("b", 1.0), ("c", 2.0))}
    result = pairwise_wilcoxon(scores)
    assert result.pvalues.shape == (3, 3)
    assert np.allclose(result.pvalues, result.pvalues.T, equal_nan=True)
    assert np.all(np.isnan(np.diag(result.pvalues)))
    assert np.allclose(result.effect, -result.effect.T, equal_nan=True)


def test_identical_methods_are_not_reported_as_different():
    values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    result = pairwise_wilcoxon({"a": values, "b": values.copy()})
    assert result.pvalues[0, 1] == 1.0
    assert result.adjusted[0, 1] == 1.0


def test_a_consistent_difference_is_detected():
    a = np.arange(8, dtype=float)
    result = pairwise_wilcoxon({"a": a + 1.0, "b": a})
    assert result.pvalues[0, 1] < 0.05


def test_mismatched_run_counts_are_rejected():
    with pytest.raises(ValueError):
        pairwise_wilcoxon({"a": np.zeros(8), "b": np.zeros(6)})


# ------------------------------------------------------------- Friedman


def test_average_ranks_puts_the_best_method_first():
    table = np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]])
    assert np.allclose(average_ranks(table), [1.0, 2.0, 3.0])


def test_friedman_power_floor_is_the_unanimous_case():
    """With 3 datasets and 5 methods no data can beat p = 0.0174."""
    floor = friedman_power_floor(3, 5)
    assert floor == pytest.approx(0.01735, abs=1e-4)

    unanimous = np.tile(np.arange(5, 0, -1.0), (3, 1))
    assert friedman_over_datasets(unanimous)["pvalue"] == pytest.approx(floor)


def test_friedman_power_floor_falls_as_datasets_are_added():
    floors = [friedman_power_floor(n, 5) for n in (3, 5, 8, 12)]
    assert floors == sorted(floors, reverse=True)
    assert floors[0] > 0.01 and floors[-1] < 1e-4


def test_friedman_reports_its_own_design_limits():
    table = np.array([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75], [0.99, 0.9, 0.8]])
    outcome = friedman_over_datasets(table)
    assert outcome["blocks"] == 3 and outcome["methods"] == 3
    assert outcome["pvalue"] >= outcome["power_floor"]


# ----------------------------------------------------------- sensitivity


def test_tolerance_region_finds_the_plateau():
    values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    scores = [0.5, 0.995, 1.0, 0.999, 0.4]
    region = tolerance_region(values, scores, tolerance=0.01)
    assert region["best_at"] == 10.0
    assert region["count"] == 3
    assert region["min"] == 1.0 and region["max"] == 100.0
    assert region["decades"] == pytest.approx(2.0)


def test_a_sharp_optimum_has_zero_width():
    region = tolerance_region([1.0, 2.0, 4.0], [0.1, 1.0, 0.1], tolerance=0.01)
    assert region["count"] == 1
    assert region["decades"] == 0.0


def test_a_tighter_tolerance_cannot_widen_the_region():
    values = [1.0, 2.0, 4.0, 8.0]
    scores = [0.9, 0.99, 1.0, 0.97]
    wide = tolerance_region(values, scores, tolerance=0.05)
    narrow = tolerance_region(values, scores, tolerance=0.001)
    assert narrow["count"] <= wide["count"]


def test_wilcoxon_power_floor_flags_an_undersized_seed_count():
    """8 seeds over 10 comparisons cannot reach 0.05, whatever the data say."""
    from karcifann.analysis import wilcoxon_power_floor

    assert wilcoxon_power_floor(8, 10) > 0.05
    assert wilcoxon_power_floor(10, 10) <= 0.05
    assert wilcoxon_power_floor(16, 10) < 0.001


def test_wilcoxon_power_floor_falls_as_pairs_are_added():
    from karcifann.analysis import wilcoxon_power_floor

    floors = [wilcoxon_power_floor(n, 10) for n in (6, 8, 10, 12, 16)]
    assert floors == sorted(floors, reverse=True)


def test_wilcoxon_power_floor_is_attained_by_a_consistent_difference():
    from karcifann.analysis import wilcoxon_power_floor

    a = np.arange(12, dtype=float)
    result = pairwise_wilcoxon({"a": a + 1.0, "b": a})
    assert result.pvalues[0, 1] == pytest.approx(wilcoxon_power_floor(12, 1))


def test_nemenyi_critical_difference_matches_the_published_formula():
    """CD = q sqrt(k(k+1)/6N); 5 methods over 8 datasets is 2.16."""
    from karcifann.analysis import nemenyi_critical_difference

    expected = 2.728 * np.sqrt(5 * 6 / (6 * 8))
    assert nemenyi_critical_difference(5, 8) == pytest.approx(expected, rel=1e-9)
    assert nemenyi_critical_difference(5, 8) == pytest.approx(2.157, abs=0.01)


def test_critical_difference_shrinks_as_datasets_are_added():
    from karcifann.analysis import nemenyi_critical_difference

    widths = [nemenyi_critical_difference(5, n) for n in (4, 8, 16, 32)]
    assert widths == sorted(widths, reverse=True)


def test_critical_difference_rejects_untabulated_designs():
    from karcifann.analysis import nemenyi_critical_difference

    with pytest.raises(ValueError, match="tabulated q"):
        nemenyi_critical_difference(20, 8)
    with pytest.raises(ValueError, match="alpha"):
        nemenyi_critical_difference(5, 8, alpha=0.01)


def test_friedman_reports_the_critical_difference_alongside_the_ranks():
    table = np.array([[0.9, 0.8, 0.7]] * 8)
    outcome = friedman_over_datasets(table)
    assert outcome["critical_difference"] > 0
    assert "ranks" in outcome and len(outcome["ranks"]) == 3


# ------------------------------------ the assumption the Wilcoxon test makes


def test_sign_test_agrees_with_wilcoxon_on_a_clean_difference():
    from karcifann.analysis import sign_test

    a = np.arange(16, dtype=float)
    assert sign_test(a + 1.0, a) < 0.001


def test_sign_test_ignores_magnitude_and_reads_only_direction():
    """One huge reversal cannot outvote fifteen consistent small wins."""
    from karcifann.analysis import sign_test

    a = np.zeros(16)
    b = np.array([-1.0] * 15 + [1000.0])
    assert sign_test(a, b) < 0.01          # 15 of 16 signs agree


def test_sign_test_of_identical_runs_is_one():
    from karcifann.analysis import sign_test

    a = np.arange(8, dtype=float)
    assert sign_test(a, a) == 1.0


def test_difference_skew_detects_asymmetry():
    from karcifann.analysis import difference_skew

    symmetric = np.array([-2.0, -1, 0, 1, 2])
    assert difference_skew(symmetric, np.zeros(5)) == pytest.approx(0.0, abs=1e-9)
    skewed = np.array([0.0, 0, 0, 0, 10.0])
    assert difference_skew(skewed, np.zeros(5)) > 1.0


def test_difference_skew_of_a_constant_difference_is_zero():
    from karcifann.analysis import difference_skew

    a = np.arange(6, dtype=float)
    assert difference_skew(a + 3.0, a) == 0.0


def test_comparison_carries_the_assumption_free_cross_check():
    rng = np.random.default_rng(0)
    scores = {"a": rng.normal(1.0, 1, 16), "b": rng.normal(0.0, 1, 16)}
    result = pairwise_wilcoxon(scores)
    assert np.isfinite(result.sign[0, 1]) and 0 <= result.sign[0, 1] <= 1
    assert result.sign[0, 1] == result.sign[1, 0]          # symmetric
    assert result.skew[0, 1] == pytest.approx(-result.skew[1, 0])

def test_paired_counts_separate_wins_ties_and_losses():
    """Ties must be counted, not folded into whichever side is printed.

    `table_headline` used to reverse a pair by computing `n - n_favouring_a`,
    which counts a tie as a win for the other method.  On Vehicle that turned
    13 wins and 3 ties into a printed "16/16" -- the most persuasive number in
    the headline table, and wrong in the flattering direction.  The effect
    summary now records all three counts so no consumer has to subtract.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "experiments"))
    from make_figures import _paired_effect

    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 2.0, 5.0, 4.0]          # win, tie, loss, tie
    effect = _paired_effect(a, b)
    assert effect["n_favouring_a"] == 1
    assert effect["n_favouring_b"] == 1
    assert effect["n_tied"] == 2
    assert (effect["n_favouring_a"] + effect["n_favouring_b"]
            + effect["n_tied"] == len(a))


def test_the_headline_table_reports_ties_from_the_saved_runs():
    """The printed W/T/L must match the saved per-seed scores.

    A guard that compares the manuscript against the generator cannot catch a
    generator that is wrong; this one recomputes from `*_e1_final.csv`.
    """
    import collections
    import csv
    import re
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    if not (root / "results" / "vehicle_e1_final.csv").exists():
        pytest.skip("no confirmation runs")
    sys.path.insert(0, str(root / "experiments"))
    import update_readme

    update_readme.RESULTS = root / "results"
    printed = update_readme.table_headline()

    for dataset, pretty in (("vehicle", "Vehicle"), ("mnist", "MNIST"),
                            ("letter", "Letter")):
        by_seed = collections.defaultdict(dict)
        with (root / "results" / f"{dataset}_e1_final.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row.get("metric") not in (None, "", "test_acc"):
                    continue
                if row.get("method") and row.get("seed"):
                    by_seed[row["seed"]][row["method"]] = float(
                        row.get("value") or row.get("test_acc") or 0)
        diffs = [v["KarciFANN"] - v["gradient descent"] for v in by_seed.values()
                 if {"KarciFANN", "gradient descent"} <= set(v)]
        expected = (f"{sum(d > 0 for d in diffs)}/{sum(d == 0 for d in diffs)}"
                    f"/{sum(d < 0 for d in diffs)}")
        line = next(l for l in printed.splitlines() if l.startswith(f"| {pretty} "))
        assert expected in line, (
            f"{pretty}: saved runs give {expected}, table prints {line}")

