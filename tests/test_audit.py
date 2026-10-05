"""Grid-edge detection, and the enforcement that keeps it honest.

The last test in this file is the important one: it scans the real results
directory and fails if any published winner sits on the boundary of its own
search grid.
"""

from pathlib import Path

import pytest

from karcifann import EDGE_SPECS, edges_in_rows, find_edge_winners
from karcifann.audit import _confounded, classify_edge


def tune(method, **axes):
    return {"stage": "tune", "method": method, **{k: str(v) for k, v in axes.items()}}


def best(method, **axes):
    return {"stage": "best", "method": method, **{k: str(v) for k, v in axes.items()}}


# ------------------------------------------------------------- reconstruction


def test_an_interior_winner_is_not_flagged():
    rows = [tune("a", scale=v) for v in (1, 2, 4, 8)] + [best("a", scale=4)]
    assert edges_in_rows(rows, ("method",), ("scale",)) == []


@pytest.mark.parametrize("winner,at", [(1, "minimum"), (8, "maximum")])
def test_a_winner_at_either_end_is_flagged(winner, at):
    rows = [tune("a", scale=v) for v in (1, 2, 4, 8)] + [best("a", scale=winner)]
    found = edges_in_rows(rows, ("method",), ("scale",))
    assert len(found) == 1
    assert found[0]["at"] == at and found[0]["winner"] == winner


def test_each_axis_is_judged_against_its_own_grid():
    rows = [tune("a", shape=s, scale=k) for s in (0.1, 1.0) for k in (1, 2, 4)]
    rows.append(best("a", shape=1.0, scale=2))       # shape at edge, scale interior
    found = edges_in_rows(rows, ("method",), ("shape", "scale"))
    assert [f["axis"] for f in found] == ["shape"]


def test_methods_searched_over_different_grids_are_kept_apart():
    """The bug this whole audit exists for: one method given a narrower grid."""
    rows = [tune("baseline", scale=v) for v in (1, 2, 4)]
    rows += [tune("rival", scale=v) for v in (1, 2, 4, 8, 16)]
    rows += [best("baseline", scale=4), best("rival", scale=4)]
    found = edges_in_rows(rows, ("method",), ("scale",))
    assert [f["method"] for f in found] == ["baseline"]


def test_a_single_valued_axis_is_not_a_search():
    rows = [tune("a", scale=2, shape=0.5) for _ in range(3)] + [best("a", scale=2, shape=0.5)]
    assert edges_in_rows(rows, ("method",), ("scale", "shape")) == []


def test_a_flat_axis_is_reported_rather_than_called_truncated():
    """An axis where every setting scored the same is not an exhausted grid."""
    rows = [tune("a", scale=v) | {"val": "0.1"} for v in (1, 2, 4, 8)]
    rows.append(best("a", scale=1))
    found = edges_in_rows(rows, ("method",), ("scale",))
    assert [f["kind"] for f in found] == ["flat"]


def test_a_monotone_climb_with_no_limit_is_truncation_not_an_exemption():
    """A baseline still improving at the edge of its grid is under-tuned.

    This was a false negative: gradient descent peaked at the smallest learning
    rate tried on Vehicle, the classifier called it "monotone" and passed it,
    and an under-tuned baseline went into a headline comparison.
    """
    rows = [tune("gradient descent", scale=v) | {"val": str(0.9 - 0.1 * i)}
            for i, v in enumerate((0.5, 1, 2, 4, 8))]
    rows.append(best("gradient descent", scale=0.5))
    found = edges_in_rows(rows, ("method",), ("scale",))
    assert [f["kind"] for f in found] == ["monotone_truncated"]


def test_a_winner_with_no_tuning_rows_is_skipped():
    assert edges_in_rows([best("a", scale=1)], ("method",), ("scale",)) == []


def test_non_numeric_values_are_ignored_rather_than_crashing():
    rows = [tune("a", scale="wide"), tune("a", scale=2), best("a", scale="wide")]
    assert edges_in_rows(rows, ("method",), ("scale",)) == []


# ------------------------------------------------------------- directory scan


def test_scan_reports_both_detection_routes(tmp_path):
    (tmp_path / "gauge_family_toy.csv").write_text(
        "stage,method,shape,scale\n"
        "tune,power,0.1,1\ntune,power,0.5,2\ntune,power,0.9,4\n"
        "best,power,0.9,2\n"                                  # shape at maximum
    )
    (tmp_path / "headroom_toy.csv").write_text(
        "stage,configuration,lr,grid_edge\n"
        "summary,A,8,1\n"                                     # self-reported
    )
    found = find_edge_winners(tmp_path)
    sources = {f["source"] for f in found}
    assert sources == {"reconstructed", "self-reported"}
    assert any(f["axis"] == "shape" and f["at"] == "maximum" for f in found)


def test_a_clean_directory_reports_nothing(tmp_path):
    (tmp_path / "gauge_family_toy.csv").write_text(
        "stage,method,shape,scale\n"
        "tune,power,0.1,1\ntune,power,0.5,2\ntune,power,0.9,4\n"
        "best,power,0.5,2\n"
    )
    assert find_edge_winners(tmp_path) == []


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert find_edge_winners(tmp_path / "nope") == []


def test_every_spec_names_a_file_pattern_and_axes():
    for pattern, (groups, axes) in EDGE_SPECS.items():
        assert pattern.endswith(".csv") and groups and axes


# --------------------------------------------------------------- enforcement


@pytest.mark.results
def test_no_published_result_sits_on_a_grid_edge():
    """Every tuned winner in results/ must be interior to its search grid.

    A winner at the boundary means the grid ran out before the optimum did, so
    the number understates that method and is not comparable with a rival whose
    optimum was interior.  Widen the grid and re-run rather than reporting it.
    """
    results = Path(__file__).resolve().parent.parent / "results"
    if not any(results.glob("*.csv")):
        pytest.skip("no results to check")

    fails = {"truncated", "monotone_truncated"}
    findings = [f for f in find_edge_winners(results) if f.get("kind") in fails]
    if findings:
        lines = sorted({
            f"  {f['file']}: {f['method']} at the {f['at']} of {f['axis']} "
            f"({f['grid_min']:g} … {f['grid_max']:g})"
            for f in findings
        })
        pytest.fail(
            f"{len(findings)} winner(s) on a grid boundary with the score still "
            f"climbing -- widen the grid and re-run:\n" + "\n".join(lines)
        )


@pytest.mark.results
def test_degenerate_winners_are_reported_rather_than_hidden():
    """Families whose tuning collapses them onto gradient descent.

    This is a finding, not a fault, so it must not fail the build -- but it
    must be visible, because a 'method' that tunes itself into the baseline is
    the single most important thing to say about it.
    """
    results = Path(__file__).resolve().parent.parent / "results"
    if not any(results.glob("*.csv")):
        pytest.skip("no results to check")

    kinds = {}
    for f in find_edge_winners(results):
        if f.get("kind") in ("degenerate", "asymptotic", "flat"):
            kinds.setdefault(f["kind"], set()).add(f"{f['file'].split('_')[-1][:-4]}/{f['method']}")
    for kind, entries in sorted(kinds.items()):
        print(f"\n{kind}: {len(entries)}")
        for e in sorted(entries):
            print(f"   {e}")


@pytest.mark.results
def test_probe_floor_is_below_every_trained_weight():
    """The identity probes must cover the weights networks actually hold.

    ``Gauge.is_identity`` decides whether a rule has collapsed onto gradient
    descent by evaluating it over ``PROBE_RANGE``.  If that range stops above
    the smallest trained weight, a gauge that still rescales the smallest
    parameters is reported as degenerate and an under-tuned winner slips
    through the audit -- which is exactly what a floor of 1e-3 did.  Experiment
    E5 records the observed magnitudes so the floor is checked, not assumed.
    """
    import csv

    from karcifann.audit import PROBE_FLOOR

    results = Path(__file__).resolve().parent.parent / "results"
    files = sorted(results.glob("*_e5_factor_spread.csv"))
    if not files:
        pytest.skip("no factor-spread measurements")

    smallest = {}
    for path in files:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("weight_min"):
                    key = f"{row['dataset']}/{row['method']}"
                    smallest[key] = float(row["weight_min"])
    if not smallest:
        pytest.skip("weight magnitudes not recorded yet")

    worst = min(smallest.values())
    offenders = {k: v for k, v in smallest.items() if v < PROBE_FLOOR}
    assert not offenders, (
        f"PROBE_FLOOR={PROBE_FLOOR:g} sits above the smallest trained weight in "
        f"{len(offenders)} run(s); lower it below {worst:.2e}:\n"
        + "\n".join(f"   {k}: {v:.2e}" for k, v in sorted(offenders.items())[:8]))


# ------------------------------------------- files that mark no winner

def test_edges_are_found_in_files_that_record_trials_only():
    """E2/E3 write every trial and keep the winner implicit.

    The auditor originally required stage=tune/best rows, so these files were
    scanned and silently yielded nothing -- a Vehicle ablation sat on the floor
    of its scale grid while the suite stayed green.
    """
    rows = [
        {"mode": "full", "scale": "0.5", "val_acc": "0.87"},
        {"mode": "full", "scale": "1", "val_acc": "0.84"},
        {"mode": "full", "scale": "2", "val_acc": "0.82"},
        {"mode": "full", "scale": "4", "val_acc": "0.78"},
        {"mode": "error", "scale": "0.5", "val_acc": "0.80"},
        {"mode": "error", "scale": "1", "val_acc": "0.86"},
        {"mode": "error", "scale": "2", "val_acc": "0.81"},
        {"mode": "error", "scale": "4", "val_acc": "0.79"},
    ]
    findings = edges_in_rows(rows, ("mode",), ("scale",))
    assert [f["method"] for f in findings] == ["full"], findings
    assert findings[0]["at"] == "minimum"
    assert findings[0]["winner"] == 0.5
    assert findings[0]["kind"] in {"truncated", "monotone_truncated"}


def test_a_stageless_file_is_not_blanket_labelled_truncated():
    """classify_edge filters on stage/method; neither exists in these files.

    Without the trials hand-off it matched nothing, fell through to its default
    and would have stamped every edge -- including flat ones -- as truncated.
    """
    flat = [{"mode": "weight", "scale": str(s), "val_acc": "0.5"}
            for s in (0.5, 1, 2, 4, 8)]
    findings = edges_in_rows(flat, ("mode",), ("scale",))
    assert findings, "a winner on the boundary should still be reported"
    assert findings[0]["kind"] == "flat", findings


def test_a_boundary_winner_that_ties_an_interior_one_is_not_truncation():
    """A plateau is not a truncated search.

    The chained comparison's baseline has an effective step of scale x alpha,
    so its best score sits along a diagonal and the argmax among equals can
    land on a boundary.  Widening the grid cannot beat a score already matched
    inside it, so this must not be reported as truncation.
    """
    # A real truncation: the winner is the best score and it is at the floor,
    # with the axis still improving outward.  No interior point matches it.
    truncated = [{"stage": "tune", "method": "m", "x": str(x), "val_acc": str(v)}
                 for x, v in [(1, 0.95), (2, 0.90), (4, 0.85), (8, 0.80), (16, 0.70)]]
    truncated.append({"stage": "best", "method": "m", "x": "1", "val_acc": "0.95"})
    assert classify_edge(truncated, "m", "x", 1.0) in {"truncated", "monotone_truncated"}

    # A plateau: the boundary winner only equals what the interior already had.
    plateau = [{"stage": "tune", "method": "m", "x": str(x), "val_acc": str(v)}
               for x, v in [(1, 0.80), (2, 0.95), (4, 0.90), (8, 0.85), (16, 0.95)]]
    plateau.append({"stage": "best", "method": "m", "x": "16", "val_acc": "0.95"})
    assert classify_edge(plateau, "m", "x", 16.0) == "tied_interior"


@pytest.mark.results
def test_every_self_reported_edge_is_accounted_for():
    """A script's own grid_edge flag must not be the end of the story.

    find_edge_winners reports self-reported flags under their own kind, which
    is not in the failing set -- so for a long time a script could flag its own
    truncation and the suite would still pass.  Every flagged winner must
    therefore be independently classified, or be one of the confounded axes
    that cannot be at an edge on their own.
    """
    results = Path(__file__).resolve().parent.parent / "results"
    if not any(results.glob("*.csv")):
        pytest.skip("no results to check")

    findings = find_edge_winners(results)
    reconstructed = {(f["file"], f["method"]) for f in findings
                     if f["source"] == "reconstructed"}
    unexplained = sorted({
        (f["file"], f["method"]) for f in findings
        if f["kind"] == "self-reported"
        and (f["file"], f["method"]) not in reconstructed
        and not _confounded(f["file"], f["method"], "")
    })
    assert not unexplained, (
        "self-reported grid edges that nothing else explains:\n"
        + "\n".join(f"  {name}: {method}" for name, method in unexplained)
    )


def test_edge_specs_cover_every_tuning_file_family():
    """A results family with a search grid must be in EDGE_SPECS.

    Three families were missing at different times -- E2, E3 and E4 -- and each
    absence was invisible: the auditor simply had nothing to say about them and
    the suite passed.  Coverage is asserted against the families on disk rather
    than remembered.
    """
    import re

    results = Path(__file__).resolve().parent.parent / "results"
    if not any(results.glob("*.csv")):
        pytest.skip("no results to check")

    families = {
        re.sub(r"^(vehicle|satimage|dry_bean|segment|optdigits|pendigits|mnist|letter|digits)_",
               "*_", path.name)
        for path in results.glob("*.csv")
    }
    tuning = {f for f in families
              if any(k in f for k in ("_tuning", "_sensitivity", "gauge_family",
                                      "headroom", "chained_vs_terminal"))}
    uncovered = sorted(
        f for f in tuning
        if not any(Path(f).match(pattern) for pattern in EDGE_SPECS)
        and not any(Path(f).match(pattern) for pattern in DEDICATED_EDGE_TESTS)
    )
    assert not uncovered, f"tuning files with no EDGE_SPECS entry: {uncovered}"

    # An exemption is only worth anything if the test it names exists.
    source = Path(__file__).read_text()
    for pattern, name in DEDICATED_EDGE_TESTS.items():
        assert f"def {name}(" in source, (
            f"{pattern} is exempted from EDGE_SPECS in favour of {name}(), "
            "which is not defined in this file")


def test_a_winner_marked_summary_is_found_like_one_marked_best():
    """headroom writes stage=summary, the sweep scripts write stage=best.

    edges_in_rows defaulted to `best` alone, so the headroom entry in
    EDGE_SPECS matched files and reported nothing for months -- an inert guard
    reads exactly like a passing one.
    """
    rows = [{"stage": "tune", "configuration": "A", "lr": str(v), "val_acc": str(s)}
            for v, s in [(1, 0.95), (2, 0.90), (4, 0.85), (8, 0.70)]]
    rows.append({"stage": "summary", "configuration": "A", "lr": "1", "val_acc": "0.95"})
    findings = edges_in_rows(rows, ("configuration",), ("lr",))
    assert findings and findings[0]["at"] == "minimum", findings


#: ``results/ filename glob -> the experiment script that writes it``.  The
#: library under karcifann/ applies to every group; a script applies only to
#: what it produces, so editing the figure pass cannot make a training result
#: look stale.
PRODUCERS = {
    "*_e1_*.csv": "run_analysis.py",
    "*_e2_*.csv": "run_analysis.py",
    "*_e3_*.csv": "run_analysis.py",
    "*_e4_*.csv": "run_analysis.py",
    "*_e5_*.csv": "run_analysis.py",
    "gauge_family_*.csv": "gauge_family.py",
    "headroom_*.csv": "headroom.py",
    "chained_vs_terminal_*.csv": "chained_vs_terminal.py",
    "fod_comparison_*.csv": "fod_comparison.py",
    "digits_experiment.csv": "digits_experiment.py",
    "xor_experiment.csv": "xor_experiment.py",
    "*_methods_10ep.csv": "mnist_experiment.py",
    "loss_scaling.csv": "loss_scaling.py",
    "split_robustness_*.csv": "split_robustness.py",
    "headroom_splits_*.csv": "headroom_splits.py",
    "multiseed_tuning_*.csv": "multiseed_tuning.py",
    "summary_*.csv": "make_figures.py",
    "stats_*.csv": "make_figures.py",
}

#: Families whose grid-edge check cannot go through EDGE_SPECS, and the test
#: that covers each instead.  The generic auditor assumes one scored row per
#: grid point; these two break that assumption -- headroom_splits records its
#: setting as a formatted string rather than a numeric axis, and
#: multiseed_tuning writes one row per tuning seed, so three per point.
#:
#: This mapping exists because the alternative was worse than no coverage:
#: "headroom_splits_*.csv" was being matched by the "headroom_*.csv" entry,
#: whose axis column it does not have, so the auditor ran against it and
#: checked nothing while this test reported it as covered.  An exemption that
#: has to name a real test is honest; accidental coverage is not.
DEDICATED_EDGE_TESTS = {
    "headroom_splits_*.csv": "test_the_nested_split_studies_have_interior_winners",
    "multiseed_tuning_*.csv": "test_the_multiseed_winners_are_interior_to_their_grids",
}

#: Pure-presentation modules cannot change a computed value, so editing one and
#: redrawing must not read as a mixed generation.  karcifann/plotting exports
#: only setup/save/colour/marker and computes nothing.  Keep this minimal --
#: anything that touches arithmetic belongs in the check.  In particular
#: karcifann/audit.py is *not* exempt: it is checking code today, but exempting
#: it would hide the day it stops being, and the failure that prompted the idea
#: turned out to be a mis-attributed producer below, not this module.
NON_PRODUCING = {"karcifann/plotting.py"}


@pytest.mark.results
def test_results_are_all_from_one_generation():
    """No source change may fall inside the span of the results it produces.

    MNIST's E2 results predated a change to the gauge modes by three days and
    were never regenerated, because the repair queues assumed its fixed slices
    made it immune.  Re-running them selected different tuned scales and moved
    a published conclusion.  Nothing detected it: every file was individually
    valid, and the grid audit cannot see that two files come from different
    versions of the code.

    The invariant is *not* freshness -- editing the code after a full run
    leaves every result equally old, which is honest and is what the
    reproduction step in run_all.sh is for.  What must never happen is a
    mixture within one producer: some of its results from before a change to
    the code that writes them, and some from after.  Grouping by producer also
    keeps a cosmetic edit to the figure pass from condemning training results
    it cannot affect.
    """
    import datetime

    root = Path(__file__).resolve().parent.parent
    results = sorted((root / "results").glob("*.csv"))
    if len(results) < 2:
        pytest.skip("no results to check")

    library = [p for p in (root / "karcifann").rglob("*.py")
               if p.relative_to(root).as_posix() not in NON_PRODUCING]

    groups: dict[str, list] = {}
    unclaimed = []
    for path in results:
        # Longest pattern wins.  "headroom_*.csv" also matches
        # "headroom_splits_vehicle.csv", and taking the first match attributed
        # eight of today's files to headroom.py -- giving that producer a span
        # two days wide and failing this test for a change that was nowhere
        # near its results.
        matches = sorted((pattern for pattern in PRODUCERS if path.match(pattern)),
                         key=len, reverse=True)
        script = PRODUCERS[matches[0]] if matches else None
        if script is None:
            unclaimed.append(path.name)
        else:
            groups.setdefault(script, []).append(path)
    assert not unclaimed, (
        f"results with no known producer, so nothing checks them: {unclaimed}")

    def when(ts):
        return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")

    problems = []
    for script, paths in sorted(groups.items()):
        stamps = {p: p.stat().st_mtime for p in paths}
        oldest, newest = min(stamps.values()), max(stamps.values())
        sources = library + [root / "experiments" / script]
        straddling = [p for p in sources
                      if p.exists() and oldest < p.stat().st_mtime < newest]
        if straddling:
            cutoff = max(s.stat().st_mtime for s in straddling)
            stale = sorted(n.name for n, t in stamps.items() if t < cutoff)
            problems.append(
                f"  {script}: its results span {when(oldest)} .. {when(newest)}, "
                f"but {', '.join(p.name for p in straddling[:3])} changed inside "
                f"that span; oldest affected: {stale[:4]}")
    if problems:
        pytest.fail(
            "results/ mixes two generations of the code:\n" + "\n".join(problems)
            + "\nre-run the affected queues so each producer's results come "
              "from one version")


@pytest.mark.results
def test_the_nested_split_studies_have_interior_winners():
    """The grid-edge guarantee must reach the two newest studies too.

    Section 5.3 claims every published winner is checked against the grid it
    came from.  That was not true when these studies were added:
    `split_robustness_*.csv` matched no EDGE_SPECS pattern at all, and
    `headroom_splits_*.csv` was swallowed by the `headroom_*.csv` glob whose
    axis column it does not have -- so the audit ran and checked nothing, which
    is the same failure the comment on that entry already records.

    EDGE_SPECS now covers the first.  The second records its setting as a
    formatted string rather than a numeric axis, so it is reconstructed here:
    for every split and configuration, the tuned winner must not sit on the
    boundary of the grid its own tune rows describe.
    """
    import csv
    import re

    results = Path(__file__).resolve().parent.parent / "results"
    files = sorted(results.glob("headroom_splits_*.csv"))
    if not files:
        pytest.skip("no nested-split headroom measurements")

    number = re.compile(r"=\s*([0-9.eE+-]+)")
    offenders = []
    for path in files:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        grids, winners = {}, {}
        for row in rows:
            key = (row["split"], row["configuration"])
            value = number.search(row["setting"] or "")
            if not value:
                continue
            if row["stage"] == "tune":
                grids.setdefault(key, []).append(float(value.group(1)))
            elif row["stage"] == "summary":
                winners[key] = float(value.group(1))
        assert winners, f"{path.name} records no tuned winners"
        for key, won in winners.items():
            grid = grids.get(key)
            assert grid, f"{path.name}: no tune rows for {key}"
            if won in (min(grid), max(grid)):
                offenders.append(f"{path.name} split {key[0]} "
                                 f"{key[1][:2].strip()} won at {won:g}, a grid edge")

    assert not offenders, (
        "tuned winners sitting on a grid boundary:\n  " + "\n  ".join(offenders))


@pytest.mark.results
def test_the_multiseed_winners_are_interior_to_their_grids():
    """Section 6.4's winners must be checked against the grid they came from.

    `multiseed_tuning_*.csv` records three tuning rows per grid point -- one per
    tuning seed -- so the generic EDGE_SPECS machinery, which assumes one score
    per point, cannot read it.  Rather than bend that machinery, the check is
    done here, the same way `headroom_splits_*.csv` is handled: for every split
    and both selection rules, the winner must not sit on the boundary of its own
    axis.  A boundary winner would mean the grid ran out before the optimum did,
    and the whole point of the section is that the *selection* is sound.
    """
    import csv

    results = Path(__file__).resolve().parent.parent / "results"
    files = sorted(results.glob("multiseed_tuning_*.csv"))
    if not files:
        pytest.skip("no multi-seed tuning measurements")

    problems = []
    for path in files:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        tune = [r for r in rows if r["stage"] == "tune"]
        best = [r for r in rows if r["stage"] == "best"]
        assert best, f"{path.name}: no selected winners recorded"

        for row in best:
            axes = {"scale": sorted({float(r["scale"]) for r in tune
                                     if r["method"] == row["method"]
                                     and r["split"] == row["split"]})}
            if row["method"] == "power":
                axes["shape"] = sorted({float(r["shape"]) for r in tune
                                        if r["method"] == "power"
                                        and r["split"] == row["split"]})
            for name, values in axes.items():
                if not row[name]:
                    continue
                won = float(row[name])
                if len(values) > 1 and won in (values[0], values[-1]):
                    problems.append(
                        f"  {path.name} split {row['split']} {row['method']} "
                        f"({row['rule']} seed): {name}={won:g} is on the "
                        f"boundary of {values[0]:g}..{values[-1]:g}")
            # The recorded flag must agree with what the tuning rows say.
            recorded = bool(int(float(row["grid_edge"])))
            derived = any(float(row[n]) in (v[0], v[-1])
                          for n, v in axes.items() if row[n] and len(v) > 1)
            assert recorded == derived, (
                f"{path.name} split {row['split']} {row['method']} "
                f"({row['rule']}): grid_edge={int(recorded)} but the tuning "
                f"rows say {int(derived)}")

    assert not problems, (
        "multi-seed winners sitting on a grid boundary:\n" + "\n".join(problems))
