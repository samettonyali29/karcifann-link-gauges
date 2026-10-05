"""Detect tuned winners that sit on the boundary of the grid searched for them.

A hyperparameter search that peaks at the edge of its own grid has not found an
optimum -- it has run out of room.  Reporting such a winner understates that
method, and comparing it against a rival whose optimum *was* interior
manufactures a difference out of the search design.

This module reconstructs, for each method in a results CSV, the set of values
that were actually tried on each axis, and flags any winner lying at the
minimum or maximum of one.  It exists because relying on a human to notice the
warnings did not work: over the course of this project several edge winners
were spotted, extended by hand in one-off scripts, and the better numbers
carried into prose -- leaving tables that the pipeline could no longer
reproduce.  :func:`find_edge_winners` is wired into the test suite so an
under-tuned winner fails a test instead of reaching a table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["EDGE_SPECS", "PROBE_FLOOR", "PROBE_RANGE", "classify_edge",
           "find_edge_winners", "edges_in_rows", "identity_distance"]

#: Smallest |weight| the identity probes must cover.  A gauge is only called
#: indistinguishable from gradient descent if ``Phi ~ 1`` across every weight
#: magnitude a trained network actually holds, so this floor has to sit below
#: the smallest one observed -- which
#: ``test_probe_floor_is_below_every_trained_weight`` checks against the
#: measurements recorded by experiment E5, rather than leaving it asserted.
#: Set from measurement, not intuition: the smallest magnitude E5 has recorded
#: in a trained network is 1.3e-7, on a KarciFANN run -- the rule's own
#: prefactor shrinks the step for small weights, so they settle far lower than
#: gradient descent leaves them.  A first guess of 1e-5, taken from a gradient
#: descent run, was two orders too high.
PROBE_FLOOR = 1e-8
PROBE_RANGE = np.array([PROBE_FLOOR, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                        1e-2, 1e-1, 1.0, 10.0])

#: ``filename glob -> (columns identifying a method, numeric grid axes)``.
#: A file with no entry here is only checked for a ``grid_edge`` column it
#: recorded itself.
EDGE_SPECS = {
    "gauge_family_*.csv": (("method",), ("shape", "scale")),
    "chained_vs_terminal_*.csv": (("method",), ("coefficient", "scale")),
    "headroom_*.csv": (("configuration",), ("lr",)),
    # E2 and E3 were absent here until a Vehicle run put the `full` ablation on
    # the floor of its scale grid: run_analysis printed "(grid edge!)" and
    # nothing failed, because stdout is not enforcement.
    "*_e2_tuning.csv": (("mode",), ("scale",)),
    "*_e3_sensitivity.csv": (("sweep",), ("value",)),
    "*_e4_tuning.csv": (("config",), ("lr",)),
    # The nested-split studies.  split_robustness records its axes numerically
    # and can be reconstructed here; headroom_splits records its setting as a
    # formatted string, so it is checked by
    # ``test_the_nested_split_studies_have_interior_winners`` instead -- and
    # note that "headroom_*.csv" above would otherwise swallow it while
    # checking nothing, the same way that entry once swallowed headroom itself.
    "split_robustness_*.csv": (("method",), ("shape", "scale")),
}


#: Column holding the selection metric, tried in order -- the scripts do not
#: all name it the same thing.
SCORE_COLUMNS = ("val", "val_acc", "val_mean")


def classify_edge(rows, method, axis, winner, score=None, identity=None,
                  approaching=False, trials=None):
    """Why a winner is sitting on the boundary.  Three cases, only one a defect.

    ``degenerate``  the rule at that setting is numerically plain gradient
                    descent, so the search converged on switching the operator
                    off -- a result, and one worth reporting;
    ``asymptotic``  the rule is monotonically approaching the identity as the
                    setting moves toward the boundary, so it tends to gradient
                    descent without ever arriving and a wider grid would only
                    chase the asymptote;
    ``monotone``    the score improves all the way to the boundary with no
                    limit to approach.  This is **truncation**: unlike an
                    asymptote there is nothing to converge on, so a wider grid
                    would keep improving and the number understates the method.
                    Kept as a separate label only so the reason is visible;
    ``tied_interior``
                    the boundary winner ties a setting inside the grid, so the
                    best score was already found in the interior and a wider
                    grid cannot beat it -- a plateau, not a truncation;
    ``flat``        every setting on the axis scored identically, so the
                    winner's position carries no information -- normally a sign
                    the method never learned on this problem;
    ``truncated``   none of the above -- the grid simply ran out before the
                    optimum did, which understates the method and must be fixed
                    by widening.
    """
    if identity:
        return "degenerate"
    if approaching:
        return "asymptotic"

    candidates = (score,) if score else SCORE_COLUMNS
    # ``trials`` lets a caller that has already grouped the rows pass them in.
    # Without it the stage/method filter below silently matches nothing on
    # files that use neither column, and every edge would be called truncated.
    tuning = (list(trials) if trials is not None
              else [r for r in rows if r.get("stage") == "tune"
                    and r.get("method") == method])
    column = next((c for c in candidates if any(_number(r.get(c)) is not None for r in tuning)),
                  None)
    if column is None:
        return "truncated"

    best_at = {}
    for row in tuning:
        value, result = _number(row.get(axis)), _number(row.get(column))
        if value is None or result is None:
            continue
        best_at[value] = max(best_at.get(value, float("-inf")), result)
    if len(best_at) < 4:
        return "truncated"

    values = sorted(best_at)
    scores = [best_at[v] for v in values]
    if len(set(scores)) == 1:
        # The axis changed nothing: every setting scored the same, so the
        # winner's position is arbitrary rather than truncated.  Usually this
        # means the method failed to learn at all on this problem.
        return "flat"

    # A boundary winner that merely ties an interior one is not truncation:
    # the same score is already reachable inside the grid, so widening it
    # cannot improve on what was found.  This is how a plateau presents -- the
    # chained comparison's baseline has an effective step of scale x alpha, so
    # its optimum lies along a diagonal and the argmax among equals lands
    # wherever the enumeration happens to reach first.
    interior = [v for v in best_at if v not in (min(best_at), max(best_at))]
    if interior and max(best_at[v] for v in interior) >= best_at[winner] - 1e-12:
        return "tied_interior"

    # Spearman rather than a strict comparison: each tuning cell is a single
    # run, so a monotone trend can still wobble by a fraction of a point.
    from scipy.stats import spearmanr

    rho = spearmanr(values, scores).statistic
    toward_minimum = winner == min(best_at)
    improving_outward = rho < 0 if toward_minimum else rho > 0
    # A monotone climb to the boundary is only acceptable when the operator is
    # provably converging on something -- that case is caught above as
    # "asymptotic".  On a step-size axis there is no limit to reach: a smaller
    # learning rate is just a smaller learning rate, so the grid must be
    # widened.  Labelling these "monotone" and passing them was a false
    # negative that let an under-tuned baseline into a headline comparison.
    return "monotone_truncated" if improving_outward and abs(rho) >= 0.9 else "truncated"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def edges_in_rows(rows, group_columns, axis_columns, tune="tune", best=("best", "summary")):
    """Winners on a grid boundary, given the rows of one results file.

    ``tune`` rows define the grid that was searched; ``best`` rows name the
    winner.  An axis with fewer than two distinct values is not a search and is
    skipped -- a single fixed value cannot be an edge.

    ``best`` accepts several labels because the scripts do not agree on one:
    the sweep scripts write ``best``, headroom writes ``summary``.  It defaulted
    to ``best`` alone, which meant the headroom entry in :data:`EDGE_SPECS`
    matched files, found no winner row, and silently reported nothing -- an
    inert guard that read as a passing one.
    """
    winner_stages = {best} if isinstance(best, str) else set(best)
    # Some scripts label their sweep rows (stage=tune/best); others -- E2 and
    # E3 -- just write every trial and keep the winner implicit.  For those,
    # the winner is the best-scoring trial, which is what the script itself
    # selected, so the grid can still be reconstructed.
    if not any(row.get("stage") for row in rows):
        return _edges_by_argmax(rows, group_columns, axis_columns)

    grids: dict[tuple, dict[str, set]] = {}
    for row in rows:
        if row.get("stage") != tune:
            continue
        key = tuple(row.get(c, "") for c in group_columns)
        axes = grids.setdefault(key, {c: set() for c in axis_columns})
        for column in axis_columns:
            value = _number(row.get(column))
            if value is not None:
                axes[column].add(value)

    findings = []
    for row in rows:
        if row.get("stage") not in winner_stages:
            continue
        key = tuple(row.get(c, "") for c in group_columns)
        axes = grids.get(key)
        if not axes:
            continue
        for column in axis_columns:
            searched = axes[column]
            winner = _number(row.get(column))
            if winner is None or len(searched) < 2 or winner not in (min(searched), max(searched)):
                continue
            findings.append({
                "method": " / ".join(v for v in key if v),
                "axis": column,
                "winner": winner,
                "grid_min": min(searched),
                "grid_max": max(searched),
                "at": "minimum" if winner == min(searched) else "maximum",
                "kind": classify_edge(
                    rows, key[0], column, winner,
                    identity=_is_identity(key[0], column, winner),
                    approaching=_approaches_identity(key[0], column, winner, searched),
                ),
            })
    return findings



def _edges_by_argmax(rows, group_columns, axis_columns):
    """Edge check for files that record trials only, with no winner marked.

    The winner is taken to be the highest-scoring trial in each group -- the
    same choice the tuning script makes -- and is reported when it sits on a
    boundary of the values actually searched.
    """
    column = next((c for c in SCORE_COLUMNS
                   if any(_number(r.get(c)) is not None for r in rows)), None)
    if column is None:
        return []

    groups: dict[tuple, list] = {}
    for row in rows:
        if _number(row.get(column)) is None:
            continue
        groups.setdefault(tuple(row.get(c, "") for c in group_columns), []).append(row)

    findings = []
    for key, trials in groups.items():
        winner_row = max(trials, key=lambda r: _number(r.get(column)))
        for axis in axis_columns:
            searched = {v for v in (_number(r.get(axis)) for r in trials) if v is not None}
            winner = _number(winner_row.get(axis))
            if winner is None or len(searched) < 2:
                continue
            if winner not in (min(searched), max(searched)):
                continue
            findings.append({
                "method": " / ".join(v for v in key if v),
                "axis": axis,
                "winner": winner,
                "grid_min": min(searched),
                "grid_max": max(searched),
                "at": "minimum" if winner == min(searched) else "maximum",
                "kind": classify_edge(
                    rows, key[0], axis, winner, trials=trials,
                    identity=_is_identity(key[0], axis, winner),
                    approaching=_approaches_identity(key[0], axis, winner, searched),
                ),
            })
    return findings


def identity_distance(method: str, axis: str, value):
    """``max |Phi - 1|`` for this rule at this setting, or ``None`` if unknown.

    Zero means the rule *is* gradient descent.  Tracking this as a setting moves
    toward a grid boundary separates a family converging on the baseline --
    where a wider grid only chases an asymptote -- from one whose optimum
    genuinely lies outside the grid.
    """
    import numpy as np

    from .gauges import GAUGE_FAMILIES

    probes = PROBE_RANGE

    if axis == "shape":
        family = GAUGE_FAMILIES.get(method)
        if not family:
            return None
        factors = family[0](value).factor(0.09, probes)
    elif axis == "coefficient" and method == "KarciFANN":
        from .fod import fod_factor

        factors = fod_factor(0.09, probes, value)
    elif axis == "coefficient" and method in ("CF terminal", "CF chained"):
        from .caputo_fabrizio import cf_factor

        factors = cf_factor(probes, min(value, 1.0))
    else:
        return None
    return float(np.max(np.abs(np.asarray(factors, float) - 1.0)))


def _is_identity(method: str, axis: str, value) -> bool:
    """Is this rule, at this setting, numerically plain gradient descent?

    Covers both parameterisations in use: the gauge families are indexed by a
    shape, while the fractional rules of the chained comparison are indexed by
    an order that degenerates at ``alpha = 1``.
    """
    import numpy as np

    from .gauges import GAUGE_FAMILIES

    if axis == "shape":
        family = GAUGE_FAMILIES.get(method)
        return bool(family) and family[0](value).is_identity()

    if axis != "coefficient":
        return False

    # Same probe range as Gauge.is_identity, so the two routes agree.
    probes = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    if method == "KarciFANN":
        from .fod import fod_factor

        factors = fod_factor(0.09, probes, value)
    elif method in ("CF terminal", "CF chained"):
        from .caputo_fabrizio import cf_factor

        factors = cf_factor(probes, min(value, 1.0))
    else:
        return False
    return bool(np.all(np.abs(np.asarray(factors, float) - 1.0) <= 1e-3))


def find_edge_winners(results_dir: Path | str = "results") -> list[dict]:
    """Every edge winner in a results directory, from both detection routes.

    ``self-reported`` picks up any row a script flagged with ``grid_edge``;
    ``reconstructed`` rebuilds the grid from the tuning rows, which catches the
    scripts that never recorded a flag at all.
    """
    results_dir = Path(results_dir)
    findings: list[dict] = []
    if not results_dir.exists():
        return findings

    for path in sorted(results_dir.glob("*.csv")):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        for row in rows:
            if row.get("grid_edge") == "1":
                findings.append({
                    "file": path.name, "source": "self-reported",
                    "method": row.get("method") or row.get("configuration", ""),
                    "axis": "", "winner": row.get("setting", ""),
                    "grid_min": "", "grid_max": "", "at": "edge",
                    "kind": "self-reported",
                })

        for pattern, (group_columns, axis_columns) in EDGE_SPECS.items():
            if not path.match(pattern):
                continue
            for finding in edges_in_rows(rows, group_columns, axis_columns):
                if _confounded(path.name, finding["method"], finding["axis"]):
                    continue
                findings.append({"file": path.name, "source": "reconstructed", **finding})
    return findings


def _approaches_identity(method: str, axis: str, winner, searched) -> bool:
    """Is the rule converging on gradient descent as it nears this boundary?

    Judged on the operator itself rather than on measured accuracy, because
    each tuning cell is a single run and a genuinely monotone trend can wobble
    by a fraction of a point.
    """
    ordered = sorted(searched, reverse=winner == max(searched))
    distances = [identity_distance(method, axis, v) for v in ordered[:4]]
    if len(distances) < 3 or any(d is None for d in distances):
        return False
    # ordered[0] is the winning end, so distance must grow as we move inward.
    return all(a < b for a, b in zip(distances, distances[1:]))


def _confounded(filename: str, method: str, axis: str) -> bool:
    """Axes that are not independent hyperparameters and cannot be at an edge.

    ``chained_vs_terminal`` gives gradient descent a learning rate of
    ``coefficient x scale``, so neither factor is a knob on its own: a winner
    at the end of one is not a truncated search as long as the product is
    interior, which the scale axis already covers.
    """
    return filename.startswith("chained_vs_terminal") and method == "gradient descent"
