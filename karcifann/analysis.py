"""Statistical tools for comparing optimizers over seeds and datasets.

The design follows Demsar, *JMLR* 7:1-30, 2006, with one deliberate departure.
Demsar's protocol blocks on **datasets**, and with only a handful of them the
Friedman test has almost no power: with 5 methods and 3 datasets the smallest
p-value the statistic can produce -- under perfect, unanimous rank agreement --
is 0.017.  :func:`friedman_power_floor` computes that bound so a paper can say
how much evidence its design is capable of carrying before it reports any.

Where the dataset count is small, the better-founded test here blocks on
**seeds** within a dataset.  Every method in these experiments is initialised
from the same seed list, so runs pair exactly and a Wilcoxon signed-rank test
on the pairs is legitimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, friedmanchisquare, rankdata, skew, wilcoxon

__all__ = [
    "ComparisonResult",
    "cliffs_delta",
    "cohens_d",
    "friedman_over_datasets",
    "friedman_power_floor",
    "nemenyi_critical_difference",
    "wilcoxon_power_floor",
    "holm",
    "pairwise_wilcoxon",
    "sign_test",
    "difference_skew",
    "paired_permutation_test",
    "average_ranks",
    "tolerance_region",
]


def holm(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni step-down correction.

    Returns ``(adjusted, rejected)``.  Adjusted values are clipped to 1 and
    made monotone, so they can be reported directly in a p-value matrix.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted, adjusted <= alpha


def cohens_d(a, b) -> float:
    """Paired Cohen's d: mean difference over the SD of the differences.

    A difference that is identical on every seed has zero variance.  That is
    not "no effect" -- it is a perfectly reliable one -- so the signed infinity
    is returned rather than zero, which would invert the reading.  Only a
    difference that is both zero and constant gives ``0``.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    spread = diff.std(ddof=1)
    if spread > 0:
        return float(diff.mean() / spread)
    return 0.0 if np.isclose(diff.mean(), 0.0) else float(np.sign(diff.mean()) * np.inf)


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]; 0 means complete overlap."""
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    return float((np.sign(a - b)).mean())


def paired_permutation_test(a, b, exact_limit: int = 1 << 20, resamples: int = 100_000,
                            seed: int = 0) -> float:
    """Two-sided exact paired permutation test on the mean difference.

    This is the test the other two are working around.  Wilcoxon needs the
    paired differences to be symmetric about their median -- an assumption
    violated in a third of the comparisons here -- and the sign test avoids it
    only by discarding the magnitudes, and with them most of the power.

    The null is that the joint distribution of the paired differences is
    invariant under flipping their signs; with independent pairs, symmetry of
    each difference about zero supplies that.  Under it the sign of each pair
    is exchangeable, so the null distribution is obtained by enumerating every
    assignment of signs.  With ``n`` pairs there are ``2**n`` of them: at 16
    pairs that is 65536, which is enumerated exactly.  Beyond ``exact_limit`` a
    random subset is sampled and the p-value is the usual
    ``(1 + #extreme) / (1 + #draws)`` estimate, which stays valid rather than
    merely approximate.

    This is *not* assumption-free, and earlier versions of this docstring and
    of Section 6.2 said it was.  What it avoids is Wilcoxon's symmetry about
    the *median* and the loss of magnitude information in the sign test; it
    still needs sign-invariance under the null, which is stronger than a zero
    mean.  Running two deterministic methods from a shared seed pairs the runs
    but does not randomly assign the labels, so exactness rests on that
    assumption rather than on randomisation.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    difference = difference[difference != 0]
    n = difference.size
    if n == 0:
        return 1.0

    observed = abs(difference.mean())
    if (1 << n) <= exact_limit:
        # Every sign pattern, as the rows of a 2**n x n matrix of +-1.
        patterns = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * -2 + 1
        means = np.abs(patterns @ difference) / n
        return float((means >= observed - 1e-15).mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(resamples, n))
    means = np.abs(signs @ difference) / n
    return float((1 + int((means >= observed - 1e-15).sum())) / (1 + resamples))


def sign_test(a, b) -> float:
    """Two-sided exact sign test on paired runs.

    The Wilcoxon signed-rank test assumes the paired differences are symmetric
    about their median.  The sign test assumes only independence, at the cost
    of power: it uses only the direction of each pair.  Reporting both shows
    whether a conclusion depends on the estimand -- a mean difference against
    sign balance.  Agreement between them does not establish that symmetry
    holds or that it did not matter; it shows only that two different
    summaries point the same way.

    With 16 pairs the exact floor is 2/2**16 = 3e-5, so it is not the binding
    constraint here.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = difference[difference != 0]
    if nonzero.size == 0:
        return 1.0
    return float(binomtest(int((nonzero > 0).sum()), nonzero.size, 0.5).pvalue)


def difference_skew(a, b) -> float:
    """Skewness of the paired differences; 0 is symmetric.

    The quantity the Wilcoxon test assumes away.  ``|skew| > 1`` is usually
    read as substantial asymmetry.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(difference, difference[0]):
        return 0.0
    return float(skew(difference))


@dataclass
class ComparisonResult:
    names: list[str]
    pvalues: np.ndarray       # raw pairwise p-values (symmetric, nan on diagonal)
    adjusted: np.ndarray      # Holm-adjusted
    effect: np.ndarray        # paired Cohen's d, row minus column
    delta: np.ndarray         # Cliff's delta, row minus column
    sign: np.ndarray          # exact sign-test p, assumption-free but low-powered
    skew: np.ndarray          # skewness of the paired differences
    permutation: np.ndarray   # exact paired permutation p -- the primary test


def pairwise_wilcoxon(scores: dict[str, np.ndarray], alpha: float = 0.05) -> ComparisonResult:
    """Pairwise Wilcoxon signed-rank tests over paired runs, Holm-corrected.

    ``scores`` maps a method name to its per-seed results, all seeds in the
    same order so the pairing is real.
    """
    names = list(scores)
    n = len(names)
    raw = np.full((n, n), np.nan)
    effect = np.full((n, n), np.nan)
    delta = np.full((n, n), np.nan)
    sign = np.full((n, n), np.nan)
    asymmetry = np.full((n, n), np.nan)
    permutation = np.full((n, n), np.nan)

    flat: list[float] = []
    index: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.asarray(scores[names[i]]), np.asarray(scores[names[j]])
            if a.shape != b.shape:
                raise ValueError(f"{names[i]} and {names[j]} have different run counts")
            if np.allclose(a, b):
                p = 1.0
            else:
                p = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
            flat.append(p)
            index.append((i, j))
            raw[i, j] = raw[j, i] = p
            effect[i, j] = cohens_d(a, b)
            effect[j, i] = -effect[i, j]
            delta[i, j] = cliffs_delta(a, b)
            delta[j, i] = -delta[i, j]
            sign[i, j] = sign[j, i] = sign_test(a, b)
            permutation[i, j] = permutation[j, i] = paired_permutation_test(a, b)
            asymmetry[i, j] = difference_skew(a, b)
            asymmetry[j, i] = -asymmetry[i, j]

    adjusted_flat, _ = holm(flat, alpha) if flat else (np.array([]), None)
    adjusted = np.full((n, n), np.nan)
    for (i, j), value in zip(index, adjusted_flat):
        adjusted[i, j] = adjusted[j, i] = value
    return ComparisonResult(names, raw, adjusted, effect, delta, sign, asymmetry, permutation)


def average_ranks(table: np.ndarray) -> np.ndarray:
    """Mean rank per method; ``table`` is (blocks, methods), higher is better."""
    return np.array([rankdata(-row) for row in np.atleast_2d(table)]).mean(axis=0)


def friedman_over_datasets(table: np.ndarray):
    """Friedman test with datasets as blocks.  ``table`` is (datasets, methods)."""
    table = np.atleast_2d(table)
    statistic, pvalue = friedmanchisquare(*table.T)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "ranks": average_ranks(table),
        "blocks": table.shape[0],
        "methods": table.shape[1],
        "power_floor": friedman_power_floor(table.shape[0], table.shape[1]),
        "critical_difference": nemenyi_critical_difference(table.shape[1], table.shape[0]),
    }


def friedman_power_floor(blocks: int, methods: int) -> float:
    """Smallest p-value Friedman can return for this design.

    Attained when every block ranks the methods identically.  If this is not
    comfortably below the intended alpha, the design cannot support a
    conclusion no matter what the data look like.
    """
    if blocks < 2 or methods < 3:
        return float("nan")
    unanimous = np.tile(np.arange(methods, 0, -1.0), (blocks, 1))
    return float(friedmanchisquare(*unanimous.T).pvalue)


#: Studentized range statistic at alpha = 0.05 divided by sqrt(2), indexed by
#: the number of methods (Demsar 2006, Table 5).
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def nemenyi_critical_difference(methods: int, blocks: int, alpha: float = 0.05) -> float:
    """Smallest average-rank gap that counts as significant after Friedman.

    ``CD = q_alpha sqrt(k(k+1) / 6N)`` for ``k`` methods over ``N`` datasets.
    This is the all-pairs post-hoc of Demsar (2006) and it is deliberately
    conservative: a significant Friedman says *some* method differs, and the
    critical difference says which pairs that licence actually covers.  Two
    methods whose ranks differ by less than the CD are not distinguished, no
    matter how the overall test came out.
    """
    if methods not in _NEMENYI_Q05:
        raise ValueError(f"no tabulated q for {methods} methods; expected 2-10")
    if alpha != 0.05:
        raise ValueError("only alpha = 0.05 is tabulated here")
    return float(_NEMENYI_Q05[methods] * np.sqrt(methods * (methods + 1) / (6.0 * blocks)))


def wilcoxon_power_floor(pairs: int, comparisons: int = 1) -> float:
    """Smallest adjusted p a paired Wilcoxon can return for this design.

    The exact two-sided test on ``n`` pairs cannot go below ``2 / 2**n``, so a
    Holm correction over ``m`` comparisons cannot go below ``m * 2 / 2**n``.
    With 8 seeds and 10 pairwise comparisons that floor is 0.078: **no result,
    however large, can be called significant at 0.05**.  Ten seeds clear it,
    sixteen clear it comfortably.  Check this before running the experiment,
    not after.
    """
    if pairs < 1:
        return float("nan")
    # Computed from the combinatorics rather than by handing SciPy a vector of
    # identical differences: that construction ties every rank, which pushes
    # SciPy's automatic method onto its normal approximation and returned
    # 6.3e-05 for 16 pairs where the exact bound is 3.1e-05.  The bound itself
    # is elementary -- only one of the 2**n sign assignments puts all the rank
    # mass on one side, and the two-sided test doubles it.
    raw = 2.0 / 2.0 ** pairs
    return float(min(raw * comparisons, 1.0))


def tolerance_region(values, scores, tolerance: float = 0.01) -> dict:
    """How wide is the region of a hyperparameter that works?

    ``scores`` are results (higher is better) at each point of ``values``.  The
    region is every point within ``tolerance`` (relative) of the best score.
    Reported as a count, a fraction of the grid, and -- for a log-spaced grid --
    the width in decades, which is the scale-free way to compare the basin of
    a learning rate against the basin of a fractional order.
    """
    values = np.asarray(values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    best = np.nanmax(scores)
    keep = scores >= best * (1.0 - tolerance)
    inside = values[keep]
    decades = float(np.log10(inside.max() / inside.min())) if inside.size and inside.min() > 0 else 0.0
    return {
        "best": float(best),
        "best_at": float(values[int(np.nanargmax(scores))]),
        "count": int(keep.sum()),
        "fraction": float(keep.mean()),
        "min": float(inside.min()) if inside.size else float("nan"),
        "max": float(inside.max()) if inside.size else float("nan"),
        "decades": decades,
    }
