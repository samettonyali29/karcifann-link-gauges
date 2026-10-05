"""Session-wide invariants that no individual test owns."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _fingerprint() -> dict[str, str]:
    if not RESULTS.exists():
        return {}
    return {
        path.name: hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted(RESULTS.glob("*.csv"))
    }


@pytest.fixture(scope="session", autouse=True)
def results_are_read_only():
    """Running the tests must not rewrite the committed results.

    The figure tests drive the real plotting functions, and those write summary
    CSVs as a side effect.  Before this was noticed, a plain `pytest` run
    silently rewrote results/summary_gauge_ablation.csv and
    results/stats_tolerance.csv -- and would happily have written half-finished
    numbers into them while experiments were still running.
    """
    before = _fingerprint()
    yield
    after = _fingerprint()

    changed = sorted(n for n in before.keys() & after.keys() if before[n] != after[n])
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    if changed or added or removed:
        pytest.fail(
            "the test run modified results/:\n"
            + "".join(f"  modified {n}\n" for n in changed)
            + "".join(f"  created  {n}\n" for n in added)
            + "".join(f"  deleted  {n}\n" for n in removed)
            + "tests must write to tmp_path, not to the committed results"
        )
