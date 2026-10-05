"""Tidy CSV output for the experiment scripts.

Every experiment prints a table to the terminal *and* records the same numbers
here, so that a result can be re-plotted or re-tested without re-running the
computation that produced it.

One row per observation, one column per variable, no merged cells and no
formatting -- the console table is the human view, the CSV is the data.
"""

from __future__ import annotations

import csv
from pathlib import Path

__all__ = ["RESULTS_DIR", "Table"]

RESULTS_DIR = Path("results")


class Table:
    """Accumulates rows of results and writes them to ``results/<name>.csv``.

    Rows may carry different keys; the union is used as the header and missing
    entries are written blank, so an experiment can add a column partway
    through without a second pass.
    """

    def __init__(self, name: str, outdir: Path | str | None = None) -> None:
        self.name = name
        self.outdir = Path(outdir) if outdir is not None else RESULTS_DIR
        self.rows: list[dict] = []

    def add(self, **row) -> dict:
        self.rows.append(row)
        return row

    def extend(self, rows) -> None:
        for row in rows:
            self.rows.append(dict(row))

    @property
    def columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def write(self, quiet: bool = False) -> Path | None:
        """Write the CSV.  Returns ``None`` when there is nothing to write."""
        if not self.rows:
            return None
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / f"{self.name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, restval="")
            writer.writeheader()
            writer.writerows(self.rows)
        if not quiet:
            print(f"   wrote {path} ({len(self.rows)} rows)", flush=True)
        return path

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Table({self.name!r}, rows={len(self.rows)})"
