"""The CSV writer every experiment shares."""

import csv

import pytest

from karcifann import Table


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_writes_one_row_per_observation(tmp_path):
    table = Table("simple", tmp_path)
    table.add(method="a", score=1.0)
    table.add(method="b", score=2.0)
    path = table.write(quiet=True)

    rows = read(path)
    assert [r["method"] for r in rows] == ["a", "b"]
    assert [r["score"] for r in rows] == ["1.0", "2.0"]


def test_a_column_added_partway_through_is_kept_and_backfilled(tmp_path):
    """An experiment may start recording a new quantity mid-run."""
    table = Table("ragged", tmp_path)
    table.add(a=1)
    table.add(a=2, b=3)
    rows = read(table.write(quiet=True))

    assert table.columns == ["a", "b"]
    assert rows[0]["b"] == ""
    assert rows[1]["b"] == "3"


def test_column_order_follows_first_appearance(tmp_path):
    table = Table("order", tmp_path)
    table.add(z=1, a=2)
    table.add(m=3)
    assert table.columns == ["z", "a", "m"]


def test_writing_nothing_creates_no_file(tmp_path):
    table = Table("empty", tmp_path)
    assert table.write(quiet=True) is None
    assert not (tmp_path / "empty.csv").exists()


def test_creates_a_missing_output_directory(tmp_path):
    table = Table("nested", tmp_path / "deep" / "deeper")
    table.add(x=1)
    assert table.write(quiet=True).exists()


def test_extend_copies_the_rows_it_is_given(tmp_path):
    source = [{"a": 1}, {"a": 2}]
    table = Table("extended", tmp_path)
    table.extend(source)
    source[0]["a"] = 99
    assert table.rows[0]["a"] == 1
    assert len(table) == 2


def test_defaults_to_the_shared_results_directory(monkeypatch, tmp_path):
    import karcifann.records as records

    monkeypatch.setattr(records, "RESULTS_DIR", tmp_path)
    table = Table("defaulted")
    table.add(x=1)
    assert table.write(quiet=True).parent == tmp_path


def test_rewriting_replaces_rather_than_appends(tmp_path):
    table = Table("rewrite", tmp_path)
    table.add(x=1)
    table.write(quiet=True)
    path = table.write(quiet=True)
    assert len(read(path)) == 1
