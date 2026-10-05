"""Figure plumbing: both vector formats, every time."""

import numpy as np
import pytest

from karcifann.plotting import METHOD_COLOURS, colour, marker, save, setup


@pytest.fixture(scope="module")
def plt():
    return setup()


def test_save_writes_both_pdf_and_svg(plt, tmp_path):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    written = save(fig, "example", tmp_path)

    assert [p.suffix for p in written] == [".pdf", ".svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert written[0].read_bytes().startswith(b"%PDF")
    assert b"<svg" in written[1].read_bytes()


def test_svg_keeps_text_as_text_not_paths(plt, tmp_path):
    """svg.fonttype='none' keeps labels editable in Inkscape or Illustrator."""
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.set_xlabel("a distinctive label")
    svg = save(fig, "text", tmp_path)[1].read_text()
    assert "a distinctive label" in svg


def test_save_creates_a_missing_directory(plt, tmp_path):
    fig, _ = plt.subplots(figsize=(1, 1))
    target = tmp_path / "deep" / "nested"
    assert save(fig, "fig", target)[0].exists()


def test_colours_are_stable_and_distinct_across_methods():
    names = ["gradient descent", "KarciFANN", "exp", "tanh", "log"]
    assigned = [colour(n) for n in names]
    assert len(set(assigned)) == len(names)
    assert colour("KarciFANN") == colour("power")     # same method, two labels


def test_unknown_method_still_gets_a_style():
    assert colour("something new").startswith("#")
    assert marker("something new")


def test_every_declared_colour_is_a_hex_triple():
    assert all(len(v) == 7 and v.startswith("#") for v in METHOD_COLOURS.values())


def test_saving_the_same_figure_twice_gives_identical_bytes(tmp_path):
    """Figure output must be byte-reproducible, not merely equivalent.

    Matplotlib salts every generated element id per process and stamps a
    creation date into both backends, so two runs of the same code produced
    files that differed in every id while drawing the same picture.  A
    clean-room regeneration then had to be checked with a normaliser instead of
    `cmp`, which is exactly the kind of "probably the same" that hides a real
    change.  setup() pins svg.hashsalt and save() suppresses the timestamp.
    """
    import numpy as np

    from karcifann.plotting import save, setup

    def draw():
        plt = setup()
        fig, ax = plt.subplots()
        ax.plot(np.arange(10), np.arange(10) ** 0.5, label="root")
        ax.fill_between(np.arange(10), 0, np.arange(10) ** 0.5, alpha=0.2)
        ax.legend()
        ax.set_title("determinism")
        return fig

    first = {p.suffix: p.read_bytes()
             for p in save(draw(), "determinism", tmp_path / "a")}
    second = {p.suffix: p.read_bytes()
              for p in save(draw(), "determinism", tmp_path / "b")}

    assert set(first) == {".pdf", ".svg"}
    for suffix in first:
        assert first[suffix] == second[suffix], (
            f"{suffix} output is not byte-reproducible across runs")
