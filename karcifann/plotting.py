"""Figure helpers.  Every figure is written as both PDF and SVG.

Matplotlib is imported lazily so the rest of the package keeps working without
it.  The style is deliberately plain: vector output, no chart junk, colours
that survive greyscale printing, and fonts large enough to stay legible when a
two-column paper shrinks the figure to 8 cm.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["FIGURE_DIR", "METHOD_COLOURS", "METHOD_MARKERS", "save", "setup"]

FIGURE_DIR = Path("figures")

#: Stable colour per method so a reader can track one across every figure.
METHOD_COLOURS = {
    "gradient descent": "#444444",
    "KarciFANN": "#0072B2",
    "power": "#0072B2",
    "exp": "#D55E00",
    "tanh": "#009E73",
    "log": "#CC79A7",
    "CF terminal": "#E69F00",
    "CF chained": "#56B4E9",
    "full": "#0072B2",
    "error": "#D55E00",
    "weight": "#009E73",
}
METHOD_MARKERS = {
    "gradient descent": "o",
    "KarciFANN": "s",
    "power": "s",
    "exp": "^",
    "tanh": "D",
    "log": "v",
    "full": "s",
    "error": "^",
    "weight": "D",
}


def setup():
    """Apply the shared style and return the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        # Type 42 keeps text selectable and editable in the PDF.
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        # Matplotlib salts every generated element id (clip paths, markers,
        # glyphs) per process, so two runs of the same code produce SVGs that
        # differ in every id while drawing exactly the same picture.  A fixed
        # salt makes the output byte-reproducible, which is what lets a
        # regeneration be checked with cmp rather than a normaliser.
        "svg.hashsalt": "karcifann",
    })
    return plt


def save(fig, name: str, outdir: Path | str = FIGURE_DIR) -> list[Path]:
    """Write ``name.pdf`` and ``name.svg`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Suppress the creation timestamp so identical inputs give identical
    # bytes; the key differs between the two backends.
    stamp = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

    written = []
    for suffix in ("pdf", "svg"):
        path = outdir / f"{name}.{suffix}"
        fig.savefig(path, format=suffix, metadata=stamp[suffix])
        written.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def colour(name: str) -> str:
    return METHOD_COLOURS.get(name, "#777777")


def marker(name: str) -> str:
    return METHOD_MARKERS.get(name, "o")
