"""Datasets for the KarciFANN experiments.

``xor`` reproduces the toy problem of Karakurt et al. (2022); ``digits`` is the
8x8 UCI digit set that ships with scikit-learn and stands in for MNIST when no
download is possible; ``mnist`` fetches the real thing through
``sklearn.datasets.fetch_openml``.

``dry_bean`` and ``letter`` are the non-image tabular sets.  Dry Bean is the
one Karakurt (2025) classifies, so results on it are directly comparable with
the published KarciFANN numbers.  Together the three classification sets span
7, 10 and 26 classes, which for one-hot targets under MSE is a 3.3x sweep of
the loss scale -- the quantity the fractional prefactor ``(J/W)^(alpha-1)``
is most sensitive to.

Downloads are cached under scikit-learn's data home (``~/scikit_learn_data``
unless ``SCIKIT_LEARN_DATA`` says otherwise).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .metrics import one_hot

__all__ = [
    "TABULAR",
    "xor",
    "digits",
    "mnist",
    "dry_bean",
    "letter",
    "openml_classification",
    "optdigits",
    "pendigits",
    "satimage",
    "segment",
    "vehicle",
    "load",
    "standardize",
    "train_val_test_split",
]

DRY_BEAN_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DRY_BEAN_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"


def xor() -> tuple[np.ndarray, np.ndarray]:
    """The 2-input XOR truth table."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    t = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x, t


def digits(normalise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """1797 8x8 handwritten digits, flattened to 64 features, one-hot targets."""
    from sklearn.datasets import load_digits

    data = load_digits()
    x = data.data.astype(float)
    if normalise:
        x = x / 16.0
    return x, one_hot(data.target, 10)


def mnist(normalise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """70000 28x28 MNIST digits flattened to 784 features (downloads once)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = data.data.astype(float)
    if normalise:
        x = x / 255.0
    return x, one_hot(data.target.astype(int), 10)


def _data_home() -> Path:
    from sklearn.datasets import get_data_home

    home = Path(get_data_home())
    home.mkdir(parents=True, exist_ok=True)
    return home


def standardize(x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Zero mean, unit variance per column; constant columns are left at zero.

    Tabular features here span six orders of magnitude in the raw data, and the
    KarciFANN prefactor depends on ``|W|``, so the input scale interacts with
    the optimizer in a way it does not for plain gradient descent.  Scaling is
    therefore part of the method, not just hygiene.
    """
    reference = x if reference is None else reference
    mean = reference.mean(axis=0)
    spread = reference.std(axis=0)
    return (x - mean) / np.where(spread > 0, spread, 1.0)


def dry_bean(scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """13611 dry beans, 16 morphological features, 7 cultivars (UCI 602).

    The dataset Karakurt (2025) uses.  Read from the UCI ARFF rather than from
    OpenML: the OpenML copy (id 45741) is a processed variant whose features
    are already standardised and whose target is a flattened multi-label array.
    """
    from scipy.io import arff

    cache = _data_home() / "dry_bean_dataset.zip"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(DRY_BEAN_URL, timeout=120).read())

    with zipfile.ZipFile(cache) as archive:
        text = archive.read(DRY_BEAN_MEMBER).decode()
    data, meta = arff.loadarff(io.StringIO(text))

    features = [name for name in meta.names() if name != "Class"]
    x = np.column_stack([data[name].astype(float) for name in features])
    labels = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in data["Class"]])
    classes = np.unique(labels)
    y = one_hot(np.searchsorted(classes, labels), len(classes))
    return (standardize(x) if scale else x), y


def letter(scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """20000 letter images reduced to 16 integer features, 26 classes (UCI)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="letter", version=1, as_frame=False, parser="liac-arff")
    x = np.asarray(data.data, dtype=float)
    labels = np.asarray(data.target).ravel()
    classes = np.unique(labels)
    y = one_hot(np.searchsorted(classes, labels), len(classes))
    return (standardize(x) if scale else x), y


def openml_classification(name: str, version: int = 1, scale: bool = True):
    """Fetch a numeric OpenML classification set as ``(x, one_hot_targets)``.

    Used for the additional tabular benchmarks.  Labels are mapped through
    ``np.unique`` so the class order is deterministic regardless of how the
    ARFF happens to list them.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name=name, version=version, as_frame=False, parser="liac-arff")
    x = np.asarray(data.data, dtype=float)
    labels = np.asarray(data.target).ravel()
    classes = np.unique(labels)
    y = one_hot(np.searchsorted(classes, labels), len(classes))
    return (standardize(x) if scale else x), y


def optdigits(scale: bool = True):
    """5620 handwritten digits as 64 integer pen-density features, 10 classes."""
    return openml_classification("optdigits", 1, scale)


def pendigits(scale: bool = True):
    """10992 digits as 16 pen-trajectory coordinates, 10 classes."""
    return openml_classification("pendigits", 1, scale)


def satimage(scale: bool = True):
    """6430 satellite image patches, 36 spectral features, 6 soil types."""
    return openml_classification("satimage", 1, scale)


def segment(scale: bool = True):
    """2310 outdoor image regions, 19 features, 7 surface classes."""
    return openml_classification("segment", 1, scale)


def vehicle(scale: bool = True):
    """846 vehicle silhouettes, 18 shape features, 4 vehicle types."""
    return openml_classification("vehicle", 1, scale)


#: Every non-image tabular benchmark, in ascending class count.  Eight
#: classification sets in total (with MNIST) is what lifts the Friedman test
#: over its power floor; see :func:`karcifann.analysis.friedman_power_floor`.
TABULAR = {
    "vehicle": vehicle,
    "satimage": satimage,
    "dry_bean": dry_bean,
    "segment": segment,
    "optdigits": optdigits,
    "pendigits": pendigits,
    "letter": letter,
}


def load(name: str, scale: bool = True):
    """Look a dataset up by name, tabular or image.

    ``scale=False`` returns the raw features so that
    :func:`train_val_test_split` can standardise from the training split alone.
    """
    if name in TABULAR:
        return TABULAR[name](scale=scale)
    if name == "mnist":
        return mnist()
    if name == "digits":
        return digits()
    raise ValueError(f"unknown dataset {name!r}; expected one of "
                     f"{sorted([*TABULAR, 'mnist', 'digits'])}")


def train_val_test_split(
    x: np.ndarray,
    t: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    stratify: bool = True,
    standardize_features: bool = False,
):
    """Split into train / validation / test.

    ``stratify`` (on by default) splits within each class so the class
    proportions of the whole set are preserved in every part.  A plain
    permutation is badly behaved on the smaller sets here: on Vehicle it left
    the class proportions 47 % adrift, enough to move the measured accuracies
    by several points.

    ``standardize_features`` fits the mean and spread on the **training split
    only** and applies them to all three.  Scaling before splitting lets the
    validation and test rows influence the statistics -- a small leak for
    standardisation, but one with no reason to accept.
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))

    if stratify and t.ndim == 2 and t.shape[1] > 1:
        labels = np.argmax(t, axis=1)
        test_idx, val_idx, train_idx = [], [], []
        for label in np.unique(labels):
            members = np.where(labels == label)[0]
            rng.shuffle(members)
            take_test = int(round(len(members) * test_fraction))
            take_val = int(round(len(members) * val_fraction))
            test_idx += list(members[:take_test])
            val_idx += list(members[take_test : take_test + take_val])
            train_idx += list(members[take_test + take_val :])
        test_idx, val_idx, train_idx = (np.array(sorted(i)) for i in
                                        (test_idx, val_idx, train_idx))
    else:
        idx = rng.permutation(n)
        test_idx, val_idx, train_idx = idx[:n_test], idx[n_test : n_test + n_val], idx[n_test + n_val :]

    parts = [(x[train_idx], t[train_idx]), (x[val_idx], t[val_idx]), (x[test_idx], t[test_idx])]
    if standardize_features:
        reference = parts[0][0]
        parts = [(standardize(features, reference=reference), targets)
                 for features, targets in parts]
    return tuple(parts)
