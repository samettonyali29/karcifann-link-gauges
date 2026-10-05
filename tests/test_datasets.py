"""Dataset loaders, scaling and splitting.

The loaders that download are marked ``network`` so the suite still runs
offline: ``pytest -m "not network"``.  They are worth testing because the
Dry Bean loader deliberately avoids the OpenML copy of that dataset, which is
a processed variant with standardised features and a flattened multi-label
target -- a silent wrong answer rather than an error.
"""

import numpy as np
import pytest

from karcifann import dry_bean, letter, standardize, train_val_test_split, xor
from karcifann.datasets import digits


# ---------------------------------------------------------------------- xor


def test_xor_is_the_truth_table():
    x, t = xor()
    assert x.shape == (4, 2) and t.shape == (4, 1)
    for inputs, target in zip(x, t):
        assert target[0] == float(int(inputs[0]) ^ int(inputs[1]))


def test_xor_is_not_linearly_separable():
    """The reason the problem is used at all: no single line splits it."""
    x, t = xor()
    augmented = np.hstack([x, np.ones((4, 1))])
    weights, *_ = np.linalg.lstsq(augmented, t.ravel(), rcond=None)
    assert not np.array_equal((augmented @ weights > 0.5).astype(float), t.ravel())


# ------------------------------------------------------------------ scaling


def test_standardize_centres_and_scales():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, (200, 4))
    z = standardize(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-12)


def test_standardize_leaves_a_constant_column_at_zero():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    z = standardize(x)
    assert np.all(z[:, 0] == 0.0)
    assert np.isfinite(z).all()


def test_standardize_can_reuse_training_statistics():
    """Validation data must be scaled by the training mean and spread."""
    train = np.arange(10.0).reshape(-1, 1)
    held_out = np.array([[20.0]])
    scaled = standardize(held_out, reference=train)
    assert scaled[0, 0] == pytest.approx((20.0 - train.mean()) / train.std())


# ------------------------------------------------------------------ splitting


def test_splits_preserve_class_proportions():
    """Stratification, on by default.

    A plain permutation left Vehicle's class proportions 47 % adrift, which
    moved the measured accuracies by several points and tripled one reported
    gap.  Balance is therefore a property of the splitter, not of luck.
    """
    rng = np.random.default_rng(0)
    labels = rng.choice(4, 800, p=[0.5, 0.25, 0.15, 0.10])      # deliberately skewed
    x = rng.normal(0, 1, (800, 5))
    t = np.eye(4)[labels]

    full = t.mean(axis=0)
    for part, _ in train_val_test_split(x, t, 0.15, 0.15, seed=0):
        pass
    for _, targets in train_val_test_split(x, t, 0.15, 0.15, seed=0):
        drift = np.max(np.abs(targets.mean(axis=0) - full) / full)
        assert drift < 0.05, f"class proportions drifted {drift:.1%}"


def test_stratification_can_be_switched_off():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 3))
    t = np.eye(2)[rng.integers(0, 2, 200)]
    a = train_val_test_split(x, t, 0.2, 0.2, seed=0, stratify=True)[2][0]
    b = train_val_test_split(x, t, 0.2, 0.2, seed=0, stratify=False)[2][0]
    assert not np.array_equal(a, b)


def test_features_are_scaled_from_the_training_split_alone():
    """Scaling before splitting lets held-out rows set the mean and spread."""
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 2.0, (400, 4))
    t = np.eye(2)[rng.integers(0, 2, 400)]
    (xtr, _), (xva, _), (xte, _) = train_val_test_split(
        x, t, 0.2, 0.2, seed=0, standardize_features=True)

    assert np.allclose(xtr.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(xtr.std(axis=0), 1.0, atol=1e-12)
    # held-out splits are scaled by the training statistics, so they are close
    # to but not exactly standardised -- that difference is the leak, removed.
    assert not np.allclose(xva.mean(axis=0), 0.0, atol=1e-12)
    assert np.all(np.abs(xva.mean(axis=0)) < 0.4)
    assert np.all(np.abs(xte.mean(axis=0)) < 0.4)


def test_split_partitions_without_overlap_or_loss():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (100, 3))
    t = np.eye(2)[rng.integers(0, 2, 100)]
    (xtr, _), (xva, _), (xte, _) = train_val_test_split(x, t, 0.2, 0.2, seed=0)

    assert len(xtr) + len(xva) + len(xte) == 100
    assert abs(len(xva) - 20) <= 2 and abs(len(xte) - 20) <= 2
    rows = {tuple(r) for r in np.vstack([xtr, xva, xte])}
    assert len(rows) == 100                      # every row appears exactly once


def test_split_keeps_inputs_aligned_with_targets():
    x = np.arange(60.0).reshape(-1, 1)
    t = x * 2.0
    for inputs, targets in train_val_test_split(x, t, 0.2, 0.2, seed=3):
        assert np.allclose(targets, inputs * 2.0)


def test_split_is_reproducible_and_seed_dependent():
    x = np.arange(50.0).reshape(-1, 1)
    t = np.eye(2)[np.arange(50) % 2]
    a = train_val_test_split(x, t, 0.2, 0.2, seed=7)[0][0]
    b = train_val_test_split(x, t, 0.2, 0.2, seed=7)[0][0]
    c = train_val_test_split(x, t, 0.2, 0.2, seed=8)[0][0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ------------------------------------------------------------------- digits


def test_digits_shape_and_scaling():
    x, t = digits()
    assert x.shape == (1797, 64) and t.shape == (1797, 10)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert np.array_equal(t.sum(axis=1), np.ones(1797))


def test_digits_can_be_left_unscaled():
    raw, _ = digits(normalise=False)
    assert raw.max() > 1.0


# ---------------------------------------------------------- tabular datasets


@pytest.mark.network
def test_dry_bean_matches_the_published_description():
    """13611 beans, 16 features, 7 imbalanced cultivars (UCI 602)."""
    x, t = dry_bean()
    assert x.shape == (13611, 16)
    assert t.shape == (13611, 7)

    counts = sorted(t.sum(axis=0).astype(int).tolist())
    assert counts == [522, 1322, 1630, 1928, 2027, 2636, 3546]
    assert np.array_equal(t.sum(axis=1), np.ones(13611))


@pytest.mark.network
def test_dry_bean_is_standardised_by_default():
    x, _ = dry_bean()
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-9)


@pytest.mark.network
def test_dry_bean_raw_features_span_six_orders_of_magnitude():
    """Why scaling is part of the method here and not just hygiene."""
    x, _ = dry_bean(scale=False)
    assert x.max() / x[x > 0].min() > 1e6


@pytest.mark.network
def test_letter_shape_and_classes():
    x, t = letter()
    assert x.shape == (20000, 16)
    assert t.shape == (20000, 26)
    assert np.array_equal(t.sum(axis=1), np.ones(20000))
    counts = t.sum(axis=0)
    assert counts.min() > 700 and counts.max() < 820      # near-balanced


@pytest.mark.network
def test_the_three_classification_sets_sweep_the_loss_scale():
    """7, 10 and 26 classes is a 3.3x sweep of the chance-level MSE.

    Under one-hot targets the chance-level MSE is (k-1)/k^2, which is the
    quantity the KarciFANN prefactor is most sensitive to; the dataset choice
    exists to vary it.
    """
    chance = lambda k: (k - 1) / k**2
    assert chance(7) / chance(26) == pytest.approx(3.3, abs=0.1)


# ------------------------------------------ the additional tabular benchmarks


@pytest.mark.network
@pytest.mark.parametrize(
    "name,shape,classes",
    [
        ("vehicle", (846, 18), 4),
        ("satimage", (6430, 36), 6),
        ("segment", (2310, 19), 7),
        ("optdigits", (5620, 64), 10),
        ("pendigits", (10992, 16), 10),
    ],
)
def test_additional_tabular_sets_have_their_published_shapes(name, shape, classes):
    from karcifann import TABULAR

    x, t = TABULAR[name]()
    assert x.shape == shape
    assert t.shape == (shape[0], classes)
    assert np.array_equal(t.sum(axis=1), np.ones(shape[0]))
    assert np.isfinite(x).all()


@pytest.mark.network
def test_every_tabular_set_is_standardised_by_default():
    """Informative columns get unit variance; dead ones are left at zero.

    ``segment`` carries one constant feature and ``optdigits`` two, so a blanket
    "every column has unit SD" assertion would be wrong: dividing a constant
    column by its zero spread is what ``standardize`` exists to avoid.
    """
    from karcifann import TABULAR

    for name, loader in TABULAR.items():
        raw, _ = loader(scale=False)
        x, _ = loader()
        assert np.isfinite(x).all(), name
        assert np.allclose(x.mean(axis=0), 0.0, atol=1e-8), name

        constant = raw.std(axis=0) == 0
        assert np.allclose(x[:, ~constant].std(axis=0), 1.0, atol=1e-8), name
        assert np.all(x[:, constant] == 0.0), name


@pytest.mark.network
@pytest.mark.parametrize("name,dead", [("segment", 1), ("optdigits", 2)])
def test_the_datasets_with_dead_features_are_known_and_handled(name, dead):
    """A constant input column contributes nothing and must not become NaN."""
    from karcifann import TABULAR

    raw, _ = TABULAR[name](scale=False)
    assert int((raw.std(axis=0) == 0).sum()) == dead


@pytest.mark.network
def test_the_benchmark_suite_spans_a_wide_range_of_loss_scales():
    """Eight sets covering 4 to 26 classes.

    Under one-hot targets and MSE the chance-level loss is (k-1)/k^2, and that
    is the quantity the KarciFANN prefactor responds to, so the suite is
    chosen to vary it by a factor of five rather than to pile up near one
    value.
    """
    from karcifann import TABULAR

    counts = sorted({loader()[1].shape[1] for loader in TABULAR.values()} | {10})
    assert min(counts) == 4 and max(counts) == 26
    chance = [(k - 1) / k**2 for k in counts]
    assert max(chance) / min(chance) > 4.5


def test_load_dispatches_by_name_and_rejects_unknown():
    from karcifann import load

    with pytest.raises(ValueError, match="unknown dataset"):
        load("not_a_dataset")


@pytest.mark.network
def test_load_agrees_with_the_direct_loader():
    from karcifann import load, vehicle

    a, b = load("vehicle"), vehicle()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


