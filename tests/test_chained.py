"""Backpropagation with a fractional gauge inserted at every link.

The headline result here is the end-to-end telescoping check: threading the
*Karci* gauge through every link of a real network reproduces, to machine
precision, the single ``(J/W)^(alpha-1)`` prefactor that KarciFANN applies as a
pure optimizer.  The same construction with a Caputo-Fabrizio gauge does not
collapse, which is what makes it a different method rather than a slower
spelling of the same one.
"""

import numpy as np
import pytest

from karcifann import (
    MLP,
    SGD,
    CaputoFabrizioGD,
    KarciFANN,
    fod_factor,
    one_hot,
    train,
)
from karcifann.chained import (
    ChainedGaugeMLP,
    caputo_gauge,
    cf_gauge,
    karci_gauge,
    newton_gauge,
)


def _problem(seed=0, samples=6, layers=(4, 5, 3)):
    rng = np.random.default_rng(seed)
    net = ChainedGaugeMLP(list(layers), activation="tanh", weight_init="glorot", seed=seed + 1)
    x = rng.normal(0, 1, (samples, layers[0]))
    t = one_hot(rng.integers(0, layers[-1], samples), layers[-1])
    return net, x, t


# ----------------------------------------------------------------- degeneracy


def test_no_gauge_is_plain_backpropagation():
    net, x, t = _problem()
    plain = MLP.gradients(net, x, t)[1]
    assert net.gauge is None
    assert all(np.array_equal(a, b) for a, b in zip(plain, net.gradients(x, t)[1]))


def test_newton_gauge_is_plain_backpropagation():
    net, x, t = _problem()
    _, plain = net.gradients(x, t)
    net.gauge = newton_gauge()
    _, gauged = net.gradients(x, t)
    assert all(np.allclose(a, b) for a, b in zip(plain, gauged))


@pytest.mark.parametrize("gauge", [karci_gauge, cf_gauge, caputo_gauge])
def test_alpha_one_recovers_the_newton_gradient(gauge):
    """Every gauge is 1 at alpha = 1, for every link."""
    net, x, t = _problem()
    _, plain = net.gradients(x, t)
    net.gauge = gauge(1.0)
    _, gauged = net.gradients(x, t)
    assert all(np.allclose(a, b) for a, b in zip(plain, gauged))


# --------------------------------------------------- the telescoping identity


@pytest.mark.parametrize("alpha", [0.4, 0.7, 1.3, 1.9])
@pytest.mark.parametrize("layers", [(4, 5, 3), (4, 5, 4, 3)])
def test_chained_karci_collapses_to_the_single_prefactor(alpha, layers):
    """(J/Y)(Y/A)...(A/W) = J/W, verified inside a real network.

    This is the whole justification for KarciFANN being implementable as an
    optimizer: the per-link derivation of the papers and the one-factor
    implementation must agree, and they do to ~1e-15.
    """
    net, x, t = _problem(layers=layers)
    loss, plain = net.gradients(x, t)
    net.gauge = karci_gauge(alpha, eps=0.0)
    _, chained = net.gradients(x, t)

    for parameter, newton, got in zip(net.parameters, plain, chained):
        collapsed = fod_factor(loss, parameter, alpha) * newton
        assert np.allclose(got, collapsed, rtol=1e-10, atol=1e-15)


def test_chained_karci_trains_identically_to_the_karcifann_optimizer():
    """The strongest form of the same claim: identical over a whole run."""
    net_chained, x, t = _problem(seed=3, samples=12)
    net_optimizer = MLP.copy(net_chained)

    net_chained.gauge = karci_gauge(0.7, eps=0.0)
    h_chained = train(net_chained, SGD(lr=1.0), x, t, epochs=40)
    h_optimizer = train(net_optimizer, KarciFANN(alpha=0.7, eps=0.0), x, t, epochs=40)

    assert h_chained.loss == pytest.approx(h_optimizer.loss, rel=1e-9)
    for a, b in zip(net_chained.parameters, net_optimizer.parameters):
        assert np.allclose(a, b, rtol=1e-9)


# ------------------------------------------------- Caputo-Fabrizio does *not*


@pytest.mark.parametrize("alpha", [0.4, 0.7, 0.95])
def test_chained_cf_does_not_collapse_to_any_scalar(alpha):
    """If it telescoped, chained/Newton would be one number per parameter."""
    net, x, t = _problem()
    _, plain = net.gradients(x, t)
    net.gauge = cf_gauge(alpha)
    _, chained = net.gradients(x, t)

    flat_chained = np.concatenate([c.ravel() for c in chained])
    flat_plain = np.concatenate([p.ravel() for p in plain])
    # Ignore entries whose Newton gradient is numerically negligible: their
    # ratio is dominated by round-off, not by the gauge.
    significant = np.abs(flat_plain) > 1e-8 * np.max(np.abs(flat_plain))
    ratios = np.abs(flat_chained[significant] / flat_plain[significant])
    assert ratios.max() / ratios.min() > 5.0


@pytest.mark.parametrize("alpha", [0.4, 0.7])
def test_chained_and_terminal_cf_are_different_methods(alpha):
    """One applies the operator once at the end, the other at every link."""
    net_chained, x, t = _problem(seed=5)
    net_terminal = MLP.copy(net_chained)

    net_chained.gauge = cf_gauge(alpha)
    train(net_chained, SGD(lr=1.0), x, t, epochs=25)
    train(net_terminal, CaputoFabrizioGD(alpha=alpha, lr=1.0), x, t, epochs=25)

    assert not np.allclose(net_chained.weights[0], net_terminal.weights[0])


# ------------------------------------------------------- implementation paths


def test_per_path_and_elementwise_paths_agree():
    """The einsum path and the cheap elementwise path must not disagree.

    ``cf_gauge`` declares ``denominator_only`` and takes the elementwise
    shortcut; stripping the declaration forces the same gauge down the general
    per-path code, which is the expensive branch used by the Karci gauge.
    """
    net, x, t = _problem(seed=7)

    net.gauge = cf_gauge(0.6)
    _, cheap = net.gradients(x, t)

    forced = cf_gauge(0.6)
    forced.denominator_only = False
    net.gauge = forced
    _, general = net.gradients(x, t)

    for a, b in zip(cheap, general):
        assert np.allclose(a, b, rtol=1e-12)


def test_gauge_survives_copy_with_the_subclass_intact():
    net, x, t = _problem()
    net.gauge = cf_gauge(0.5)
    clone = net.copy()
    assert isinstance(clone, ChainedGaugeMLP)
    assert clone.gauge is net.gauge
    assert all(np.array_equal(a, b) for a, b in zip(net.parameters, clone.parameters))


def test_works_without_biases():
    rng = np.random.default_rng(1)
    net = ChainedGaugeMLP([3, 4, 2], activation="tanh", weight_init="glorot",
                          use_bias=False, seed=1, gauge=cf_gauge(0.5))
    x = rng.normal(0, 1, (5, 3))
    t = one_hot(rng.integers(0, 2, 5), 2)
    _, grads_b = net.backward(net.forward(x), t)
    assert grads_b is None


def test_target_shape_is_still_validated():
    net, x, t = _problem()
    net.gauge = cf_gauge(0.5)
    with pytest.raises(ValueError):
        net.backward(net.forward(x), np.zeros((6, 5)))


@pytest.mark.parametrize("gauge", [karci_gauge, cf_gauge, caputo_gauge])
def test_chained_gauges_train_without_diverging(gauge):
    net, x, t = _problem(seed=9, samples=40)
    net.gauge = gauge(0.6)
    history = train(net, SGD(lr=1.0), x, t, epochs=150)
    assert history.diverged_at is None
    assert history.loss[-1] < history.loss[0]


# ------------------------------- which link gauges admit a telescoping chain


def _telescopes(phi, values=(0.09, 0.7, 0.4, 0.25)):
    j, y, a, w = values
    return phi(j, y) * phi(y, a) * phi(a, w) == pytest.approx(phi(j, w), rel=1e-12)


@pytest.mark.parametrize(
    "g",
    [
        lambda v: v**-0.4,          # Karci, alpha = 0.6
        lambda v: v**2.7,           # a power, but not a fractional order
        lambda v: np.log1p(v),      # not a power at all
        lambda v: np.exp(v),
        lambda v: np.sinh(3 * v),
    ],
)
def test_any_gauge_of_the_form_g_num_over_g_den_telescopes(g):
    """The general characterisation of a collapsing link gauge.

    ``Phi(a,b) Phi(b,c) = Phi(a,c)`` holds exactly when ``Phi(a,b) = g(a)/g(b)``
    for some ``g``: sufficiency is immediate, and fixing ``c`` recovers
    ``g(x) = Phi(x,c)``.  So the learning-rate-free family that KarciFANN
    belongs to is the whole of

        W <- W - [g(J)/g(W)] . dJ/dW ,

    and Karci's operator is the member with ``g(x) = x^(alpha-1)`` -- not the
    only possibility, and not special for being fractional.
    """
    assert _telescopes(lambda a, b: g(a) / g(b))


def test_the_collapsed_factor_is_g_of_the_loss_over_g_of_the_weight():
    g = np.log1p
    j, w = 0.09, 0.25
    assert _telescopes(lambda a, b: g(a) / g(b))
    chain = (g(j) / g(0.7)) * (g(0.7) / g(0.4)) * (g(0.4) / g(w))
    assert chain == pytest.approx(g(j) / g(w))


def test_no_nontrivial_denominator_only_gauge_can_telescope():
    """One line that rules out both Caputo and Caputo-Fabrizio at once.

    A gauge of the form ``Phi(a,b) = phi(b)`` telescopes only if
    ``g(a) = phi(b) g(b)`` for every pair, which forces ``g`` constant and
    ``Phi == 1``.  Both truncated prefactors depend on the denominator alone,
    so neither can ever collapse a chain.
    """
    assert _telescopes(lambda a, b: 1.0)                       # the trivial case
    for phi in (lambda v: 1.0 - np.exp(-v), lambda v: v**0.4, np.log1p):
        assert not _telescopes(lambda a, b, f=phi: f(b))
        assert not _telescopes(lambda a, b, f=phi: f(a))


def test_a_gauge_outside_the_family_does_not_telescope():
    assert not _telescopes(lambda a, b: np.sqrt(a * b))
    assert not _telescopes(lambda a, b: a - b + 1.0)
