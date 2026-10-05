"""The Caputo-Fabrizio derivative, and the optimizers built on it.

Expected values are the closed forms of the exponential-kernel integral

    CF D^a f(t) = M/(1-a) . int_a^t f'(tau) exp(-lam (t - tau)) dtau ,
    lam = a / (1 - a)

worked out by hand for each test function.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from karcifann import (
    MLP,
    CaputoFabrizioGD,
    CaputoGD,
    KarciFANN,
    SGD,
    caputo_fabrizio,
    caputo_factor,
    cf_factor,
    cf_normalization,
    cf_rate,
    fod_factor,
    one_hot,
    train,
)
from karcifann.optimizers import get_optimizer

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ------------------------------------------------------- the operator itself


@pytest.mark.parametrize("alpha", ALPHAS)
def test_constant_function_has_zero_cf_derivative(alpha):
    """f' = 0 kills the integrand -- the property Riemann-Liouville lacks."""
    got = caputo_fabrizio(lambda v: np.full_like(v, 3.7), 2.0, alpha, a=0.0,
                          df=lambda v: np.zeros_like(v))
    assert got == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_identity_matches_closed_form(alpha):
    """CF D^a t = (M/a)(1 - exp(-lam t)) from the origin."""
    t, lam = 2.0, cf_rate(alpha)
    expected = (1.0 / alpha) * (1.0 - math.exp(-lam * t))
    got = caputo_fabrizio(lambda v: v, t, alpha, a=0.0, df=lambda v: np.ones_like(v))
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("b", [0.5, 1.0, -0.7])
def test_exponential_matches_closed_form(alpha, b):
    """CF D^a e^{bt} = M/(1-a) . b (e^{bt} - e^{-lam t}) / (b + lam)."""
    t, lam = 1.5, cf_rate(alpha)
    expected = (1.0 / (1.0 - alpha)) * b * (math.exp(b * t) - math.exp(-lam * t)) / (b + lam)
    got = caputo_fabrizio(lambda v: np.exp(b * v), t, alpha, a=0.0,
                          df=lambda v: b * np.exp(b * v))
    assert got == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_sine_matches_closed_form(alpha):
    """CF D^a sin(bt) = M/(1-a) . b [lam cos bt + b sin bt - lam e^{-lam t}] / (lam^2 + b^2)."""
    t, b, lam = 1.5, 2.0, cf_rate(alpha)
    expected = (
        (1.0 / (1.0 - alpha))
        * b
        * (lam * math.cos(b * t) + b * math.sin(b * t) - lam * math.exp(-lam * t))
        / (lam**2 + b**2)
    )
    got = caputo_fabrizio(lambda v: np.sin(b * v), t, alpha, a=0.0,
                          df=lambda v: b * np.cos(b * v))
    assert got == pytest.approx(expected, rel=1e-8)


def test_alpha_one_is_the_newton_derivative():
    assert caputo_fabrizio(np.sin, 1.0, 1.0, df=np.cos) == pytest.approx(math.cos(1.0))


def test_alpha_approaching_one_converges_to_the_newton_derivative():
    """The kernel narrows into a delta; quadrature must follow it there."""
    errors = [
        abs(caputo_fabrizio(np.sin, 1.0, alpha, a=0.0, df=np.cos) - math.cos(1.0))
        for alpha in (0.9, 0.99, 0.999, 0.9999)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3


def test_alpha_zero_is_the_increment_of_the_function():
    got = caputo_fabrizio(np.sin, 1.0, 0.0, a=0.0, df=np.cos)
    assert got == pytest.approx(math.sin(1.0) - math.sin(0.0))


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cf_is_linear_while_karci_is_not(alpha):
    """The clearest structural split between the two definitions."""
    t = 1.4
    f, df = np.sin, np.cos
    g, dg = np.exp, np.exp
    p, q = 2.0, -3.0

    combined = caputo_fabrizio(lambda v: p * f(v) + q * g(v), t, alpha, a=0.0,
                               df=lambda v: p * df(v) + q * dg(v))
    separate = p * caputo_fabrizio(f, t, alpha, a=0.0, df=df) + q * caputo_fabrizio(
        g, t, alpha, a=0.0, df=dg
    )
    assert combined == pytest.approx(separate, rel=1e-9)

    # Karci's operator, on the same combination, is not additive.
    x = np.array([t])
    karci = lambda fn, dfn: fod_factor(fn(x), x, alpha) * dfn(x)
    karci_combined = fod_factor(p * f(x) + q * g(x), x, alpha) * (p * df(x) + q * dg(x))
    assert not np.allclose(karci_combined, p * karci(f, df) + q * karci(g, dg))


@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
def test_cf_is_nonlocal_while_karci_is_pointwise(alpha):
    """Two functions that agree near t but not before it.

    ``bump`` is identically zero for ``x >= 0.5`` and C^2 everywhere, so at
    ``t = 1`` the two functions share both value and slope.  The Karci
    derivative, being pointwise, cannot tell them apart; the Caputo-Fabrizio
    derivative integrates the earlier history and does.
    """
    t = 1.0
    f, df = (lambda v: v), (lambda v: np.ones_like(v))
    bump = lambda v: np.where(v < 0.5, (0.5 - v) ** 3, 0.0)
    dbump = lambda v: np.where(v < 0.5, -3.0 * (0.5 - v) ** 2, 0.0)
    g, dg = (lambda v: v + bump(v)), (lambda v: 1.0 + dbump(v))

    x = np.array([t])
    assert g(x)[0] == pytest.approx(f(x)[0])
    assert dg(x)[0] == pytest.approx(df(x)[0])

    karci_f = fod_factor(f(x), x, alpha) * df(x)
    karci_g = fod_factor(g(x), x, alpha) * dg(x)
    assert np.allclose(karci_f, karci_g)

    cf_f = caputo_fabrizio(f, t, alpha, a=0.0, df=df)
    cf_g = caputo_fabrizio(g, t, alpha, a=0.0, df=dg)
    assert cf_f != pytest.approx(cf_g, rel=1e-6)


def test_normalization_schemes():
    assert cf_normalization(0.4, "unit") == 1.0
    assert cf_normalization(0.0, "losada_nieto") == pytest.approx(1.0)
    assert cf_normalization(0.4, "losada_nieto") == pytest.approx(2.0 / 1.6)
    with pytest.raises(ValueError):
        cf_normalization(0.4, "banana")


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_rate_rejects_orders_outside_the_open_unit_interval(alpha):
    """The classical Caputo-Fabrizio definition only covers 0 < alpha < 1."""
    with pytest.raises(ValueError):
        cf_rate(alpha)


def test_full_range_quadrature_is_available():
    """tail=inf integrates the whole interval instead of the kernel support."""
    truncated = caputo_fabrizio(np.sin, 1.0, 0.5, a=-40.0, df=np.cos)
    full = caputo_fabrizio(np.sin, 1.0, 0.5, a=-40.0, df=np.cos, tail=np.inf, n=20000)
    assert truncated == pytest.approx(full, rel=1e-6)


# ----------------------------------------------------------- the prefactors


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cf_factor_matches_the_truncated_integral(alpha):
    """Phi is exactly the CF integral of a function with constant slope."""
    d, lam = 0.4, cf_rate(alpha)
    expected = (1.0 / alpha) * (1.0 - math.exp(-lam * d))
    assert float(cf_factor(d, alpha)) == pytest.approx(expected)


@pytest.mark.parametrize("factor", [cf_factor, caputo_factor])
def test_every_prefactor_is_exactly_one_at_alpha_one(factor):
    """All three fractional rules must degenerate to plain gradient descent."""
    assert np.allclose(factor(np.array([-2.0, 0.3, 5.0]), 1.0), 1.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cf_factor_is_bounded_by_m_over_alpha(alpha):
    """The exponential kernel saturates; the Caputo power-law kernel does not."""
    huge = np.array([1e3, 1e6])
    assert np.all(cf_factor(huge, alpha) <= 1.0 / alpha + 1e-12)
    assert np.all(cf_factor(huge, alpha) == pytest.approx(1.0 / alpha))
    assert np.all(caputo_factor(huge, alpha) > 1.0 / alpha)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_prefactors_grow_with_distance_from_the_terminal(alpha):
    d = np.array([0.01, 0.1, 1.0, 10.0])
    assert np.all(np.diff(cf_factor(d, alpha)) > 0)
    assert np.all(np.diff(caputo_factor(d, alpha)) > 0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_prefactors_vanish_at_the_terminal(alpha):
    """Phi(0) = 0: a weight that has not moved gets no update.

    This is the stalling mechanism of Caputo-style fractional gradient descent,
    and it has no analogue in KarciFANN, whose prefactor is driven by the error.
    """
    assert float(cf_factor(0.0, alpha)) == pytest.approx(0.0)
    assert float(caputo_factor(0.0, alpha)) == pytest.approx(0.0)


@pytest.mark.parametrize("factor", [cf_factor, caputo_factor])
def test_unsigned_prefactors_never_reverse_the_step(factor):
    assert np.all(factor(np.array([-0.5, -2.0]), 0.5) > 0)
    assert factor(np.array([-0.5]), 0.5, signed=True)[0] < 0


def test_caputo_factor_matches_its_closed_form():
    assert float(caputo_factor(0.4, 0.3)) == pytest.approx(0.4**0.7 / math.gamma(1.7))


# ------------------------------------------------- truncation vs the real thing


def test_truncated_prefactor_converges_to_the_exact_cf_derivative():
    """Phi . dJ/dW is a truncation; it becomes exact as the interval shrinks.

    The exact partial CF derivative of the loss with respect to one weight is
    computed by quadrature, re-running backpropagation at every quadrature
    node.  Shrinking the lower terminal towards the current weight must drive
    the relative error of the truncation to zero.
    """
    rng = np.random.default_rng(0)
    net = MLP([3, 4, 2], weight_init="glorot", seed=1)
    x = rng.normal(0, 1, (6, 3))
    t = one_hot(rng.integers(0, 2, 6), 2)

    layer, index = 0, (1, 2)
    w_now = float(net.weights[layer][index])

    def dj_dw(values):
        """dJ/dw at each of the given values of that one weight."""
        out = np.empty_like(values)
        original = net.weights[layer][index]
        for i, v in enumerate(np.atleast_1d(values)):
            net.weights[layer][index] = v
            _, grads = net.gradients(x, t)
            out[i] = grads[layer][index]
        net.weights[layer][index] = original
        return out

    alpha = 0.6
    errors = []
    for width in (0.4, 0.2, 0.1, 0.05):
        exact = caputo_fabrizio(None, w_now, alpha, a=w_now - width, df=dj_dw, n=64)
        approx = float(cf_factor(width, alpha)) * float(dj_dw(np.array([w_now]))[0])
        errors.append(abs(approx - exact) / abs(exact))

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 0.02


# ---------------------------------------------------------------- optimizers


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([4, 6, 3], weight_init="glorot", seed=seed)
    x = rng.normal(0, 1, (10, 4))
    t = one_hot(rng.integers(0, 3, 10), 3)
    return net, x, t


@pytest.mark.parametrize("optimizer", [CaputoFabrizioGD, CaputoGD])
@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
def test_update_equals_prefactor_times_newton_gradient(optimizer, alpha):
    net, x, t = _problem()
    loss, grads = net.gradients(x, t)
    before = [p.copy() for p in net.parameters]
    opt = optimizer(alpha=alpha, lr=0.7)
    expected_factors = opt.factors(net.parameters)

    opt.step(net.parameters, grads, loss)

    for w0, g, phi, w1 in zip(before, grads, expected_factors, net.parameters):
        assert np.allclose(w1, w0 - 0.7 * phi * g)


@pytest.mark.parametrize("optimizer", [CaputoFabrizioGD, CaputoGD])
def test_alpha_one_is_bit_identical_to_gradient_descent(optimizer):
    """Same degeneracy check as for KarciFANN, on the other two definitions."""
    net_f, x, t = _problem(3)
    net_s = net_f.copy()
    hf = train(net_f, optimizer(alpha=1.0, lr=0.8), x, t, epochs=50)
    hs = train(net_s, SGD(lr=0.8), x, t, epochs=50)
    assert hf.loss == hs.loss
    for a, b in zip(net_f.parameters, net_s.parameters):
        assert np.array_equal(a, b)


def test_prefactor_ignores_the_error_unlike_karcifann():
    """CF sees only weight geometry; Karci sees the error as well."""
    params = [np.array([0.3, -0.7])]
    cf = CaputoFabrizioGD(alpha=0.6)
    assert np.allclose(cf.factors(params, loss=1e-6), cf.factors(params, loss=10.0))

    karci = KarciFANN(alpha=0.6)
    assert not np.allclose(
        karci.factors(params, loss=1e-6), karci.factors(params, loss=10.0)
    )


def test_previous_terminal_stalls_as_the_steps_shrink():
    """With W0 = W_{t-1} the prefactor is driven by the last step length.

    Small steps therefore beget smaller steps, which is the documented failure
    mode of Caputo-style fractional gradient descent.
    """
    net, x, t = _problem(2)
    history = train(net, CaputoFabrizioGD(alpha=0.5, terminal="previous"), x, t, epochs=200)
    assert history.factor_mean[-1] < history.factor_mean[0]
    assert history.factor_mean[-1] < 1e-3

    net_o, x, t = _problem(2)
    origin = train(net_o, CaputoFabrizioGD(alpha=0.5, terminal="origin"), x, t, epochs=200)
    assert origin.loss[-1] < history.loss[-1]


def test_first_step_of_the_previous_terminal_falls_back_to_the_origin():
    net, x, t = _problem(4)
    net_o = net.copy()
    loss, grads = net.gradients(x, t)
    CaputoFabrizioGD(alpha=0.5, terminal="previous").step(net.parameters, grads, loss)
    CaputoFabrizioGD(alpha=0.5, terminal="origin").step(net_o.parameters, grads, loss)
    for a, b in zip(net.parameters, net_o.parameters):
        assert np.array_equal(a, b)


def test_reset_clears_the_stored_previous_iterate():
    net, x, t = _problem(5)
    opt = CaputoFabrizioGD(alpha=0.5, terminal="previous")
    loss, grads = net.gradients(x, t)
    opt.step(net.parameters, grads, loss)
    assert opt._previous is not None
    opt.reset()
    assert opt._previous is None


def test_unknown_terminal_rejected():
    with pytest.raises(ValueError):
        CaputoFabrizioGD(alpha=0.5, terminal="banana")


@pytest.mark.parametrize("name", ["caputo_fabrizio", "cf", "caputo"])
def test_optimizer_registry(name):
    assert get_optimizer(name) is not None


# ------------------------------------------------ the comparison, controlled


def test_raw_sweep_is_dominated_by_the_scale_of_the_prefactor():
    """The whole point of the comparison, stated as a test.

    At the same nominal alpha the three rules differ mostly in how large their
    prefactor is, not in its shape.  Ranking them by final loss reproduces the
    ranking of their mean prefactor -- so an uncontrolled sweep measures step
    size, not mechanism.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (60, 8))
    t = one_hot(rng.integers(0, 4, 60), 4)

    alpha = 0.5
    results = {}
    for label, optimizer in (
        ("karci", KarciFANN(alpha=alpha)),
        ("cf", CaputoFabrizioGD(alpha=alpha)),
        ("caputo", CaputoGD(alpha=alpha)),
    ):
        net = MLP([8, 10, 4], weight_init="glorot", seed=21)
        history = train(net, optimizer, x, t, epochs=200)
        results[label] = (history.factor_mean[-1], history.loss[-1])

    by_factor = sorted(results, key=lambda k: -results[k][0])
    by_loss = sorted(results, key=lambda k: results[k][1])
    assert by_factor == by_loss


def test_karcifann_scale_knob_is_neutral_at_one():
    net, x, t = _problem(6)
    net_scaled = net.copy()
    h1 = train(net, KarciFANN(alpha=0.7), x, t, epochs=30)
    h2 = train(net_scaled, KarciFANN(alpha=0.7, scale=1.0), x, t, epochs=30)
    assert h1.loss == h2.loss


def test_karcifann_scale_knob_rescales_the_step():
    net, x, t = _problem(7)
    net_scaled = net.copy()
    loss, grads = net.gradients(x, t)
    before = [p.copy() for p in net.parameters]

    KarciFANN(alpha=0.7).step(net.parameters, grads, loss)
    KarciFANN(alpha=0.7, scale=3.0).step(net_scaled.parameters, grads, loss)

    for w0, plain, scaled in zip(before, net.parameters, net_scaled.parameters):
        assert np.allclose(scaled - w0, 3.0 * (plain - w0))


# ---------------------------- why CF cannot be threaded through backpropagation


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_cf_does_not_telescope_through_a_chain(alpha):
    """The reason there is no "CF backpropagation".

    KarciFANN threads its derivative through every link of the chain and the
    prefactors collapse.  Doing the same with Caputo-Fabrizio -- each link
    integrated along its own axis -- does not reproduce the derivative of the
    composition, because the exponential kernel measures distance along one
    specific axis and a change of variable moves it to a different one.
    """
    h, dh = np.sin, np.cos                      # u = h(x)
    g, dg = np.exp, np.exp                      # y = g(u)
    x, x0 = 1.2, 0.3
    u, u0 = float(h(x)), float(h(x0))
    y = float(g(u))

    karci_chain = (fod_factor(y, u, alpha) * dg(u)) * (fod_factor(u, x, alpha) * dh(x))
    karci_direct = fod_factor(y, x, alpha) * (dg(u) * dh(x))
    assert karci_chain == pytest.approx(karci_direct, rel=1e-12)

    cf_chain = caputo_fabrizio(g, u, alpha, a=u0, df=dg) * caputo_fabrizio(
        h, x, alpha, a=x0, df=dh
    )
    cf_direct = caputo_fabrizio(
        lambda v: g(h(v)), x, alpha, a=x0, df=lambda v: dg(h(v)) * dh(v)
    )
    assert cf_chain != pytest.approx(cf_direct, rel=1e-3)


def test_the_measured_cf_residue_is_large_where_the_method_is_used():
    """The recorded residue must stay large, not merely nonzero.

    ``test_cf_does_not_telescope_through_a_chain`` asserts the two routes
    differ.  The manuscript makes the stronger claim that they differ by
    something of the parameter's own order at the orders anyone would tune to,
    and quotes ``results/chained_vs_terminal_digits.csv`` for it.  If the
    measurement ever shrank to round-off the claim would be wrong while that
    test still passed, so the size is checked here too.
    """
    import csv

    path = (Path(__file__).resolve().parent.parent / "results"
            / "chained_vs_terminal_digits.csv")
    if not path.exists():
        pytest.skip("no chained-vs-terminal measurements")
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["stage"] == "cf_residue"]
    assert rows, "the cf_residue stage is missing from the recorded results"

    by_alpha = {float(r["coefficient"]): (float(r["median_relative_error"]),
                                          float(r["max_relative_error"]))
                for r in rows}
    for alpha in (0.2, 0.4, 0.6, 0.8):
        median, worst = by_alpha[alpha]
        assert median > 0.2, f"median residue at alpha={alpha} is only {median}"
        assert worst > 100, f"worst residue at alpha={alpha} is only {worst}"
    # ... and collapses at the identity limit, which is the point of 4.5.
    assert by_alpha[0.999][0] < 0.05


def test_cf_chain_error_vanishes_only_in_the_newton_limit():
    """Both agree at alpha = 1 -- where each is just the ordinary derivative."""
    h, dh, g, dg = np.sin, np.cos, np.exp, np.exp
    x, x0 = 1.2, 0.3
    u0 = float(h(x0))

    errors = []
    for alpha in (0.5, 0.9, 0.99, 0.999):
        chain = caputo_fabrizio(g, float(h(x)), alpha, a=u0, df=dg) * caputo_fabrizio(
            h, x, alpha, a=x0, df=dh
        )
        direct = caputo_fabrizio(
            lambda v: g(h(v)), x, alpha, a=x0, df=lambda v: dg(h(v)) * dh(v)
        )
        errors.append(abs(chain - direct) / abs(direct))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-3


def test_only_a_power_gauge_telescopes():
    """Why Karci's form is the one that works, and CF's shape never can.

    Write the per-link prefactor as ``Phi(numerator/denominator)``.  Collapsing
    a chain requires ``Phi(r1) Phi(r2) Phi(r3) = Phi(r1 r2 r3)`` for all
    positive ratios -- i.e. ``Phi`` must be multiplicative, and the continuous
    solutions of that functional equation are exactly the powers ``r^k``.
    Karci's ``(J/W)^(alpha-1)`` is that form; the saturating shape of the
    Caputo-Fabrizio prefactor is not, and neither is any other bounded one.

    Note what this also shows: the collapse has nothing to do with the operator
    being *fractional*.  Any exponent telescopes, fractional or not.
    """
    j, y, a, w = 0.09, 0.7, 0.4, 0.25

    def telescopes(phi):
        return phi(j / y) * phi(y / a) * phi(a / w) == pytest.approx(phi(j / w), rel=1e-12)

    for exponent in (-1.3, -0.4, 0.5, 2.7):
        assert telescopes(lambda r, k=exponent: r**k)

    for shape in (
        lambda r: 1.0 - math.exp(-r),       # the Caputo-Fabrizio shape
        lambda r: math.log1p(r),
        lambda r: r / (1.0 + r),
    ):
        assert not telescopes(shape)
