"""The Karci fractional order derivative itself.

Every expected value here is taken from the closed forms published in
A. Karci, Universal J. Engineering Science 1(3), 2013 and 3(3), 2015.
"""

import numpy as np
import pytest

from karcifann import fod_factor, karci_fod, karci_fod_limit


ALPHAS = [0.4, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]


@pytest.mark.parametrize("alpha", ALPHAS)
def test_constant_function_has_zero_fod(alpha):
    """f(x) = c  ->  D^a f = 0, unlike Riemann-Liouville or Caputo."""
    x = np.array([0.5, 1.0, 3.0, 7.5])
    got = karci_fod(lambda v: np.full_like(v, 4.2), x, alpha, df=lambda v: np.zeros_like(v))
    assert np.allclose(got, 0.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_identity_function_has_unit_fod(alpha):
    """f(x) = x  ->  D^a f = 1 for every order; the motivating example."""
    x = np.array([0.25, 1.0, 2.0, 10.0])
    got = karci_fod(lambda v: v, x, alpha, df=lambda v: np.ones_like(v))
    assert np.allclose(got, 1.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_sine_matches_published_closed_form(alpha):
    """D^a sin(x) = cos(x) sin(x)^(a-1) / x^(a-1)."""
    x = np.array([0.3, 0.9, 1.5, 3.0])
    expected = np.cos(x) * np.power(np.sin(x), alpha - 1) / np.power(x, alpha - 1)
    got = karci_fod(np.sin, x, alpha, df=np.cos)
    assert np.allclose(got, expected)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_log_matches_published_closed_form(alpha):
    """D^a ln(x) = (1/x) (ln x / x)^(a-1)."""
    x = np.array([1.5, 2.0, 5.0])
    expected = (1.0 / x) * np.power(np.log(x) / x, alpha - 1)
    got = karci_fod(np.log, x, alpha, df=lambda v: 1.0 / v)
    assert np.allclose(got, expected)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("power", [1, 2, 3])
def test_monomial_matches_published_closed_form(alpha, power):
    """D^a x^n = n x^(n-1) . x^((n-1)(a-1))."""
    x = np.array([0.7, 1.3, 4.0])
    expected = power * x ** (power - 1) * x ** ((power - 1) * (alpha - 1))
    got = karci_fod(lambda v: v**power, x, alpha, df=lambda v: power * v ** (power - 1))
    assert np.allclose(got, expected)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_alpha_one_is_the_newton_derivative(alpha):
    """The a = 1 case must collapse onto f'(x) for any f."""
    x = np.array([0.4, 1.1, 2.7])
    got = karci_fod(np.exp, x, 1.0, df=np.exp)
    assert np.allclose(got, np.exp(x))


@pytest.mark.parametrize("alpha", [0.6, 0.9, 1.3, 1.7, 2.2])
def test_closed_form_agrees_with_the_limit_definition(alpha):
    """(f(x+h)^a - f(x)^a) / ((x+h)^a - x^a) -> (f/x)^(a-1) f'(x) as h -> 0.

    This is the step from Definition 1 to the closed form, done numerically.
    """
    x = np.array([0.8, 1.6, 2.4])
    closed = karci_fod(np.sin, x, alpha, df=np.cos)
    limit = karci_fod_limit(np.sin, x, alpha, h=1e-7)
    assert np.allclose(closed, limit, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.7, 1.4, 2.1])
def test_chain_rule_telescopes(alpha):
    """D^a(y/x) = D^a(y/u) . D^a(u/x) -- the identity KarciFANN relies on.

    With y = g(u) and u = h(x), each factor contributes its own
    ``(.)^(a-1)`` and all the intermediate ones cancel.
    """
    x = np.array([0.6, 1.2, 2.0])
    h, dh = np.sin, np.cos                      # u = sin x
    g, dg = np.exp, np.exp                      # y = exp u

    u = h(x)
    y = g(u)

    d_y_u = fod_factor(y, u, alpha) * dg(u)     # D^a (y / u)
    d_u_x = fod_factor(u, x, alpha) * dh(x)     # D^a (u / x)
    d_y_x = fod_factor(y, x, alpha) * (dg(u) * dh(x))  # D^a (y / x)

    assert np.allclose(d_y_u * d_u_x, d_y_x)


def test_numeric_derivative_matches_analytic_derivative():
    """karci_fod falls back to a finite difference when df is not supplied."""
    x = np.array([0.5, 1.0, 2.0])
    analytic = karci_fod(np.sin, x, 1.3, df=np.cos)
    numeric = karci_fod(np.sin, x, 1.3)
    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


class TestFodFactor:
    def test_alpha_one_gives_exactly_one(self):
        w = np.array([[-3.0, 0.0, 2.5]])
        assert np.array_equal(fod_factor(0.37, w, 1.0), np.ones_like(w))

    def test_positive_base_is_plain_power(self):
        got = fod_factor(0.25, np.array([0.5, 2.0]), 1.5)
        assert np.allclose(got, np.sqrt(np.array([0.5, 0.125])))

    def test_abs_mode_keeps_the_factor_positive_for_negative_weights(self):
        got = fod_factor(0.25, np.array([-0.5, -2.0]), 1.5, mode="abs")
        assert np.all(got > 0)
        assert np.allclose(got, fod_factor(0.25, np.array([0.5, 2.0]), 1.5))

    def test_signed_mode_flips_the_sign_for_negative_weights(self):
        got = fod_factor(0.25, np.array([-0.5]), 1.5, mode="signed")
        assert got[0] < 0

    def test_real_mode_is_the_real_part_of_the_principal_power(self):
        base = 0.25 / -0.5
        expected = abs(base) ** 0.5 * np.cos(np.pi * 0.5)
        assert np.allclose(fod_factor(0.25, np.array([-0.5]), 1.5, mode="real"), expected)

    def test_complex_mode_returns_the_principal_complex_power(self):
        got = fod_factor(0.25, np.array([-0.5]), 1.5, mode="complex")
        assert np.iscomplexobj(got)
        assert np.allclose(got, complex(0.25 / -0.5) ** 0.5)

    def test_nan_mode_refuses_negative_bases(self):
        got = fod_factor(0.25, np.array([-0.5, 0.5]), 1.5, mode="nan")
        assert np.isnan(got[0]) and np.isfinite(got[1])

    def test_eps_floor_keeps_a_zero_weight_finite(self):
        raw = fod_factor(0.25, np.array([0.0]), 1.5, eps=0.0)
        floored = fod_factor(0.25, np.array([0.0]), 1.5, eps=1e-12)
        assert not np.isfinite(raw[0])
        assert np.isfinite(floored[0])

    def test_eps_floor_keeps_a_zero_error_finite(self):
        """alpha < 1 puts the error in the denominator, so J = 0 explodes."""
        raw = fod_factor(0.0, np.array([0.5]), 0.5, eps=0.0)
        floored = fod_factor(0.0, np.array([0.5]), 0.5, eps=1e-12)
        assert not np.isfinite(raw[0])
        assert np.isfinite(floored[0])

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            fod_factor(0.1, np.array([1.0]), 1.5, mode="nope")
