"""Caputo-Fabrizio fractional order derivative, and the Caputo derivative.

Caputo & Fabrizio (*Progr. Fract. Differ. Appl.* 1(2):73-85, 2015) replaced the
singular power-law kernel of the Caputo derivative with a decaying exponential.
For ``0 < alpha < 1`` and ``f`` in ``H^1(a, b)``::

                        M(alpha)   / t                (   alpha         )
    CF D^alpha f(t)  =  --------   |    f'(tau) . exp ( - -------- (t-tau) )  dtau
                        1 - alpha  / a                ( 1 - alpha        )

``M(alpha)`` is a normalisation with ``M(0) = M(1) = 1``; ``M = 1`` satisfies
that and is the default here.  The two limits then come out classical:
``alpha -> 1`` turns the kernel into a Dirac delta and recovers ``f'(t)``,
while ``alpha -> 0`` gives ``f(t) - f(a)``.

How this differs from Karci's derivative, and why it matters for a network
-------------------------------------------------------------------------

============  =========================  ================================
                Karci                      Caputo-Fabrizio
============  =========================  ================================
form          ``(f(x)/x)^(a-1) f'(x)``   integral over ``[a, t]``
locality      pointwise (local)          nonlocal, exponential memory
linearity     no                         yes
chain rule    yes, multiplicative        no
depends on    the *value* of ``f``       the *history* of ``f'``
order range   any real (or complex)      ``0 < alpha < 1``
============  =========================  ================================

The missing chain rule is the important one.  KarciFANN is cheap because
Karci's chain rule telescopes an entire backpropagation into one
``(J/W)^(a-1)`` prefactor.  Nothing of the kind exists for Caputo-Fabrizio, so
there is no "CF backpropagation" to write down.  What *can* be done -- and what
the fractional-gradient-descent literature does with the Caputo derivative -- is
to replace the partial derivative ``dJ/dW`` in the update rule by the fractional
derivative of ``J`` with respect to that same weight, from a lower terminal
``W0`` up to the current ``W``.  Truncating the integrand at its value at ``W``
gives a closed-form prefactor again, so all three methods end up as

    W  <-  W  -  Phi(.) . dJ/dW

and differ only in the shape of ``Phi``:

    Karci             Phi = (J / W)^(alpha-1)
    Caputo-Fabrizio   Phi = (M/alpha) . (1 - exp(-alpha |W - W0| / (1 - alpha)))
    Caputo            Phi = |W - W0|^(1-alpha) / Gamma(2 - alpha)

all three of which are exactly ``1`` at ``alpha = 1``.  Note where the
dependence sits: Karci's prefactor is driven by the *error*, the other two only
by the *distance the weight has travelled*.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "NORMALIZATIONS",
    "caputo_fabrizio",
    "caputo_factor",
    "cf_factor",
    "cf_normalization",
    "cf_rate",
]

#: Normalisation ``M(alpha)`` of the Caputo-Fabrizio kernel.
#:
#: ``"unit"``          -- ``M = 1``; satisfies the ``M(0) = M(1) = 1`` condition
#:                        of the original paper and gives both classical limits.
#: ``"losada_nieto"``  -- ``M = 2 / (2 - alpha)``, the constant of Losada &
#:                        Nieto (2015).  The literature is not consistent about
#:                        this factor, so it is exposed rather than hard-coded.
NORMALIZATIONS = ("unit", "losada_nieto")


def cf_normalization(alpha: float, scheme: str = "unit") -> float:
    if scheme == "unit":
        return 1.0
    if scheme == "losada_nieto":
        return 2.0 / (2.0 - alpha)
    raise ValueError(f"unknown normalization {scheme!r}; expected one of {NORMALIZATIONS}")


def cf_rate(alpha: float) -> float:
    """Decay rate ``lambda = alpha / (1 - alpha)`` of the exponential kernel."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"the Caputo-Fabrizio derivative needs 0 < alpha < 1, got {alpha}")
    return alpha / (1.0 - alpha)


def caputo_fabrizio(
    f,
    t,
    alpha: float,
    a: float = 0.0,
    df=None,
    n: int = 2048,
    normalization: str = "unit",
    tail: float = 40.0,
):
    """Caputo-Fabrizio derivative of ``f`` at ``t``, lower terminal ``a``.

    The integral is evaluated by composite Simpson quadrature on ``n``
    subintervals.  ``df`` is the ordinary first derivative; a central finite
    difference is used when it is omitted.

    The kernel decays over a length ``1/lambda = (1 - alpha)/alpha``, which goes
    to zero as ``alpha -> 1``.  Spreading a uniform grid over the whole of
    ``[a, t]`` then fails to resolve it, so quadrature is restricted to
    ``[t - tail/lambda, t]``; what is dropped is weighted by ``exp(-tail)`` and
    is negligible for any well-behaved integrand.  Pass ``tail=inf`` to
    integrate over the full interval regardless.

    ``alpha = 1`` returns ``f'(t)`` and ``alpha = 0`` returns ``f(t) - f(a)``,
    which are the limits of the operator rather than special cases of the
    integral.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))

    if alpha == 1.0:
        out = _derivative(f, t_arr, df)
    elif alpha == 0.0:
        at_a = np.asarray(f(np.array([a], dtype=float)), dtype=float).ravel()[0]
        out = np.asarray(f(t_arr), dtype=float) - at_a
    else:
        lam = cf_rate(alpha)
        scale = cf_normalization(alpha, normalization) / (1.0 - alpha)
        if n % 2:
            n += 1  # Simpson needs an even number of subintervals
        out = np.empty_like(t_arr)
        for i, upper in enumerate(t_arr):
            lower = a if np.isinf(tail) else max(a, upper - tail / lam)
            tau = np.linspace(lower, upper, n + 1)
            integrand = _derivative(f, tau, df) * np.exp(-lam * (upper - tau))
            out[i] = scale * _simpson(integrand, tau)

    return out if np.ndim(t) else float(out[0])


def _derivative(f, x: np.ndarray, df=None, h: float = 1e-6) -> np.ndarray:
    if df is not None:
        return np.asarray(df(x), dtype=float)
    return (np.asarray(f(x + h), dtype=float) - np.asarray(f(x - h), dtype=float)) / (2.0 * h)


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson rule on a uniform grid with an even number of panels."""
    if x[-1] == x[0]:
        return 0.0
    step = (x[-1] - x[0]) / (len(x) - 1)
    return float(step / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def cf_factor(
    delta,
    alpha: float,
    normalization: str = "unit",
    signed: bool = False,
    eps: float = 0.0,
):
    """Truncated Caputo-Fabrizio prefactor ``(M/alpha)(1 - exp(-alpha|d|/(1-alpha)))``.

    This is what the CF derivative of ``J`` with respect to a weight collapses
    to when the integrand ``dJ/dtau`` is frozen at its value at the upper
    terminal::

        CF D^a_{W0} J(W)  =  M/(1-a) . int_{W0}^{W} J'(tau) e^{-lam (W-tau)} dtau
                          ~= M/(1-a) . J'(W) . (1 - e^{-lam d}) / lam
                           = (M/a) (1 - e^{-lam d}) . J'(W)

    ``delta`` is ``W - W0``.  With ``signed=False`` (the default) the magnitude
    ``|d|`` is used, which keeps the prefactor non-negative so the update is
    always a descent step -- the same convention this package applies to the
    negative bases of the Karci prefactor.  ``eps`` floors ``|d|``.

    Unlike the Caputo prefactor this one is *bounded*: it can never exceed
    ``M/alpha``, no matter how far the weight has travelled.
    """
    delta = np.asarray(delta, dtype=float)
    if alpha == 1.0:
        return np.ones_like(delta)

    lam = cf_rate(alpha)
    magnitude = np.abs(delta)
    if eps > 0.0:
        magnitude = np.maximum(magnitude, eps)
    factor = (cf_normalization(alpha, normalization) / alpha) * (
        1.0 - np.exp(-lam * magnitude)
    )
    return np.where(delta < 0.0, -factor, factor) if signed else factor


def caputo_factor(delta, alpha: float, signed: bool = False, eps: float = 0.0):
    """Truncated Caputo prefactor ``|W - W0|^(1-alpha) / Gamma(2 - alpha)``.

    The first term of the Caputo series expansion, and the standard basis of
    fractional gradient descent (Chen et al., Wang et al.).  Included as the
    reference point that Caputo-Fabrizio was designed to improve on: its kernel
    is singular, and unlike the CF prefactor it is unbounded in ``|W - W0|``.
    """
    delta = np.asarray(delta, dtype=float)
    if alpha == 1.0:
        return np.ones_like(delta)

    magnitude = np.abs(delta)
    if eps > 0.0:
        magnitude = np.maximum(magnitude, eps)
    factor = np.power(magnitude, 1.0 - alpha) / math.gamma(2.0 - alpha)
    return np.where(delta < 0.0, -factor, factor) if signed else factor
