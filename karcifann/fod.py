"""Karci fractional order derivative (FOD).

Karci (2013) defines the fractional order derivative through the L'Hospital
limit of the ratio of the alpha-th powers of the dependent and the independent
variable::

                           f(x + h)^a - f(x)^a          d/dh f(x + h)^a
    f^(a)(x) = lim_{h->0} --------------------- = lim  -----------------
                            (x + h)^a - x^a             d/dh (x + h)^a

Applying L'Hospital's rule once gives the closed form used everywhere in the
KarciFANN papers (Eq. 5 of Karakurt, Saygili & Karci, Turk J Elec Eng &
Comp Sci 33(3), 2025)::

                  ( f(x) )^(a-1)
    D^a_K f(x) =  ( ---- )        . f'(x)
                  (  x   )

For ``a = 1`` this collapses to the ordinary (Newton) derivative, which is why
KarciFANN with ``alpha = 1`` behaves exactly like classical gradient descent
with a learning rate of one.

The operator is *not* linear and it obeys a telescoping chain rule
(Karci, Science Innovation 3(6), 2015)::

    D^a_K(y/x) = D^a_K(y/u) . D^a_K(u/x)

which is what makes the whole backpropagation chain in KarciFANN collapse to a
single ``(J/W)^(a-1)`` prefactor in front of the ordinary gradient.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "POWER_MODES",
    "fod_factor",
    "karci_fod",
    "karci_fod_limit",
]

#: How ``(f/x)^(a-1)`` is evaluated when the base ``f/x`` is negative.
#:
#: ``"abs"``      -- ``|f/x|^(a-1)``; always a positive real number.  This is the
#:                   convention used by the KarciFANN reference implementations:
#:                   the fractional factor only rescales the gradient, it never
#:                   flips the descent direction.
#: ``"signed"``   -- ``sign(f/x) * |f/x|^(a-1)``; keeps the sign of the base.
#: ``"real"``     -- real part of the principal complex power,
#:                   ``|f/x|^(a-1) * cos(pi*(a-1))`` for a negative base.
#: ``"complex"``  -- the principal complex power itself (returns complex dtype).
#: ``"nan"``      -- leave ``nan`` for negative bases (strict real-valued FOD).
POWER_MODES = ("abs", "signed", "real", "complex", "nan")

def _power(base: np.ndarray, exponent: float, mode: str) -> np.ndarray:
    """Raise ``base`` to ``exponent`` honouring the negative-base convention."""
    if mode not in POWER_MODES:
        raise ValueError(f"unknown power mode {mode!r}; expected one of {POWER_MODES}")

    base = np.asarray(base, dtype=float)
    if exponent == 0.0:
        # 0^0 is taken as 1 so that alpha == 1 is exactly the Newton derivative.
        return np.ones_like(base)

    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.power(np.abs(base), exponent)
    if mode == "abs":
        return magnitude
    if mode == "signed":
        return np.where(base < 0.0, -magnitude, magnitude)
    if mode == "real":
        return np.where(base < 0.0, magnitude * np.cos(np.pi * exponent), magnitude)
    if mode == "complex":
        return np.power(base.astype(complex), exponent)
    # mode == "nan"
    return np.where(base < 0.0, np.nan, magnitude)


def fod_factor(f_val, x_val, alpha: float, mode: str = "abs", eps: float = 0.0):
    """Return the Karci prefactor ``(f/x)^(alpha-1)``.

    Parameters
    ----------
    f_val, x_val:
        Value of the function and of the independent variable.  Broadcasting
        follows the usual numpy rules, so these may be arrays.
    alpha:
        Order of the fractional derivative.
    mode:
        Negative-base convention, see :data:`POWER_MODES`.
    eps:
        If positive, ``|f|`` and ``|x|`` are floored at ``eps`` before the
        division.  This keeps the factor finite when a weight crosses zero or
        when the error reaches zero, which is essential for a stable optimizer
        but is *not* part of the mathematical definition.
    """
    f_val = np.asarray(f_val, dtype=float)
    x_val = np.asarray(x_val, dtype=float)
    if alpha == 1.0:
        return np.ones(np.broadcast(f_val, x_val).shape)

    if eps > 0.0:
        f_val = np.sign(f_val) * np.maximum(np.abs(f_val), eps)
        # sign(0) == 0, so a genuine zero would be wiped out; put it back.
        f_val = np.where(f_val == 0.0, eps, f_val)
        x_val = np.where(x_val < 0.0, -1.0, 1.0) * np.maximum(np.abs(x_val), eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        base = f_val / x_val
        return _power(base, alpha - 1.0, mode)


def karci_fod(f, x, alpha: float, df=None, h: float = 1e-6, mode: str = "abs"):
    """Karci fractional order derivative of ``f`` at ``x``.

    ``df`` is the ordinary first derivative of ``f``.  When it is omitted a
    central finite difference of step ``h`` is used instead.
    """
    x = np.asarray(x, dtype=float)
    f_val = np.asarray(f(x), dtype=float)
    if df is None:
        derivative = (np.asarray(f(x + h), dtype=float) - np.asarray(f(x - h), dtype=float)) / (2.0 * h)
    else:
        derivative = np.asarray(df(x), dtype=float)
    return fod_factor(f_val, x, alpha, mode=mode) * derivative


def karci_fod_limit(f, x, alpha: float, h: float = 1e-6):
    """Karci FOD evaluated straight from the limit definition (Definition 1).

    This is the un-simplified quotient ``(f(x+h)^a - f(x)^a) / ((x+h)^a - x^a)``
    and only makes sense where ``f(x) > 0`` and ``x > 0``.  It exists purely so
    the closed form can be cross-checked against the definition it came from.
    """
    x = np.asarray(x, dtype=float)
    numerator = np.power(np.asarray(f(x + h), dtype=float), alpha) - np.power(
        np.asarray(f(x), dtype=float), alpha
    )
    denominator = np.power(x + h, alpha) - np.power(x, alpha)
    return numerator / denominator
