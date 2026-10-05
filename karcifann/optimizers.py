"""Optimizers, including the KarciFANN update rule.

Classical rules (GD/SGD, momentum, AdaGrad, RMSProp) follow the formulation in
Karakurt, Saygili & Karci, "Mathematical model of the KarciFANN machine
learning method", Firat Univ. J. Eng. Sci. 38(1), 2026.

The KarciFANN rule replaces the learning rate by the prefactor that falls out
of the Karci fractional order derivative.  Because the Karci chain rule
telescopes,

    D^a_K(J/W) = D^a_K(J/Y_c) . D^a_K(Y_c/Y) . ... . D^a_K(A/W)
               = (J/W)^(a-1) . dJ/dY_c . dY_c/dY . ... . dA/dW
               = (J/W)^(a-1) . dJ/dW

every intermediate ``(.)^(a-1)`` cancels and the whole fractional
backpropagation reduces to *one* factor in front of the ordinary gradient::

    W <- W - (J / W)^(a-1) . dJ/dW

with ``J`` the scalar error of the current iteration.  There is no learning
rate: the step size is set by the network's own error and by the magnitude of
each individual weight, and it is recomputed every iteration.  At ``alpha = 1``
the factor is exactly ``1``, so KarciFANN degenerates into gradient descent
with a learning rate of one -- the equivalence the papers report at 1.0.
"""

from __future__ import annotations

import numpy as np

from .caputo_fabrizio import caputo_factor, cf_factor
from .fod import fod_factor

__all__ = [
    "Optimizer",
    "SGD",
    "Momentum",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "KarciFANN",
    "CaputoFabrizioGD",
    "CaputoGD",
    "get_optimizer",
]


class Optimizer:
    """Base class.  ``step`` mutates the parameter arrays in place."""

    name = "optimizer"

    def step(self, params: list[np.ndarray], grads: list[np.ndarray], loss: float) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any accumulated state (called at the start of training)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


class SGD(Optimizer):
    """``W <- W - lr * (dJ/dW + weight_decay * W)``.

    With the full training set per step this is batch gradient descent; with
    mini-batches it is (mini-batch) SGD.  This is the "classical ANN" baseline
    the papers compare against.
    """

    name = "sgd"

    def __init__(self, lr: float = 0.1, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def step(self, params, grads, loss):
        for p, g in zip(params, grads):
            update = g + self.weight_decay * p if self.weight_decay else g
            p -= self.lr * update

    def __repr__(self):  # pragma: no cover
        return f"SGD(lr={self.lr})"


class Momentum(Optimizer):
    """``V <- beta V + (1 - beta) g``,  ``W <- W - lr V``."""

    name = "momentum"

    def __init__(self, lr: float = 0.1, beta: float = 0.9, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self._v: list[np.ndarray] | None = None

    def reset(self):
        self._v = None

    def step(self, params, grads, loss):
        if self._v is None:
            self._v = [np.zeros_like(p) for p in params]
        for v, p, g in zip(self._v, params, grads):
            if self.weight_decay:
                g = g + self.weight_decay * p
            v *= self.beta
            v += (1.0 - self.beta) * g
            p -= self.lr * v


class AdaGrad(Optimizer):
    """``G <- G + g^2``,  ``W <- W - lr / sqrt(G + eps) * g``."""

    name = "adagrad"

    def __init__(self, lr: float = 0.1, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._g2: list[np.ndarray] | None = None

    def reset(self):
        self._g2 = None

    def step(self, params, grads, loss):
        if self._g2 is None:
            self._g2 = [np.zeros_like(p) for p in params]
        for acc, p, g in zip(self._g2, params, grads):
            if self.weight_decay:
                g = g + self.weight_decay * p
            acc += g * g
            p -= self.lr * g / np.sqrt(acc + self.eps)


class RMSProp(Optimizer):
    """``V <- beta V + (1 - beta) g^2``,  ``W <- W - lr / sqrt(V + eps) * g``."""

    name = "rmsprop"

    def __init__(
        self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8, weight_decay: float = 0.0
    ) -> None:
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._v: list[np.ndarray] | None = None

    def reset(self):
        self._v = None

    def step(self, params, grads, loss):
        if self._v is None:
            self._v = [np.zeros_like(p) for p in params]
        for v, p, g in zip(self._v, params, grads):
            if self.weight_decay:
                g = g + self.weight_decay * p
            v *= self.beta
            v += (1.0 - self.beta) * g * g
            p -= self.lr * g / np.sqrt(v + self.eps)


class Adam(Optimizer):
    """Kingma & Ba (2015).  Not part of the KarciFANN line, but the reference
    every modern practitioner would actually reach for, and therefore the right
    yardstick for how much headroom a sigmoid/MSE/plain-descent setup leaves.

    ``m <- b1 m + (1-b1) g``,  ``v <- b2 v + (1-b2) g^2``, both bias-corrected,
    then ``W <- W - lr . m_hat / (sqrt(v_hat) + eps)``.
    """

    name = "adam"

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._t = 0

    def reset(self):
        self._m = self._v = None
        self._t = 0

    def step(self, params, grads, loss):
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        correction1 = 1.0 - self.beta1**self._t
        correction2 = 1.0 - self.beta2**self._t

        for m, v, p, g in zip(self._m, self._v, params, grads):
            if self.weight_decay:
                g = g + self.weight_decay * p
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def __repr__(self):  # pragma: no cover
        return f"Adam(lr={self.lr})"


class KarciFANN(Optimizer):
    """Karci fractional artificial neural network weight update.

    ``W <- W - (J / W)^(alpha - 1) . (dJ/dW + weight_decay . W)``

    Parameters
    ----------
    alpha:
        Order of the Karci fractional derivative.  ``1.0`` reproduces
        :class:`SGD` with ``lr=1``.  The papers report useful behaviour roughly
        in ``0.8 <= alpha <= 1.8`` for MSE-trained sigmoid networks.
    power_mode:
        How ``(J/W)^(alpha-1)`` is evaluated for a negative weight, where the
        base is negative and the real power is undefined.  ``"abs"`` (the
        default, and what the reference implementations do) uses ``|J/W|``, so
        the factor is a positive rescaling and never reverses the descent
        direction.  See :data:`karcifann.fod.POWER_MODES` for the alternatives;
        ``"signed"`` in particular turns descent into ascent for every negative
        weight and is provided only for study.
    eps:
        Floor applied to ``|J|`` and ``|W|`` before dividing.  Without it the
        factor blows up to infinity whenever a weight crosses zero (for
        ``alpha > 1``) or whenever the error reaches zero (for ``alpha < 1``).
    weight_decay:
        Coefficient ``lambda`` of the decay term of Karakurt,
        "The effect of weight decay in the KarciFANN method" (2025):
        ``W <- W - D^a_K (dJ/dW + lambda W)``.
    decoupled_decay:
        Apply the decay outside the fractional factor
        (``W <- W - D^a_K dJ/dW - lambda W``) instead, AdamW-style.
    max_step:
        Optional absolute clip on ``|Delta W|``.  ``None`` (default) leaves the
        raw rule untouched.
    scale:
        Plain multiplier on the whole update.  The published method has no such
        knob and ``scale = 1`` is the method as written; it exists so that
        KarciFANN can be compared against other rules at a *matched* effective
        step size, separating the shape of the prefactor from its magnitude.
    """

    name = "karcifann"

    def __init__(
        self,
        alpha: float = 1.0,
        power_mode: str = "abs",
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        decoupled_decay: bool = False,
        max_step: float | None = None,
        scale: float = 1.0,
    ) -> None:
        self.alpha = float(alpha)
        self.power_mode = power_mode
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.decoupled_decay = bool(decoupled_decay)
        self.max_step = max_step
        self.scale = float(scale)
        #: Diagnostics from the most recent step, filled in by :meth:`step`.
        self.last_factor_stats: dict[str, float] = {}

    def factors(self, params, loss: float) -> list[np.ndarray]:
        """The per-parameter ``(J/W)^(alpha-1)`` prefactors for this step."""
        return [
            fod_factor(loss, p, self.alpha, mode=self.power_mode, eps=self.eps)
            for p in params
        ]

    def step(self, params, grads, loss):
        if self.power_mode == "complex":
            raise ValueError(
                "power_mode='complex' produces a complex update that cannot be written "
                "back into real-valued weights; use it through factors() for inspection, "
                "or pick 'abs' / 'real' / 'signed' to train."
            )
        stats_min, stats_max, stats_sum, stats_n = np.inf, -np.inf, 0.0, 0
        for p, g in zip(params, grads):
            factor = fod_factor(loss, p, self.alpha, mode=self.power_mode, eps=self.eps)
            update = g + self.weight_decay * p if (self.weight_decay and not self.decoupled_decay) else g
            delta = self.scale * factor * update
            if self.decoupled_decay and self.weight_decay:
                delta = delta + self.weight_decay * p
            if self.max_step is not None:
                delta = np.clip(delta, -self.max_step, self.max_step)
            p -= delta

            finite = factor[np.isfinite(factor)] if factor.ndim else factor
            if np.size(finite):
                stats_min = min(stats_min, float(np.min(finite)))
                stats_max = max(stats_max, float(np.max(finite)))
                stats_sum += float(np.sum(finite))
                stats_n += int(np.size(finite))
        self.last_factor_stats = {
            "min": stats_min if stats_n else float("nan"),
            "max": stats_max if stats_n else float("nan"),
            "mean": stats_sum / stats_n if stats_n else float("nan"),
        }

    def __repr__(self):  # pragma: no cover
        return f"KarciFANN(alpha={self.alpha}, power_mode={self.power_mode!r})"


class _DistanceFractionalGD(Optimizer):
    """Fractional gradient descent driven by how far a weight has travelled.

    Both the Caputo and the Caputo-Fabrizio derivative of ``J`` with respect to
    a weight are integrals from a lower terminal ``W0`` up to the current
    ``W``.  Freezing the integrand at the upper terminal -- the standard
    truncation of the fractional-gradient-descent literature -- turns each into
    a closed-form prefactor on the ordinary gradient::

        W  <-  W  -  lr . Phi(W - W0; alpha) . dJ/dW

    The essential structural difference from KarciFANN is that ``Phi`` here does
    not see the error at all.  It is a function of weight geometry only, so
    these rules cannot anneal themselves as the network converges the way the
    Karci prefactor ``(J/W)^(alpha-1)`` does.

    Parameters
    ----------
    alpha:
        Fractional order.  ``1.0`` makes every prefactor exactly ``1``, so the
        rule degenerates to ``SGD(lr)``.
    lr:
        Step multiplier.  Unlike KarciFANN these rules keep a learning rate:
        the truncated prefactor is a preconditioner, not a step size.  Set it
        to ``1.0`` to compare the raw prefactors head to head.
    terminal:
        ``"origin"`` fixes ``W0 = 0``, so the prefactor is a function of ``|W|``.
        ``"previous"`` sets ``W0`` to the previous iterate, so it is a function
        of the previous step length -- which shrinks towards zero as the method
        converges and stalls it, the known failure mode of Caputo fractional
        gradient descent.  The first iteration has no previous value and falls
        back to the origin.
    signed:
        Keep the sign of ``W - W0`` in the prefactor.  Off by default, for the
        same reason as in :class:`KarciFANN`: a negative prefactor turns the
        step into gradient ascent.
    """

    name = "fractional_gd"

    def __init__(
        self,
        alpha: float = 0.5,
        lr: float = 1.0,
        terminal: str = "origin",
        signed: bool = False,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        max_step: float | None = None,
    ) -> None:
        if terminal not in ("origin", "previous"):
            raise ValueError(f"unknown terminal {terminal!r}; expected 'origin' or 'previous'")
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.terminal = terminal
        self.signed = bool(signed)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_step = max_step
        self._previous: list[np.ndarray] | None = None
        self.last_factor_stats: dict[str, float] = {}

    def reset(self) -> None:
        self._previous = None

    def _prefactor(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _displacements(self, params) -> list[np.ndarray]:
        """``W - W0`` per parameter; copies, so mutating the weights is safe."""
        if self.terminal == "origin" or self._previous is None:
            return [np.array(p, dtype=float, copy=True) for p in params]
        return [p - q for p, q in zip(params, self._previous)]

    def factors(self, params, loss: float | None = None) -> list[np.ndarray]:
        """The per-parameter prefactors for this step (``loss`` is unused)."""
        return [self._prefactor(d) for d in self._displacements(params)]

    def step(self, params, grads, loss):
        displacements = self._displacements(params)
        self._previous = [np.array(p, dtype=float, copy=True) for p in params]

        stats_min, stats_max, stats_sum, stats_n = np.inf, -np.inf, 0.0, 0
        for p, g, d in zip(params, grads, displacements):
            factor = self._prefactor(d)
            update = g + self.weight_decay * p if self.weight_decay else g
            delta = self.lr * factor * update
            if self.max_step is not None:
                delta = np.clip(delta, -self.max_step, self.max_step)
            p -= delta

            finite = factor[np.isfinite(factor)]
            if finite.size:
                stats_min = min(stats_min, float(np.min(finite)))
                stats_max = max(stats_max, float(np.max(finite)))
                stats_sum += float(np.sum(finite))
                stats_n += int(finite.size)
        self.last_factor_stats = {
            "min": stats_min if stats_n else float("nan"),
            "max": stats_max if stats_n else float("nan"),
            "mean": stats_sum / stats_n if stats_n else float("nan"),
        }

    def __repr__(self):  # pragma: no cover
        return f"{type(self).__name__}(alpha={self.alpha}, lr={self.lr}, terminal={self.terminal!r})"


class CaputoFabrizioGD(_DistanceFractionalGD):
    """Fractional gradient descent on the Caputo-Fabrizio derivative.

    ``Phi = (M(alpha)/alpha) . (1 - exp(-alpha |W - W0| / (1 - alpha)))``

    The prefactor is **bounded above by** ``M(alpha)/alpha``: the exponential
    kernel saturates, so however far a weight has travelled the step can never
    be scaled beyond that.  This is the practical difference from the Caputo
    rule, whose ``|W - W0|^(1-alpha)`` grows without limit.

    Valid for ``0 < alpha <= 1``.
    """

    name = "caputo_fabrizio"

    def __init__(self, alpha: float = 0.5, normalization: str = "unit", **kwargs) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.normalization = normalization

    def _prefactor(self, delta):
        return cf_factor(
            delta, self.alpha, normalization=self.normalization, signed=self.signed, eps=self.eps
        )


class CaputoGD(_DistanceFractionalGD):
    """Fractional gradient descent on the Caputo derivative.

    ``Phi = |W - W0|^(1-alpha) / Gamma(2 - alpha)``

    The classical baseline of the fractional-gradient-descent literature, and
    the definition Caputo-Fabrizio was proposed to improve on.
    """

    name = "caputo"

    def _prefactor(self, delta):
        return caputo_factor(delta, self.alpha, signed=self.signed, eps=self.eps)


_REGISTRY = {
    "sgd": SGD,
    "gd": SGD,
    "momentum": Momentum,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "karcifann": KarciFANN,
    "karci": KarciFANN,
    "caputo_fabrizio": CaputoFabrizioGD,
    "cf": CaputoFabrizioGD,
    "caputo": CaputoGD,
}


def get_optimizer(spec, **kwargs) -> Optimizer:
    if isinstance(spec, Optimizer):
        return spec
    try:
        return _REGISTRY[str(spec).lower()](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown optimizer {spec!r}; expected one of {sorted(_REGISTRY)}"
        ) from None
