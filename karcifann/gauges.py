"""The complete family of learning-rate-free gauged-backpropagation rules.

A per-link gauge ``Phi(numerator, denominator)`` collapses a backpropagation
chain exactly when ``Phi(a,b) Phi(b,c) = Phi(a,c)``, and the general solution of
that is ``Phi(a,b) = g(a)/g(b)`` for an arbitrary ``g`` (fix ``c`` and read off
``g(x) = Phi(x,c)``).  Every such rule therefore collapses to

    W  <-  W  -  [ g(J) / g(|W|) ] . dJ/dW

with no learning rate, and KarciFANN is the single member with
``g(x) = x^(alpha-1)``.

That raises the question this module exists to answer: **is the power form
actually the best member, or merely the analytically convenient one?**

Two structural remarks worth carrying into the comparison:

* The powers are exactly the *scale-invariant* members.  ``|J/W|^k`` depends
  only on the ratio, so it has no characteristic magnitude and is unbounded in
  both directions.  Every other ``g`` introduces a scale ``c`` and saturates.
* ``g(x) = exp(x/c)`` gives ``Phi = exp((J - |W|)/c)`` -- an exponential,
  translation-based gauge in the spirit of the Caputo-Fabrizio kernel, but one
  that *does* telescope.  It is the closest a CF-flavoured rule can come to
  being a KarciFANN.
"""

from __future__ import annotations

import numpy as np

from .optimizers import Optimizer

__all__ = [
    "Gauge",
    "PowerGauge",
    "LogGauge",
    "TanhGauge",
    "ExpGauge",
    "GaugedDescent",
    "GAUGE_FAMILIES",
]


class Gauge:
    """``Phi(a, b) = g(a) / g(b)``."""

    family = "gauge"

    def __init__(self, shape: float) -> None:
        self.shape = float(shape)

    def g(self, x):
        """The generating function.  Only used for the telescoping checks."""
        raise NotImplementedError

    def factor(self, loss: float, weight: np.ndarray) -> np.ndarray:
        """``Phi(J, |W|)``.  Overridden where the ratio is unstable to form."""
        return np.asarray(self.g(loss), dtype=float) / self.g(weight)

    def is_identity(self, tolerance: float = 1e-3) -> bool:
        """True when this gauge is indistinguishable from plain descent.

        Several families have a degenerate limit in their shape parameter where
        ``Phi -> 1`` everywhere, and the rule collapses onto gradient descent:
        ``tanh(x/c)`` saturates for ``c`` below about 1e-3, and
        ``log(1 + x/c)`` tends to 1 logarithmically.  A search that picks such a
        shape has *converged on switching the operator off*, which is a result,
        not a grid that ran out of room -- and the two have to be told apart
        before either is reported.

        The check is numeric rather than per-family algebra, so it stays correct
        for any generating function added later.
        """
        import numpy as _np

        # The probe range has to cover the weights a trained network actually
        # holds, not a convenient subset.  Measured over these experiments the
        # smallest |weight| after training is around 1e-4, so probing only down
        # to 1e-3 declared gauges identical that still differ by 30% on the
        # smallest 0.2% of parameters.  1e-5 is deliberately below the observed
        # floor: a false "degenerate" label would let an under-tuned winner
        # through the audit, so the test errs toward flagging.
        from .audit import PROBE_RANGE

        losses = _np.array([1e-3, 1e-2, 0.09, 0.3])
        weights = PROBE_RANGE
        for loss in losses:
            factors = _np.asarray(self.factor(float(loss), weights), dtype=float)
            if not _np.all(_np.abs(factors - 1.0) <= tolerance):
                return False
        return True

    @property
    def name(self) -> str:
        return f"{self.family}({self.shape:g})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


class PowerGauge(Gauge):
    """``g(x) = x^k``, giving ``Phi = |J/W|^k``.  KarciFANN is ``k = alpha - 1``."""

    family = "power"

    def g(self, x):
        return np.power(np.asarray(x, dtype=float), self.shape)

    def factor(self, loss, weight):
        # Formed as a single ratio: J^k and W^k separately overflow long before
        # their quotient does.
        return np.power(loss / weight, self.shape)

    @classmethod
    def from_alpha(cls, alpha: float) -> "PowerGauge":
        return cls(alpha - 1.0)

    @property
    def alpha(self) -> float:
        return self.shape + 1.0


class LogGauge(Gauge):
    """``g(x) = log(1 + x/c)``.  Linear for ``x << c``, logarithmic beyond."""

    family = "log"

    def g(self, x):
        return np.log1p(np.asarray(x, dtype=float) / self.shape)


class TanhGauge(Gauge):
    """``g(x) = tanh(x/c)``.  Saturating, so ``Phi -> 1`` once both ends exceed ``c``."""

    family = "tanh"

    def g(self, x):
        return np.tanh(np.asarray(x, dtype=float) / self.shape)


class ExpGauge(Gauge):
    """``g(x) = exp(x/c)``, i.e. ``Phi = exp((J - |W|)/c)``.

    The telescoping counterpart of an exponential kernel: it responds to the
    *difference* between loss and weight rather than their ratio.
    """

    family = "exp"

    def g(self, x):
        return np.exp(np.clip(np.asarray(x, dtype=float) / self.shape, -700.0, 700.0))

    def factor(self, loss, weight):
        return np.exp(np.clip((loss - weight) / self.shape, -700.0, 700.0))


#: Shape grids that put each family on a comparable footing.  ``power`` is
#: indexed by ``k = alpha - 1``; the rest by their characteristic scale ``c``.
#: Grids wide enough that the optimum is interior on every dataset in the
#: suite.  The first version stopped at 0.003 for ``log`` and ``tanh`` and at 30
#: for ``exp``, and those bounds turned out to be the winner on 14 of 32
#: dataset-family pairs -- meaning the search ran out of room rather than
#: finding an optimum.  ``karcifann.audit`` now fails a test if that recurs.
GAUGE_FAMILIES = {
    "power": (PowerGauge, [-1.5, -1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.5, 1.0, 1.5]),
    "log": (LogGauge, [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
    "tanh": (TanhGauge, [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
    "exp": (ExpGauge, [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]),
}


class GaugedDescent(Optimizer):
    """``W <- W - scale . [g(J)/g(|W|)] . dJ/dW``.

    The generalisation of :class:`~karcifann.optimizers.KarciFANN` to an
    arbitrary generating function.  ``GaugedDescent(PowerGauge(alpha - 1))``
    reproduces ``KarciFANN(alpha)`` exactly.

    ``scale`` exists for the same reason as on ``KarciFANN``: comparing gauges
    at their natural magnitude compares step sizes, so it has to be tunable
    away before the *shape* of a gauge can be judged.

    ``mode`` splits the prefactor into its two halves, which is the ablation the
    decomposition makes possible.  Since ``Phi = g(J)/g(c) . g(c)/g(|W|)`` for
    any reference ``c``:

    ``"full"``    the rule as published, ``g(J)/g(|W|)``;
    ``"error"``   ``g(J)/g(c)`` -- tracks the error over iterations but is the
                  same for every parameter, so it is a *step schedule*;
    ``"weight"``  ``g(c)/g(|W|)`` -- varies across parameters but not with the
                  error, so it is a *diagonal preconditioner*.

    Running all three answers which half of KarciFANN actually does the work.
    The reference ``c`` only shifts the overall magnitude, which ``scale``
    absorbs, so it is fixed at 1 by default.
    """

    MODES = ("full", "error", "weight")

    name = "gauged"

    def __init__(
        self,
        gauge: Gauge,
        scale: float = 1.0,
        mode: str = "full",
        reference: float = 1.0,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        max_step: float | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {self.MODES}")
        self.gauge = gauge
        self.mode = mode
        self.reference = float(reference)
        self.scale = float(scale)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.max_step = max_step
        self.last_factor_stats: dict[str, float] = {}

    def _factor(self, magnitude: float, parameter: np.ndarray) -> np.ndarray:
        weight = np.maximum(np.abs(parameter), self.eps)
        if self.mode == "full":
            return self.gauge.factor(magnitude, weight)
        if self.mode == "error":
            return np.broadcast_to(
                self.gauge.factor(magnitude, np.array(self.reference)), weight.shape
            )
        return self.gauge.factor(self.reference, weight)

    def factors(self, params, loss: float) -> list[np.ndarray]:
        magnitude = max(abs(float(loss)), self.eps)
        return [self._factor(magnitude, p) for p in params]

    def step(self, params, grads, loss):
        stats_min, stats_max, stats_sum, stats_n = np.inf, -np.inf, 0.0, 0
        magnitude = max(abs(float(loss)), self.eps)
        for p, g in zip(params, grads):
            factor = self._factor(magnitude, p)
            update = g + self.weight_decay * p if self.weight_decay else g
            delta = self.scale * factor * update
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
        return f"GaugedDescent({self.gauge.name}, mode={self.mode!r}, scale={self.scale})"
