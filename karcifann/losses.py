"""Loss functions.

MSE is the loss used in the original KarciFANN papers; MAE and RMSE come from
the loss-function comparison of Saygili, Karakurt & Karci (IDAP'24), and the
cross-entropy variants from Karakurt (J. Computer Science 10(2), 2025).

Every loss returns a **scalar** value.  That scalar is the ``H`` (or ``J``,
or ``Xi``) that the KarciFANN optimizer feeds into its ``(H/W)^(alpha-1)``
prefactor, so the choice of loss directly rescales the effective step size --
which is exactly why the papers report very different optimal ``alpha`` values
for MSE (~0.8) and for RMSE (~7).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Loss",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "CategoricalCrossEntropy",
    "BinaryCrossEntropy",
    "get_loss",
]

_EPS = 1e-12


class Loss:
    name = "loss"
    #: Set by losses that can absorb the output activation's Jacobian.
    fuses_with = ()

    def value(self, y: np.ndarray, t: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """dL/dy, same shape as ``y``."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


class MeanSquaredError(Loss):
    """``mean((y - t)**2)`` over every sample and every output unit."""

    name = "mse"

    def value(self, y, t):
        return float(np.mean((y - t) ** 2))

    def gradient(self, y, t):
        return 2.0 * (y - t) / y.size


class MeanAbsoluteError(Loss):
    """``mean(|y - t|)``.  Sub-gradient is 0 at the kink."""

    name = "mae"

    def value(self, y, t):
        return float(np.mean(np.abs(y - t)))

    def gradient(self, y, t):
        return np.sign(y - t) / y.size


class RootMeanSquaredError(Loss):
    """``sqrt(mean((y - t)**2))``."""

    name = "rmse"

    def value(self, y, t):
        return float(np.sqrt(np.mean((y - t) ** 2)))

    def gradient(self, y, t):
        rmse = np.sqrt(np.mean((y - t) ** 2))
        if rmse < _EPS:
            return np.zeros_like(y)
        return (y - t) / (y.size * rmse)


class CategoricalCrossEntropy(Loss):
    """``-mean_over_samples(sum_k t_k log y_k)`` for one-hot targets."""

    name = "categorical_crossentropy"
    fuses_with = ("softmax",)

    def value(self, y, t):
        y = np.clip(y, _EPS, None)
        return float(-np.sum(t * np.log(y)) / y.shape[0])

    def gradient(self, y, t):
        y = np.clip(y, _EPS, None)
        return -t / (y * y.shape[0])

    def fused_gradient(self, y, t):
        """dL/d(pre-activation) when the output layer is a softmax."""
        return (y - t) / y.shape[0]


class BinaryCrossEntropy(Loss):
    """Element-wise ``-(t log y + (1 - t) log(1 - y))``, averaged."""

    name = "binary_crossentropy"
    fuses_with = ("sigmoid",)

    def value(self, y, t):
        y = np.clip(y, _EPS, 1.0 - _EPS)
        return float(-np.mean(t * np.log(y) + (1.0 - t) * np.log(1.0 - y)))

    def gradient(self, y, t):
        y = np.clip(y, _EPS, 1.0 - _EPS)
        return (y - t) / (y * (1.0 - y) * y.size)

    def fused_gradient(self, y, t):
        return (y - t) / y.size


_REGISTRY = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "rmse": RootMeanSquaredError,
    "root_mean_squared_error": RootMeanSquaredError,
    "ce": CategoricalCrossEntropy,
    "crossentropy": CategoricalCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "bce": BinaryCrossEntropy,
    "binary_crossentropy": BinaryCrossEntropy,
}


def get_loss(spec) -> Loss:
    if isinstance(spec, Loss):
        return spec
    try:
        return _REGISTRY[str(spec).lower()]()
    except KeyError:
        raise ValueError(f"unknown loss {spec!r}; expected one of {sorted(_REGISTRY)}") from None
