"""Activation functions used in the KarciFANN papers.

The reference studies use the logistic sigmoid throughout; tanh, the
"hyperbolic tangent sigmoid" (tansig) of Karakurt et al. (IDAP'24) and ReLU
appear in the follow-up comparisons.  Softmax is added so cross-entropy
classification heads can be built as well.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Activation", "Sigmoid", "Tanh", "TanSig", "ReLU", "Identity", "Softmax", "get_activation"]


class Activation:
    """Element-wise activation with its derivative expressed via the output."""

    name = "activation"
    #: True when ``backward`` needs the pre-activation instead of the output.
    needs_pre_activation = False

    def forward(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, out: np.ndarray, pre: np.ndarray) -> np.ndarray:
        """d(out)/d(pre), evaluated element-wise."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, a):
        # Branch-free stable logistic: exp is only ever applied to <= 0.
        positive = a >= 0
        z = np.empty_like(a, dtype=float)
        ez = np.exp(-np.abs(a))
        z[positive] = 1.0 / (1.0 + ez[positive])
        z[~positive] = ez[~positive] / (1.0 + ez[~positive])
        return z

    def backward(self, out, pre):
        return out * (1.0 - out)


class Tanh(Activation):
    name = "tanh"

    def forward(self, a):
        return np.tanh(a)

    def backward(self, out, pre):
        return 1.0 - out * out


class TanSig(Activation):
    """``tansig(a) = 2 / (1 + exp(-2a)) - 1``, numerically identical to tanh."""

    name = "tansig"

    def forward(self, a):
        return 2.0 / (1.0 + np.exp(-2.0 * np.clip(a, -350.0, 350.0))) - 1.0

    def backward(self, out, pre):
        return 1.0 - out * out


class ReLU(Activation):
    name = "relu"

    def forward(self, a):
        return np.maximum(a, 0.0)

    def backward(self, out, pre):
        return (pre > 0.0).astype(float)


class Identity(Activation):
    name = "identity"

    def forward(self, a):
        return a

    def backward(self, out, pre):
        return np.ones_like(out)


class Softmax(Activation):
    """Row-wise softmax.

    ``backward`` returns the diagonal of the Jacobian only, so it must not be
    combined with an arbitrary loss.  Pair it with
    :class:`~karcifann.losses.CategoricalCrossEntropy`, which detects the
    combination and short-circuits to the ``p - t`` gradient.
    """

    name = "softmax"

    def forward(self, a):
        shifted = a - np.max(a, axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / np.sum(e, axis=-1, keepdims=True)

    def backward(self, out, pre):
        return out * (1.0 - out)


_REGISTRY = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "tansig": TanSig,
    "relu": ReLU,
    "identity": Identity,
    "linear": Identity,
    "softmax": Softmax,
}


def get_activation(spec) -> Activation:
    if isinstance(spec, Activation):
        return spec
    try:
        return _REGISTRY[str(spec).lower()]()
    except KeyError:
        raise ValueError(
            f"unknown activation {spec!r}; expected one of {sorted(_REGISTRY)}"
        ) from None
