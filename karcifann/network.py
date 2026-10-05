"""A plain fully connected feed-forward network.

The architecture matches the KarciFANN papers: an input layer, one or more
hidden layers and an output layer, every layer followed by the same activation
function (sigmoid in the original work), trained by backpropagation on a
differentiable loss.

Nothing here knows about fractional derivatives.  The network only produces the
ordinary Newton gradients ``dJ/dW``; turning those into a KarciFANN update is
the optimizer's job (see :mod:`karcifann.optimizers`).  That separation is what
lets the exact same model be trained with SGD or with KarciFANN and compared
under identical initial conditions, as the papers require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .activations import Activation, get_activation
from .losses import Loss, get_loss

__all__ = ["MLP", "ForwardCache"]


@dataclass
class ForwardCache:
    """Everything backpropagation needs from one forward pass."""

    inputs: list[np.ndarray]   # input fed to each layer
    pre: list[np.ndarray]      # W.x + b per layer
    outputs: list[np.ndarray]  # activation(pre) per layer

    @property
    def prediction(self) -> np.ndarray:
        return self.outputs[-1]


class MLP:
    """Multilayer perceptron trained by backpropagation.

    Parameters
    ----------
    layer_sizes:
        ``[n_input, n_hidden..., n_output]``.  ``[784, 50, 10]`` reproduces the
        network of Karakurt et al. (2025).
    activation:
        Activation for the hidden layers.
    output_activation:
        Activation for the output layer; defaults to ``activation`` because the
        papers put a sigmoid on the output layer too.
    loss:
        Loss function; see :mod:`karcifann.losses`.
    weight_init:
        ``"uniform01"``  -- ``U(0, 1)``, the initialisation shown in the weight
        tables of Karakurt et al. (2022) and the default here;
        ``"glorot"``/``"he"``/``"normal"`` -- the usual alternatives.
    bias_init:
        ``"random"`` (default) draws biases from the same distribution as the
        weights; ``"zeros"`` is the usual deep-learning default.  Beware that a
        parameter sitting exactly at zero is a singularity of the KarciFANN
        prefactor ``(J/W)^(alpha-1)``: for ``alpha > 1`` it produces an
        effectively infinite first step and the network saturates immediately.
        The papers fold the bias in as one more randomly initialised weight,
        which is what ``"random"`` reproduces.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation="sigmoid",
        output_activation=None,
        loss="mse",
        weight_init: str = "uniform01",
        bias_init: str = "random",
        use_bias: bool = True,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1
        self.use_bias = use_bias
        self.loss: Loss = get_loss(loss)

        hidden = get_activation(activation)
        out = hidden if output_activation is None else get_activation(output_activation)
        self.activations: list[Activation] = [
            type(hidden)() for _ in range(self.n_layers - 1)
        ] + [out]

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._initialise()

        # Set when the loss can absorb the output activation's Jacobian.
        self._fused_output = (
            self.activations[-1].name in getattr(self.loss, "fuses_with", ())
            and hasattr(self.loss, "fused_gradient")
        )
        if self.activations[-1].name == "softmax" and not self._fused_output:
            # Softmax has a full Jacobian; only a loss that absorbs it (cross
            # entropy) can be backpropagated through the element-wise path.
            raise ValueError(
                "a softmax output layer needs loss='categorical_crossentropy'; "
                f"got {self.loss.name!r}"
            )

    # ------------------------------------------------------------------ setup

    def _draw(self, shape: tuple[int, ...], n_in: int, n_out: int) -> np.ndarray:
        if self.weight_init == "uniform01":
            return self.rng.random(shape)
        if self.weight_init == "glorot":
            limit = np.sqrt(6.0 / (n_in + n_out))
            return self.rng.uniform(-limit, limit, shape)
        if self.weight_init == "he":
            return self.rng.normal(0.0, np.sqrt(2.0 / n_in), shape)
        if self.weight_init == "normal":
            return self.rng.normal(0.0, 0.1, shape)
        raise ValueError(f"unknown weight_init {self.weight_init!r}")

    def _initialise(self) -> None:
        if self.bias_init not in ("random", "zeros"):
            raise ValueError(f"unknown bias_init {self.bias_init!r}")
        self.weights.clear()
        self.biases.clear()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(self._draw((n_in, n_out), n_in, n_out))
            if self.use_bias and self.bias_init == "random":
                self.biases.append(self._draw((n_out,), n_in, n_out))
            else:
                self.biases.append(np.zeros(n_out))

    @property
    def parameters(self) -> list[np.ndarray]:
        """Weights first, then biases -- the order the optimizers expect."""
        return self.weights + (self.biases if self.use_bias else [])

    def copy(self) -> "MLP":
        """Deep copy sharing nothing but the hyperparameters.

        Used to give KarciFANN and classical ANN *identical* starting weights,
        which is how every comparison in the papers is set up.
        """
        clone = type(self).__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.layer_sizes = list(self.layer_sizes)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.activations = [type(a)() for a in self.activations]
        clone.loss = type(self.loss)()
        clone.rng = np.random.default_rng()
        return clone

    # ---------------------------------------------------------------- forward

    def forward(self, x: np.ndarray) -> ForwardCache:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        inputs, pres, outs = [], [], []
        current = x
        for w, b, act in zip(self.weights, self.biases, self.activations):
            inputs.append(current)
            pre = current @ w
            if self.use_bias:
                pre = pre + b
            out = act.forward(pre)
            pres.append(pre)
            outs.append(out)
            current = out
        return ForwardCache(inputs, pres, outs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).prediction

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict(x), axis=1)

    def compute_loss(self, x: np.ndarray, t: np.ndarray) -> float:
        return self.loss.value(self.predict(x), np.atleast_2d(t))

    # --------------------------------------------------------------- backward

    def backward(self, cache: ForwardCache, t: np.ndarray):
        """Ordinary Newton gradients of the loss w.r.t. every parameter.

        Returns ``(grads_w, grads_b)``; ``grads_b`` is ``None`` when the network
        has no biases.
        """
        t = np.atleast_2d(np.asarray(t, dtype=float))
        y = cache.prediction
        if y.shape != t.shape:
            raise ValueError(f"target shape {t.shape} does not match output shape {y.shape}")

        last = self.activations[-1]
        if self._fused_output:
            delta = self.loss.fused_gradient(y, t)
        else:
            delta = self.loss.gradient(y, t) * last.backward(y, cache.pre[-1])

        grads_w: list[np.ndarray | None] = [None] * self.n_layers
        grads_b: list[np.ndarray | None] = [None] * self.n_layers
        for layer in range(self.n_layers - 1, -1, -1):
            grads_w[layer] = cache.inputs[layer].T @ delta
            grads_b[layer] = np.sum(delta, axis=0)
            if layer > 0:
                act = self.activations[layer - 1]
                delta = (delta @ self.weights[layer].T) * act.backward(
                    cache.outputs[layer - 1], cache.pre[layer - 1]
                )
        return grads_w, (grads_b if self.use_bias else None)

    def gradients(self, x: np.ndarray, t: np.ndarray):
        """Convenience wrapper: ``(loss_value, flat_parameter_gradients)``."""
        cache = self.forward(x)
        value = self.loss.value(cache.prediction, np.atleast_2d(t))
        grads_w, grads_b = self.backward(cache, t)
        grads = list(grads_w) + (list(grads_b) if grads_b is not None else [])
        return value, grads

    # ------------------------------------------------------------------ misc

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        arch = "-".join(str(n) for n in self.layer_sizes)
        return (
            f"MLP({arch}, activation={self.activations[0].name!r}, "
            f"output={self.activations[-1].name!r}, loss={self.loss.name!r})"
        )
