"""Backpropagation with a fractional gauge inserted at *every* link.

KarciFANN is derived by replacing each factor of the backpropagation chain with
a fractional derivative of the numerator with respect to the denominator::

      Xi              Xi        Fo(3,k)      F(3,k)
  D^a ------  =  D^a -------  D^a -------  D^a ---------
      W(3,j,k)       Fo(3,k)     F(3,k)       W(3,j,k)

For the Karci operator the per-link prefactors telescope and the whole thing
collapses to ``(J/W)^(a-1) . dJ/dW``, which is why :class:`~karcifann.optimizers.KarciFANN`
can be a pure optimizer that never touches the backward pass.

For Caputo-Fabrizio they do **not** telescope.  So there are two distinct
readings of "use CF the way KarciFANN uses Karci's derivative", and they are
genuinely different methods:

* **terminal** -- apply the fractional derivative once, to ``J`` along the
  weight axis, leaving backpropagation Newtonian.  That is
  :class:`~karcifann.optimizers.CaputoFabrizioGD`.
* **chained** -- apply it at every link and let the prefactors compound instead
  of collapsing.  That is what this module does.

The chained form is not the fractional derivative *of* anything: the product of
per-link CF derivatives is not ``CF D^a(J/W)``.  It is a consistent deformation
of backpropagation (it returns the exact Newton gradient at ``a = 1``) and it is
the honest way to ask what KarciFANN's construction does when the operator
underneath it lacks a chain rule.

Cost note
---------
A gauge that depends on both ends of a link is per-*path*, not per-element: the
link from ``W[i,j]`` to ``pre[s,j]`` needs an ``(s, i, j)`` tensor.  Gauges that
depend only on the denominator -- the truncated Caputo and Caputo-Fabrizio
prefactors both do -- declare ``denominator_only`` and take a cheap elementwise
path instead.
"""

from __future__ import annotations

import numpy as np

from .caputo_fabrizio import caputo_factor, cf_factor
from .fod import fod_factor
from .network import MLP

__all__ = ["ChainedGaugeMLP", "caputo_gauge", "cf_gauge", "karci_gauge", "newton_gauge"]


def _tag(fn, denominator_only: bool, name: str):
    fn.denominator_only = denominator_only
    fn.gauge_name = name
    return fn


def karci_gauge(alpha: float, mode: str = "abs", eps: float = 1e-12):
    """``Phi(num, den) = (num/den)^(alpha-1)`` -- the Karci prefactor.

    Depends on both ends, so it is a per-path gauge.  Because ``|a/b| . |b/c| =
    |a/c|``, chaining it reproduces the single ``(J/W)^(alpha-1)`` factor
    exactly; that identity is the end-to-end check on this module.
    """
    return _tag(
        lambda num, den: fod_factor(num, den, alpha, mode=mode, eps=eps),
        False,
        f"karci(alpha={alpha})",
    )


def cf_gauge(alpha: float, normalization: str = "unit", eps: float = 1e-12):
    """Truncated Caputo-Fabrizio prefactor of one link, terminal at the origin.

    ``CF D^a(num/den) ~= Phi_CF(den) . d(num)/d(den)``, so the gauge is a
    function of the denominator alone.
    """
    return _tag(
        lambda num, den: cf_factor(den, alpha, normalization=normalization, eps=eps),
        True,
        f"cf(alpha={alpha})",
    )


def caputo_gauge(alpha: float, eps: float = 1e-12):
    """Truncated Caputo prefactor of one link, terminal at the origin."""
    return _tag(
        lambda num, den: caputo_factor(den, alpha, eps=eps),
        True,
        f"caputo(alpha={alpha})",
    )


def newton_gauge():
    """A gauge of 1 everywhere; recovers ordinary backpropagation."""
    return _tag(lambda num, den: np.ones_like(np.asarray(den, dtype=float)), True, "newton")


class ChainedGaugeMLP(MLP):
    """An :class:`~karcifann.network.MLP` whose backward pass is gauged link by link.

    Pair it with ``SGD(lr=1.0)``: the fractional part lives in the derivative,
    not in the optimizer, which is exactly the split the KarciFANN derivation
    assumes.

    ``gauge=None`` (the default) leaves the network identical to ``MLP``.
    """

    def __init__(self, *args, gauge=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gauge = gauge

    # ------------------------------------------------------------------ links

    def backward(self, cache, t: np.ndarray):
        if self.gauge is None:
            return super().backward(cache, t)

        t = np.atleast_2d(np.asarray(t, dtype=float))
        y = cache.prediction
        if y.shape != t.shape:
            raise ValueError(f"target shape {t.shape} does not match output shape {y.shape}")

        gauge = self.gauge
        cheap = getattr(gauge, "denominator_only", False)
        loss_value = self.loss.value(y, t)

        # Link J -> y, then y -> pre[-1].
        if self._fused_output:
            delta = self.loss.fused_gradient(y, t) * gauge(loss_value, cache.pre[-1])
        else:
            delta = self.loss.gradient(y, t) * gauge(loss_value, y)
            delta = delta * self.activations[-1].backward(y, cache.pre[-1])
            delta = delta * gauge(y, cache.pre[-1])

        grads_w: list[np.ndarray | None] = [None] * self.n_layers
        grads_b: list[np.ndarray | None] = [None] * self.n_layers

        for layer in range(self.n_layers - 1, -1, -1):
            inputs, pre, w = cache.inputs[layer], cache.pre[layer], self.weights[layer]

            # Link pre[layer] -> W[layer]  (and -> b[layer]).
            if cheap:
                grads_w[layer] = (inputs.T @ delta) * gauge(None, w)
                grads_b[layer] = np.sum(delta, axis=0) * gauge(None, self.biases[layer])
            else:
                g = gauge(pre[:, None, :], w[None, :, :])       # (samples, in, out)
                grads_w[layer] = np.einsum("si,sj,sij->ij", inputs, delta, g, optimize=True)
                gb = gauge(pre, self.biases[layer][None, :])     # (samples, out)
                grads_b[layer] = np.sum(delta * gb, axis=0)

            if layer == 0:
                break

            # Link pre[layer] -> out[layer-1], then out[layer-1] -> pre[layer-1].
            previous_out, previous_pre = cache.outputs[layer - 1], cache.pre[layer - 1]
            if cheap:
                delta = (delta @ w.T) * gauge(None, previous_out)
            else:
                g = gauge(pre[:, :, None], previous_out[:, None, :])  # (samples, out, in)
                delta = np.einsum("sj,ij,sji->si", delta, w, g, optimize=True)
            delta = delta * self.activations[layer - 1].backward(previous_out, previous_pre)
            delta = delta * gauge(previous_out, previous_pre)

        return grads_w, (grads_b if self.use_bias else None)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = getattr(self.gauge, "gauge_name", None) or "none"
        return f"ChainedGaugeMLP({'-'.join(map(str, self.layer_sizes))}, gauge={name})"
