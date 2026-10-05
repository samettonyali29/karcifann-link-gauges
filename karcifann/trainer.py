"""Training loop shared by the KarciFANN and the classical-ANN experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import accuracy
from .network import MLP
from .optimizers import Optimizer

__all__ = ["History", "train"]


@dataclass
class History:
    """Per-epoch record of a training run."""

    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    factor_mean: list[float] = field(default_factory=list)
    diverged_at: int | None = None

    @property
    def epochs(self) -> int:
        return len(self.loss)

    @property
    def final(self) -> dict[str, float]:
        out = {"loss": self.loss[-1] if self.loss else float("nan")}
        if self.accuracy:
            out["accuracy"] = self.accuracy[-1]
        if self.val_loss:
            out["val_loss"] = self.val_loss[-1]
        if self.val_accuracy:
            out["val_accuracy"] = self.val_accuracy[-1]
        return out


def train(
    model: MLP,
    optimizer: Optimizer,
    x: np.ndarray,
    t: np.ndarray,
    epochs: int = 100,
    batch_size: int | None = None,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    track_accuracy: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
    verbose: int = 0,
    stop_on_nonfinite: bool = True,
) -> History:
    """Train ``model`` in place.

    ``batch_size=None`` means full-batch, which is what the KarciFANN papers
    do: one forward pass over the whole training set, one error value ``J``,
    one weight update per epoch.

    Training stops early and records ``History.diverged_at`` if the loss or the
    weights become non-finite -- a real failure mode of KarciFANN for
    unfavourable ``alpha``, and one worth reporting rather than hiding.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    t = np.atleast_2d(np.asarray(t, dtype=float))
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    optimizer.reset()

    history = History()
    params = model.parameters

    for epoch in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        size = n if batch_size is None else min(batch_size, n)

        epoch_loss, batches = 0.0, 0
        for start in range(0, n, size):
            idx = order[start : start + size]
            cache = model.forward(x[idx])
            loss_value = model.loss.value(cache.prediction, t[idx])
            grads_w, grads_b = model.backward(cache, t[idx])
            grads = list(grads_w) + (list(grads_b) if grads_b is not None else [])
            optimizer.step(params, grads, loss_value)
            epoch_loss += loss_value
            batches += 1

        # Report the loss of the *updated* model, so the curve reflects the
        # network the epoch actually produced.
        prediction = model.predict(x)
        epoch_loss = model.loss.value(prediction, t)
        history.loss.append(epoch_loss)
        if track_accuracy:
            history.accuracy.append(accuracy(prediction, t))
        stats = getattr(optimizer, "last_factor_stats", None)
        if stats:
            history.factor_mean.append(stats.get("mean", float("nan")))

        if validation_data is not None:
            vx, vt = validation_data
            vp = model.predict(vx)
            history.val_loss.append(model.loss.value(vp, np.atleast_2d(vt)))
            if track_accuracy:
                history.val_accuracy.append(accuracy(vp, vt))

        if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
            msg = f"epoch {epoch + 1:5d}/{epochs}  loss={epoch_loss:.6f}"
            if track_accuracy:
                msg += f"  acc={history.accuracy[-1] * 100:6.2f}%"
            if validation_data is not None:
                msg += f"  val_loss={history.val_loss[-1]:.6f}"
                if track_accuracy:
                    msg += f"  val_acc={history.val_accuracy[-1] * 100:6.2f}%"
            print(msg)

        if stop_on_nonfinite and (
            not np.isfinite(epoch_loss) or any(not np.all(np.isfinite(p)) for p in params)
        ):
            history.diverged_at = epoch
            break

    return history
