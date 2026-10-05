"""The training loop."""

import numpy as np
import pytest

from karcifann import MLP, KarciFANN, SGD, one_hot, train, xor


def _small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (20, 4))
    t = one_hot(rng.integers(0, 3, 20), 3)
    net = MLP([4, 6, 3], weight_init="glorot", seed=1)
    return net, x, t


def test_history_length_matches_epochs():
    net, x, t = _small()
    history = train(net, SGD(lr=0.5), x, t, epochs=12)
    assert history.epochs == 12
    assert len(history.accuracy) == 12


def test_validation_metrics_are_recorded():
    net, x, t = _small()
    history = train(net, SGD(lr=0.5), x, t, epochs=5, validation_data=(x[:5], t[:5]))
    assert len(history.val_loss) == 5
    assert len(history.val_accuracy) == 5
    assert set(history.final) == {"loss", "accuracy", "val_loss", "val_accuracy"}


def test_loss_decreases():
    net, x, t = _small()
    history = train(net, SGD(lr=1.0), x, t, epochs=200)
    assert history.loss[-1] < history.loss[0]


def test_full_batch_is_the_default():
    """One update per epoch, so 1 epoch on N samples == 1 manual step."""
    net_a, x, t = _small()
    net_b = net_a.copy()

    train(net_a, SGD(lr=0.5), x, t, epochs=1)

    loss, grads = net_b.gradients(x, t)
    SGD(lr=0.5).step(net_b.parameters, grads, loss)

    for a, b in zip(net_a.parameters, net_b.parameters):
        assert np.array_equal(a, b)


def test_mini_batches_perform_more_updates_than_full_batch():
    net_a, x, t = _small()
    net_b = net_a.copy()
    train(net_a, SGD(lr=0.5), x, t, epochs=1)
    train(net_b, SGD(lr=0.5), x, t, epochs=1, batch_size=5)
    assert not np.allclose(net_a.weights[0], net_b.weights[0])


def test_shuffle_is_reproducible_for_a_fixed_seed():
    net_a, x, t = _small()
    net_b = net_a.copy()
    train(net_a, SGD(lr=0.5), x, t, epochs=3, batch_size=4, shuffle=True, seed=5)
    train(net_b, SGD(lr=0.5), x, t, epochs=3, batch_size=4, shuffle=True, seed=5)
    assert np.array_equal(net_a.weights[0], net_b.weights[0])


def test_karcifann_factor_history_is_recorded():
    net, x, t = _small()
    history = train(net, KarciFANN(alpha=1.5), x, t, epochs=7)
    assert len(history.factor_mean) == 7


def test_divergence_is_detected_and_reported():
    """A zero-valued parameter is a singularity of (J/W)^(a-1).

    With the eps floor switched off, the very first update of a bias that
    starts at zero is infinite.  That must be flagged, not silently absorbed.
    """
    net = MLP([2, 4, 1], weight_init="uniform01", bias_init="zeros", seed=3)
    x, t = xor()
    history = train(net, KarciFANN(alpha=1.5, eps=0.0), x, t, epochs=50)
    assert history.diverged_at == 0
    assert history.epochs == 1


def test_training_can_be_allowed_to_run_through_nonfinite_values():
    net = MLP([2, 4, 1], weight_init="uniform01", bias_init="zeros", seed=3)
    x, t = xor()
    history = train(
        net, KarciFANN(alpha=1.5, eps=0.0), x, t, epochs=6, stop_on_nonfinite=False
    )
    assert history.diverged_at is None
    assert history.epochs == 6
