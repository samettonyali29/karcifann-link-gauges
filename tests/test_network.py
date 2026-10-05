"""Forward pass, backpropagation and the activation/loss primitives."""

import numpy as np
import pytest

from karcifann import MLP, get_activation, get_loss, one_hot
from karcifann.activations import _REGISTRY as ACTIVATIONS
from karcifann.losses import _REGISTRY as LOSSES


# --------------------------------------------------------------- activations


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "tansig", "relu", "identity"])
def test_activation_derivative_matches_finite_difference(name):
    act = get_activation(name)
    a = np.array([[-3.0, -0.4, 0.35, 2.7]])
    h = 1e-6
    numeric = (act.forward(a + h) - act.forward(a - h)) / (2 * h)
    analytic = act.backward(act.forward(a), a)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_sigmoid_is_stable_for_large_magnitudes():
    out = get_activation("sigmoid").forward(np.array([[-1e4, 0.0, 1e4]]))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, [[0.0, 0.5, 1.0]])


def test_tansig_equals_tanh():
    a = np.linspace(-5, 5, 21).reshape(1, -1)
    assert np.allclose(get_activation("tansig").forward(a), np.tanh(a))


def test_softmax_rows_sum_to_one():
    p = get_activation("softmax").forward(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        get_activation("banana")


# --------------------------------------------------------------------- losses


@pytest.mark.parametrize("name", sorted(set(LOSSES)))
def test_loss_gradient_matches_finite_difference(name):
    loss = get_loss(name)
    rng = np.random.default_rng(0)
    y = rng.uniform(0.15, 0.85, (4, 3))
    t = one_hot(np.array([0, 1, 2, 1]), 3)

    analytic = loss.gradient(y, t)
    numeric = np.zeros_like(y)
    h = 1e-6
    for i in range(y.size):
        up, down = y.copy(), y.copy()
        up.flat[i] += h
        down.flat[i] -= h
        numeric.flat[i] = (loss.value(up, t) - loss.value(down, t)) / (2 * h)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_mse_matches_its_definition():
    y = np.array([[0.2, 0.9]])
    t = np.array([[0.0, 1.0]])
    assert get_loss("mse").value(y, t) == pytest.approx((0.04 + 0.01) / 2)


def test_rmse_is_the_square_root_of_mse():
    y = np.array([[0.2, 0.9], [0.4, 0.1]])
    t = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert get_loss("rmse").value(y, t) == pytest.approx(np.sqrt(get_loss("mse").value(y, t)))


def test_mae_matches_its_definition():
    y = np.array([[0.2, 0.9]])
    t = np.array([[0.0, 1.0]])
    assert get_loss("mae").value(y, t) == pytest.approx((0.2 + 0.1) / 2)


def test_unknown_loss_rejected():
    with pytest.raises(ValueError):
        get_loss("banana")


# ---------------------------------------------------------------- forward pass


def test_forward_reproduces_a_hand_computed_two_layer_network():
    """Equations (1)-(2) of the mathematical-model paper, worked by hand."""
    net = MLP([2, 2, 1], activation="sigmoid", loss="mse", seed=0)
    net.weights[0] = np.array([[0.10, 0.20], [0.30, 0.40]])
    net.weights[1] = np.array([[0.50], [0.60]])
    net.biases[0] = np.array([0.05, 0.07])
    net.biases[1] = np.array([0.09])

    x = np.array([[1.0, 2.0]])
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))

    y1 = sig(1.0 * 0.10 + 2.0 * 0.30 + 0.05)
    y2 = sig(1.0 * 0.20 + 2.0 * 0.40 + 0.07)
    z = sig(y1 * 0.50 + y2 * 0.60 + 0.09)

    assert float(net.predict(x)[0, 0]) == pytest.approx(float(z))


def test_bias_can_be_switched_off():
    net = MLP([2, 3, 1], use_bias=False, seed=1)
    assert net.parameters == net.weights
    cache = net.forward(np.zeros((1, 2)))
    _, grads_b = net.backward(cache, np.zeros((1, 1)))
    assert grads_b is None


# ------------------------------------------------------------- backpropagation


def _numeric_parameter_gradients(net, x, t, h=1e-6):
    numeric = []
    for p in net.parameters:
        g = np.zeros_like(p)
        for i in range(p.size):
            original = p.flat[i]
            p.flat[i] = original + h
            up = net.compute_loss(x, t)
            p.flat[i] = original - h
            down = net.compute_loss(x, t)
            p.flat[i] = original
            g.flat[i] = (up - down) / (2 * h)
        numeric.append(g)
    return numeric


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "tansig", "relu"])
@pytest.mark.parametrize("loss", ["mse", "rmse", "mae"])
def test_backprop_matches_finite_differences(activation, loss):
    rng = np.random.default_rng(7)
    net = MLP([4, 5, 3], activation=activation, loss=loss, weight_init="glorot", seed=3)
    # Nudge biases off zero so the ReLU kink is not sitting on the test point.
    net.biases = [b + rng.normal(0, 0.3, b.shape) for b in net.biases]

    x = rng.normal(0, 1, (6, 4))
    t = one_hot(rng.integers(0, 3, 6), 3)

    _, analytic = net.gradients(x, t)
    numeric = _numeric_parameter_gradients(net, x, t)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize(
    "output_activation,loss",
    [("softmax", "categorical_crossentropy"), ("sigmoid", "binary_crossentropy")],
)
def test_fused_output_gradients_match_finite_differences(output_activation, loss):
    """The p - t shortcut must agree with differentiating the real loss."""
    rng = np.random.default_rng(11)
    net = MLP(
        [4, 6, 3],
        activation="sigmoid",
        output_activation=output_activation,
        loss=loss,
        weight_init="glorot",
        seed=5,
    )
    assert net._fused_output

    x = rng.normal(0, 1, (5, 4))
    t = one_hot(rng.integers(0, 3, 5), 3)

    _, analytic = net.gradients(x, t)
    numeric = _numeric_parameter_gradients(net, x, t)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, rtol=1e-4, atol=1e-7)


def test_deep_network_backprop_matches_finite_differences():
    rng = np.random.default_rng(13)
    net = MLP([3, 4, 4, 2], activation="sigmoid", loss="mse", weight_init="glorot", seed=9)
    x = rng.normal(0, 1, (4, 3))
    t = one_hot(rng.integers(0, 2, 4), 2)

    _, analytic = net.gradients(x, t)
    numeric = _numeric_parameter_gradients(net, x, t)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, rtol=1e-4, atol=1e-7)


# ------------------------------------------------------------------- plumbing


def test_copy_shares_no_arrays_but_starts_identical():
    net = MLP([3, 4, 2], seed=42)
    clone = net.copy()
    assert all(np.array_equal(a, b) for a, b in zip(net.weights, clone.weights))
    clone.weights[0][0, 0] += 1.0
    assert not np.array_equal(net.weights[0], clone.weights[0])


def test_uniform01_init_matches_the_papers_weight_tables():
    net = MLP([3, 4, 2], weight_init="uniform01", seed=0)
    assert all(np.all((w >= 0) & (w < 1)) for w in net.weights)


def test_target_shape_is_validated():
    net = MLP([2, 3, 2], seed=0)
    cache = net.forward(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        net.backward(cache, np.zeros((1, 3)))


def test_too_few_layers_rejected():
    with pytest.raises(ValueError):
        MLP([5])


def test_softmax_output_requires_cross_entropy():
    with pytest.raises(ValueError, match="softmax"):
        MLP([3, 4, 2], output_activation="softmax", loss="mse", seed=0)


def test_copy_does_not_share_the_layer_size_list():
    net = MLP([3, 4, 2], seed=0)
    clone = net.copy()
    clone.layer_sizes.append(99)
    assert net.layer_sizes == [3, 4, 2]
