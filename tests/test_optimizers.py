"""The KarciFANN weight update, and the classical rules it is compared against."""

import numpy as np
import pytest

from karcifann import MLP, AdaGrad, KarciFANN, Momentum, RMSProp, SGD, fod_factor, one_hot, train
from karcifann.optimizers import get_optimizer


def _tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([3, 4, 2], activation="sigmoid", loss="mse", weight_init="glorot", seed=seed)
    x = rng.normal(0, 1, (5, 3))
    t = one_hot(rng.integers(0, 2, 5), 2)
    return net, x, t


# ------------------------------------------------------ the defining identity


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0, 1.3, 1.8, 2.4])
def test_update_equals_fractional_factor_times_newton_gradient(alpha):
    """W_new = W - (J/W)^(a-1) . dJ/dW, element by element."""
    net, x, t = _tiny_problem()
    loss, grads = net.gradients(x, t)
    before = [p.copy() for p in net.parameters]

    KarciFANN(alpha=alpha, eps=0.0).step(net.parameters, grads, loss)

    for w0, g, w1 in zip(before, grads, net.parameters):
        with np.errstate(divide="ignore"):
            expected = w0 - np.power(np.abs(loss / w0), alpha - 1.0) * g
        assert np.allclose(w1, expected, rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0.6, 1.2, 1.9])
def test_explicit_karci_chain_telescopes_to_a_single_prefactor(alpha):
    """Reproduce Eq. (24) factor by factor and check it collapses to Eq. (3).

    The paper writes the output-layer update as a product of three Karci
    fractional derivatives, ``D^a(H/Zc) . D^a(Zc/Z) . D^a(Z/W)``.  Each
    contributes its own ``(.)^(a-1)`` prefactor; the claim is that they
    telescope into ``(H/W)^(a-1)``.  Here the long form is built explicitly and
    compared with what the optimizer computes.
    """
    net = MLP([2, 2, 1], activation="sigmoid", loss="mse", weight_init="glorot", seed=4)
    x = np.array([[0.6, -1.1]])
    t = np.array([[1.0]])

    cache = net.forward(x)
    zc = cache.outputs[-1]          # Y(3,1,2), post-activation output
    z = cache.pre[-1]               # Y(3,1,1), pre-activation output
    w = net.weights[-1]             # W(2,j,1)
    hidden = cache.inputs[-1]       # Y(2,j,2)
    h_value = net.loss.value(zc, t)

    d_h_d_zc = net.loss.gradient(zc, t)                                  # dH/dZc
    d_zc_d_z = net.activations[-1].backward(zc, z)                       # dZc/dZ
    d_z_d_w = hidden.T                                                   # dZ/dW

    long_form = (
        fod_factor(h_value, zc, alpha)
        * d_h_d_zc
        * fod_factor(zc, z, alpha)
        * d_zc_d_z
    ).T * fod_factor(z, w.T, alpha).T * d_z_d_w

    short_form = fod_factor(h_value, w, alpha) * (hidden.T @ (d_h_d_zc * d_zc_d_z))
    assert np.allclose(long_form, short_form)


# ------------------------------------------------------- alpha = 1 equivalence


@pytest.mark.parametrize("weight_init", ["uniform01", "glorot"])
def test_alpha_one_is_bit_identical_to_gradient_descent_with_lr_one(weight_init):
    """The papers' headline sanity check: at 1.0 KarciFANN *is* classical ANN."""
    net_k = MLP([4, 6, 3], activation="sigmoid", loss="mse", weight_init=weight_init, seed=17)
    net_s = net_k.copy()

    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (12, 4))
    t = one_hot(rng.integers(0, 3, 12), 3)

    h_k = train(net_k, KarciFANN(alpha=1.0), x, t, epochs=60)
    h_s = train(net_s, SGD(lr=1.0), x, t, epochs=60)

    for a, b in zip(net_k.parameters, net_s.parameters):
        assert np.array_equal(a, b)
    assert h_k.loss == h_s.loss


def test_alpha_one_differs_from_gradient_descent_with_another_lr():
    """Guard against the equivalence test passing for a trivial reason."""
    net_k, x, t = _tiny_problem(1)
    net_s = net_k.copy()
    train(net_k, KarciFANN(alpha=1.0), x, t, epochs=25)
    train(net_s, SGD(lr=0.5), x, t, epochs=25)
    assert not np.allclose(net_k.weights[0], net_s.weights[0])


# --------------------------------------------------- behaviour of the prefactor


def test_factor_grows_as_the_error_shrinks_for_alpha_below_one():
    opt = KarciFANN(alpha=0.5)
    w = [np.array([0.5])]
    big = opt.factors(w, loss=1e-2)[0][0]
    small = opt.factors(w, loss=1e-6)[0][0]
    assert small > big


def test_factor_shrinks_as_the_error_shrinks_for_alpha_above_one():
    """Above 1 the step self-anneals: converging shrinks the step size."""
    opt = KarciFANN(alpha=1.5)
    w = [np.array([0.5])]
    big = opt.factors(w, loss=1e-2)[0][0]
    small = opt.factors(w, loss=1e-6)[0][0]
    assert small < big


def test_factor_is_one_everywhere_at_alpha_one():
    opt = KarciFANN(alpha=1.0)
    factors = opt.factors([np.array([[-2.0, 0.0, 3.0]])], loss=0.31)
    assert np.array_equal(factors[0], np.ones((1, 3)))


def test_step_records_factor_statistics():
    net, x, t = _tiny_problem(3)
    loss, grads = net.gradients(x, t)
    opt = KarciFANN(alpha=1.4)
    opt.step(net.parameters, grads, loss)
    stats = opt.last_factor_stats
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert np.isfinite(list(stats.values())).all()


# ------------------------------------------------------- negative-base handling


def test_abs_mode_never_reverses_the_descent_direction():
    w = np.array([[-0.4, 0.4]])
    factors = KarciFANN(alpha=1.7, power_mode="abs").factors([w], loss=0.2)[0]
    assert np.all(factors > 0)


def test_signed_mode_turns_descent_into_ascent_for_negative_weights():
    """Documented pathology of the sign-preserving reading of (J/W)^(a-1)."""
    w = np.array([[-0.4]])
    factor = KarciFANN(alpha=1.5, power_mode="signed").factors([w], loss=0.2)[0]
    assert factor[0, 0] < 0


def test_nan_mode_makes_negative_weights_undefined():
    w = np.array([[-0.4, 0.4]])
    factors = KarciFANN(alpha=1.5, power_mode="nan").factors([w], loss=0.2)[0]
    assert np.isnan(factors[0, 0]) and np.isfinite(factors[0, 1])


# -------------------------------------------------------------- numerical care


def test_eps_floor_keeps_a_weight_crossing_zero_finite():
    grads = [np.array([1.0])]
    w_unprotected = [np.array([0.0])]
    w_protected = [np.array([0.0])]

    KarciFANN(alpha=1.5, eps=0.0).step(w_unprotected, grads, 0.25)
    KarciFANN(alpha=1.5, eps=1e-12).step(w_protected, grads, 0.25)

    assert not np.isfinite(w_unprotected[0][0])
    assert np.isfinite(w_protected[0][0])


def test_max_step_clips_the_update():
    w = [np.array([1.0])]
    KarciFANN(alpha=0.5, max_step=0.1).step(w, [np.array([1e6])], 0.25)
    assert w[0][0] == pytest.approx(0.9)


# ---------------------------------------------------------------- weight decay


def test_coupled_weight_decay_is_scaled_by_the_fractional_factor():
    """W <- W - D^a_K (dJ/dW + lambda W), Eq. (4) of the weight-decay paper."""
    w0, g, loss, alpha, lam = 0.4, 0.3, 0.2, 1.6, 1e-2
    w = [np.array([w0])]
    KarciFANN(alpha=alpha, weight_decay=lam, eps=0.0).step(w, [np.array([g])], loss)
    factor = abs(loss / w0) ** (alpha - 1)
    assert w[0][0] == pytest.approx(w0 - factor * (g + lam * w0))


def test_decoupled_weight_decay_sits_outside_the_fractional_factor():
    w0, g, loss, alpha, lam = 0.4, 0.3, 0.2, 1.6, 1e-2
    w = [np.array([w0])]
    KarciFANN(alpha=alpha, weight_decay=lam, decoupled_decay=True, eps=0.0).step(
        w, [np.array([g])], loss
    )
    factor = abs(loss / w0) ** (alpha - 1)
    assert w[0][0] == pytest.approx(w0 - factor * g - lam * w0)


def test_weight_decay_shrinks_weights_when_there_is_no_gradient():
    w = [np.array([2.0])]
    KarciFANN(alpha=1.0, weight_decay=0.1).step(w, [np.array([0.0])], 0.5)
    assert w[0][0] == pytest.approx(1.8)


# ----------------------------------------------------------- classical optimizers


def test_sgd_matches_its_formula():
    w = [np.array([1.0, -2.0])]
    SGD(lr=0.25).step(w, [np.array([0.4, 0.8])], 0.1)
    assert np.allclose(w[0], [1.0 - 0.1, -2.0 - 0.2])


def test_momentum_matches_its_formula():
    opt = Momentum(lr=0.5, beta=0.9)
    w = [np.array([1.0])]
    g = np.array([2.0])
    opt.step(w, [g], 0.1)
    v1 = 0.9 * 0.0 + 0.1 * 2.0
    assert w[0][0] == pytest.approx(1.0 - 0.5 * v1)
    opt.step(w, [g], 0.1)
    v2 = 0.9 * v1 + 0.1 * 2.0
    assert w[0][0] == pytest.approx(1.0 - 0.5 * v1 - 0.5 * v2)


def test_adagrad_accumulates_squared_gradients():
    opt = AdaGrad(lr=0.5, eps=0.0)
    w = [np.array([1.0])]
    opt.step(w, [np.array([2.0])], 0.1)
    assert w[0][0] == pytest.approx(1.0 - 0.5 * 2.0 / np.sqrt(4.0))
    opt.step(w, [np.array([2.0])], 0.1)
    assert w[0][0] == pytest.approx(1.0 - 0.5 - 0.5 * 2.0 / np.sqrt(8.0))


def test_rmsprop_matches_its_formula():
    opt = RMSProp(lr=0.5, beta=0.9, eps=0.0)
    w = [np.array([1.0])]
    opt.step(w, [np.array([2.0])], 0.1)
    v = 0.1 * 4.0
    assert w[0][0] == pytest.approx(1.0 - 0.5 * 2.0 / np.sqrt(v))


def test_optimizer_state_is_cleared_by_reset():
    opt = Momentum(lr=0.5)
    w = [np.array([1.0])]
    opt.step(w, [np.array([2.0])], 0.1)
    opt.reset()
    assert opt._v is None


@pytest.mark.parametrize("name", ["sgd", "momentum", "adagrad", "rmsprop", "karcifann"])
def test_optimizer_registry(name):
    assert get_optimizer(name) is not None


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError):
        get_optimizer("banana")


def test_complex_mode_cannot_be_used_to_train():
    """A complex update has nowhere to go in a real-valued weight array."""
    opt = KarciFANN(alpha=1.5, power_mode="complex")
    assert np.iscomplexobj(opt.factors([np.array([-0.4])], loss=0.2)[0])
    with pytest.raises(ValueError, match="complex"):
        opt.step([np.array([-0.4])], [np.array([0.1])], 0.2)


def test_adam_matches_its_formula():
    from karcifann import Adam

    opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=0.0)
    w = [np.array([1.0])]
    g = np.array([2.0])
    opt.step(w, [g], 0.1)
    m, v = 0.1 * 2.0, 0.001 * 4.0
    expected = 1.0 - 0.1 * (m / 0.1) / np.sqrt(v / 0.001)
    assert w[0][0] == pytest.approx(expected)


def test_adam_bias_correction_makes_the_first_step_lr_sized():
    from karcifann import Adam

    w = [np.array([0.0])]
    Adam(lr=0.05).step(w, [np.array([7.0])], 0.1)
    assert w[0][0] == pytest.approx(-0.05, rel=1e-4)


def test_adam_state_is_cleared_by_reset():
    from karcifann import Adam

    opt = Adam()
    opt.step([np.array([1.0])], [np.array([1.0])], 0.1)
    assert opt._t == 1
    opt.reset()
    assert opt._m is None and opt._t == 0
