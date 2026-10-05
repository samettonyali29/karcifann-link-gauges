"""End-to-end learning behaviour, and the qualitative claims of the papers."""

import numpy as np
import pytest

from karcifann import (
    MLP,
    KarciFANN,
    SGD,
    digits,
    train,
    train_val_test_split,
    xor,
)

# Mini-batch settings shared by the classification tests.  Full-batch training
# reaches the same place (see experiments/digits_experiment.py) but needs
# thousands of epochs, which is too slow for a test suite.
BATCH = dict(epochs=60, batch_size=32, shuffle=True, seed=0)


def _xor_net(seed=1):
    return MLP([2, 4, 1], activation="sigmoid", loss="mse", weight_init="uniform01", seed=seed)


def _digit_net(layers=(64, 50, 10), **kwargs):
    kwargs.setdefault("activation", "sigmoid")
    kwargs.setdefault("loss", "mse")
    kwargs.setdefault("weight_init", "glorot")
    kwargs.setdefault("seed", 11)
    return MLP(list(layers), **kwargs)


@pytest.fixture(scope="module")
def digit_split():
    x, t = digits()
    return train_val_test_split(x, t, 0.15, 0.15, seed=0)


# ------------------------------------------------------------------------ XOR


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.2, 1.4, 1.6])
def test_karcifann_solves_xor_without_any_learning_rate(alpha):
    """The experiment of Karakurt et al. (2022): no learning rate anywhere."""
    x, t = xor()
    net = _xor_net()
    history = train(net, KarciFANN(alpha=alpha), x, t, epochs=4000)

    assert history.diverged_at is None
    assert history.accuracy[-1] == 1.0
    assert history.loss[-1] < 0.05
    assert np.all(np.round(net.predict(x)) == t)


def test_xor_error_grows_with_alpha_above_one():
    """Reproduces the trend of Table 3 of Karakurt et al. (2022).

    Above 1 the (J/W)^(a-1) factor shrinks as the error falls, so the network
    anneals itself into ever smaller steps and a fixed budget of 4000 epochs
    buys less progress the larger alpha is.
    """
    x, t = xor()
    losses = [
        train(_xor_net(), KarciFANN(alpha=alpha), x, t, epochs=4000).loss[-1]
        for alpha in (1.0, 1.2, 1.4, 1.6, 1.8)
    ]
    assert losses == sorted(losses)


def test_alpha_one_reproduces_classical_ann_on_xor():
    x, t = xor()
    hk = train(_xor_net(), KarciFANN(alpha=1.0), x, t, epochs=500)
    hs = train(_xor_net(), SGD(lr=1.0), x, t, epochs=500)
    assert hk.loss == hs.loss


def test_zero_initialised_bias_is_a_singularity_for_alpha_above_one():
    """A parameter at exactly zero makes (J/W)^(a-1) blow up.

    This is why biases are drawn at random by default; the papers fold the bias
    in as one more randomly initialised weight.  With zero biases and
    alpha > 1 the first step is enormous and the sigmoids saturate for good.
    """
    x, t = xor()
    net = MLP([2, 4, 1], weight_init="uniform01", bias_init="zeros", seed=1)
    assert train(net, KarciFANN(alpha=1.4), x, t, epochs=2000).accuracy[-1] < 1.0
    assert train(_xor_net(), KarciFANN(alpha=1.4), x, t, epochs=2000).accuracy[-1] == 1.0


# --------------------------------------------------------------- classification


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
def test_karcifann_classifies_digits(digit_split, alpha):
    (xtr, ttr), (xva, tva), _ = digit_split
    history = train(_digit_net(), KarciFANN(alpha=alpha), xtr, ttr,
                    validation_data=(xva, tva), **BATCH)

    assert history.diverged_at is None
    assert history.loss[-1] < history.loss[0]
    assert history.accuracy[-1] > 0.90
    assert history.val_accuracy[-1] > 0.85


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.8])
def test_karcifann_below_one_learns_faster_than_gradient_descent(digit_split, alpha):
    """The papers' central practical claim, under identical initial conditions.

    With MSE well below one, alpha < 1 puts the error in the denominator: the
    effective step is J^(alpha-1) > 1 and shrinks only as the network improves.
    That self-scaling step beats a fixed learning rate of the same numeric
    value over the same budget.
    """
    (xtr, ttr), _, _ = digit_split
    net_k = _digit_net()
    net_s = net_k.copy()

    hk = train(net_k, KarciFANN(alpha=alpha), xtr, ttr, **BATCH)
    hs = train(net_s, SGD(lr=alpha), xtr, ttr, **BATCH)

    assert hk.loss[-1] < hs.loss[-1]
    assert hk.accuracy[-1] > hs.accuracy[-1]


def test_alpha_one_matches_classical_ann_on_digits(digit_split):
    (xtr, ttr), _, _ = digit_split
    net_k = _digit_net()
    net_s = net_k.copy()

    hk = train(net_k, KarciFANN(alpha=1.0), xtr, ttr, **BATCH)
    hs = train(net_s, SGD(lr=1.0), xtr, ttr, **BATCH)

    assert hk.loss == hs.loss
    for a, b in zip(net_k.parameters, net_s.parameters):
        assert np.array_equal(a, b)


def test_runs_are_reproducible(digit_split):
    """Every comparison in the papers starts both networks from one seed."""
    (xtr, ttr), _, _ = digit_split
    ha = train(_digit_net(seed=5), KarciFANN(alpha=0.4), xtr, ttr, **BATCH)
    hb = train(_digit_net(seed=5), KarciFANN(alpha=0.4), xtr, ttr, **BATCH)
    assert ha.loss == hb.loss


def test_karcifann_trains_a_four_layer_network(digit_split):
    (xtr, ttr), _, _ = digit_split
    history = train(_digit_net((64, 40, 30, 10), seed=6), KarciFANN(alpha=0.2), xtr, ttr, **BATCH)
    assert history.diverged_at is None
    assert history.accuracy[-1] > 0.75


def test_karcifann_trains_a_softmax_cross_entropy_head(digit_split):
    (xtr, ttr), _, _ = digit_split
    net = _digit_net(
        (64, 30, 10),
        output_activation="softmax",
        loss="categorical_crossentropy",
        seed=4,
    )
    history = train(net, KarciFANN(alpha=0.5), xtr, ttr, **BATCH)
    assert history.diverged_at is None
    assert history.accuracy[-1] > 0.95


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "tansig", "relu"])
def test_karcifann_trains_with_every_activation(digit_split, activation):
    """Karakurt et al. (IDAP'24) compare sigmoid, tanh and tansig."""
    (xtr, ttr), _, _ = digit_split
    net = _digit_net((64, 30, 10), activation=activation, output_activation="sigmoid", seed=3)
    history = train(net, KarciFANN(alpha=0.4), xtr, ttr, **BATCH)
    assert history.diverged_at is None
    assert history.accuracy[-1] > 0.85


def test_mse_and_rmse_train_but_mae_does_not(digit_split):
    """Saygili et al. (IDAP'24) find MSE best and MAE worst; so does this.

    MAE is minimised element-wise by the median of the target, and for one-hot
    targets that median is zero -- so the network is pulled towards predicting
    all-zeros rather than towards discriminating.  The failure is a property of
    the loss, not of KarciFANN.
    """
    (xtr, ttr), _, _ = digit_split
    scores = {}
    for loss, alpha in (("mse", 0.4), ("rmse", 0.4), ("mae", 0.4)):
        net = _digit_net((64, 30, 10), loss=loss, seed=2)
        history = train(net, KarciFANN(alpha=alpha), xtr, ttr, **BATCH)
        assert history.diverged_at is None
        assert history.loss[-1] < history.loss[0]
        scores[loss] = history.accuracy[-1]

    assert scores["mse"] > 0.95
    assert scores["rmse"] > 0.95
    assert scores["mae"] < 0.5


def test_weight_decay_keeps_the_weights_smaller(digit_split):
    (xtr, ttr), _, _ = digit_split
    net_plain = _digit_net((64, 20, 10), seed=8)
    net_decay = net_plain.copy()

    train(net_plain, KarciFANN(alpha=0.4), xtr, ttr, **BATCH)
    train(net_decay, KarciFANN(alpha=0.4, weight_decay=1e-2), xtr, ttr, **BATCH)

    norm = lambda net: sum(float(np.sum(w**2)) for w in net.weights)
    assert norm(net_decay) < norm(net_plain)


def test_large_alpha_stalls_learning(digit_split):
    """The upper end of the reported usable range, and what lies past it.

    At alpha = 3 the factor J^2/W^2 with J ~ 0.09 is a per-step shrink of
    roughly 1e-2, so the network barely moves -- the "failure to learn" the
    papers report outside 0.8 <= alpha <= 1.8.
    """
    (xtr, ttr), _, _ = digit_split
    good = train(_digit_net(), KarciFANN(alpha=0.4), xtr, ttr, **BATCH)
    stalled = train(_digit_net(), KarciFANN(alpha=3.0), xtr, ttr, **BATCH)
    assert stalled.accuracy[-1] < 0.5 < good.accuracy[-1]
