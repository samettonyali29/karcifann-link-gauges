"""The generalised gauge family, of which KarciFANN is one member.

The theorem says a link gauge collapses a chain iff it has the form
``g(a)/g(b)``.  This module implements that family; the tests check that every
member really does telescope, that the power member reproduces KarciFANN
exactly, and that the numerically-stabilised ``factor`` overrides agree with
the definition they replace.
"""

import numpy as np
import pytest

from karcifann import MLP, KarciFANN, one_hot, train
from karcifann.gauges import (
    GAUGE_FAMILIES,
    ExpGauge,
    GaugedDescent,
    LogGauge,
    PowerGauge,
    TanhGauge,
)

ALL_GAUGES = [
    PowerGauge(-0.4), PowerGauge(0.5), PowerGauge(2.0),
    LogGauge(0.05), LogGauge(1.0),
    TanhGauge(0.05), TanhGauge(1.0),
    ExpGauge(0.1), ExpGauge(1.0),
]


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([4, 6, 3], weight_init="glorot", seed=seed + 1)
    x = rng.normal(0, 1, (12, 4))
    t = one_hot(rng.integers(0, 3, 12), 3)
    return net, x, t


# ------------------------------------------------------- the defining property


@pytest.mark.parametrize("gauge", ALL_GAUGES, ids=lambda g: g.name)
def test_every_gauge_telescopes(gauge):
    """Phi(a,b) Phi(b,c) Phi(c,d) = Phi(a,d) -- the whole point of the family."""
    j, y, a, w = 0.09, 0.7, 0.4, 0.25
    phi = lambda p, q: float(gauge.g(p) / gauge.g(q))
    assert phi(j, y) * phi(y, a) * phi(a, w) == pytest.approx(phi(j, w), rel=1e-11)


@pytest.mark.parametrize("gauge", ALL_GAUGES, ids=lambda g: g.name)
def test_stabilised_factor_matches_the_generating_function(gauge):
    """``factor`` is allowed to be reformulated, not to be a different gauge."""
    loss = 0.09
    weight = np.array([0.01, 0.25, 1.0, 4.0])
    assert np.allclose(gauge.factor(loss, weight), gauge.g(loss) / gauge.g(weight), rtol=1e-10)


@pytest.mark.parametrize("gauge", ALL_GAUGES, ids=lambda g: g.name)
def test_factor_is_one_when_loss_equals_weight(gauge):
    """g(a)/g(a) = 1 for any g, so every family shares a fixed point."""
    assert float(gauge.factor(0.3, np.array([0.3]))[0]) == pytest.approx(1.0)


def test_power_gauge_is_scale_invariant_and_the_others_are_not():
    """The powers depend only on J/W; every other family has a scale."""
    power = PowerGauge(-0.4)
    assert float(power.factor(0.09, np.array([0.3]))[0]) == pytest.approx(
        float(power.factor(0.9, np.array([3.0]))[0])
    )
    for gauge in (LogGauge(0.1), TanhGauge(0.1), ExpGauge(0.1)):
        assert float(gauge.factor(0.09, np.array([0.3]))[0]) != pytest.approx(
            float(gauge.factor(0.9, np.array([3.0]))[0])
        )


def test_exp_gauge_responds_to_the_difference_not_the_ratio():
    gauge = ExpGauge(0.5)
    assert float(gauge.factor(0.6, np.array([0.2]))[0]) == pytest.approx(
        float(gauge.factor(1.1, np.array([0.7]))[0])
    )


# ---------------------------------------------------- agreement with KarciFANN


@pytest.mark.parametrize("alpha", [0.4, 0.8, 1.0, 1.3, 1.9])
def test_power_member_is_bit_identical_to_karcifann(alpha):
    net_gauged, x, t = _problem(3)
    net_karci = net_gauged.copy()

    h_gauged = train(net_gauged, GaugedDescent(PowerGauge.from_alpha(alpha)), x, t, epochs=60)
    h_karci = train(net_karci, KarciFANN(alpha=alpha), x, t, epochs=60)

    assert h_gauged.loss == h_karci.loss
    for a, b in zip(net_gauged.parameters, net_karci.parameters):
        assert np.array_equal(a, b)


def test_alpha_round_trips_through_the_power_shape():
    assert PowerGauge.from_alpha(0.65).shape == pytest.approx(-0.35)
    assert PowerGauge(-0.35).alpha == pytest.approx(0.65)


def test_shape_zero_power_is_plain_gradient_descent():
    """g(x) = x^0 = 1, so Phi == 1 and the rule degenerates to SGD(scale)."""
    from karcifann import SGD

    net_gauged, x, t = _problem(4)
    net_sgd = net_gauged.copy()
    train(net_gauged, GaugedDescent(PowerGauge(0.0), scale=0.7), x, t, epochs=40)
    train(net_sgd, SGD(lr=0.7), x, t, epochs=40)
    for a, b in zip(net_gauged.parameters, net_sgd.parameters):
        assert np.allclose(a, b)


# ------------------------------------------------------------------- optimizer


@pytest.mark.parametrize("gauge", ALL_GAUGES, ids=lambda g: g.name)
def test_update_equals_scale_times_factor_times_gradient(gauge):
    net, x, t = _problem(5)
    loss, grads = net.gradients(x, t)
    before = [p.copy() for p in net.parameters]
    optimizer = GaugedDescent(gauge, scale=0.6)
    factors = optimizer.factors(net.parameters, loss)

    optimizer.step(net.parameters, grads, loss)

    for w0, g, phi, w1 in zip(before, grads, factors, net.parameters):
        assert np.allclose(w1, w0 - 0.6 * phi * g)


def test_eps_floors_a_zero_weight():
    weights = [np.array([0.0])]
    GaugedDescent(PowerGauge(-0.5), eps=1e-12).step(weights, [np.array([1.0])], 0.25)
    assert np.isfinite(weights[0][0])


def test_max_step_clips_the_update():
    weights = [np.array([1.0])]
    GaugedDescent(PowerGauge(-1.0), max_step=0.1).step(weights, [np.array([1e6])], 0.25)
    assert weights[0][0] == pytest.approx(0.9)


def test_step_records_factor_statistics():
    net, x, t = _problem(6)
    loss, grads = net.gradients(x, t)
    optimizer = GaugedDescent(LogGauge(0.1))
    optimizer.step(net.parameters, grads, loss)
    stats = optimizer.last_factor_stats
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert np.isfinite(list(stats.values())).all()


@pytest.mark.parametrize("family", sorted(GAUGE_FAMILIES))
def test_every_family_trains_without_diverging(family):
    cls, shapes = GAUGE_FAMILIES[family]
    net, x, t = _problem(7)
    history = train(net, GaugedDescent(cls(shapes[len(shapes) // 2])), x, t, epochs=100)
    assert history.diverged_at is None


def test_family_grids_are_well_formed():
    for family, (cls, shapes) in GAUGE_FAMILIES.items():
        assert len(shapes) >= 4
        assert all(cls(s).family == family for s in shapes)


def test_identity_probes_reach_below_the_smallest_trained_weight():
    """A too-narrow probe range would call a gauge 'identity' that is not.

    Training leaves |weight| values down to about 1e-4; probing only to 1e-3
    declared ``tanh(1e-4)`` indistinguishable from gradient descent when it
    still scales the smallest parameters by ~1.3.  A false negative here lets
    an under-tuned winner past the grid-edge audit, so the range must sit below
    anything the networks hold.
    """
    gauge = TanhGauge(1e-4)
    assert float(gauge.factor(0.09, np.array([1e-3]))[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(gauge.factor(0.09, np.array([1e-4]))[0]) > 1.2
    assert not gauge.is_identity()


def test_a_genuinely_degenerate_gauge_is_still_recognised():
    """Shrinking the shape far enough really does reproduce plain descent."""
    assert TanhGauge(1e-9).is_identity()
    assert PowerGauge(0.0).is_identity()


# ------------------------------------ transformed objective and gradient geometry


@pytest.mark.parametrize("k", [-0.4, -0.2, 0.3, 1.0])
def test_gauged_chain_factor_is_a_derivative_in_transformed_coordinates(k):
    """Proposition 7(a): g(f)f'/g = d H(f(x)) / d H(x) when H' = g.

    The manuscript claims the telescoping factor is not an exotic object but an
    ordinary derivative in the coordinate H.  If that identity broke, the
    mirror-descent reading of the whole family would go with it.
    """
    g = lambda x: x ** k
    H = lambda x: x ** (k + 1) / (k + 1)
    f = lambda x: 2.0 * x ** 1.7 + 0.5
    fprime = lambda x: 2.0 * 1.7 * x ** 0.7

    x, h = 1.3, 1e-6
    numeric = (H(f(x + h)) - H(f(x - h))) / (H(x + h) - H(x - h))
    closed = g(f(x)) * fprime(x) / g(x)
    assert numeric == pytest.approx(closed, rel=1e-7)


def test_gauged_update_is_a_hessian_metric_gradient_step():
    """Proposition 7(b): the rule's direction is grad of H(J) in the psi metric.

    With psi separable and psi'' = g(|w|), the Hessian-metric gradient of
    H(J) reproduces (g(J)/g(|w_i|)) dJ/dw_i exactly -- the direction of the
    family update, which the manuscript labels eq:family.  (Not the update
    itself: that is w - s times this vector.)
    """
    import numpy as np

    rng = np.random.default_rng(0)
    k = -0.2
    g = lambda x: x ** k

    m = 6
    w = rng.uniform(0.2, 2.0, m)
    a = rng.normal(0, 1, (m, m))
    q = a.T @ a + m * np.eye(m)
    loss = 0.5 * w @ q @ w + 1.0
    grad = q @ w

    gauged = (g(loss) / g(np.abs(w))) * grad
    metric = np.diag(1.0 / g(np.abs(w))) @ (g(loss) * grad)   # inv-Hessian times grad F
    assert np.allclose(gauged, metric, rtol=0, atol=1e-12)


def test_the_metric_reading_is_not_descent_in_the_transformed_coordinate():
    """Proposition 7's two halves are different statements, and the review of
    2026-09-23 was right that conflating them would be an error.

    Reading the rule as Euclidean gradient descent on F after substituting
    z = H(w) is *not* the same as the Hessian-metric reading.  Pulling the flow
    in z back to w puts the gauge in the denominator twice, because one factor
    comes from differentiating F with respect to z and a second from converting
    the velocity.  The rule carries one factor, so the two agree only where
    g(|w|) = 1.  The manuscript states the metric reading; this pins the
    difference so no later edit can quietly swap them.
    """
    import numpy as np

    k = -0.2
    g = lambda x: x ** k
    H = lambda x: x ** (k + 1) / (k + 1)

    w, loss, grad = 1.7, 3.1, 0.8          # one coordinate is enough

    rule = g(loss) * grad / g(w)

    # d/dz F(w(z)) at z = H(w), then w-dot = z-dot / H'(w), both by finite
    # differences so the algebra is not assumed.
    h = 1e-6
    dz = H(w + h) - H(w - h)
    dF_dz = (g(loss) * grad) * (2 * h) / dz     # F'(w) dw/dz, with F' = g(J) J'
    pullback = dF_dz / g(w)

    assert pullback == pytest.approx(g(loss) * grad / g(w) ** 2, rel=1e-6)
    assert not np.isclose(pullback, rule)


def test_expected_gauged_direction_is_a_gradient_of_transformed_batch_losses():
    """Equation (18): E_B[D0 g(J_B) grad J_B] = D0 grad E_B[H(J_B)].

    The manuscript uses this to say something precise about minibatching -- the
    expected direction targets the average of the *transformed* batch losses,
    not the transform of the objective -- in place of repeatedly observing that
    the stochastic expectation does not factor.  The right-hand gradient is
    taken by finite differences so the identity is checked rather than assumed,
    and the last assertion is the part that carries the claim: the two targets
    genuinely differ.
    """
    import numpy as np

    alpha = 0.6
    g = lambda x: x ** (alpha - 1.0)
    H = lambda x: x ** alpha / alpha           # H' = g
    h = lambda w: g(np.abs(w))

    rng = np.random.default_rng(3)
    m, n = 5, 4                                 # five batches, four coordinates
    a = rng.uniform(0.3, 1.5, (m, n))
    b = rng.uniform(0.5, 2.0, m)
    # J_B(w) = b_B + sum_i a_Bi w_i^2, positive and smooth on the orthant
    J = lambda w: b + a @ (w ** 2)
    gradJ = lambda w: 2.0 * a * w[None, :]

    w = rng.uniform(0.4, 1.6, n)
    D0 = 1.0 / h(w)

    left = (D0[None, :] * g(J(w))[:, None] * gradJ(w)).mean(axis=0)

    step = 1e-6
    right = np.empty(n)
    for i in range(n):
        up, down = w.copy(), w.copy()
        up[i] += step
        down[i] -= step
        right[i] = D0[i] * (H(J(up)).mean() - H(J(down)).mean()) / (2 * step)

    assert np.allclose(left, right, rtol=1e-6)

    # ... and E_B[H(J_B)] is not H(E_B[J_B]): the transform does not commute
    # with the average, which is the point the manuscript draws from it.
    assert not np.isclose(H(J(w)).mean(), H(J(w).mean()))


def test_the_graph_collapse_holds_with_branching_and_a_shared_parameter():
    """Proposition 5, which nothing else in this suite covers.

    Every other telescoping test here runs a *chain*.  Proposition 5 is the
    extension to a computational graph, where a parameter is reached along
    several paths and the derivative is a sum over them; the claim is that the
    same endpoint factor comes outside that sum.  A chain test cannot see
    whether that step is right, so the graph is built explicitly here.

    The second half checks the proposition's own warning: the cancellation uses
    one generator at every edge, and mixing two leaves a residue.
    """
    import numpy as np

    k = -0.3
    g = lambda x: x ** k

    # J <- u, v ; u <- W ; v <- W.  Node values are positive, as the
    # proposition requires, and W is reached along both paths.
    w = 1.4
    u, v = 2.0 * w + 1.0, 0.5 * w ** 2 + 0.7
    j = 0.3 * u + 1.1 * v + 2.0
    dj_du, dj_dv, du_dw, dv_dw = 0.3, 1.1, 2.0, w

    plain = dj_du * du_dw + dj_dv * dv_dw
    gauged = ((g(j) / g(u)) * (g(u) / g(w)) * dj_du * du_dw
              + (g(j) / g(v)) * (g(v) / g(w)) * dj_dv * dv_dw)
    collapsed = (g(j) / g(w)) * plain
    assert gauged == pytest.approx(collapsed, rel=1e-12), (
        "the sum over paths did not collapse to the endpoint factor")

    # One generator throughout is not decoration: swap it on the second link
    # and the endpoint form no longer holds.
    h = lambda x: x ** (k + 0.4)
    mixed = ((g(j) / g(u)) * (h(u) / h(w)) * dj_du * du_dw
             + (g(j) / g(v)) * (h(v) / h(w)) * dj_dv * dv_dw)
    assert not np.isclose(mixed, collapsed, rtol=0, atol=1e-12), (
        "mixing generators still collapsed, which would contradict the "
        "converse warning in Proposition 5")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_descent_bound_and_its_step_limit_hold(seed):
    """Proposition 4: equation (11), and that s < 2/(LM) strictly decreases J.

    The proposition is the only optimisation guarantee the paper states, and
    nothing pinned it.  Checked on a quadratic, where L is exactly the largest
    eigenvalue of the Hessian, so the bound can be tested at the edge of its
    own validity rather than somewhere comfortably inside it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 5
    a = rng.normal(size=(n, n))
    q = a.T @ a + n * np.eye(n)
    smooth = np.linalg.eigvalsh(q).max()          # L for a quadratic
    d = rng.uniform(0.2, 2.0, n)
    bound = d.max()                               # M >= lambda_max(D)
    w = rng.normal(size=n)
    grad = q @ w
    objective = lambda x: 0.5 * x @ q @ x

    # ||Dv||^2 <= M v'Dv, the step the proof turns on
    assert grad @ (d ** 2 * grad) <= bound * (grad @ (d * grad)) + 1e-12

    for s in np.linspace(1e-4, 0.999 * 2 / (smooth * bound), 200):
        moved = objective(w - s * (d * grad))
        predicted = objective(w) - s * (1 - 0.5 * smooth * s * bound) * (grad @ (d * grad))
        assert moved <= predicted + 1e-9, f"bound (11) violated at s={s}"
        assert moved < objective(w), f"no strict decrease at s={s} < 2/(LM)"


def test_the_batch_transform_counterexample_is_correct():
    """Section 3.6's two-loss example, which the manuscript states exactly.

    The claim is that averaging *transformed* batch losses moves the minimiser:
    the mean of the two losses is minimised at 0, but the mean of their square
    roots is minimised at 1/3.  Both figures appear in the text, so both are
    checked here rather than trusted.
    """
    import math

    import numpy as np
    from scipy.optimize import minimize_scalar

    l1 = lambda w: (w - 1) ** 2 + 1
    l2 = lambda w: (w + 1) ** 2 + 4

    mean_loss = lambda w: 0.5 * (l1(w) + l2(w))
    for w in (-2.0, 0.0, 1.7):
        assert mean_loss(w) == pytest.approx(w ** 2 + 3.5)
    assert minimize_scalar(mean_loss).x == pytest.approx(0.0, abs=1e-8)

    # alpha = 1/2 gives H(u) = 2 sqrt(u); constants do not move a minimiser.
    transformed = lambda w: math.sqrt(l1(w)) + math.sqrt(l2(w))
    step = 1e-6
    derivative = (transformed(step) - transformed(-step)) / (2 * step)
    assert derivative == pytest.approx(-1 / math.sqrt(2) + 1 / math.sqrt(5), rel=1e-5)
    assert not np.isclose(derivative, 0.0)
    assert minimize_scalar(transformed).x == pytest.approx(1 / 3, abs=1e-6)
