"""Classification metrics."""

import numpy as np
import pytest

from karcifann import accuracy, confusion_matrix, one_hot, precision_recall_f1


# ------------------------------------------------------------------ one_hot


def test_one_hot_round_trips_through_argmax():
    labels = np.array([0, 3, 1, 2, 3])
    encoded = one_hot(labels, 4)
    assert encoded.shape == (5, 4)
    assert np.array_equal(np.argmax(encoded, axis=1), labels)
    assert np.array_equal(encoded.sum(axis=1), np.ones(5))


def test_one_hot_infers_the_class_count():
    assert one_hot(np.array([0, 1, 2])).shape == (3, 3)


def test_one_hot_accepts_more_classes_than_appear():
    encoded = one_hot(np.array([0, 1]), 5)
    assert encoded.shape == (2, 5)
    assert encoded[:, 2:].sum() == 0


# ----------------------------------------------------------------- accuracy


def test_accuracy_on_a_perfect_prediction():
    t = one_hot(np.array([0, 1, 2]), 3)
    assert accuracy(t, t) == 1.0


def test_accuracy_counts_argmax_not_probability():
    """A confident wrong answer and a marginal right one both count once."""
    pred = np.array([[0.34, 0.33, 0.33], [0.0, 0.0, 1.0]])
    true = one_hot(np.array([0, 0]), 3)
    assert accuracy(pred, true) == 0.5


def test_accuracy_handles_binary_column_vectors():
    pred = np.array([[0.9], [0.2], [0.6]])
    true = np.array([[1.0], [0.0], [0.0]])
    assert accuracy(pred, true) == pytest.approx(2 / 3)


# --------------------------------------------------------- confusion matrix


def test_confusion_matrix_is_true_by_predicted():
    pred = one_hot(np.array([0, 1, 1, 2]), 3)
    true = one_hot(np.array([0, 1, 2, 2]), 3)
    cm = confusion_matrix(pred, true)
    assert cm.shape == (3, 3)
    assert cm[2, 1] == 1        # a class-2 sample predicted as class 1
    assert cm[1, 1] == 1
    assert cm.sum() == 4
    assert np.array_equal(cm.sum(axis=1), [1, 1, 2])   # rows are the true counts


def test_confusion_matrix_of_a_perfect_classifier_is_diagonal():
    t = one_hot(np.array([0, 1, 2, 2]), 3)
    cm = confusion_matrix(t, t)
    assert np.array_equal(cm, np.diag(np.diag(cm)))


# ------------------------------------------------------ precision / recall / F1


def test_perfect_prediction_scores_one_everywhere():
    t = one_hot(np.array([0, 1, 2, 1]), 3)
    assert precision_recall_f1(t, t) == pytest.approx((1.0, 1.0, 1.0))


def test_macro_scores_match_a_hand_computation():
    """Two classes: one perfectly recalled, one missed entirely."""
    pred = one_hot(np.array([0, 0, 0, 0]), 2)
    true = one_hot(np.array([0, 0, 1, 1]), 2)
    precision, recall, f1 = precision_recall_f1(pred, true)
    # class 0: precision 2/4, recall 2/2 ; class 1: never predicted -> 0, 0
    assert precision == pytest.approx((0.5 + 0.0) / 2)
    assert recall == pytest.approx((1.0 + 0.0) / 2)
    assert f1 == pytest.approx(((2 * 0.5 * 1.0 / 1.5) + 0.0) / 2)


def test_macro_average_weights_a_rare_class_equally():
    """The reason macro-F1 is reported for the imbalanced Dry Bean set."""
    pred = one_hot(np.array([0] * 9 + [0]), 2)
    true = one_hot(np.array([0] * 9 + [1]), 2)
    assert accuracy(pred, true) == pytest.approx(0.9)
    _, recall, _ = precision_recall_f1(pred, true)
    assert recall == pytest.approx(0.5)      # the missed minority class halves it


def test_micro_average_equals_accuracy():
    pred = one_hot(np.array([0, 1, 1, 2]), 3)
    true = one_hot(np.array([0, 1, 2, 2]), 3)
    micro = precision_recall_f1(pred, true, average="micro")
    assert micro[0] == pytest.approx(accuracy(pred, true))


def test_per_class_scores_are_returned_unaveraged():
    pred = one_hot(np.array([0, 0, 1]), 2)
    true = one_hot(np.array([0, 1, 1]), 2)
    precision, recall, f1 = precision_recall_f1(pred, true, average=None)
    assert precision.shape == recall.shape == f1.shape == (2,)


def test_a_class_never_predicted_scores_zero_not_nan():
    pred = one_hot(np.array([0, 0]), 2)
    true = one_hot(np.array([0, 1]), 2)
    scores = precision_recall_f1(pred, true, average=None)
    assert all(np.all(np.isfinite(s)) for s in scores)


def test_unknown_average_is_rejected():
    t = one_hot(np.array([0, 1]), 2)
    with pytest.raises(ValueError):
        precision_recall_f1(t, t, average="geometric")
