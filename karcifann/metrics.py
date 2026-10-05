"""Classification metrics used to report KarciFANN results."""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "one_hot",
]


def one_hot(labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    labels = np.asarray(labels).astype(int).ravel()
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    out = np.zeros((labels.size, n_classes))
    out[np.arange(labels.size), labels] = 1.0
    return out


def _to_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return (y.ravel() > 0.5).astype(int)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(_to_labels(y_pred) == _to_labels(y_true)))


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, n_classes: int | None = None):
    p, t = _to_labels(y_pred), _to_labels(y_true)
    if n_classes is None:
        n_classes = int(max(p.max(), t.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (t, p), 1)
    return cm


def precision_recall_f1(y_pred: np.ndarray, y_true: np.ndarray, average: str = "macro"):
    """Macro-averaged precision, recall and F1 (the metrics the papers report)."""
    cm = confusion_matrix(y_pred, y_true)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "micro":
        total = cm.sum()
        micro = float(tp.sum() / total) if total else 0.0
        return micro, micro, micro
    if average is None:
        return precision, recall, f1
    raise ValueError(f"unknown average {average!r}")
