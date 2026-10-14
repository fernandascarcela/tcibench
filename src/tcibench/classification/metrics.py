"""Classification metrics used in TCI-Bench."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_N_BOOTSTRAP = 2000


def evaluate_classification(
    y_true: Any,
    y_pred: Any,
    minority_proba: Any,
    *,
    minority_class: Any,
) -> dict[str, float]:
    """Compute the classification metrics used in TCI-Bench."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    minority_proba = np.asarray(minority_proba)

    # Fix the label set so Macro-F1 always averages over the same classes.
    labels = np.unique(y_true)

    minority_true = (y_true == minority_class).astype(int)
    minority_pred = (y_pred == minority_class).astype(int)

    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "auprc": float(
            average_precision_score(minority_true, minority_proba)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "minority_precision": float(
            precision_score(minority_true, minority_pred, zero_division=0)
        ),
        "minority_recall": float(
            recall_score(minority_true, minority_pred, zero_division=0)
        ),
        "minority_f1": float(
            f1_score(minority_true, minority_pred, zero_division=0)
        ),
    }


def bootstrap_metric_differences(
    y_true: Any,
    pred_a: Any,
    proba_a: Any,
    pred_b: Any,
    proba_b: Any,
    *,
    minority_class: Any,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Paired, class-stratified bootstrap of the difference A - B for every metric.

    Both methods are evaluated on the same resampled test indices. Resampling is
    done within each class, so every replicate keeps the original number of
    positives and negatives.

    Returns, for each metric:
        difference  observed A - B on the original test set (the reported Δ)
        ci_lower    2.5th percentile of the bootstrap differences
        ci_upper    97.5th percentile of the bootstrap differences
    """

    y_true = np.asarray(y_true)
    pred_a, proba_a = np.asarray(pred_a), np.asarray(proba_a)
    pred_b, proba_b = np.asarray(pred_b), np.asarray(proba_b)

    if not (len(y_true) == len(pred_a) == len(proba_a) == len(pred_b) == len(proba_b)):
        raise ValueError("All inputs must have the same length.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    # Point estimate: the difference on the real test set, not the bootstrap mean.
    observed_a = evaluate_classification(y_true, pred_a, proba_a, minority_class=minority_class)
    observed_b = evaluate_classification(y_true, pred_b, proba_b, minority_class=minority_class)
    metric_names = list(observed_a)

    if len(np.unique(y_true)) != 2:
        raise ValueError("The bootstrap expects a binary target.")

    positive_true = (y_true == minority_class).astype(int)
    positive_pred_a = (pred_a == minority_class).astype(int)
    positive_pred_b = (pred_b == minority_class).astype(int)
    proba_a = proba_a.astype(float)
    proba_b = proba_b.astype(float)

    indices_by_class = [
        np.flatnonzero(y_true == label)
        for label in np.unique(y_true)
    ]

    rng = np.random.default_rng(random_seed)
    differences = {name: np.empty(n_bootstrap) for name in metric_names}

    for b in range(n_bootstrap):
        indices = np.concatenate([
            rng.choice(class_indices, size=len(class_indices), replace=True)
            for class_indices in indices_by_class
        ])

        # One evaluation per method yields the difference for every metric.
        metrics_a = _binary_metrics_fast(
            positive_true[indices], positive_pred_a[indices], proba_a[indices]
        )
        metrics_b = _binary_metrics_fast(
            positive_true[indices], positive_pred_b[indices], proba_b[indices]
        )

        for name in metric_names:
            differences[name][b] = metrics_a[name] - metrics_b[name]

    return {
        name: {
            "difference": observed_a[name] - observed_b[name],
            "ci_lower": float(np.percentile(differences[name], 2.5)),
            "ci_upper": float(np.percentile(differences[name], 97.5)),
        }
        for name in metric_names
    }


def _binary_metrics_fast(
    positive_true: np.ndarray,
    positive_pred: np.ndarray,
    positive_proba: np.ndarray,
) -> dict[str, float]:
    """
    NumPy version of evaluate_classification for binary targets.

    Gives the same values as the scikit-learn implementation, without its
    per-call overhead, which dominates thousands of bootstrap replicates.
    """

    tp = int(np.sum((positive_true == 1) & (positive_pred == 1)))
    fp = int(np.sum((positive_true == 0) & (positive_pred == 1)))
    fn = int(np.sum((positive_true == 1) & (positive_pred == 0)))
    tn = int(np.sum((positive_true == 0) & (positive_pred == 0)))

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1_pos = ratio(2 * tp, 2 * tp + fp + fn)
    f1_neg = ratio(2 * tn, 2 * tn + fp + fn)
    specificity = ratio(tn, tn + fp)

    return {
        "macro_f1": (f1_pos + f1_neg) / 2,
        "auprc": _average_precision(positive_true, positive_proba),
        "balanced_accuracy": (recall + specificity) / 2,
        "minority_precision": precision,
        "minority_recall": recall,
        "minority_f1": f1_pos,
    }


def _average_precision(
    positive_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Average precision with the same tie handling as scikit-learn."""

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_true = positive_true[order]

    # Last position of each distinct score: one threshold per distinct value.
    distinct = np.flatnonzero(np.diff(sorted_scores))
    threshold_idx = np.r_[distinct, len(sorted_true) - 1]

    tps = np.cumsum(sorted_true)[threshold_idx]
    fps = threshold_idx + 1 - tps

    if tps[-1] == 0:
        return 0.0

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))