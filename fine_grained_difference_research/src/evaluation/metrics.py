"""Small, dependency-light metrics with explicit empty-mask conventions."""

from __future__ import annotations

import numpy as np


def binary_pair_metrics(truth: list[bool], prediction: list[bool]) -> dict[str, float | int]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and equally sized")
    tp = sum(t and p for t, p in zip(truth, prediction))
    tn = sum((not t) and (not p) for t, p in zip(truth, prediction))
    fp = sum((not t) and p for t, p in zip(truth, prediction))
    fn = sum(t and (not p) for t, p in zip(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(truth), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def mask_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("truth and prediction masks must be same-size 2-D arrays")
    t, p = truth.astype(bool), prediction.astype(bool)
    tp = int(np.count_nonzero(t & p))
    fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p))
    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    return {"pixels": int(t.size), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0),
            "recall": tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0),
            "iou": tp / union if union else 1.0,
            "dice": 2 * tp / denom if denom else 1.0}


def selective_pair_metrics(truth: list[bool], prediction: list[bool | None]) -> dict:
    """Evaluate pair recognition while preserving abstentions as uncovered cases."""
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be non-empty and equally sized")
    covered = [(t, p) for t, p in zip(truth, prediction) if p is not None]
    correct = sum(t == p for t, p in covered)
    metrics = binary_pair_metrics([t for t, _ in covered], [bool(p) for _, p in covered]) if covered else None
    return {
        "n": len(truth),
        "covered": len(covered),
        "abstained": len(truth) - len(covered),
        "coverage": len(covered) / len(truth),
        "selective_accuracy": correct / len(covered) if covered else None,
        "selective_error_rate": 1 - correct / len(covered) if covered else None,
        "covered_confusion_metrics": metrics,
    }


def expected_calibration_error(confidence: list[float], correctness: list[bool], bins: int = 10) -> float:
    """Equal-width ECE; report only for confidence values validated as probabilities."""
    if len(confidence) != len(correctness) or not confidence or bins < 1:
        raise ValueError("confidence and correctness must be non-empty, equal-length; bins >= 1")
    values = np.asarray(confidence, dtype=float)
    labels = np.asarray(correctness, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("confidence values must be finite and in [0,1]")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = ((values >= edges[index]) & (values <= edges[index + 1])
                    if index == bins - 1 else
                    (values >= edges[index]) & (values < edges[index + 1]))
        if selected.any():
            ece += float(selected.mean() * abs(values[selected].mean() - labels[selected].mean()))
    return ece
