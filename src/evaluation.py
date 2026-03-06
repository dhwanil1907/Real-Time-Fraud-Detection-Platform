"""Evaluation: PR-AUC, ROC-AUC, precision/recall/F1, confusion matrix, threshold tuning, PR curve plot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    Compute precision, recall, F1, PR-AUC, ROC-AUC at given threshold.
    y_true: binary (0/1), y_prob: probability of positive class.
    """
    y_pred = (y_prob >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.5,
) -> tuple[float, float]:
    """
    Sweep thresholds to maximize F1 subject to precision >= min_precision.
    Returns (best_threshold, best_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_f1 = 0.0
    best_t = 0.5
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= min_precision:
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
    return best_t, best_f1


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix (TN, FP, FN, TP)."""
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix:\n%s", cm)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        logger.info("TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> None:
    """Plot precision-recall curve and save to save_path."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved PR curve to %s", save_path)


def run_full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    config: Any,
    min_precision: float = 0.5,
) -> dict[str, Any]:
    """
    Run full evaluation: metrics, optimal threshold, confusion matrix, PR curve plot.
    Returns dict with metrics, best_threshold, best_f1.
    """
    metrics = compute_metrics(y_true, y_prob)
    best_threshold, best_f1 = find_optimal_threshold(y_true, y_prob, min_precision=min_precision)
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    logger.info("[%s] PR-AUC=%.4f ROC-AUC=%.4f F1@0.5=%.4f", label, metrics["pr_auc"], metrics["roc_auc"], metrics["f1"])
    logger.info("[%s] Optimal threshold=%.4f (F1=%.4f at min_precision=%.2f)", label, best_threshold, best_f1, min_precision)
    print_confusion_matrix(y_true, y_pred_opt)
    plots_dir = getattr(config, "plots_dir", Path("models/plots"))
    plot_path = Path(plots_dir) / f"pr_curve_{label.replace(' ', '_')}.png"
    plot_pr_curve(y_true, y_prob, plot_path)
    return {
        "metrics": metrics,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
    }
