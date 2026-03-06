"""Meta-model fusion: combine XGB proba, LogReg proba, and autoencoder anomaly score."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def build_meta_features(
    xgb_proba: np.ndarray,
    logreg_proba: np.ndarray,
    ae_score: np.ndarray,
) -> np.ndarray:
    """
    Stack [xgb_proba, logreg_proba, ae_score] into (N, 3).
    Normalize ae_score to [0,1] by min-max so it behaves like a probability.
    """
    ae_min, ae_max = ae_score.min(), ae_score.max()
    if ae_max > ae_min:
        ae_norm = (ae_score - ae_min) / (ae_max - ae_min)
    else:
        ae_norm = np.zeros_like(ae_score)
    return np.column_stack([xgb_proba, logreg_proba, ae_norm])


def train_meta_model(
    meta_X: np.ndarray,
    y: np.ndarray,
    config: Any,
) -> tuple[LogisticRegression, np.ndarray]:
    """Train Logistic Regression on the 3 stacked scores. Returns (model, train proba)."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(meta_X, y)
    proba = model.predict_proba(meta_X)[:, 1]
    return model, proba


def evaluate_fusion(
    meta_model: LogisticRegression,
    meta_X_test: np.ndarray,
    y_test: np.ndarray,
    xgb_proba_test: np.ndarray,
) -> dict[str, float]:
    """
    Compute PR-AUC for fusion and XGBoost alone; log comparison.
    Returns dict with fusion_pr_auc, xgb_pr_auc.
    """
    fusion_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    fusion_pr_auc = average_precision_score(y_test, fusion_proba)
    xgb_pr_auc = average_precision_score(y_test, xgb_proba_test)
    logger.info("Fusion PR-AUC: %.4f | XGBoost-only PR-AUC: %.4f", fusion_pr_auc, xgb_pr_auc)
    if fusion_pr_auc > xgb_pr_auc:
        logger.info("Fusion improves PR-AUC by %.4f", fusion_pr_auc - xgb_pr_auc)
    return {"fusion_pr_auc": fusion_pr_auc, "xgb_pr_auc": xgb_pr_auc}


def save_meta_model(model: LogisticRegression, path: Path) -> None:
    """Save meta model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved meta model to %s", path)


def load_meta_model(path: Path) -> LogisticRegression:
    """Load meta model from joblib."""
    return joblib.load(path)
