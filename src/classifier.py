"""Supervised classifiers: Logistic Regression baseline and XGBoost with imbalance handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_logreg(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    config: Any,
) -> tuple[LogisticRegression, np.ndarray]:
    """
    Train Logistic Regression with class_weight='balanced'.
    Returns (model, predicted probabilities for positive class).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    max_iter = getattr(config, "logreg_max_iter", 1000)
    C = getattr(config, "logreg_C", 1.0)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        C=C,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    logger.info("LogReg trained; train pred shape %s", proba.shape)
    return model, proba


def train_xgboost(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    config: Any,
) -> tuple[XGBClassifier, np.ndarray]:
    """
    Train XGBoost with scale_pos_weight = n_neg / n_pos for imbalance.
    Returns (model, predicted probabilities for positive class).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = dict(config.xgb_params)
    params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    proba = model.predict_proba(X)[:, 1]
    logger.info("XGBoost trained; scale_pos_weight=%.2f, train pred shape %s", scale_pos_weight, proba.shape)
    return model, proba


def save_model(model: Any, path: Path) -> None:
    """Save sklearn/XGB model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: Path) -> Any:
    """Load model from joblib."""
    return joblib.load(path)
