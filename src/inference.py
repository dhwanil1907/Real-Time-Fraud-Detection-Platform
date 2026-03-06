"""Real-time inference: load artifacts, predict single row, simulate streaming."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.autoencoder import FraudAutoencoder, compute_anomaly_scores, load_autoencoder
from src.classifier import load_model
from src.fusion import build_meta_features, load_meta_model

# Default autoencoder hidden dims (must match training)
_DEFAULT_AE_HIDDEN_DIMS = (128, 64, 32)

logger = logging.getLogger(__name__)

# Risk buckets
RISK_LOW = "low"
RISK_MED = "medium"
RISK_HIGH = "high"


def _risk_bucket(prob: float) -> str:
    if prob < 0.3:
        return RISK_LOW
    if prob <= 0.7:
        return RISK_MED
    return RISK_HIGH


def load_artifacts(config: Any) -> dict[str, Any]:
    """
    Load scaler, freq_map, feature_columns, autoencoder, logreg, xgb, meta_model, ae_bounds.
    Returns dict with all artifacts needed for predict_single.
    """
    models_dir = Path(config.models_dir)
    artifacts: dict[str, Any] = {}
    # Scaler and encoder (saved by preprocessing/training)
    scaler_path = models_dir / "scaler.pkl"
    if scaler_path.exists():
        artifacts["scaler"] = joblib.load(scaler_path)
    else:
        artifacts["scaler"] = None
    freq_map_path = models_dir / "freq_map.pkl"
    if freq_map_path.exists():
        artifacts["freq_map"] = joblib.load(freq_map_path)
    else:
        artifacts["freq_map"] = {}
    feature_columns_path = models_dir / "feature_columns.pkl"
    if feature_columns_path.exists():
        artifacts["feature_columns"] = joblib.load(feature_columns_path)
    else:
        artifacts["feature_columns"] = None
    ae_bounds_path = models_dir / "ae_bounds.pkl"
    if ae_bounds_path.exists():
        artifacts["ae_bounds"] = joblib.load(ae_bounds_path)
    else:
        artifacts["ae_bounds"] = {"min": 0.0, "max": 1.0}
    # Models
    ae_path = models_dir / "autoencoder.pt"
    if ae_path.exists():
        ae_state = torch.load(ae_path, map_location="cpu")
        input_dim = ae_state["encoder.0.weight"].shape[1]
        artifacts["autoencoder"] = load_autoencoder(ae_path, input_dim=input_dim, hidden_dims=_DEFAULT_AE_HIDDEN_DIMS)
    else:
        artifacts["autoencoder"] = None
    artifacts["logreg"] = load_model(models_dir / "logreg.pkl") if (models_dir / "logreg.pkl").exists() else None
    artifacts["xgb"] = load_model(models_dir / "xgb.pkl") if (models_dir / "xgb.pkl").exists() else None
    artifacts["meta_model"] = load_meta_model(models_dir / "meta_model.pkl") if (models_dir / "meta_model.pkl").exists() else None
    return artifacts


def _row_to_features(row: pd.Series | dict, feature_columns: list[str], scaler: Any, freq_map: dict) -> np.ndarray:
    """
    Convert a single row (series or dict) to feature vector aligned to feature_columns.
    Fill missing columns with 0; apply freq_map for known categoricals; scale numeric.
    """
    if isinstance(row, dict):
        row = pd.Series(row)
    X = pd.DataFrame([row])
    # Encode categoricals if freq_map has entries
    for col, mapping in freq_map.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(0.0)
    # Align to feature_columns
    vec = np.zeros(len(feature_columns))
    for i, c in enumerate(feature_columns):
        if c in X.columns:
            try:
                vec[i] = float(X[c].iloc[0])
            except (TypeError, ValueError):
                vec[i] = 0.0
    if scaler is not None:
        vec = scaler.transform(vec.reshape(1, -1))[0]
    return vec.astype(np.float32)


def predict_single(
    row: pd.Series | dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply preprocessing/feature alignment and run models. Return fraud probability and risk bucket.
    row: single transaction as Series or dict (must contain columns used in training).
    For single-row inference, rolling and historical-fraud features are taken from the provided
    row; if keys are missing they are filled with 0.
    """
    feature_columns = artifacts.get("feature_columns")
    scaler = artifacts.get("scaler")
    freq_map = artifacts.get("freq_map", {})
    if feature_columns is None:
        raise ValueError("feature_columns not in artifacts; run training first")
    row_keys = set(row.keys()) if isinstance(row, dict) else set(row.index)
    overlap = row_keys & set(feature_columns)
    if len(overlap) == 0:
        logger.warning("Row has no keys overlapping feature_columns; prediction uses all defaults.")
    X = _row_to_features(row, feature_columns, scaler, freq_map).reshape(1, -1)
    proba = 0.0
    # Prefer meta model; else fallback to XGB
    if artifacts.get("meta_model") is not None and artifacts.get("autoencoder") is not None:
        ae = artifacts["autoencoder"]
        ae_scores = compute_anomaly_scores(ae, X)
        ae_min = artifacts["ae_bounds"].get("min", 0.0)
        ae_max = artifacts["ae_bounds"].get("max", 1.0)
        ae_norm = (ae_scores[0] - ae_min) / (ae_max - ae_min + 1e-9)
        xgb_p = artifacts["xgb"].predict_proba(X)[:, 1][0]
        logreg_p = artifacts["logreg"].predict_proba(X)[:, 1][0]
        meta_X = np.array([[xgb_p, logreg_p, ae_norm]], dtype=np.float32)
        proba = artifacts["meta_model"].predict_proba(meta_X)[:, 1][0]
    elif artifacts.get("xgb") is not None:
        proba = artifacts["xgb"].predict_proba(X)[:, 1][0]
    else:
        raise ValueError("No model in artifacts")
    return {
        "fraud_probability": float(proba),
        "risk_bucket": _risk_bucket(proba),
    }


def simulate_stream(
    test_df: pd.DataFrame,
    artifacts: dict[str, Any],
    n: int = 100,
) -> None:
    """
    Simulate streaming by iterating over the last N rows of test_df.
    For each row, call predict_single and print result with latency.
    """
    subset = test_df.tail(n)
    logger.info("Simulating stream over last %d rows", len(subset))
    for i, (idx, row) in enumerate(subset.iterrows()):
        t0 = time.perf_counter()
        out = predict_single(row.to_dict(), artifacts)
        elapsed = time.perf_counter() - t0
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("Row %d: prob=%.4f bucket=%s latency_ms=%.2f", i + 1, out["fraud_probability"], out["risk_bucket"], elapsed * 1000)
    logger.info("Stream simulation done (%d rows)", len(subset))
