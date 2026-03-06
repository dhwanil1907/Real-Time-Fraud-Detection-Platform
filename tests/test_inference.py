"""Tests for inference."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference import _risk_bucket, predict_single


def test_risk_bucket() -> None:
    assert _risk_bucket(0.0) == "low"
    assert _risk_bucket(0.29) == "low"
    assert _risk_bucket(0.3) == "medium"
    assert _risk_bucket(0.7) == "medium"
    assert _risk_bucket(0.71) == "high"
    assert _risk_bucket(1.0) == "high"


def test_predict_single_mock_artifacts() -> None:
    feature_columns = ["TransactionAmt", "log_amt", "hour"]
    row = {"TransactionAmt": 50.0, "log_amt": 3.93, "hour": 12}
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.2, 0.8]])
    artifacts = {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "freq_map": {},
        "xgb": mock_xgb,
        "meta_model": None,
        "autoencoder": None,
    }
    out = predict_single(row, artifacts)
    assert "fraud_probability" in out
    assert "risk_bucket" in out
    assert out["fraud_probability"] == 0.8
    assert out["risk_bucket"] in ("low", "medium", "high")


def test_predict_single_raises_when_no_feature_columns() -> None:
    artifacts = {"feature_columns": None, "scaler": None, "freq_map": {}}
    with pytest.raises(ValueError, match="feature_columns"):
        predict_single({"TransactionAmt": 50}, artifacts)
