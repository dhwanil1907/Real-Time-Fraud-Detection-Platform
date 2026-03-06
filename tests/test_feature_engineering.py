"""Tests for feature_engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import add_log_amount, add_time_features, build_features


def test_add_time_features() -> None:
    df = pd.DataFrame({"TransactionDT": [0, 3600, 86400 * 2 + 7200]})
    out = add_time_features(df)
    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert out["hour"].iloc[0] == 0
    assert out["hour"].iloc[1] == 1
    assert out["hour"].iloc[2] == 2
    assert out["day_of_week"].iloc[2] == 2


def test_add_time_features_missing_col() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    out = add_time_features(df)
    assert "hour" not in out.columns
    assert out.equals(df) or len(out.columns) == 1


def test_add_log_amount() -> None:
    df = pd.DataFrame({"TransactionAmt": [1.0, 10.0, 100.0]})
    out = add_log_amount(df)
    assert "log_amt" in out.columns
    assert np.allclose(out["log_amt"], np.log1p([1, 10, 100]))


def test_build_features_minimal() -> None:
    df = pd.DataFrame({
        "TransactionDT": [1000, 2000],
        "TransactionAmt": [50.0, 75.0],
        "card1": [100, 100],
    })
    out = build_features(df, include_fraud_rate=False)
    assert "hour" in out.columns
    assert "log_amt" in out.columns
