"""Tests for preprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    encode_categoricals,
    merge_tables,
    separate_target,
    time_based_split,
)


def test_merge_tables() -> None:
    trans = pd.DataFrame({"TransactionID": [1, 2], "Amt": [10, 20]})
    identity = pd.DataFrame({"TransactionID": [1, 2], "id_01": [1.0, 2.0]})
    out = merge_tables(trans, identity)
    assert len(out) == 2
    assert "id_01" in out.columns
    assert out["Amt"].tolist() == [10, 20]


def test_merge_tables_identity_none() -> None:
    trans = pd.DataFrame({"TransactionID": [1], "Amt": [10]})
    out = merge_tables(trans, None)
    assert len(out) == 1
    assert out["Amt"].iloc[0] == 10


def test_separate_target() -> None:
    df = pd.DataFrame({"A": [1, 2], "isFraud": [0, 1]})
    X, y = separate_target(df)
    assert "isFraud" not in X.columns
    assert y is not None
    assert list(y) == [0, 1]


def test_separate_target_no_target() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    X, y = separate_target(df)
    assert "B" in X.columns
    assert y is None


def test_time_based_split() -> None:
    X = pd.DataFrame({"TransactionDT": [100, 200, 300, 400, 500], "x": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 1, 0, 1, 0])
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_frac=0.8)
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert y_train is not None and len(y_train) == 4
    assert y_test is not None and len(y_test) == 1


def test_time_based_split_empty_raises() -> None:
    X = pd.DataFrame({"TransactionDT": [100, 200], "x": [1, 2]})
    y = pd.Series([0, 1])
    # train_frac=0.5 with 2 rows -> 1 train, 1 test. Use 0.01 so train is empty.
    with pytest.raises(ValueError, match="empty"):
        time_based_split(X, y, train_frac=0.01)


def test_encode_categoricals_fit_transform() -> None:
    X = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1, 2, 3, 4]})
    X_enc, freq_map = encode_categoricals(X)
    assert "cat" in freq_map
    assert X_enc["cat"].dtype in (np.floating, float)
    X2 = pd.DataFrame({"cat": ["a", "c"], "num": [1, 2]})
    X2_enc, _ = encode_categoricals(X2, freq_map=freq_map)
    assert list(X2_enc["cat"]) == [freq_map["cat"]["a"], 0.0]
