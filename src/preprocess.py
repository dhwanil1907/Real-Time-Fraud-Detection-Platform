"""Preprocessing: merge transaction/identity, handle missing, encode categoricals, scale, split."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Column name for target
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"


def merge_tables(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left join identity onto transaction on TransactionID."""
    if identity_df is None:
        return transaction_df.copy()
    out = transaction_df.merge(
        identity_df,
        on=ID_COL,
        how="left",
        suffixes=("", "_id"),
    )
    logger.info("Merged shape: %s (transaction %s + identity)", out.shape, transaction_df.shape)
    return out


def separate_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Remove isFraud from dataframe; return (X, y). y is None if no target column."""
    if TARGET_COL not in df.columns:
        return df.copy(), None
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])
    return X, y


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def handle_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric: fill with median. Categorical: fill with 'missing'."""
    X = X.copy()
    num_cols = _get_numeric_columns(X)
    cat_cols = _get_categorical_columns(X)
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
    return X


def encode_categoricals(
    X: pd.DataFrame,
    freq_map: dict[str, dict[Any, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, float]]]:
    """
    Frequency-encode categorical/object columns. If freq_map provided, use it (transform);
    otherwise fit on X and return the map for later use.
    """
    X = X.copy()
    cat_cols = _get_categorical_columns(X)
    if freq_map is None:
        freq_map = {}

    for c in cat_cols:
        if c not in X.columns:
            continue
        if freq_map and c in freq_map:
            X[c] = X[c].map(freq_map[c]).fillna(0.0)
        else:
            counts = X[c].value_counts()
            total = len(X)
            freq = (counts / total).to_dict()
            freq_map[c] = freq
            X[c] = X[c].map(freq).fillna(0.0)

    # Ensure numeric output for encoded cols
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, freq_map


def scale_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    numeric_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler, list[str]]:
    """
    Fit StandardScaler on X_train (numeric_cols only), transform train and test.
    Returns (X_train_scaled, X_test_scaled, scaler, numeric_cols).
    If numeric_cols is None, uses all numeric columns in X_train.
    """
    if numeric_cols is None:
        numeric_cols = _get_numeric_columns(X_train)
    # Restrict to cols present in both
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        present = [c for c in numeric_cols if c in X_test_scaled.columns]
        if present:
            X_test_scaled[present] = scaler.transform(X_test[present])

    return X_train_scaled, X_test_scaled, scaler, numeric_cols


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series | None,
    time_col: str = TIME_COL,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    """
    Sort by time_col and split at train_frac quantile.
    Returns (X_train, X_test, y_train, y_test). y can be None.
    """
    if time_col not in X.columns:
        logger.warning("Column %s not found; using index order for split", time_col)
        idx = np.arange(len(X))
        np.random.RandomState(42).shuffle(idx)
        split_idx = int(len(idx) * train_frac)
        train_idx, test_idx = idx[:split_idx], idx[split_idx:]
    else:
        X_sorted = X.sort_values(time_col)
        n = len(X_sorted)
        split_idx = int(n * train_frac)
        train_idx = X_sorted.index[:split_idx].tolist()
        test_idx = X_sorted.index[split_idx:].tolist()

    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy() if y is not None else None
    y_test = y.loc[test_idx].copy() if y is not None else None
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Time-based split produced empty train or test set; check data and train_frac."
        )
    logger.info("Time-based split: train %d, test %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to path as parquet (or csv if parquet fails)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        logger.info("Saved processed data to %s", path)
    except Exception as e:
        logger.warning("Parquet save failed (%s), using CSV", e)
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s", path)


def load_processed(path: Path) -> pd.DataFrame:
    """Load processed data from parquet or csv."""
    path = Path(path)
    parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No processed file found at {parquet_path} or {csv_path}")
