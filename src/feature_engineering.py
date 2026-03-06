"""Feature engineering with no leakage: time features, log amt, rolling stats, historical fraud rate."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIME_COL = "TransactionDT"
AMT_COL = "TransactionAmt"
TARGET_COL = "isFraud"
# Card/group columns for rolling features (IEEE-CIS has card1, card2, etc.)
CARD_GROUP_COLS = ["card1", "card2", "card3", "card4", "card5"]
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def add_time_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Add hour and day_of_week from TransactionDT (seconds since reference)."""
    df = df.copy()
    if time_col not in df.columns:
        logger.warning("Time column %s not found; skipping time features", time_col)
        return df
    s = df[time_col]
    df["hour"] = (s % SECONDS_PER_DAY) // SECONDS_PER_HOUR
    df["day_of_week"] = (s // SECONDS_PER_DAY) % 7
    return df


def add_log_amount(df: pd.DataFrame, amt_col: str = AMT_COL) -> pd.DataFrame:
    """Add log1p(TransactionAmt)."""
    df = df.copy()
    if amt_col not in df.columns:
        logger.warning("Amount column %s not found; skipping log_amt", amt_col)
        return df
    df["log_amt"] = np.log1p(df[amt_col].astype(float))
    return df


def _rolling_count_per_group(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    window_seconds: int = SECONDS_PER_HOUR,
) -> pd.Series:
    """
    For each row, count of rows in same group with TransactionDT in (row_time - window, row_time).
    Implemented via expanding window and shift to avoid leakage: we use past only.
    Approximation: sort by time, then per group use expanding count of rows in last window_seconds.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    out = np.zeros(len(df))
    for _, grp in df.groupby(group_col):
        idx = grp.index.tolist()
        times = grp[time_col].values
        for i, (ii, t) in enumerate(zip(idx, times)):
            # Past only: times < t and >= t - window
            mask = (times < t) & (times >= t - window_seconds)
            out[ii] = mask.sum()
    return pd.Series(out, index=df.index)


def _expanding_mean_per_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    time_col: str,
    shift: bool = True,
) -> pd.Series:
    """Expanding mean of value_col per group_col (sorted by time). Shifted by 1 if shift=True."""
    df = df.sort_values(time_col).reset_index(drop=True)
    if shift:
        # Expanding mean of previous rows only
        agg = df.groupby(group_col)[value_col].apply(
            lambda x: x.shift(1).expanding().mean(),
            include_groups=False,
        )
    else:
        agg = df.groupby(group_col)[value_col].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        return agg
    return agg.reindex(df.index).reset_index(level=0, drop=True)


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "card1",
    time_col: str = TIME_COL,
    amt_col: str = AMT_COL,
    window_seconds: int = SECONDS_PER_HOUR,
) -> pd.DataFrame:
    """
    Add card_txn_count_1h and card_mean_amt (expanding mean of amount per group, shifted).
    Skips if group_col or required cols missing.
    """
    df = df.copy()
    if group_col not in df.columns:
        logger.warning("Group column %s not found; skipping rolling features", group_col)
        return df
    if time_col not in df.columns or amt_col not in df.columns:
        logger.warning("Time or amount column missing; skipping rolling features")
        return df

    df = df.sort_values(time_col).reset_index(drop=True)
    # Rolling count in last hour (approximation: past rows within window)
    count_series = _rolling_count_per_group(df, group_col, time_col, window_seconds)
    df["card_txn_count_1h"] = count_series.values
    df["card_txn_count_1h"] = df["card_txn_count_1h"].fillna(0)

    # Expanding mean amount per group, shifted
    mean_amt = (
        df.groupby(group_col)[amt_col]
        .transform(lambda x: x.shift(1).expanding().mean())
        .values
    )
    df["card_mean_amt"] = mean_amt
    df["card_mean_amt"] = df["card_mean_amt"].fillna(0.0)
    return df


def add_historical_fraud_rate_simple(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """
    Expanding mean of target per group sorted by time, shifted by 1 (past only, no leakage).
    Use only on training data; for test use apply_train_fraud_rates.
    """
    df = df.copy()
    if group_col not in df.columns or target_col not in df.columns or time_col not in df.columns:
        return df
    df = df.sort_values(time_col).reset_index(drop=True)
    df["card_fraud_rate"] = (
        df.groupby(group_col)[target_col]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )
    return df


def get_train_fraud_rates_per_group(
    train_df: pd.DataFrame,
    group_col: str,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
) -> dict[Any, float]:
    """
    Compute last (most recent) expanding mean fraud rate per group on train.
    Use to fill test card_fraud_rate via apply_train_fraud_rates (no leakage).
    """
    if group_col not in train_df.columns or target_col not in train_df.columns:
        return {}
    train_sorted = train_df.sort_values(time_col)
    last_rate = (
        train_sorted.groupby(group_col)[target_col]
        .apply(lambda x: x.shift(1).expanding().mean().iloc[-1] if len(x) > 0 else 0.0, include_groups=False)
    )
    return last_rate.to_dict()


def apply_train_fraud_rates(
    df: pd.DataFrame,
    group_col: str,
    train_group_rates: dict[Any, float],
) -> pd.DataFrame:
    """Set card_fraud_rate on df (e.g. test) from train-derived group rates. Missing group -> 0."""
    df = df.copy()
    if group_col not in df.columns:
        return df
    df["card_fraud_rate"] = df[group_col].map(train_group_rates).fillna(0.0)
    return df


def build_features(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    amt_col: str = AMT_COL,
    target_col: str = TARGET_COL,
    group_col: str = "card1",
    include_fraud_rate: bool = True,
    train_fraud_rates: dict[Any, float] | None = None,
) -> pd.DataFrame:
    """
    Add all leak-free features. If include_fraud_rate and target_col in df, add expanding
    shifted fraud rate on train. If train_fraud_rates is provided (e.g. for test set),
    use that to set card_fraud_rate instead.
    """
    df = add_time_features(df, time_col)
    df = add_log_amount(df, amt_col)
    df = add_rolling_features(df, group_col=group_col, time_col=time_col, amt_col=amt_col)
    if include_fraud_rate:
        if train_fraud_rates is not None:
            df = apply_train_fraud_rates(df, group_col=group_col, train_group_rates=train_fraud_rates)
        elif target_col in df.columns:
            df = add_historical_fraud_rate_simple(
                df, group_col=group_col, target_col=target_col, time_col=time_col
            )
        else:
            df["card_fraud_rate"] = 0.0
    return df
