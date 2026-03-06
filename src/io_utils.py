"""Robust data loading: discover CSVs in data dir and load with efficient dtypes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Expected substring patterns (lowercase) for matching filenames
TRAIN_TRANSACTION_PATTERN = "train_transaction"
TRAIN_IDENTITY_PATTERN = "train_identity"
TEST_TRANSACTION_PATTERN = "test_transaction"
TEST_IDENTITY_PATTERN = "test_identity"


def discover_files(data_dir: Path) -> dict[str, Path | None]:
    """
    List CSV files in data_dir and match by substring patterns.
    Returns dict with keys: train_transaction, train_identity, test_transaction, test_identity.
    Values are Path or None if not found.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return {
            "train_transaction": None,
            "train_identity": None,
            "test_transaction": None,
            "test_identity": None,
        }

    csv_files = list(data_dir.glob("*.csv"))
    result: dict[str, Path | None] = {
        "train_transaction": None,
        "train_identity": None,
        "test_transaction": None,
        "test_identity": None,
    }
    patterns = [
        ("train_transaction", TRAIN_TRANSACTION_PATTERN),
        ("train_identity", TRAIN_IDENTITY_PATTERN),
        ("test_transaction", TEST_TRANSACTION_PATTERN),
        ("test_identity", TEST_IDENTITY_PATTERN),
    ]
    for key, pattern in patterns:
        for p in csv_files:
            if pattern in p.name.lower():
                result[key] = p
                break
    return result


def _infer_dtypes(df: pd.DataFrame) -> dict[str, Any]:
    """Infer efficient dtypes: float32 for float cols, keep int/category where sensible."""
    dtypes: dict[str, Any] = {}
    for col in df.columns:
        if df[col].dtype == "float64":
            dtypes[col] = "float32"
    return dtypes


def _print_summary(name: str, df: pd.DataFrame, has_target: bool) -> None:
    """Print shape, columns sample, target presence, and missingness summary."""
    logger.info("[%s] shape: %s", name, df.shape)
    logger.info("[%s] columns (%d): %s", name, len(df.columns), list(df.columns[:20]))
    if len(df.columns) > 20:
        logger.info("  ... and %d more", len(df.columns) - 20)
    if has_target and "isFraud" in df.columns:
        logger.info("[%s] target 'isFraud' present: mean=%.4f", name, df["isFraud"].mean())
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        logger.info("[%s] columns with missing (top 10): %s", name, missing.head(10).to_dict())
    else:
        logger.info("[%s] no missing values", name)


def load_csv(
    path: Path,
    use_efficient_dtypes: bool = True,
) -> pd.DataFrame:
    """Load a single CSV with optional dtype optimization. Prints no summary."""
    kwargs: dict[str, Any] = {}
    if use_efficient_dtypes:
        sample = pd.read_csv(path, nrows=1000)
        kwargs["dtype"] = _infer_dtypes(sample)
    df = pd.read_csv(path, **kwargs)
    if len(df) == 0:
        raise ValueError(f"CSV is empty: {path}")
    return df


def load_dataset(
    transaction_path: Path | None,
    identity_path: Path | None,
    use_efficient_dtypes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load transaction and (optional) identity CSVs. Does not merge.
    Returns (transaction_df, identity_df). identity_df is None if path missing.
    Prints summary: shape, columns, target presence, missingness.
    """
    if transaction_path is None or not Path(transaction_path).exists():
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")

    trans_df = load_csv(transaction_path, use_efficient_dtypes)
    if "TransactionID" not in trans_df.columns:
        raise ValueError("Transaction file missing required column: TransactionID")
    has_target = "isFraud" in trans_df.columns
    _print_summary("transaction", trans_df, has_target)

    id_df: pd.DataFrame | None = None
    if identity_path is not None and Path(identity_path).exists():
        id_df = load_csv(identity_path, use_efficient_dtypes)
        _print_summary("identity", id_df, False)
    elif identity_path is not None:
        logger.warning("Identity file not found or None: %s", identity_path)

    return trans_df, id_df


def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame | None]:
    """
    Discover and load all available train/test transaction and identity files.
    Returns dict with keys: train_transaction, train_identity, test_transaction, test_identity.
    Values are DataFrame or None if file not found. Preprocess will merge transaction + identity.
    """
    files = discover_files(data_dir)
    result: dict[str, pd.DataFrame | None] = {}

    train_trans_path = files["train_transaction"]
    train_id_path = files["train_identity"]
    if train_trans_path is None:
        raise FileNotFoundError(f"No train transaction CSV found in {data_dir}")
    trans_df, id_df = load_dataset(train_trans_path, train_id_path)
    result["train_transaction"] = trans_df
    result["train_identity"] = id_df

    test_trans_path = files["test_transaction"]
    test_id_path = files["test_identity"]
    if test_trans_path is not None:
        test_trans_df, test_id_df = load_dataset(test_trans_path, test_id_path)
        result["test_transaction"] = test_trans_df
        result["test_identity"] = test_id_df
    else:
        result["test_transaction"] = None
        result["test_identity"] = None
        logger.info("No test transaction CSV found; only train will be used.")

    return result
