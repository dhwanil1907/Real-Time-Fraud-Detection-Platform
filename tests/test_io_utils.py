"""Tests for io_utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.io_utils import discover_files, load_csv, load_dataset


def test_discover_files_empty_dir(tmp_path: Path) -> None:
    result = discover_files(tmp_path)
    assert result["train_transaction"] is None
    assert result["train_identity"] is None
    assert result["test_transaction"] is None
    assert result["test_identity"] is None


def test_discover_files_finds_train_transaction(temp_data_dir: Path, sample_train_transaction_csv: Path) -> None:
    result = discover_files(temp_data_dir)
    assert result["train_transaction"] is not None
    assert "train_transaction" in str(result["train_transaction"]).lower()


def test_discover_files_finds_train_identity(temp_data_dir: Path, sample_train_transaction_csv: Path, sample_identity_csv: Path) -> None:
    result = discover_files(temp_data_dir)
    assert result["train_identity"] is not None


def test_load_dataset_raises_on_empty_csv(tmp_path: Path) -> None:
    empty = tmp_path / "train_transaction.csv"
    empty.write_text("TransactionID,TransactionDT\n")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty, None)


def test_load_dataset_raises_on_missing_transaction_id(tmp_path: Path) -> None:
    bad = tmp_path / "train_transaction.csv"
    bad.write_text("TransactionDT,TransactionAmt\n1,50.0\n")
    with pytest.raises(ValueError, match="TransactionID"):
        load_dataset(bad, None)


def test_load_dataset_returns_transaction_and_optional_identity(
    sample_train_transaction_csv: Path, sample_identity_csv: Path
) -> None:
    trans, id_df = load_dataset(sample_train_transaction_csv, sample_identity_csv)
    assert len(trans) == 5
    assert "TransactionID" in trans.columns
    assert "isFraud" in trans.columns
    assert id_df is not None
    assert len(id_df) == 5
