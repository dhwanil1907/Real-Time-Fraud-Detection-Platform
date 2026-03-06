"""Pytest fixtures for fraud detection tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Ensure project root on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def sample_train_transaction_csv(temp_data_dir: Path) -> Path:
    path = temp_data_dir / "train_transaction.csv"
    path.write_text(
        "TransactionID,TransactionDT,TransactionAmt,isFraud,ProductCD,card1,card2\n"
        "1,1000,50.0,0,P,100,200\n"
        "2,2000,75.0,0,H,101,201\n"
        "3,3000,100.0,1,P,100,200\n"
        "4,4000,25.0,0,H,102,202\n"
        "5,5000,200.0,0,P,100,200\n"
    )
    return path


@pytest.fixture
def sample_identity_csv(temp_data_dir: Path) -> Path:
    path = temp_data_dir / "train_identity.csv"
    path.write_text(
        "TransactionID,id_01,id_02\n"
        "1,1.0,2.0\n"
        "2,1.0,3.0\n"
        "3,2.0,2.0\n"
        "4,1.0,2.0\n"
        "5,2.0,3.0\n"
    )
    return path
