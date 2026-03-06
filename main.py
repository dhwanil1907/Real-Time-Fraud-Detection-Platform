#!/usr/bin/env python3
"""CLI entry point for the Fraud Detection Platform."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.evaluation import run_full_evaluation
from src.feature_engineering import build_features, get_train_fraud_rates_per_group
from src.fusion import (
    build_meta_features,
    evaluate_fusion,
    save_meta_model,
    train_meta_model,
)
from src.inference import load_artifacts, simulate_stream
from src.io_utils import load_raw_data
from src.preprocess import (
    encode_categoricals,
    handle_missing,
    load_processed,
    merge_tables,
    save_processed,
    scale_numeric,
    separate_target,
    time_based_split,
)
from src.autoencoder import (
    compute_anomaly_scores,
    load_autoencoder,
    save_autoencoder,
    train_autoencoder,
)
from src.classifier import train_logreg, train_xgboost, save_model
import joblib
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set numpy and torch seeds for reproducibility. sklearn uses numpy RNG."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_preprocess(config: Config) -> None:
    """Load raw data, merge, clean, encode, time-based split, save processed data and freq_map."""
    set_global_seed(config.seed)
    config.ensure_dirs()
    raw = load_raw_data(config.raw_data_dir)
    train_trans = raw["train_transaction"]
    train_id = raw["train_identity"]
    if train_trans is None:
        raise FileNotFoundError("Train transaction data not found")
    merged = merge_tables(train_trans, train_id)
    X, y = separate_target(merged)
    X = handle_missing(X)
    # Split first so encoding is fit only on train (no test leakage)
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_frac=config.train_frac)
    X_train, freq_map = encode_categoricals(X_train)
    X_test, _ = encode_categoricals(X_test, freq_map=freq_map)
    # Save splits (with target if present)
    train_path = config.processed_data_dir / "train.parquet"
    test_path = config.processed_data_dir / "test.parquet"
    if y_train is not None:
        X_train = X_train.join(y_train)
    if y_test is not None:
        X_test = X_test.join(y_test)
    save_processed(X_train, train_path)
    save_processed(X_test, test_path)
    joblib.dump(freq_map, config.models_dir / "freq_map.pkl")
    logger.info("Preprocess done. Train %s, test %s", train_path, test_path)


def cmd_features(config: Config) -> None:
    """Load processed train/test, add features, save featured data."""
    config.ensure_dirs()
    train_df = load_processed(config.processed_data_dir / "train")
    test_df = load_processed(config.processed_data_dir / "test")
    if "isFraud" in train_df.columns:
        train_rates = get_train_fraud_rates_per_group(train_df, "card1", "isFraud", "TransactionDT")
        train_df = build_features(train_df, include_fraud_rate=True)
        test_df = build_features(test_df, include_fraud_rate=True, train_fraud_rates=train_rates)
    else:
        train_df = build_features(train_df, include_fraud_rate=False)
        test_df = build_features(test_df, include_fraud_rate=False)
    save_processed(train_df, config.processed_data_dir / "train_featured.parquet")
    save_processed(test_df, config.processed_data_dir / "test_featured.parquet")
    logger.info("Features saved.")


def cmd_train(config: Config) -> None:
    """Load featured data, scale, train AE + classifiers + fusion, save all artifacts."""
    set_global_seed(config.seed)
    config.ensure_dirs()
    train_df = load_processed(config.processed_data_dir / "train_featured")
    test_df = load_processed(config.processed_data_dir / "test_featured")
    y_train = train_df.pop("isFraud") if "isFraud" in train_df.columns else None
    y_test = test_df.pop("isFraud") if "isFraud" in test_df.columns else None
    # Drop non-feature columns for model matrix
    drop_cols = [c for c in train_df.columns if c in ("TransactionID",)]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")
    # Numeric only for scaling and models
    feature_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    train_df = train_df[feature_columns]
    test_df = test_df[[c for c in feature_columns if c in test_df.columns]]
    # Align test columns to train
    for c in feature_columns:
        if c not in test_df.columns:
            test_df[c] = 0.0
    test_df = test_df[feature_columns]
    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, config.models_dir / "scaler.pkl")
    joblib.dump(feature_columns, config.models_dir / "feature_columns.pkl")

    if y_train is None:
        raise ValueError("No target in train; cannot train classifiers")

    # Autoencoder on normal only
    X_train_normal = X_train[y_train.values == 0]
    if len(X_train_normal) == 0:
        raise ValueError("No normal (non-fraud) samples in train; cannot train autoencoder.")
    ae_model, _ = train_autoencoder(X_train_normal, config)
    save_autoencoder(ae_model, config.models_dir / "autoencoder.pt")
    ae_train = compute_anomaly_scores(ae_model, X_train)
    ae_test = compute_anomaly_scores(ae_model, X_test)
    ae_bounds = {"min": float(ae_train.min()), "max": float(ae_train.max())}
    joblib.dump(ae_bounds, config.models_dir / "ae_bounds.pkl")

    # Classifiers
    logreg, logreg_proba_train = train_logreg(X_train, y_train, config)
    save_model(logreg, config.models_dir / "logreg.pkl")
    xgb, xgb_proba_train = train_xgboost(X_train, y_train, config)
    save_model(xgb, config.models_dir / "xgb.pkl")
    xgb_proba_test = xgb.predict_proba(X_test)[:, 1]
    logreg_proba_test = logreg.predict_proba(X_test)[:, 1]

    # Fusion
    meta_X_train = build_meta_features(xgb_proba_train, logreg_proba_train, ae_train)
    meta_X_test = build_meta_features(xgb_proba_test, logreg_proba_test, ae_test)
    meta_model, _ = train_meta_model(meta_X_train, y_train.values, config)
    save_meta_model(meta_model, config.models_dir / "meta_model.pkl")
    fusion_proba_test = meta_model.predict_proba(meta_X_test)[:, 1]
    evaluate_fusion(meta_model, meta_X_test, y_test.values, xgb_proba_test)
    run_full_evaluation(y_test.values, fusion_proba_test, "fusion", config)
    logger.info("Training done.")


def cmd_evaluate(config: Config) -> None:
    """Load featured test and models, run full evaluation."""
    config.ensure_dirs()
    test_df = load_processed(config.processed_data_dir / "test_featured")
    if "isFraud" not in test_df.columns:
        logger.warning("No target in test; skipping evaluation")
        return
    y_test = test_df["isFraud"]
    feature_columns = joblib.load(config.models_dir / "feature_columns.pkl")
    X_test = test_df[feature_columns].values.astype(np.float32)
    scaler = joblib.load(config.models_dir / "scaler.pkl")
    X_test = scaler.transform(X_test)
    meta_model = joblib.load(config.models_dir / "meta_model.pkl")
    xgb = joblib.load(config.models_dir / "xgb.pkl")
    ae = load_autoencoder(config.models_dir / "autoencoder.pt", input_dim=X_test.shape[1])
    ae_test = compute_anomaly_scores(ae, X_test)
    ae_bounds = joblib.load(config.models_dir / "ae_bounds.pkl")
    ae_norm = (ae_test - ae_bounds["min"]) / (ae_bounds["max"] - ae_bounds["min"] + 1e-9)
    xgb_p = xgb.predict_proba(X_test)[:, 1]
    logreg = joblib.load(config.models_dir / "logreg.pkl")
    logreg_p = logreg.predict_proba(X_test)[:, 1]
    meta_X = np.column_stack([xgb_p, logreg_p, ae_norm])
    fusion_proba = meta_model.predict_proba(meta_X)[:, 1]
    run_full_evaluation(y_test.values, fusion_proba, "fusion", config)
    logger.info("Evaluation done.")


def cmd_infer(config: Config) -> None:
    """Load artifacts and simulate streaming on last N test rows."""
    config.ensure_dirs()
    artifacts = load_artifacts(config)
    test_df = load_processed(config.processed_data_dir / "test_featured")
    simulate_stream(test_df, artifacts, n=100)


def cmd_pipeline(config: Config) -> None:
    """Run preprocess -> features -> train -> evaluate -> infer."""
    cmd_preprocess(config)
    cmd_features(config)
    cmd_train(config)
    cmd_evaluate(config)
    cmd_infer(config)
    logger.info("Pipeline complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fraud Detection Platform")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preprocess", help="Load, merge, clean, split, save")
    sub.add_parser("features", help="Engineer features on processed data")
    sub.add_parser("train", help="Train AE + classifiers + fusion")
    sub.add_parser("evaluate", help="Run evaluation on test split")
    sub.add_parser("infer", help="Simulate streaming inference")
    sub.add_parser("pipeline", help="Run full pipeline")
    args = parser.parse_args()
    config = Config()
    set_global_seed(config.seed)
    if args.command == "preprocess":
        cmd_preprocess(config)
    elif args.command == "features":
        cmd_features(config)
    elif args.command == "train":
        cmd_train(config)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    elif args.command == "infer":
        cmd_infer(config)
    elif args.command == "pipeline":
        cmd_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
