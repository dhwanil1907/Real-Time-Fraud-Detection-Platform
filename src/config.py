"""Central configuration for the Fraud Detection Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Single source of truth for paths, seeds, and hyperparameters."""

    # --- Paths (relative to project root) ---
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    plots_dir: Path = field(default_factory=lambda: Path("models/plots"))

    # --- General ---
    seed: int = 42
    train_frac: float = 0.8

    # --- Autoencoder ---
    ae_epochs: int = 30
    ae_lr: float = 1e-3
    ae_hidden_dims: tuple[int, ...] = (128, 64, 32)
    ae_batch_size: int = 256
    ae_val_frac: float = 0.1
    ae_early_stop_patience: int = 5

    # --- XGBoost ---
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # --- Logistic Regression ---
    logreg_max_iter: int = 1000
    logreg_C: float = 1.0

    def __post_init__(self) -> None:
        """Resolve paths relative to project root (parent of src)."""
        project_root = Path(__file__).resolve().parent.parent
        self.raw_data_dir = project_root / self.raw_data_dir
        self.processed_data_dir = project_root / self.processed_data_dir
        self.models_dir = project_root / self.models_dir
        self.plots_dir = project_root / self.plots_dir

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.plots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def xgb_params(self) -> dict:
        """XGBoost classifier params; scale_pos_weight should be set at fit time."""
        return {
            "n_estimators": self.xgb_n_estimators,
            "max_depth": self.xgb_max_depth,
            "learning_rate": self.xgb_learning_rate,
            "subsample": self.xgb_subsample,
            "colsample_bytree": self.xgb_colsample_bytree,
            "random_state": self.seed,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "verbosity": 0,
        }
