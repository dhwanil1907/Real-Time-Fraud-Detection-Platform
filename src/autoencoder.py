"""PyTorch autoencoder for anomaly detection: train on normal (non-fraud) only, score by reconstruction error."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class FraudAutoencoder(nn.Module):
    """
    Dense autoencoder: input_dim -> hidden_dims -> bottleneck -> hidden_dims -> input_dim.
    ReLU in encoder/decoder, no activation on final output (MSE loss).
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64, 32)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        layers_enc = []
        prev = input_dim
        for h in hidden_dims:
            layers_enc.append(nn.Linear(prev, h))
            layers_enc.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*layers_enc)
        layers_dec = []
        for h in reversed(hidden_dims[:-1]):
            layers_dec.append(nn.Linear(prev, h))
            layers_dec.append(nn.ReLU())
            prev = h
        layers_dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers_dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(X.astype(np.float32))
    return t.to(device)


def train_autoencoder(
    X_train_normal: np.ndarray,
    config: Any,
    X_val: np.ndarray | None = None,
) -> tuple[FraudAutoencoder, list[float]]:
    """
    Train autoencoder on normal (non-fraud) samples only.
    Returns (trained model, list of train losses per epoch).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_normal.shape[1]
    hidden_dims = getattr(config, "ae_hidden_dims", (128, 64, 32))
    if isinstance(hidden_dims, list):
        hidden_dims = tuple(hidden_dims)
    model = FraudAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ae_lr)
    criterion = nn.MSELoss()

    batch_size = getattr(config, "ae_batch_size", 256)
    epochs = config.ae_epochs
    val_frac = getattr(config, "ae_val_frac", 0.1)
    patience = getattr(config, "ae_early_stop_patience", 5)

    n_val = int(len(X_train_normal) * val_frac) if X_val is None else 0
    if X_val is None and n_val > 0:
        np.random.RandomState(config.seed).shuffle(X_train_normal)
        X_val = X_train_normal[-n_val:]
        X_train_normal = X_train_normal[:-n_val]

    train_ds = TensorDataset(_to_tensor(X_train_normal, device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_val_loss = float("inf")
    best_state: dict[str, Any] = {}
    patience_counter = 0
    train_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_t = _to_tensor(X_val, device)
                val_recon = model(val_t)
                val_loss = criterion(val_recon, val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break
        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d train loss %.6f", epoch + 1, train_losses[-1])
    if best_state:
        model.load_state_dict(best_state)  # Serialized model is now the best by validation loss.
    return model, train_losses


def compute_anomaly_scores(model: FraudAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row MSE reconstruction error as anomaly score."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t = _to_tensor(X, device)
        recon = model(t)
        mse = ((t - recon) ** 2).mean(dim=1)
    return mse.cpu().numpy()


def save_autoencoder(model: FraudAutoencoder, path: Path) -> None:
    """Save model state dict to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved autoencoder to %s", path)


def load_autoencoder(path: Path, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64, 32)) -> FraudAutoencoder:
    """Load model from path."""
    model = FraudAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model
