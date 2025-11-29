"""Autoencoder model for anomaly detection using PyTorch."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel

logger = logging.getLogger(__name__)


class AutoencoderNetwork(nn.Module):
    """PyTorch Autoencoder Network."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [64],
        dropout_rate: float = 0.1,
    ):
        """
        Initialize autoencoder architecture.

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the bottleneck layer
            hidden_dims: List of hidden layer dimensions (encoder side)
            dropout_rate: Dropout rate
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = dim

        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim

        # Reverse hidden dims for decoder
        for dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # No activation for output if data is standardized (can be negative)
        # If normalized to [0,1], use Sigmoid. Assuming standardization here.
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderModel(BaseModel):
    """Autoencoder model for one-class anomaly detection."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [64],
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        contamination: float = 0.01,  # Expected proportion of outliers in data
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize Autoencoder model.

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the bottleneck layer
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Max epochs
            early_stopping_patience: Patience for early stopping
            contamination: Expected contamination (used for thresholding if needed)
            device: 'cuda' or 'cpu'
            random_state: Random seed
        """
        super().__init__(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            contamination=contamination,
            random_state=random_state,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": []}
        self.threshold = None

        # Set random seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        logger.info(f"Using device: {self.device}")

    def build(self):
        """Build autoencoder model."""
        if self.params["input_dim"] is None:
            raise ValueError("input_dim must be specified")

        self.model = AutoencoderNetwork(
            input_dim=self.params["input_dim"],
            encoding_dim=self.params["encoding_dim"],
            hidden_dims=self.params["hidden_dims"],
            dropout_rate=self.params["dropout_rate"],
        ).to(self.device)

        logger.info("Built Autoencoder")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train autoencoder.

        # Note: If y_train is provided, only samples with y_train == 0 (normal)
        # will be used for training.
        """
        # Build model if not already built
        if self.model is None:
            self.params["input_dim"] = X_train.shape[1]
            self.build()

        # Filter for normal class if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_train_normal = X_train[normal_mask]
            logger.info(
                f"Filtered training data: {len(X_train_normal)} "
                f"normal samples out of {len(X_train)}"
            )
        else:
            X_train_normal = X_train
            logger.info(
                f"Using all {len(X_train)} samples for training (assuming mostly normal)"
            )

        # Prepare validation data (only normal samples for validation loss)
        if X_val is not None:
            if y_val is not None:
                val_normal_mask = y_val == 0
                X_val_normal = X_val[val_normal_mask]
            else:
                X_val_normal = X_val
        else:
            X_val_normal = None

        # Prepare data loaders
        # Ensure data is writable to avoid PyTorch warnings
        if not X_train_normal.flags.writeable:
            X_train_normal = X_train_normal.copy()

        train_dataset = TensorDataset(torch.FloatTensor(X_train_normal))
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        val_loader = None
        if X_val_normal is not None and len(X_val_normal) > 0:
            if not X_val_normal.flags.writeable:
                X_val_normal = X_val_normal.copy()
            val_dataset = TensorDataset(torch.FloatTensor(X_val_normal))
            val_loader = DataLoader(
                val_dataset, batch_size=self.params["batch_size"], shuffle=False
            )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch_X = batch[0].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_X = batch[0].to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_X)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{self.params['epochs']}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.params["early_stopping_patience"]:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.params['epochs']}: "
                    f"train_loss={train_loss:.4f}"
                )

        self.is_trained = True

        # Determine threshold based on training data reconstruction error
        self._set_threshold(X_train_normal)

        logger.info("Autoencoder training completed")
        return {"history": self.history}

    def _set_threshold(self, X: np.ndarray):
        """Calculate and set threshold based on reconstruction error percentiles."""
        self.model.eval()
        errors = []

        # Process in batches to avoid memory issues
        batch_size = self.params["batch_size"]
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_data = X[i : i + batch_size]
                if not batch_data.flags.writeable:
                    batch_data = batch_data.copy()
                batch = torch.FloatTensor(batch_data).to(self.device)
                outputs = self.model(batch)
                # MSE per sample
                batch_errors = torch.mean((outputs - batch) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())

        errors = np.array(errors)
        # Set threshold at (1 - contamination) quantile
        # e.g., if contamination is 0.01, we set threshold at 99th percentile
        self.threshold = np.quantile(errors, 1 - self.params["contamination"])
        logger.info(
            f"Threshold set to {self.threshold:.6f} "
            f"(contamination={self.params['contamination']})"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly, 0 = normal).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        errors = self._get_reconstruction_errors(X)
        return (errors > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return reconstruction errors as "probabilities" (not true probabilities).
        We'll normalize them roughly or just return error scores.
        For compatibility, we return [1-score, score].
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        errors = self._get_reconstruction_errors(X)
        # Simple scaling for "probability" interpretation
        # This is heuristic. A sigmoid on (error - threshold) could work too.
        # Let's use a sigmoid centered at threshold

        # Scale factor to make the transition reasonable
        scale = self.threshold if self.threshold > 0 else 1.0
        logits = (errors - self.threshold) / scale * 5  # *5 to make it steeper
        probs = 1 / (1 + np.exp(-logits))

        return np.column_stack([1 - probs, probs])

    def _get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        errors = []
        batch_size = self.params["batch_size"]

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_data = X[i : i + batch_size]
                if not batch_data.flags.writeable:
                    batch_data = batch_data.copy()
                batch = torch.FloatTensor(batch_data).to(self.device)
                outputs = self.model(batch)
                batch_errors = torch.mean((outputs - batch) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())

        return np.array(errors)

    def save(self, path: str):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "params": self.params,
                "history": self.history,
                "threshold": self.threshold,
            },
            path,
        )
        logger.info(f"Saved Autoencoder model to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.params = checkpoint["params"]
        self.history = checkpoint["history"]
        self.threshold = checkpoint.get("threshold")

        self.build()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True

        logger.info(f"Loaded Autoencoder model from {path}")
