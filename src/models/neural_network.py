"""Neural Network model for anomaly detection using PyTorch."""

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


class NeuralNetworkClassifier(nn.Module):
    """PyTorch Neural Network for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
    ):
        """
        Initialize neural network architecture.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class NeuralNetworkModel(BaseModel):
    """Neural Network model for anomaly detection."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        class_weight: Optional[List[float]] = None,
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize Neural Network model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            class_weight: Weights for imbalanced classes [weight_class_0, weight_class_1]
            device: Device to use ('cuda' or 'cpu')
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs,
        )

        self.class_weight = class_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        # Set random seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

        logger.info(f"Using device: {self.device}")

    def build(self):
        """Build neural network model."""
        if self.params["input_dim"] is None:
            raise ValueError("input_dim must be specified")

        self.model = NeuralNetworkClassifier(
            input_dim=self.params["input_dim"],
            hidden_dims=self.params["hidden_dims"],
            dropout_rate=self.params["dropout_rate"],
        ).to(self.device)

        logger.info(
            f"Built Neural Network: {self.params['input_dim']} -> "
            f"{' -> '.join(map(str, self.params['hidden_dims']))} -> 1"
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train neural network model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            **kwargs: Additional training parameters

        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.params["input_dim"] = X_train.shape[1]
            self.build()

        logger.info(
            f"Training Neural Network on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.params["batch_size"], shuffle=False
            )

        # Loss function with class weights
        if self.class_weight is not None:
            pos_weight = torch.tensor([self.class_weight[1] / self.class_weight[0]]).to(
                self.device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCELoss()

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

                logger.info(
                    f"Epoch {epoch + 1}/{self.params['epochs']}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

                # Early stopping
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
        logger.info("Neural Network training completed")

        return {"history": self.history}

    def _evaluate(self, data_loader, criterion):
        """Evaluate model on data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).cpu().numpy().astype(int).flatten()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities [P(class=0), P(class=1)]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()

        # Return probabilities for both classes
        proba = np.column_stack([1 - outputs, outputs])
        return proba

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "params": self.params,
                "history": self.history,
            },
            path,
        )
        logger.info(f"Saved Neural Network model to {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.params = checkpoint["params"]
        self.history = checkpoint["history"]

        self.build()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True

        logger.info(f"Loaded Neural Network model from {path}")
