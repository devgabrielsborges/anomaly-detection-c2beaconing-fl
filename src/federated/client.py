"""Flower client implementation for federated learning."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

logger = logging.getLogger(__name__)


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""

    def __init__(
        self,
        client_id: str,
        model: any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs_per_round: int = 1,
    ):
        """
        Initialize Flower client.

        Args:
            client_id: Unique client identifier
            model: Model instance (RandomForest, XGBoost, or NeuralNetwork)
            X_train: Training features for this client
            y_train: Training labels for this client
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs_per_round: Number of local epochs per federated round
        """
        super().__init__()
        self.client_id = client_id
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs_per_round = epochs_per_round

        logger.info(
            f"Client {client_id} initialized with {len(X_train)} training samples"
        )

    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """
        Get model parameters.

        Args:
            config: Configuration dictionary

        Returns:
            List of model parameters as numpy arrays
        """
        # For sklearn models (RandomForest, XGBoost)
        if hasattr(self.model.model, "get_booster"):
            # XGBoost
            return [self.model.model.get_booster().save_raw("json").encode()]
        elif hasattr(self.model.model, "estimators_"):
            # Random Forest - not directly supported in FL
            logger.warning("Random Forest does not support parameter aggregation")
            return []
        else:
            # PyTorch Neural Network
            import torch

            return [val.cpu().numpy() for val in self.model.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.

        Args:
            parameters: List of model parameters as numpy arrays
        """
        if not parameters:
            return

        # For PyTorch Neural Network
        if hasattr(self.model.model, "state_dict"):
            import torch

            params_dict = zip(self.model.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.model.load_state_dict(state_dict, strict=True)
        # For XGBoost
        elif hasattr(self.model.model, "get_booster"):
            self.model.model.get_booster().load_model(parameters[0].tobytes())

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, any]]:
        """
        Train model on local data.

        Args:
            parameters: Model parameters from server
            config: Training configuration

        Returns:
            Tuple of (updated parameters, number of samples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training")

        # Set parameters received from server
        self.set_parameters(parameters)

        # Train model
        if hasattr(self.model, "model") and hasattr(self.model.model, "state_dict"):
            # Neural Network - train for specified epochs
            history = self.model.fit(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                epochs=self.epochs_per_round,
            )
        else:
            # Tree-based models
            history = self.model.fit(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
            )

        # Get updated parameters
        updated_parameters = self.get_parameters({})

        # Compute training metrics
        y_pred = self.model.predict(self.X_train)
        train_accuracy = np.mean(y_pred == self.y_train)

        metrics = {
            "train_accuracy": float(train_accuracy),
            "train_samples": len(self.X_train),
        }

        logger.info(
            f"Client {self.client_id}: Training completed. "
            f"Accuracy: {train_accuracy:.4f}"
        )

        return updated_parameters, len(self.X_train), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, any]
    ) -> Tuple[float, int, Dict[str, any]]:
        """
        Evaluate model on local data.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, number of samples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting evaluation")

        # Set parameters
        self.set_parameters(parameters)

        # Use validation set if available, otherwise use training set
        X_eval = self.X_val if self.X_val is not None else self.X_train
        y_eval = self.y_val if self.y_val is not None else self.y_train

        # Make predictions
        y_pred = self.model.predict(X_eval)
        y_proba = self.model.predict_proba(X_eval)

        # Compute metrics
        accuracy = np.mean(y_pred == y_eval)

        # Compute binary cross-entropy loss
        epsilon = 1e-15
        y_proba_clipped = np.clip(y_proba[:, 1], epsilon, 1 - epsilon)
        loss = -np.mean(
            y_eval * np.log(y_proba_clipped)
            + (1 - y_eval) * np.log(1 - y_proba_clipped)
        )

        metrics = {
            "accuracy": float(accuracy),
            "eval_samples": len(X_eval),
        }

        logger.info(
            f"Client {self.client_id}: Evaluation completed. "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        return float(loss), len(X_eval), metrics


def create_client_fn(
    model_class: any,
    model_params: Dict[str, any],
    data_partitions: Dict[str, Dict[str, np.ndarray]],
    epochs_per_round: int = 1,
) -> Callable[[str], FlowerClient]:
    """
    Create a client function for Flower simulation.

    Args:
        model_class: Model class (e.g., NeuralNetworkModel)
        model_params: Parameters for model initialization
        data_partitions: Dictionary mapping client_id to data dict
        epochs_per_round: Number of local epochs per round

    Returns:
        Client function that creates a FlowerClient instance
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client for given client ID."""
        # Get client's data
        client_data = data_partitions[cid]

        # Create model instance
        model = model_class(**model_params)

        # Create and return client
        return FlowerClient(
            client_id=cid,
            model=model,
            X_train=client_data["X_train"],
            y_train=client_data["y_train"],
            X_val=client_data.get("X_val"),
            y_val=client_data.get("y_val"),
            epochs_per_round=epochs_per_round,
        )

    return client_fn
