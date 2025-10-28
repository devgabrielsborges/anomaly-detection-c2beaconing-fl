"""Base model interface for all models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs):
        """
        Initialize the model.

        Args:
            **kwargs: Model-specific parameters
        """
        self.model = None
        self.is_trained = False
        self.params = kwargs

    @abstractmethod
    def build(self):
        """Build the model architecture."""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.params

    def set_params(self, **params):
        """
        Set model parameters.

        Args:
            **params: Parameters to set
        """
        self.params.update(params)
