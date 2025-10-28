"""Random Forest model for anomaly detection."""

import logging
import pickle
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for anomaly detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            class_weight: Class weights ('balanced', 'balanced_subsample')
            random_state: Random seed
            n_jobs: Number of parallel jobs
            **kwargs: Additional sklearn RandomForest parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.build()

    def build(self):
        """Build Random Forest model."""
        self.model = RandomForestClassifier(**self.params)
        logger.info(f"Built Random Forest with {self.params['n_estimators']} trees")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for RF)
            y_val: Validation labels (not used for RF)
            **kwargs: Additional fit parameters

        Returns:
            Training history (empty for RF)
        """
        logger.info(
            f"Training Random Forest on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        self.model.fit(X_train, y_train, **kwargs)
        self.is_trained = True

        # Get feature importances
        feature_importances = self.model.feature_importances_

        logger.info("Random Forest training completed")

        return {
            "feature_importances": feature_importances,
            "n_features": X_train.shape[1],
        }

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

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict_proba(X)

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved Random Forest model to {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Loaded Random Forest model from {path}")

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances.

        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        return self.model.feature_importances_
