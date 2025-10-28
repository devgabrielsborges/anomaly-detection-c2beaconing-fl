"""XGBoost model for anomaly detection."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier for anomaly detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = 10,
        **kwargs,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            scale_pos_weight: Balancing of positive and negative weights
            random_state: Random seed
            n_jobs: Number of parallel threads
            early_stopping_rounds: Early stopping rounds
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.build()

    def build(self):
        """Build XGBoost model."""
        # Remove early_stopping_rounds from model params
        model_params = {
            k: v for k, v in self.params.items() if k != "early_stopping_rounds"
        }

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "aucpr"],
            **model_params,
        )
        logger.info(f"Built XGBoost with {self.params['n_estimators']} rounds")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            **kwargs: Additional fit parameters

        Returns:
            Training history with eval results
        """
        logger.info(
            f"Training XGBoost on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        fit_params = {}

        # Add early stopping if validation set provided
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            if self.early_stopping_rounds is not None:
                fit_params["early_stopping_rounds"] = self.early_stopping_rounds
            fit_params["verbose"] = kwargs.get("verbose", False)

        # Merge with any additional kwargs
        fit_params.update(kwargs)

        self.model.fit(X_train, y_train, **fit_params)
        self.is_trained = True

        # Get eval results
        history = {}
        if hasattr(self.model, "evals_result_"):
            history = self.model.evals_result_

        # Get feature importances
        feature_importances = self.model.feature_importances_

        logger.info("XGBoost training completed")

        return {
            "history": history,
            "feature_importances": feature_importances,
            "best_iteration": (
                self.model.best_iteration
                if hasattr(self.model, "best_iteration")
                else None
            ),
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
        self.model.save_model(path)
        logger.info(f"Saved XGBoost model to {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        self.model.load_model(path)
        self.is_trained = True
        logger.info(f"Loaded XGBoost model from {path}")

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances.

        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        return self.model.feature_importances_
