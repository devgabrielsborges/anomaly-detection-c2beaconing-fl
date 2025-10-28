"""Data loading and partitioning for federated learning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPartitioner:
    """Partition data across federated learning clients."""

    def __init__(
        self,
        num_clients: int,
        partition_strategy: str = "iid",
        alpha: float = 0.5,
        random_state: int = 42,
    ):
        """
        Initialize data partitioner.

        Args:
            num_clients: Number of federated clients
            partition_strategy: Partitioning strategy ('iid', 'non-iid-dirichlet')
            alpha: Concentration parameter for Dirichlet (lower = more non-IID)
            random_state: Random seed
        """
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.alpha = alpha
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        logger.info(
            f"Initialized DataPartitioner: {num_clients} clients, "
            f"strategy={partition_strategy}, alpha={alpha}"
        )

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.0,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Partition data across clients.

        Args:
            X: Features
            y: Labels
            val_size: Fraction of each client's data for validation

        Returns:
            Dictionary mapping client_id to data dict
        """
        logger.info(f"Partitioning {len(X)} samples across {self.num_clients} clients")

        if self.partition_strategy == "iid":
            partitions = self._partition_iid(X, y)
        elif self.partition_strategy == "non-iid-dirichlet":
            partitions = self._partition_non_iid_dirichlet(X, y)
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")

        # Split each client's data into train/val if requested
        if val_size > 0:
            partitions = self._add_validation_split(partitions, val_size)

        # Log partition statistics
        self._log_partition_stats(partitions)

        return partitions

    def _partition_iid(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Partition data in IID fashion (random shuffle and split).

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary mapping client_id to data dict
        """
        num_samples = len(X)
        indices = self.rng.permutation(num_samples)

        # Split indices evenly across clients
        client_indices = np.array_split(indices, self.num_clients)

        partitions = {}
        for client_id, idx in enumerate(client_indices):
            partitions[str(client_id)] = {
                "X_train": X[idx],
                "y_train": y[idx],
            }

        logger.info("Created IID partitions")
        return partitions

    def _partition_non_iid_dirichlet(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Partition data using Dirichlet distribution (label non-IID).

        This creates realistic non-IID distributions where each client
        has different class distributions.

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary mapping client_id to data dict
        """
        num_classes = len(np.unique(y))
        num_samples = len(X)

        # Get indices for each class
        class_indices = [np.where(y == i)[0] for i in range(num_classes)]

        # Initialize client data indices
        client_indices = [[] for _ in range(self.num_clients)]

        # For each class, distribute samples across clients using Dirichlet
        for class_idx in class_indices:
            # Sample proportions from Dirichlet distribution
            proportions = self.rng.dirichlet(np.repeat(self.alpha, self.num_clients))

            # Scale proportions to match number of samples in this class
            proportions = (proportions * len(class_idx)).astype(int)

            # Adjust to ensure all samples are distributed
            diff = len(class_idx) - proportions.sum()
            proportions[0] += diff

            # Shuffle class indices
            shuffled_idx = self.rng.permutation(class_idx)

            # Distribute to clients
            start_idx = 0
            for client_id, count in enumerate(proportions):
                end_idx = start_idx + count
                client_indices[client_id].extend(shuffled_idx[start_idx:end_idx])
                start_idx = end_idx

        # Create partitions
        partitions = {}
        for client_id, idx in enumerate(client_indices):
            if len(idx) > 0:
                idx = np.array(idx)
                # Shuffle client's data
                shuffled = self.rng.permutation(len(idx))
                partitions[str(client_id)] = {
                    "X_train": X[idx[shuffled]],
                    "y_train": y[idx[shuffled]],
                }

        logger.info(f"Created non-IID Dirichlet partitions (alpha={self.alpha})")
        return partitions

    def _add_validation_split(
        self,
        partitions: Dict[str, Dict[str, np.ndarray]],
        val_size: float,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Add validation split to each client's data.

        Args:
            partitions: Client partitions
            val_size: Validation fraction

        Returns:
            Updated partitions with validation data
        """
        for client_id in partitions:
            X_train = partitions[client_id]["X_train"]
            y_train = partitions[client_id]["y_train"]

            if len(X_train) > 1:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=val_size,
                    random_state=self.random_state,
                    stratify=y_train if len(np.unique(y_train)) > 1 else None,
                )

                partitions[client_id] = {
                    "X_train": X_tr,
                    "y_train": y_tr,
                    "X_val": X_val,
                    "y_val": y_val,
                }

        return partitions

    def _log_partition_stats(
        self,
        partitions: Dict[str, Dict[str, np.ndarray]],
    ):
        """Log statistics about partitions."""
        logger.info("=" * 60)
        logger.info("Partition Statistics:")
        logger.info("=" * 60)

        total_train = 0
        total_val = 0

        for client_id, data in partitions.items():
            n_train = len(data["y_train"])
            n_val = len(data.get("y_val", []))
            total_train += n_train
            total_val += n_val

            # Class distribution
            unique, counts = np.unique(data["y_train"], return_counts=True)
            class_dist = dict(zip(unique, counts))

            logger.info(
                f"Client {client_id}: "
                f"{n_train} train, {n_val} val, "
                f"class_dist={class_dist}"
            )

        logger.info(f"Total: {total_train} train, {total_val} val")
        logger.info("=" * 60)


def load_preprocessed_data(
    data_path: str,
    dataset: str = "ctu13",
    features_to_drop: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load preprocessed data for training.

    Args:
        data_path: Path to processed data directory
        dataset: Dataset name ('ctu13' or 'ugr16')
        features_to_drop: List of feature names to drop (e.g., identifiers)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names)
    """
    data_path = Path(data_path)

    logger.info(f"Loading {dataset} data from {data_path}")

    # Load train and test data
    train_df = pd.read_parquet(data_path / f"{dataset}_train.parquet")
    test_df = pd.read_parquet(data_path / f"{dataset}_test.parquet")

    logger.info(f"Loaded {len(train_df)} train, {len(test_df)} test samples")

    # Determine label column
    label_col = "label" if dataset == "ctu13" else "binary_label"

    # Default features to drop
    default_drop = [
        "StartTime",
        "SrcAddr",
        "DstAddr",
        "Label",
        "scenario",
        "Timestamp",
        "Te",
    ]

    if features_to_drop is None:
        features_to_drop = []

    # Combine default and user-specified drops
    all_drops = list(set(default_drop + features_to_drop))

    # Separate features and labels - only select numeric columns
    feature_cols = [
        col
        for col in train_df.columns
        if col != label_col
        and col not in all_drops
        and train_df[col].dtype in ["int64", "int32", "float64", "float32", "bool"]
    ]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int64)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int64)

    logger.info(
        f"Features: {len(feature_cols)}, Train: {X_train.shape}, Test: {X_test.shape}"
    )
    logger.info(f"Train label distribution: {np.bincount(y_train.astype(int))}")
    logger.info(f"Test label distribution: {np.bincount(y_test.astype(int))}")

    return X_train, y_train, X_test, y_test, feature_cols
