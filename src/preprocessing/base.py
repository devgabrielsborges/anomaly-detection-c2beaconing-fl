"""
Base preprocessor class for network flow data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """Base class for dataset preprocessing."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the preprocessor.

        Args:
            input_path: Path to input parquet file
            output_dir: Directory for processed output files
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}

    def load_data(self) -> pd.DataFrame:
        """Load data from parquet file."""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path, engine="fastparquet")
        logger.info(f"Loaded {len(self.df):,} rows")
        return self.df

    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess raw data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_labels(self) -> pd.DataFrame:
        """Create binary labels for anomaly detection."""
        pass

    @abstractmethod
    def extract_features(self) -> pd.DataFrame:
        """Extract relevant features for modeling."""
        pass

    def handle_missing_values(self, strategy: str = "drop") -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            strategy: 'drop', 'fill_zero', or 'fill_median'
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")

            if strategy == "drop":
                original_len = len(self.df)
                self.df = self.df.dropna()
                logger.info(
                    f"Dropped {original_len - len(self.df):,} rows with missing values"
                )
            elif strategy == "fill_zero":
                self.df = self.df.fillna(0)
                logger.info("Filled missing values with 0")
            elif strategy == "fill_median":
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].fillna(
                    self.df[numeric_cols].median()
                )
                logger.info("Filled missing numeric values with median")

        return self.df

    def split_data(
        self, df: pd.DataFrame, label_col: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with stratification.

        Args:
            df: DataFrame to split
            label_col: Name of the label column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test sets")

        # First split: train+val / test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[label_col] if label_col in df.columns else None,
        )

        # Second split: train / val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=(
                train_val_df[label_col] if label_col in train_val_df.columns else None
            ),
        )

        logger.info(f"Train set: {len(train_df):,} samples")
        logger.info(f"Val set: {len(val_df):,} samples")
        logger.info(f"Test set: {len(test_df):,} samples")

        # Log class distribution
        if label_col in df.columns:
            for split_name, split_df in [
                ("Train", train_df),
                ("Val", val_df),
                ("Test", test_df),
            ]:
                dist = split_df[label_col].value_counts()
                logger.info(f"{split_name} class distribution:\n{dist}")

        return train_df, val_df, test_df

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "processed",
    ) -> None:
        """
        Save processed datasets to parquet files.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            prefix: Prefix for output filenames
        """
        logger.info(f"Saving processed data to {self.output_dir}")

        # Save splits
        train_path = self.output_dir / f"{prefix}_train.parquet"
        val_path = self.output_dir / f"{prefix}_val.parquet"
        test_path = self.output_dir / f"{prefix}_test.parquet"

        train_df.to_parquet(
            train_path, engine="fastparquet", compression="snappy", index=False
        )
        val_df.to_parquet(
            val_path, engine="fastparquet", compression="snappy", index=False
        )
        test_df.to_parquet(
            test_path, engine="fastparquet", compression="snappy", index=False
        )

        logger.info(f"✓ Saved train set to {train_path}")
        logger.info(f"✓ Saved val set to {val_path}")
        logger.info(f"✓ Saved test set to {test_path}")

        self.metadata.update(
            {
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "total_samples": len(train_df) + len(val_df) + len(test_df),
                "features": list(train_df.columns),
                "test_size": self.test_size,
                "val_size": self.val_size,
                "random_state": self.random_state,
            }
        )

        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"✓ Saved metadata to {metadata_path}")

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full preprocessing pipeline.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 80)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 80)

        self.load_data()

        self.df = self.clean_data()

        self.df = self.create_labels()

        self.df = self.extract_features()

        self.df = self.handle_missing_values(strategy="drop")

        train_df, val_df, test_df = self.split_data(self.df)

        dataset_name = self.__class__.__name__.replace("Preprocessor", "").lower()
        self.save_processed_data(train_df, val_df, test_df, prefix=dataset_name)

        logger.info("=" * 80)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 80)

        return train_df, val_df, test_df
