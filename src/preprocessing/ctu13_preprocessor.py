"""
CTU-13 dataset preprocessor for C2 beaconing detection.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class CTU13Preprocessor(BasePreprocessor):
    """Preprocessor for CTU-13 NetFlow dataset."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__(input_path, output_dir, test_size, val_size, random_state)

    def _convert_port(self, port) -> int:
        """Convert port to integer, handling hex and missing values."""
        if pd.isna(port) or port == "-":
            return 0
        try:
            # Check if it's hex format (0x...)
            if isinstance(port, str) and port.startswith("0x"):
                return int(port, 16)
            return int(port)
        except (ValueError, TypeError):
            return 0

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess CTU-13 data."""
        logger.info("Cleaning CTU-13 data")

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert timestamp
        self.df["StartTime"] = pd.to_datetime(self.df["StartTime"], errors="coerce")

        # Convert ports to integers (handling hex format)
        logger.info("Converting port numbers")
        self.df["Sport"] = self.df["Sport"].apply(self._convert_port)
        self.df["Dport"] = self.df["Dport"].apply(self._convert_port)

        # Convert ToS values to integers
        self.df["sTos"] = self.df["sTos"].fillna(0).astype("int32")
        self.df["dTos"] = self.df["dTos"].fillna(0).astype("int32")

        # Ensure numeric columns are proper types
        numeric_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical columns
        categorical_cols = ["Proto", "Dir", "State"]
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Drop rows with invalid timestamps
        original_len = len(self.df)
        self.df = self.df.dropna(subset=["StartTime"])
        if len(self.df) < original_len:
            logger.info(
                f"Dropped {original_len - len(self.df):,} rows with invalid timestamps"
            )

        self.metadata["cleaning"] = {
            "port_conversion": "hex and string to integer",
            "timestamp_format": "converted to datetime",
            "rows_after_cleaning": len(self.df),
        }

        return self.df

    def create_labels(self) -> pd.DataFrame:
        """Create binary labels for botnet detection."""
        logger.info("Creating binary labels")

        if self.df is None:
            raise ValueError("Data not loaded.")

        # Create binary label: 1 for botnet, 0 for background
        self.df["label"] = (
            self.df["Label"].str.contains("Botnet", case=False, na=False).astype(int)
        )

        # Extract scenario information
        self.df["scenario"] = (
            self.df["Label"].str.extract(r"(V\d{2})", expand=False).fillna("Unknown")
        )

        # Log label distribution
        label_dist = self.df["label"].value_counts()
        logger.info(f"Label distribution:\n{label_dist}")
        logger.info(
            f"Imbalance ratio: "
            f"{label_dist[0] / label_dist[1]:.2f}:1 (background:botnet)"
        )

        self.metadata["labels"] = {
            "botnet_samples": int(label_dist.get(1, 0)),
            "background_samples": int(label_dist.get(0, 0)),
            "imbalance_ratio": float(label_dist[0] / label_dist[1]),
        }

        return self.df

    def extract_features(self) -> pd.DataFrame:
        """Extract features for C2 beaconing detection."""
        logger.info("Extracting features")

        if self.df is None:
            raise ValueError("Data not loaded.")

        # Temporal features
        self.df["hour"] = self.df["StartTime"].dt.hour
        self.df["day_of_week"] = self.df["StartTime"].dt.dayofweek
        self.df["timestamp_seconds"] = self.df["StartTime"].astype("int64") // 10**9

        # Statistical features
        # Bytes per packet ratio
        self.df["bytes_per_packet"] = self.df["TotBytes"] / (self.df["TotPkts"] + 1)

        # Source to total bytes ratio
        self.df["src_bytes_ratio"] = self.df["SrcBytes"] / (self.df["TotBytes"] + 1)

        # Log transformations for skewed distributions
        for col in ["Dur", "TotPkts", "TotBytes", "SrcBytes"]:
            if col in self.df.columns:
                self.df[f"{col}_log"] = np.log1p(self.df[col])

        # Protocol encoding (one-hot for top protocols)
        top_protocols = self.df["Proto"].value_counts().head(10).index
        for proto in top_protocols:
            self.df[f"proto_{proto}"] = (self.df["Proto"] == proto).astype(int)

        # Port features
        # Check if port is in common service port range
        self.df["is_common_src_port"] = (
            (self.df["Sport"] > 0) & (self.df["Sport"] < 1024)
        ).astype(int)
        self.df["is_common_dst_port"] = (
            (self.df["Dport"] > 0) & (self.df["Dport"] < 1024)
        ).astype(int)

        # Connection state features
        state_counts = self.df["State"].value_counts().head(10).index
        for state in state_counts:
            self.df[f"state_{state}"] = (self.df["State"] == state).astype(int)

        self.metadata["features"] = {
            "temporal_features": ["hour", "day_of_week", "timestamp_seconds"],
            "statistical_features": [
                "bytes_per_packet",
                "src_bytes_ratio",
                "Dur_log",
                "TotPkts_log",
                "TotBytes_log",
                "SrcBytes_log",
            ],
            "protocol_features": [f"proto_{p}" for p in top_protocols],
            "port_features": ["is_common_src_port", "is_common_dst_port"],
            "total_features": len(
                [
                    c
                    for c in self.df.columns
                    if c not in ["Label", "SrcAddr", "DstAddr", "StartTime", "scenario"]
                ]
            ),
        }

        logger.info(f"Created {self.metadata['features']['total_features']} features")

        return self.df
