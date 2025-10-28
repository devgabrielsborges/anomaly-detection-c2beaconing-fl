"""
UGR'16 dataset preprocessor for C2 beaconing detection.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from .base import BasePreprocessor

logger = logging.getLogger(__name__)


class UGR16Preprocessor(BasePreprocessor):
    """Preprocessor for UGR'16 network traffic dataset."""

    MALICIOUS_KEYWORDS = [
        "botnet",
        "attack",
        "anomaly",
        "malicious",
        "ddos",
        "worm",
        "spam",
        "blacklist",
    ]

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__(input_path, output_dir, test_size, val_size, random_state)

    def _standardize_column_names(self) -> pd.DataFrame:
        """Standardize column names from columnXX format to readable names."""
        column_mapping = {
            "column0": "timestamp",
            "column00": "timestamp",
            "column1": "duration",
            "column01": "duration",
            "column2": "src_ip",
            "column02": "src_ip",
            "column3": "dst_ip",
            "column03": "dst_ip",
            "column4": "src_port",
            "column04": "src_port",
            "column5": "dst_port",
            "column05": "dst_port",
            "column6": "protocol",
            "column06": "protocol",
            "column7": "flags",
            "column07": "flags",
            "column8": "tos",
            "column08": "tos",
            "column9": "packets_fwd",
            "column09": "packets_fwd",
            "column10": "packets_bwd",
            "column11": "bytes_total",
            "column12": "label",
        }

        rename_map = {}
        for original in self.df.columns:
            if original in column_mapping:
                rename_map[original] = column_mapping[original]

        if rename_map:
            self.df = self.df.rename(columns=rename_map)
            logger.info(f"Renamed {len(rename_map)} columns")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess UGR'16 data."""
        logger.info("Cleaning UGR'16 data")

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._standardize_column_names()

        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

        numeric_cols = {
            "duration": float,
            "src_port": "Int64",
            "dst_port": "Int64",
            "packets_fwd": "Int64",
            "packets_bwd": "Int64",
            "bytes_total": float,
            "tos": "Int64",
        }

        for col, dtype in numeric_cols.items():
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(
                    dtype
                )

        categorical_cols = ["protocol", "flags", "label"]
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Drop rows with invalid timestamps
        if "timestamp" in self.df.columns:
            original_len = len(self.df)
            self.df = self.df.dropna(subset=["timestamp"])
            if len(self.df) < original_len:
                logger.info(
                    f"Dropped {original_len - len(self.df):,} rows "
                    "with invalid timestamps"
                )

        self.metadata["cleaning"] = {
            "column_standardization": "applied",
            "timestamp_format": "converted to datetime",
            "rows_after_cleaning": len(self.df),
        }

        return self.df

    def create_labels(self) -> pd.DataFrame:
        """Create binary labels for malicious traffic detection."""
        logger.info("Creating binary labels")

        if self.df is None or "label" not in self.df.columns:
            raise ValueError("Data not loaded or 'label' column not found.")

        # Create binary label: 1 for malicious, 0 for benign
        pattern = "|".join(self.MALICIOUS_KEYWORDS)
        self.df["is_malicious"] = (
            self.df["label"]
            .astype(str)
            .str.contains(pattern, case=False, na=False)
            .astype(int)
        )

        self.df["binary_label"] = self.df["is_malicious"]

        # Log label distribution
        label_dist = self.df["binary_label"].value_counts()
        logger.info(f"Label distribution:\n{label_dist}")

        if label_dist.get(1, 0) > 0:
            imbalance_ratio = label_dist[0] / label_dist[1]
            logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1 (benign:malicious)")
        else:
            imbalance_ratio = float("inf")
            logger.warning("No malicious samples detected!")

        self.metadata["labels"] = {
            "malicious_samples": int(label_dist.get(1, 0)),
            "benign_samples": int(label_dist.get(0, 0)),
            "imbalance_ratio": float(imbalance_ratio),
            "keywords_used": self.MALICIOUS_KEYWORDS,
        }

        return self.df

    def extract_features(self) -> pd.DataFrame:
        """Extract features for C2 beaconing detection."""
        logger.info("Extracting features")

        if self.df is None:
            raise ValueError("Data not loaded.")

        if "timestamp" in self.df.columns:
            self.df["hour"] = self.df["timestamp"].dt.hour
            self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek
            self.df["timestamp_seconds"] = self.df["timestamp"].astype("int64") // 10**9

        # Statistical features
        if "packets_fwd" in self.df.columns and "packets_bwd" in self.df.columns:
            self.df["total_packets"] = self.df["packets_fwd"].fillna(0) + self.df[
                "packets_bwd"
            ].fillna(0)

            # Packet direction ratio
            self.df["fwd_packet_ratio"] = self.df["packets_fwd"] / (
                self.df["total_packets"] + 1
            )

        # Bytes per packet
        if "bytes_total" in self.df.columns and "total_packets" in self.df.columns:
            self.df["bytes_per_packet"] = self.df["bytes_total"] / (
                self.df["total_packets"] + 1
            )

        # Log transformations for skewed distributions
        for col in ["duration", "packets_fwd", "packets_bwd", "bytes_total"]:
            if col in self.df.columns:
                self.df[f"{col}_log"] = np.log1p(self.df[col].fillna(0))

        # Protocol encoding (one-hot for top protocols)
        if "protocol" in self.df.columns:
            top_protocols = self.df["protocol"].value_counts().head(10).index
            for proto in top_protocols:
                self.df[f"proto_{proto}"] = (self.df["protocol"] == proto).astype(int)

        # Port features
        if "src_port" in self.df.columns:
            self.df["is_common_src_port"] = (
                (self.df["src_port"] > 0) & (self.df["src_port"] < 1024)
            ).astype(int)

        if "dst_port" in self.df.columns:
            self.df["is_common_dst_port"] = (
                (self.df["dst_port"] > 0) & (self.df["dst_port"] < 1024)
            ).astype(int)

            # Top destination ports
            top_ports = self.df["dst_port"].value_counts().head(20).index
            for port in top_ports:
                self.df[f"dst_port_{port}"] = (self.df["dst_port"] == port).astype(int)

        # Flag features (if available)
        if "flags" in self.df.columns:
            top_flags = self.df["flags"].value_counts().head(10).index
            for flag in top_flags:
                self.df[f"flag_{flag}"] = (self.df["flags"] == flag).astype(int)

        # Count features
        feature_list = [
            c
            for c in self.df.columns
            if c not in ["label", "src_ip", "dst_ip", "timestamp", "is_malicious"]
        ]

        self.metadata["features"] = {
            "temporal_features": ["hour", "day_of_week", "timestamp_seconds"],
            "statistical_features": [
                "total_packets",
                "fwd_packet_ratio",
                "bytes_per_packet",
            ],
            "total_features": len(feature_list),
        }

        logger.info(f"Created {len(feature_list)} features")

        return self.df
