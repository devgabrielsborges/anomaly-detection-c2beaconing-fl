"""
Advanced feature engineering for C2 beaconing detection.

This module provides utilities for extracting behavioral and temporal features
that are indicative of command-and-control beaconing patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for C2 beaconing detection.

    Focuses on:
    - Periodicity detection
    - Flow aggregation statistics
    - Behavioral patterns
    - Temporal consistency
    """

    def __init__(self, time_window: str = "1H"):
        """
        Initialize feature engineer.

        Args:
            time_window: Time window for aggregation (pandas offset string)
        """
        self.time_window = time_window
        self.fitted_ = False
        self.statistics_ = {
            "periodicity": {},
            "aggregation": {},
            "entropy": {},
            "consistency": {},
        }
        self.group_cols_ = None
        self.timestamp_col_ = None

    def _determine_group_cols(self, df: pd.DataFrame) -> List[str]:
        """Determine appropriate grouping columns from DataFrame."""
        possible_src = ["SrcAddr", "src_ip"]
        possible_dst = ["DstAddr", "dst_ip"]

        src_col = next((c for c in possible_src if c in df.columns), None)
        dst_col = next((c for c in possible_dst if c in df.columns), None)

        if src_col and dst_col:
            return [src_col, dst_col]
        elif src_col:
            return [src_col]
        else:
            logger.warning("No suitable grouping columns found")
            return []

    def fit(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_cols: Optional[List[str]] = None,
        enable_periodicity: bool = True,
        enable_aggregation: bool = True,
        enable_entropy: bool = True,
        enable_consistency: bool = True,
    ) -> "FeatureEngineer":
        """
        Learn aggregation statistics from TRAINING data only.

        CRITICAL: This must be called ONLY on training data after train/test split.

        Args:
            df: Training DataFrame (post-split)
            timestamp_col: Name of timestamp column
            group_cols: Columns to group by (e.g., src_ip, dst_ip)
            enable_*: Flags to enable specific feature types

        Returns:
            self (for method chaining)
        """
        logger.info("=" * 80)
        logger.info("FITTING FeatureEngineer on TRAINING data only")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(df):,}")

        self.timestamp_col_ = timestamp_col

        # Determine grouping columns
        if group_cols is None:
            group_cols = self._determine_group_cols(df)
        self.group_cols_ = group_cols

        if not self.group_cols_:
            logger.warning(
                "No grouping columns available - skipping aggregated features"
            )
            self.fitted_ = True
            return self

        # Compute statistics on training data only
        if enable_periodicity and timestamp_col in df.columns:
            self._fit_periodicity_features(df)

        if enable_aggregation and timestamp_col in df.columns:
            self._fit_aggregation_features(df)

        if enable_entropy:
            self._fit_entropy_features(df)

        if enable_consistency:
            self._fit_consistency_features(df)

        self.fitted_ = True
        logger.info("✓ FeatureEngineer fitted successfully")
        total_stats = sum(
            len(v) if isinstance(v, dict) else 0 for v in self.statistics_.values()
        )
        logger.info(f"✓ Statistics computed for {total_stats} unique groups")
        logger.info("=" * 80)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned statistics to new data (validation/test).

        This uses ONLY the statistics computed during fit() on training data.
        No new statistics are computed from this data.

        Args:
            df: DataFrame to transform (can be train/val/test)

        Returns:
            DataFrame with engineered features added
        """
        if not self.fitted_:
            raise ValueError(
                "FeatureEngineer must be fitted before transform. "
                "Call fit() on training data first."
            )

        logger.info(f"Transforming {len(df):,} samples using learned statistics")

        df = df.copy()

        if not self.group_cols_:
            logger.warning("No grouping columns - returning original DataFrame")
            return df

        # Apply learned statistics
        if self.statistics_["periodicity"]:
            df = self._transform_periodicity_features(df)

        if self.statistics_["aggregation"]:
            df = self._transform_aggregation_features(df)

        if self.statistics_["entropy"]:
            df = self._transform_entropy_features(df)

        if self.statistics_["consistency"]:
            df = self._transform_consistency_features(df)

        logger.info("✓ Transform completed")

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_cols: Optional[List[str]] = None,
        enable_periodicity: bool = True,
        enable_aggregation: bool = True,
        enable_entropy: bool = True,
        enable_consistency: bool = True,
    ) -> pd.DataFrame:
        """
        Fit on training data and transform it in one step.

        This should ONLY be used on training data.
        For val/test, use transform() separately.
        """
        self.fit(
            df,
            timestamp_col,
            group_cols,
            enable_periodicity,
            enable_aggregation,
            enable_entropy,
            enable_consistency,
        )
        return self.transform(df)

    def _fit_periodicity_features(self, df: pd.DataFrame):
        """Learn inter-arrival time statistics from training data."""
        logger.info("  Computing periodicity statistics...")

        df_sorted = df.sort_values(self.timestamp_col_)

        # Compute inter-arrival times
        df_sorted = df_sorted.copy()
        df_sorted["_iat"] = (
            df_sorted.groupby(self.group_cols_, observed=True)[self.timestamp_col_]
            .diff()
            .dt.total_seconds()
        )

        # Aggregate statistics per group
        periodicity_stats = (
            df_sorted.groupby(self.group_cols_, observed=True)["_iat"]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
        )

        # Compute CV
        periodicity_stats["cv"] = periodicity_stats["std"] / (
            periodicity_stats["mean"] + 1e-9
        )
        periodicity_stats["cv"] = (
            periodicity_stats["cv"].replace([np.inf, -np.inf], 0).fillna(0)
        )

        # Store as dictionary for fast lookup
        if len(self.group_cols_) == 1:
            self.statistics_["periodicity"] = periodicity_stats.set_index(
                self.group_cols_[0]
            ).to_dict("index")
        else:
            self.statistics_["periodicity"] = periodicity_stats.set_index(
                self.group_cols_
            ).to_dict("index")

        logger.info(
            f"    Stored stats for {len(self.statistics_['periodicity'])} groups"
        )

    def _transform_periodicity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned periodicity statistics."""
        # Create DataFrame from stored statistics
        stats_df = pd.DataFrame.from_dict(
            self.statistics_["periodicity"], orient="index"
        )

        # Reset index and handle column naming
        if len(self.group_cols_) == 1:
            # Single group column - simple case
            stats_df = stats_df.reset_index()
            stats_df = stats_df.rename(columns={"index": self.group_cols_[0]})
        else:
            stats_df = stats_df.reset_index()
            if "level_0" in stats_df.columns:
                # Generic names were created, need to rename
                rename_dict = {}
                for i, col in enumerate(self.group_cols_):
                    rename_dict[f"level_{i}"] = col
                stats_df = stats_df.rename(columns=rename_dict)

        # Merge with data
        df = df.merge(
            stats_df, left_on=self.group_cols_, right_on=self.group_cols_, how="left"
        )

        # Rename columns
        df = df.rename(
            columns={
                "mean": "iat_mean",
                "std": "iat_std",
                "median": "iat_median",
                "min": "iat_min",
                "max": "iat_max",
                "cv": "iat_cv",
            }
        )

        # Fill missing values (for groups not seen in training)
        fill_values = {
            "iat_mean": 0,
            "iat_std": 0,
            "iat_median": 0,
            "iat_min": 0,
            "iat_max": 0,
            "iat_cv": 0,
        }
        df = df.fillna(fill_values)

        # Compute derived features
        df["periodicity_score"] = 1 / (df["iat_cv"] + 1e-9)

        return df

    def _fit_aggregation_features(self, df: pd.DataFrame):
        """Learn flow aggregation statistics from training data."""
        logger.info("  Computing aggregation statistics...")

        # Use only source for aggregation
        src_col = self.group_cols_[0]

        df_sorted = df.sort_values(self.timestamp_col_)
        df_indexed = df_sorted.set_index(self.timestamp_col_)

        flow_counts = (
            df_indexed.groupby([src_col], observed=True)
            .resample(self.time_window)
            .size()
            .reset_index(name="flows_per_window")
        )

        flow_stats = (
            flow_counts.groupby([src_col], observed=True)["flows_per_window"]
            .agg(["mean", "std", "max", "sum"])
            .reset_index()
        )

        flow_stats["std"] = flow_stats["std"].fillna(0)

        self.statistics_["aggregation"] = flow_stats.set_index(src_col).to_dict("index")

        logger.info(
            f"    Stored stats for {len(self.statistics_['aggregation'])} sources"
        )

    def _transform_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned aggregation statistics."""
        src_col = self.group_cols_[0]

        # Create DataFrame from stored statistics
        stats_df = pd.DataFrame.from_dict(
            self.statistics_["aggregation"], orient="index"
        ).reset_index()
        stats_df = stats_df.rename(columns={"index": src_col})

        # Merge with data
        df = df.merge(stats_df, on=src_col, how="left", suffixes=("", "_agg"))

        # Rename columns
        df = df.rename(
            columns={
                "mean": "flow_count_mean",
                "std": "flow_count_std",
                "max": "flow_count_max",
                "sum": "flow_count_total",
            }
        )

        # Fill missing values
        fill_values = {
            "flow_count_mean": 0,
            "flow_count_std": 0,
            "flow_count_max": 0,
            "flow_count_total": 0,
        }
        df = df.fillna(fill_values)

        return df

    def _fit_entropy_features(self, df: pd.DataFrame):
        """Learn entropy statistics from training data."""
        logger.info("  Computing entropy statistics...")

        src_col = self.group_cols_[0]

        def compute_entropy(series):
            value_counts = series.value_counts(normalize=True)
            return stats.entropy(value_counts) if len(value_counts) > 0 else 0.0

        entropy_stats = {}

        # Port entropy
        port_cols = ["Dport", "dst_port"]
        port_col = next((c for c in port_cols if c in df.columns), None)

        if port_col:
            port_entropy = (
                df.groupby([src_col], observed=True)[port_col]
                .apply(compute_entropy)
                .to_dict()
            )
            entropy_stats["port"] = port_entropy

        # Protocol entropy
        proto_cols = ["Proto", "protocol"]
        proto_col = next((c for c in proto_cols if c in df.columns), None)

        if proto_col:
            proto_entropy = (
                df.groupby([src_col], observed=True)[proto_col]
                .apply(compute_entropy)
                .to_dict()
            )
            entropy_stats["protocol"] = proto_entropy

        self.statistics_["entropy"] = entropy_stats

        logger.info(f"    Stored entropy for {len(entropy_stats)} feature types")

    def _transform_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned entropy statistics."""
        src_col = self.group_cols_[0]

        if "port" in self.statistics_["entropy"]:
            df["dst_port_entropy"] = (
                df[src_col].map(self.statistics_["entropy"]["port"]).fillna(0.0)
            )

        if "protocol" in self.statistics_["entropy"]:
            df["protocol_entropy"] = (
                df[src_col].map(self.statistics_["entropy"]["protocol"]).fillna(0.0)
            )

        return df

    def _fit_consistency_features(self, df: pd.DataFrame):
        """Learn size consistency statistics from training data."""
        logger.info("  Computing consistency statistics...")

        src_col = self.group_cols_[0]
        consistency_stats = {}

        # Bytes consistency
        bytes_cols = ["TotBytes", "bytes_total"]
        bytes_col = next((c for c in bytes_cols if c in df.columns), None)

        if bytes_col:
            bytes_stats = (
                df.groupby([src_col], observed=True)[bytes_col]
                .agg(["mean", "std"])
                .reset_index()
            )
            bytes_stats["cv"] = bytes_stats["std"] / (bytes_stats["mean"] + 1e-9)
            bytes_stats["cv"] = (
                bytes_stats["cv"].replace([np.inf, -np.inf], 0).fillna(0)
            )
            consistency_stats["bytes"] = bytes_stats.set_index(src_col).to_dict("index")

        # Packets consistency
        pkts_cols = ["TotPkts", "total_packets"]
        pkts_col = next((c for c in pkts_cols if c in df.columns), None)

        if pkts_col:
            pkts_stats = (
                df.groupby([src_col], observed=True)[pkts_col]
                .agg(["mean", "std"])
                .reset_index()
            )
            pkts_stats["cv"] = pkts_stats["std"] / (pkts_stats["mean"] + 1e-9)
            pkts_stats["cv"] = pkts_stats["cv"].replace([np.inf, -np.inf], 0).fillna(0)
            consistency_stats["pkts"] = pkts_stats.set_index(src_col).to_dict("index")

        self.statistics_["consistency"] = consistency_stats

        logger.info(
            f"    Stored consistency for {len(consistency_stats)} feature types"
        )

    def _transform_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned consistency statistics."""
        src_col = self.group_cols_[0]

        if "bytes" in self.statistics_["consistency"]:
            bytes_df = pd.DataFrame.from_dict(
                self.statistics_["consistency"]["bytes"], orient="index"
            ).reset_index()
            bytes_df = bytes_df.rename(
                columns={
                    "index": src_col,
                    "mean": "bytes_mean",
                    "std": "bytes_std",
                    "cv": "bytes_cv",
                }
            )
            df = df.merge(bytes_df, on=src_col, how="left")
            df["bytes_mean"] = df["bytes_mean"].fillna(0)
            df["bytes_std"] = df["bytes_std"].fillna(0)
            df["bytes_cv"] = df["bytes_cv"].fillna(0)
            df["bytes_consistency_score"] = 1 / (df["bytes_cv"] + 1e-9)

        if "pkts" in self.statistics_["consistency"]:
            pkts_df = pd.DataFrame.from_dict(
                self.statistics_["consistency"]["pkts"], orient="index"
            ).reset_index()
            pkts_df = pkts_df.rename(
                columns={
                    "index": src_col,
                    "mean": "pkts_mean",
                    "std": "pkts_std",
                    "cv": "pkts_cv",
                }
            )
            df = df.merge(pkts_df, on=src_col, how="left")
            df["pkts_mean"] = df["pkts_mean"].fillna(0)
            df["pkts_std"] = df["pkts_std"].fillna(0)
            df["pkts_cv"] = df["pkts_cv"].fillna(0)
            df["pkts_consistency_score"] = 1 / (df["pkts_cv"] + 1e-9)

        return df
