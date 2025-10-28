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

    def compute_periodicity_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute periodicity features for beaconing detection.

        C2 beaconing typically exhibits periodic behavior with regular
        intervals between connections.

        Args:
            df: DataFrame with flow data
            timestamp_col: Name of timestamp column
            group_cols: Columns to group by (e.g., src_ip, dst_ip)

        Returns:
            DataFrame with added periodicity features
        """
        logger.info("Computing periodicity features")

        if timestamp_col not in df.columns:
            logger.warning(
                f"Timestamp column '{timestamp_col}' not found. "
                "Skipping periodicity features."
            )
            return df

        if group_cols is None:
            # Default grouping
            possible_src = ["SrcAddr", "src_ip"]
            possible_dst = ["DstAddr", "dst_ip"]

            src_col = next((c for c in possible_src if c in df.columns), None)
            dst_col = next((c for c in possible_dst if c in df.columns), None)

            if src_col and dst_col:
                group_cols = [src_col, dst_col]
            else:
                logger.warning(
                    "No suitable grouping columns found. Skipping periodicity features."
                )
                return df

        # Sort by time
        df = df.sort_values(timestamp_col)

        # Compute inter-arrival times
        df["inter_arrival_time"] = (
            df.groupby(group_cols, observed=True)[timestamp_col]
            .diff()
            .dt.total_seconds()
        )

        # Aggregate periodicity metrics per group
        periodicity_stats = (
            df.groupby(group_cols, observed=True)["inter_arrival_time"]
            .agg(
                [
                    ("iat_mean", "mean"),
                    ("iat_std", "std"),
                    ("iat_median", "median"),
                    ("iat_min", "min"),
                    ("iat_max", "max"),
                    (
                        "iat_cv",
                        lambda x: x.std() / (x.mean() + 1e-9),
                    ),  # Coefficient of variation
                ]
            )
            .reset_index()
        )

        df = df.merge(periodicity_stats, on=group_cols, how="left")

        df["periodicity_score"] = 1 / (df["iat_cv"] + 1e-9)

        logger.info("Added periodicity features")

        return df

    def compute_flow_aggregation_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute aggregated flow statistics over time windows.

        Args:
            df: DataFrame with flow data
            timestamp_col: Name of timestamp column
            group_cols: Columns to group by

        Returns:
            DataFrame with aggregation features
        """
        logger.info("Computing flow aggregation features")

        if timestamp_col not in df.columns:
            logger.warning(
                f"Timestamp column '{timestamp_col}' not found. "
                "Skipping aggregation features."
            )
            return df

        if group_cols is None:
            possible_src = ["SrcAddr", "src_ip"]
            src_col = next((c for c in possible_src if c in df.columns), None)
            if src_col:
                group_cols = [src_col]
            else:
                logger.warning("No suitable grouping column found.")
                return df

        df = df.sort_values(timestamp_col)
        df_indexed = df.set_index(timestamp_col)

        flow_counts = (
            df_indexed.groupby(group_cols, observed=True)
            .resample(self.time_window)
            .size()
            .reset_index(name="flows_per_window")
        )

        flow_stats = (
            flow_counts.groupby(group_cols, observed=True)["flows_per_window"]
            .agg(
                [
                    ("flow_count_mean", "mean"),
                    ("flow_count_std", "std"),
                    ("flow_count_max", "max"),
                    ("flow_count_total", "sum"),
                ]
            )
            .reset_index()
        )

        df = df.merge(flow_stats, on=group_cols, how="left")

        logger.info("Added flow aggregation features")

        return df

    def compute_entropy_features(
        self, df: pd.DataFrame, group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute entropy-based features for behavioral analysis.

        Low entropy in destination ports/IPs suggests focused communication
        typical of C2 beaconing.

        Args:
            df: DataFrame with flow data
            group_cols: Columns to group by

        Returns:
            DataFrame with entropy features
        """
        logger.info("Computing entropy features")

        if group_cols is None:
            possible_src = ["SrcAddr", "src_ip"]
            src_col = next((c for c in possible_src if c in df.columns), None)
            if src_col:
                group_cols = [src_col]
            else:
                return df

        def compute_entropy(series):
            """Compute Shannon entropy of a series."""
            value_counts = series.value_counts(normalize=True)
            return stats.entropy(value_counts)

        # Destination port entropy
        port_cols = ["Dport", "dst_port"]
        port_col = next((c for c in port_cols if c in df.columns), None)

        if port_col:
            port_entropy = (
                df.groupby(group_cols, observed=True)[port_col]
                .apply(compute_entropy)
                .reset_index(name="dst_port_entropy")
            )
            df = df.merge(port_entropy, on=group_cols, how="left")

        # Protocol entropy
        if "Proto" in df.columns or "protocol" in df.columns:
            proto_col = "Proto" if "Proto" in df.columns else "protocol"
            proto_entropy = (
                df.groupby(group_cols, observed=True)[proto_col]
                .apply(compute_entropy)
                .reset_index(name="protocol_entropy")
            )
            df = df.merge(proto_entropy, on=group_cols, how="left")

        logger.info("Added entropy features")

        return df

    def compute_size_consistency_features(
        self, df: pd.DataFrame, group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute packet/byte size consistency features.

        C2 beaconing often has consistent packet sizes.

        Args:
            df: DataFrame with flow data
            group_cols: Columns to group by

        Returns:
            DataFrame with size consistency features
        """
        logger.info("Computing size consistency features")

        if group_cols is None:
            possible_src = ["SrcAddr", "src_ip"]
            src_col = next((c for c in possible_src if c in df.columns), None)
            if src_col:
                group_cols = [src_col]
            else:
                return df

        # Bytes consistency
        bytes_cols = ["TotBytes", "bytes_total"]
        bytes_col = next((c for c in bytes_cols if c in df.columns), None)

        if bytes_col:
            bytes_stats = (
                df.groupby(group_cols, observed=True)[bytes_col]
                .agg(
                    [
                        ("bytes_mean", "mean"),
                        ("bytes_std", "std"),
                        ("bytes_cv", lambda x: x.std() / (x.mean() + 1e-9)),
                    ]
                )
                .reset_index()
            )

            df = df.merge(bytes_stats, on=group_cols, how="left")

            df["bytes_consistency_score"] = 1 / (df["bytes_cv"] + 1e-9)

        pkts_cols = ["TotPkts", "total_packets"]
        pkts_col = next((c for c in pkts_cols if c in df.columns), None)

        if pkts_col:
            pkts_stats = (
                df.groupby(group_cols, observed=True)[pkts_col]
                .agg(
                    [
                        ("pkts_mean", "mean"),
                        ("pkts_std", "std"),
                        ("pkts_cv", lambda x: x.std() / (x.mean() + 1e-9)),
                    ]
                )
                .reset_index()
            )

            df = df.merge(pkts_stats, on=group_cols, how="left")
            df["pkts_consistency_score"] = 1 / (df["pkts_cv"] + 1e-9)

        logger.info("Added size consistency features")

        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        enable_periodicity: bool = True,
        enable_aggregation: bool = True,
        enable_entropy: bool = True,
        enable_consistency: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            enable_periodicity: Enable periodicity features
            enable_aggregation: Enable aggregation features
            enable_entropy: Enable entropy features
            enable_consistency: Enable consistency features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")

        original_cols = len(df.columns)

        if enable_periodicity:
            df = self.compute_periodicity_features(df, timestamp_col)

        if enable_aggregation:
            df = self.compute_flow_aggregation_features(df, timestamp_col)

        if enable_entropy:
            df = self.compute_entropy_features(df)

        if enable_consistency:
            df = self.compute_size_consistency_features(df)

        new_cols = len(df.columns) - original_cols
        logger.info(f"Feature engineering complete. Added {new_cols} features")

        return df
