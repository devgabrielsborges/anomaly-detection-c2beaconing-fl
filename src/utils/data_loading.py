"""Reusable helpers for loading processed datasets with leak-aware filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_DROPS = [
    "StartTime",
    "SrcAddr",
    "DstAddr",
    "src_ip",
    "dst_ip",
    "Label",
    "label",
    "is_malicious",
    "scenario",
    "Timestamp",
    "timestamp",
    "Te",
]


@dataclass
class DataSchema:
    """Normalized description of how to load processed parquet splits."""

    dataset: str
    data_path: Path
    label_column: str
    features_to_drop: List[str]
    group_column: Optional[str]
    time_column: Optional[str]
    filters: Dict[str, Dict]
    split_column: str = "__split"


def normalize_data_section(
    raw_section: Optional[dict],
    *,
    fallback_dataset: str = "ctu13",
    fallback_path: str = "data/processed",
    fallback_label: Optional[str] = None,
) -> DataSchema:
    """Normalize a `data` config section into a :class:`DataSchema`."""

    section = dict(raw_section or {})
    dataset = section.get("dataset", fallback_dataset)
    data_path = Path(section.get("data_path", fallback_path))

    custom_drops = section.get("features_to_drop")
    if custom_drops is None:
        features_to_drop = list(dict.fromkeys(DEFAULT_FEATURE_DROPS))
    else:
        features_to_drop = list({*DEFAULT_FEATURE_DROPS, *custom_drops})

    if fallback_label:
        default_label = fallback_label
    else:
        default_label = "label" if dataset == "ctu13" else "binary_label"

    label_column = section.get("label_column", default_label)

    schema = DataSchema(
        dataset=dataset,
        data_path=data_path,
        label_column=label_column,
        features_to_drop=features_to_drop,
        group_column=section.get("group_column"),
        time_column=section.get("time_column"),
        filters=section.get("filters", {}),
        split_column=section.get("split_column", "__split"),
    )

    return schema


def load_splits(
    schema: DataSchema,
    splits: Sequence[str],
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Load and concatenate the requested dataset splits, applying filters."""

    frames: List[pd.DataFrame] = []
    for split in splits:
        split_path = schema.data_path / f"{schema.dataset}_{split}.parquet"
        if not split_path.exists():
            msg = f"Split '{split}' not found at {split_path}"
            if allow_missing:
                logger.warning(msg)
                continue
            raise FileNotFoundError(msg)

        df_split = pd.read_parquet(split_path)
        df_split[schema.split_column] = split
        frames.append(df_split)

    if not frames:
        raise FileNotFoundError(
            "None of the requested splits were available. "
            "Ensure preprocessing was executed or adjust data.splits."
        )

    df = pd.concat(frames, ignore_index=True)
    df = _apply_filters(df, schema)

    if schema.time_column and schema.time_column in df.columns:
        column = schema.time_column
        if not np.issubdtype(df[column].dtype, np.datetime64):
            df[column] = pd.to_datetime(df[column], errors="coerce", unit="s")

    return df.reset_index(drop=True)


def _apply_filters(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Run include/exclude/min/max filters defined in the schema."""

    filtered = df
    for column, rules in schema.filters.items():
        if column not in filtered.columns:
            raise KeyError(f"Filter column '{column}' not present in dataframe")

        include = rules.get("include")
        exclude = rules.get("exclude")
        min_value = rules.get("min")
        max_value = rules.get("max")

        if include is not None:
            filtered = filtered[filtered[column].isin(include)]
        if exclude is not None:
            filtered = filtered[~filtered[column].isin(exclude)]
        if min_value is not None:
            filtered = filtered[filtered[column] >= min_value]
        if max_value is not None:
            filtered = filtered[filtered[column] <= max_value]

    return filtered


def dataframe_to_arrays(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    feature_order: Optional[Sequence[str]] = None,
    return_groups: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """Convert a filtered dataframe into numpy arrays for modeling."""

    if feature_order is None:
        feature_cols = [
            col
            for col in df.columns
            if col not in schema.features_to_drop
            and col != schema.label_column
            and df[col].dtype in ["int64", "int32", "float64", "float32", "bool"]
        ]
    else:
        missing = [col for col in feature_order if col not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        feature_cols = list(feature_order)

    if not feature_cols:
        raise ValueError("No numeric feature columns available after filtering")

    X = df[feature_cols].values.astype(np.float32)
    y = df[schema.label_column].values.astype(np.int64)

    groups = None
    if return_groups and schema.group_column:
        if schema.group_column not in df.columns:
            raise KeyError(
                f"Group column '{schema.group_column}' not found in dataframe"
            )
        groups = df[schema.group_column].astype("category").cat.codes.values

    return X, y, feature_cols, groups
