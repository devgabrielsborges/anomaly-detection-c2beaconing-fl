#!/usr/bin/env python3
"""
Preprocess CTU-13 dataset for anomaly detection.

Usage:
    python preprocess_ctu13.py [--input PATH] [--output PATH]
"""

import argparse
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.ctu13_preprocessor import CTU13Preprocessor  # noqa: E402
from preprocessing.feature_engineering import FeatureEngineer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CTU-13 dataset for C2 beaconing detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/raw/ctu13_sample.parquet",
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/processed",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of training data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("CTU-13 Dataset Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Validation size: {args.val_size}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("=" * 80)

    preprocessor = CTU13Preprocessor(
        input_path=input_path,
        output_dir=output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_seed,
    )

    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.create_labels()
    preprocessor.extract_features()
    preprocessor.handle_missing_values(strategy="drop")

    logger.info("=" * 80)
    logger.info("SPLITTING DATA (before feature engineering to prevent leakage)")
    logger.info("=" * 80)
    train_df, val_df, test_df = preprocessor.split_data(
        preprocessor.df, label_col="label"
    )

    logger.info("=" * 80)
    logger.info(
        "APPLYING C2 BEACONING FEATURE ENGINEERING (with proper train/test separation)"
    )
    logger.info("=" * 80)
    engineer = FeatureEngineer(time_window="1H")

    # FIT on training data ONLY
    logger.info("Fitting feature engineer on TRAINING data...")
    train_df = engineer.fit_transform(
        train_df,
        timestamp_col="StartTime",
        enable_periodicity=True,
        enable_aggregation=True,
        enable_entropy=True,
        enable_consistency=True,
    )

    logger.info("Transforming VALIDATION data using training statistics...")
    val_df = engineer.transform(val_df)

    logger.info("Transforming TEST data using training statistics...")
    test_df = engineer.transform(test_df)

    logger.info("✓ Feature engineering completed without data leakage")

    prefix = "ctu13"
    preprocessor.save_processed_data(train_df, val_df, test_df, prefix=prefix)

    logger.info("=" * 80)
    logger.info("Preprocessing completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Train set: {len(train_df):,} samples")
    logger.info(f"Val set: {len(val_df):,} samples")
    logger.info(f"Test set: {len(test_df):,} samples")
    logger.info(f"Total features: {len(train_df.columns)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
