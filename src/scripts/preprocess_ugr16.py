#!/usr/bin/env python3
"""
Preprocess UGR'16 dataset for anomaly detection.

Usage:
    python preprocess_ugr16.py [--input PATH] [--output PATH] [--advanced]
"""

import argparse
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.ugr16_preprocessor import UGR16Preprocessor  # noqa: E402
from preprocessing.feature_engineering import FeatureEngineer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess UGR'16 dataset for C2 beaconing detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/raw/ugr16_sample.parquet",
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
        "--advanced",
        action="store_true",
        help="Enable advanced C2 beaconing feature engineering",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("UGR'16 Dataset Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Validation size: {args.val_size}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Advanced features: {args.advanced}")
    logger.info("=" * 80)

    preprocessor = UGR16Preprocessor(
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

    if args.advanced:
        logger.info("Applying advanced feature engineering")
        engineer = FeatureEngineer(time_window="1H")

        preprocessor.df = engineer.engineer_features(
            preprocessor.df,
            timestamp_col="timestamp",
            enable_periodicity=True,
            enable_aggregation=True,
            enable_entropy=True,
            enable_consistency=True,
        )

    preprocessor.handle_missing_values(strategy="drop")

    train_df, val_df, test_df = preprocessor.split_data(
        preprocessor.df, label_col="binary_label"
    )

    prefix = "ugr16_advanced" if args.advanced else "ugr16"
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
