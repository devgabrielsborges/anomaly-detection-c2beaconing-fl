#!/usr/bin/env python3
"""
Efficient reservoir sampling script for UGR'16 dataset using DuckDB.

This script uses DuckDB's built-in reservoir sampling which is highly optimized
and doesn't require loading the entire dataset into memory or reading all rows.

Usage:
    python reservoir_sample_ugr16.py
    python reservoir_sample_ugr16.py --sample-size 2800000 --seed 42
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd
import os
from dotenv import load_dotenv

try:
    import duckdb
except ImportError:
    print("ERROR: DuckDB not installed. Install with: pip install duckdb")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ReservoirSampler:
    """Implement efficient reservoir sampling using DuckDB."""

    def __init__(self, sample_size: int, random_seed: int = 42):
        """
        Initialize reservoir sampler.

        Args:
            sample_size: Number of rows to sample
            random_seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.random_seed = random_seed

    def sample_csv(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict:
        """
        Perform efficient reservoir sampling on a CSV file using DuckDB.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save parquet output

        Returns:
            Dictionary with sampling metadata
        """
        logger.info("=" * 80)
        logger.info("EFFICIENT RESERVOIR SAMPLING - UGR'16 Dataset")
        logger.info("=" * 80)
        logger.info(f"Input file: {input_path}")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Sample size: {self.sample_size:,}")
        logger.info(f"Random seed: {self.random_seed}")
        logger.info("")
        logger.info("Using DuckDB for optimized reservoir sampling...")
        logger.info("(No full pass required - efficient single-pass algorithm)")
        logger.info("")

        try:
            # Connect to DuckDB
            con = duckdb.connect(":memory:")

            # Set random seed
            con.execute(f"SELECT setseed({self.random_seed / (2**31 - 1)})")

            # Column names for UGR'16 dataset
            column_names = [
                "timestamp",
                "duration",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "flags",
                "tos",
                "packets_fwd",
                "packets_bwd",
                "bytes_total",
                "label",
            ]

            logger.info("Step 1: Getting dataset statistics...")

            # Get total row count efficiently
            count_query = f"""
                SELECT COUNT(*) as total_rows
                FROM read_csv_auto('{input_path}',
                    header=false,
                    names={column_names},
                    ignore_errors=true)
            """
            total_rows = con.execute(count_query).fetchone()[0]
            logger.info(f"  Total rows in dataset: {total_rows:,}")
            logger.info("")

            # Perform reservoir sampling using DuckDB's USING SAMPLE
            logger.info(f"Step 2: Reservoir sampling {self.sample_size:,} rows...")

            sample_query = f"""
                SELECT *
                FROM read_csv_auto('{input_path}',
                    header=false,
                    names={column_names},
                    ignore_errors=true)
                USING SAMPLE reservoir({self.sample_size} ROWS)
                REPEATABLE ({self.random_seed})
            """  # Execute sampling and convert to pandas
            sampled_df = con.execute(sample_query).df()

            logger.info(f"✓ Sampled {len(sampled_df):,} rows")
            logger.info("")

            # Save to parquet
            self._save_parquet(sampled_df, output_path)

            # Generate metadata
            metadata = self._generate_metadata(
                sampled_df, total_rows, input_path, output_path
            )

            # Close connection
            con.close()

            return metadata

        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            raise

    def _save_parquet(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save dataframe to parquet with compression."""
        logger.info("")
        logger.info(f"Saving to parquet: {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(
                output_path, engine="pyarrow", compression="snappy", index=False
            )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info("✓ Successfully saved parquet file")
            logger.info(f"  File size: {file_size_mb:.2f} MB")

            # Verification
            logger.info("")
            logger.info("Verifying saved file...")
            verification_df = pd.read_parquet(output_path, engine="pyarrow")
            logger.info(f"✓ Verification passed: {len(verification_df):,} rows")

        except Exception as e:
            logger.error(f"Failed to save parquet: {e}")
            raise

    def _generate_metadata(
        self, df: pd.DataFrame, total_rows: int, input_path: Path, output_path: Path
    ) -> dict:
        """Generate sampling metadata."""

        # Analyze label distribution
        malicious_keywords = [
            "botnet",
            "attack",
            "anomaly",
            "malicious",
            "ddos",
            "worm",
            "spam",
            "blacklist",
        ]
        pattern = "|".join(malicious_keywords)

        if "label" in df.columns:
            is_malicious = (
                df["label"].astype(str).str.contains(pattern, case=False, na=False)
            )
            n_malicious = is_malicious.sum()
            n_benign = (~is_malicious).sum()
            label_counts = df["label"].value_counts().to_dict()
        else:
            n_malicious = 0
            n_benign = len(df)
            label_counts = {}

        metadata = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "sampling_method": "reservoir_sampling",
            "random_seed": self.random_seed,
            "total_rows_in_source": total_rows,
            "sample_size": len(df),
            "sampling_ratio": len(df) / total_rows if total_rows > 0 else 0,
            "malicious_samples": int(n_malicious),
            "benign_samples": int(n_benign),
            "malicious_ratio": float(n_malicious / len(df)) if len(df) > 0 else 0,
            "columns": list(df.columns),
            "top_10_labels": (
                dict(list(label_counts.items())[:10]) if label_counts else {}
            ),
        }

        logger.info("")
        logger.info("=" * 80)
        logger.info("SAMPLING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Source rows:       {metadata['total_rows_in_source']:,}")
        logger.info(f"Sampled rows:      {metadata['sample_size']:,}")
        logger.info(f"Sampling ratio:    {metadata['sampling_ratio']:.2%}")
        logger.info("")
        logger.info(
            f"Malicious:         {metadata['malicious_samples']:,} "
            f"({metadata['malicious_ratio']:.2%})"
        )
        logger.info(f"Benign:            {metadata['benign_samples']:,}")
        logger.info("")
        logger.info("Top 10 labels in sample:")
        for label, count in list(metadata["top_10_labels"].items())[:10]:
            pct = (count / len(df)) * 100
            logger.info(f"  {label}: {count:,} ({pct:.2f}%)")
        logger.info("=" * 80)

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Reservoir sampling for UGR'16 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ugr-path",
        type=str,
        default=None,
        help="Path to UGR'16 CSV file (default: from UGR_PATH env var)",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/raw/ugr16_sample.parquet",
        help="Output path for parquet file (default: data/raw/ugr16_sample.parquet)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=2_800_000,
        help="Number of rows to sample (default: 2,800,000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Get UGR path from env if not provided
    ugr_path = args.ugr_path or os.getenv("UGR_PATH")

    if not ugr_path:
        logger.error("ERROR: UGR_PATH not provided!")
        logger.error("Either:")
        logger.error("  1. Set UGR_PATH in .env file")
        logger.error("  2. Use --ugr-path argument")
        sys.exit(1)

    ugr_path = Path(ugr_path)
    output_path = Path(args.output_path)

    # Validate input file exists
    if not ugr_path.exists():
        logger.error(f"ERROR: Input file not found: {ugr_path}")
        sys.exit(1)

    # Check file size
    file_size_mb = ugr_path.stat().st_size / (1024 * 1024)
    logger.info(f"Input file size: {file_size_mb:.2f} MB")
    logger.info("")

    # Create sampler and run
    sampler = ReservoirSampler(sample_size=args.sample_size, random_seed=args.seed)

    try:
        sampler.sample_csv(input_path=ugr_path, output_path=output_path)

        logger.info("")
        logger.info("✓ Sampling completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Update SAMPLE_PATH in .env to: {output_path}")
        logger.info("  2. Run EDA notebook: jupyter notebook notebooks/eda_ugr16.ipynb")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"✗ Sampling failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
