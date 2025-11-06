#!/usr/bin/env python3
"""
Create stratified parquet samples from CTU-13 and UGR'16 datasets.

This script generates properly-sampled subsets to avoid low variability
and ensure adequate representation of both benign and malicious traffic.

Usage:
    python create_dataset_samples.py [--ctu-path PATH] [--ugr-path PATH]
                                     [--output-dir PATH] [--sample-size N]
                                     [--min-malicious-ratio RATIO]
                                     [--seed SEED]
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetSampler:
    """Create stratified samples from network traffic datasets."""

    def __init__(
        self,
        sample_size: int = 2_800_000,
        min_malicious_ratio: float = 0.01,
        random_seed: int = 42,
    ):
        """
        Initialize sampler.

        Args:
            sample_size: Target number of rows in sample
            min_malicious_ratio: Minimum proportion of malicious samples
            random_seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.min_malicious_ratio = min_malicious_ratio
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def stratified_sample(
        self, df: pd.DataFrame, label_column: str, malicious_condition: callable
    ) -> pd.DataFrame:
        """
        Perform stratified sampling to ensure adequate malicious samples.

        Args:
            df: Input dataframe
            label_column: Column containing labels
            malicious_condition: Function that returns True for malicious rows

        Returns:
            Stratified sample dataframe
        """
        logger.info(f"Starting stratified sampling from {len(df):,} rows")

        # Identify malicious and benign samples
        malicious_mask = malicious_condition(df[label_column])
        df_malicious = df[malicious_mask].copy()
        df_benign = df[~malicious_mask].copy()

        n_malicious_orig = len(df_malicious)
        n_benign_orig = len(df_benign)

        logger.info("Original distribution:")
        mal_pct = n_malicious_orig / len(df) * 100
        ben_pct = n_benign_orig / len(df) * 100
        logger.info(f"  Malicious: {n_malicious_orig:,} ({mal_pct:.2f}%)")
        logger.info(f"  Benign: {n_benign_orig:,} ({ben_pct:.2f}%)")

        # If dataset is smaller than sample size, use all data
        if len(df) <= self.sample_size:
            logger.warning(
                f"Dataset size ({len(df):,}) ≤ sample size ({self.sample_size:,})"
            )
            logger.info("Using entire dataset without sampling")
            return df.copy()

        # Calculate sample sizes maintaining original ratio
        # but ensuring minimum malicious
        original_malicious_ratio = n_malicious_orig / len(df)
        target_malicious_ratio = max(original_malicious_ratio, self.min_malicious_ratio)

        # Calculate how many samples from each class
        n_malicious_sample = min(
            int(self.sample_size * target_malicious_ratio), n_malicious_orig
        )
        n_benign_sample = min(self.sample_size - n_malicious_sample, n_benign_orig)

        # Adjust if we can't get enough malicious samples
        target_n = int(self.sample_size * target_malicious_ratio)
        if n_malicious_sample < target_n:
            logger.warning(
                f"Only {n_malicious_sample:,} malicious samples available "
                f"(target: {target_n:,})"
            )
            # Use all malicious samples and fill rest with benign
            n_malicious_sample = n_malicious_orig
            n_benign_sample = min(self.sample_size - n_malicious_sample, n_benign_orig)

        logger.info("Target sample distribution:")
        total_sample = n_malicious_sample + n_benign_sample
        mal_pct_sample = n_malicious_sample / total_sample * 100
        ben_pct_sample = n_benign_sample / total_sample * 100
        logger.info(f"  Malicious: {n_malicious_sample:,} ({mal_pct_sample:.2f}%)")
        logger.info(f"  Benign: {n_benign_sample:,} ({ben_pct_sample:.2f}%)")

        # Sample from each class
        sampled_malicious = df_malicious.sample(
            n=n_malicious_sample, random_state=self.random_seed
        )
        sampled_benign = df_benign.sample(
            n=n_benign_sample, random_state=self.random_seed
        )

        # Combine and shuffle
        sampled_df = pd.concat([sampled_malicious, sampled_benign], ignore_index=True)
        sampled_df = sampled_df.sample(
            frac=1, random_state=self.random_seed
        ).reset_index(drop=True)

        logger.info(f"Final sample size: {len(sampled_df):,}")
        sample_ratio = len(sampled_df) / len(df) * 100
        logger.info(f"Sampling ratio: {sample_ratio:.2f}%")

        return sampled_df

    def sample_ctu13(
        self, input_path: Path, output_path: Path
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Sample CTU-13 dataset with stratification.

        Args:
            input_path: Path to CTU-13 binetflow file
            output_path: Path to save parquet sample

        Returns:
            Tuple of (sampled dataframe, metadata dict)
        """
        logger.info("=" * 80)
        logger.info("CTU-13 Dataset Sampling")
        logger.info("=" * 80)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        # Load data
        logger.info("Loading CTU-13 dataset...")
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to load CTU-13 dataset: {e}")
            raise

        # Check for required columns
        if "Label" not in df.columns:
            raise ValueError("CTU-13 dataset missing 'Label' column")

        # Define malicious condition (contains "Botnet")
        def is_botnet(labels):
            return labels.astype(str).str.contains("Botnet", case=False, na=False)

        # Perform stratified sampling
        sampled_df = self.stratified_sample(df, "Label", is_botnet)

        # Save to parquet
        self._save_parquet(sampled_df, output_path)

        # Generate metadata
        metadata = self._generate_metadata(sampled_df, "Label", is_botnet, "CTU-13")

        return sampled_df, metadata

    def sample_ugr16(
        self, input_path: Path, output_path: Path
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Sample UGR'16 dataset with DuckDB reservoir sampling.

        Uses DuckDB to avoid loading entire dataset into memory.

        Args:
            input_path: Path to UGR'16 CSV file
            output_path: Path to save parquet sample

        Returns:
            Tuple of (sampled dataframe, metadata dict)
        """
        logger.info("=" * 80)
        logger.info("UGR'16 Dataset Sampling")
        logger.info("=" * 80)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        try:
            import duckdb
        except ImportError:
            logger.error("DuckDB not installed. Install with: pip install duckdb")
            logger.error("Falling back to memory-intensive pandas method...")
            return self._sample_ugr16_pandas(input_path, output_path)

        logger.info("Using DuckDB for memory-efficient reservoir sampling...")

        try:
            # Connect to DuckDB
            con = duckdb.connect(":memory:")

            # Set random seed
            con.execute(f"SELECT setseed({self.random_seed / (2**31 - 1)})")

            # Column names for UGR'16
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

            logger.info("Getting dataset statistics...")

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

            # Perform reservoir sampling using DuckDB
            logger.info(
                f"Reservoir sampling {self.sample_size:,} rows "
                "(no full dataset load required)..."
            )

            sample_query = f"""
                SELECT *
                FROM read_csv_auto('{input_path}',
                    header=false,
                    names={column_names},
                    ignore_errors=true)
                USING SAMPLE reservoir({self.sample_size} ROWS)
                REPEATABLE ({self.random_seed})
            """

            # Execute sampling and convert to pandas
            sampled_df = con.execute(sample_query).df()

            logger.info(f"✓ Sampled {len(sampled_df):,} rows")
            logger.info(f"  Sampling ratio: {len(sampled_df) / total_rows * 100:.2f}%")

            # Close connection
            con.close()

            # Save to parquet
            self._save_parquet(sampled_df, output_path)

            # Define malicious condition for metadata
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

            def is_malicious(labels):
                pattern = "|".join(malicious_keywords)
                return labels.astype(str).str.contains(pattern, case=False, na=False)

            # Generate metadata
            metadata = self._generate_metadata(
                sampled_df, "label", is_malicious, "UGR16"
            )
            metadata["total_rows_in_source"] = total_rows
            metadata["sampling_method"] = "duckdb_reservoir"

            return sampled_df, metadata

        except Exception as e:
            logger.error(f"DuckDB sampling failed: {e}")
            raise

    def _sample_ugr16_pandas(
        self, input_path: Path, output_path: Path
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Fallback: Sample UGR'16 dataset with pandas (memory-intensive).

        Args:
            input_path: Path to UGR'16 CSV file
            output_path: Path to save parquet sample

        Returns:
            Tuple of (sampled dataframe, metadata dict)
        """
        logger.warning("Using pandas method - may require significant RAM!")

        # Load data
        logger.info("Loading UGR'16 dataset...")
        try:
            # UGR'16 doesn't have headers, need to add them
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
            df = pd.read_csv(input_path, names=column_names, low_memory=False)
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to load UGR'16 dataset: {e}")
            raise

        # Check for required columns
        if "label" not in df.columns:
            raise ValueError("UGR'16 dataset missing 'label' column")

        # Define malicious condition (contains attack keywords)
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

        def is_malicious(labels):
            pattern = "|".join(malicious_keywords)
            return labels.astype(str).str.contains(pattern, case=False, na=False)

        # Perform stratified sampling
        sampled_df = self.stratified_sample(df, "label", is_malicious)

        # Save to parquet
        self._save_parquet(sampled_df, output_path)

        # Generate metadata
        metadata = self._generate_metadata(sampled_df, "label", is_malicious, "UGR16")
        metadata["sampling_method"] = "pandas_stratified"

        return sampled_df, metadata

    def _save_parquet(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save dataframe to parquet with compression."""
        logger.info(f"Saving to {output_path}...")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(
                output_path, engine="pyarrow", compression="snappy", index=False
            )

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Successfully saved to {output_path}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")

            # Verification
            verification_df = pd.read_parquet(output_path, engine="pyarrow")
            logger.info(f"  Verification: Reloaded {len(verification_df):,} rows")

        except Exception as e:
            logger.error(f"Failed to save parquet: {e}")
            raise

    def _generate_metadata(
        self,
        df: pd.DataFrame,
        label_column: str,
        malicious_condition: callable,
        dataset_name: str,
    ) -> dict:
        """Generate metadata for sampled dataset."""
        malicious_mask = malicious_condition(df[label_column])
        n_malicious = malicious_mask.sum()
        n_benign = (~malicious_mask).sum()

        imbalance = None
        if n_malicious > 0:
            imbalance = float(n_benign / n_malicious)

        metadata = {
            "dataset": dataset_name,
            "sample_size": len(df),
            "random_seed": self.random_seed,
            "malicious_samples": int(n_malicious),
            "benign_samples": int(n_benign),
            "malicious_ratio": float(n_malicious / len(df)),
            "imbalance_ratio": imbalance,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        logger.info("=" * 80)
        logger.info("Sample Metadata:")
        logger.info(f"  Dataset: {metadata['dataset']}")
        logger.info(f"  Total samples: {metadata['sample_size']:,}")
        mal_ratio = metadata["malicious_ratio"] * 100
        logger.info(
            f"  Malicious: {metadata['malicious_samples']:,} ({mal_ratio:.2f}%)"
        )
        logger.info(f"  Benign: {metadata['benign_samples']:,}")
        if imbalance:
            logger.info(f"  Imbalance ratio: {imbalance:.2f}:1")
        logger.info("=" * 80)

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified parquet samples from CTU-13 and UGR'16 datasets"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ctu-path",
        type=str,
        help="Path to CTU-13 binetflow CSV file (overrides CTU_PATH env var)",
    )
    parser.add_argument(
        "--ugr-path",
        type=str,
        help="Path to UGR'16 CSV file (overrides UGR_PATH env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for parquet samples (default: data/raw)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2_800_000,
        help="Target sample size (default: 2,800,000)",
    )
    parser.add_argument(
        "--min-malicious-ratio",
        type=float,
        default=0.01,
        help="Minimum proportion of malicious samples (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument("--skip-ctu", action="store_true", help="Skip CTU-13 sampling")
    parser.add_argument("--skip-ugr", action="store_true", help="Skip UGR'16 sampling")

    args = parser.parse_args()

    # Load environment variables if not provided
    if args.ctu_path is None or args.ugr_path is None:
        try:
            from dotenv import load_dotenv
            import os

            load_dotenv()

            if args.ctu_path is None:
                args.ctu_path = os.getenv("CTU_PATH")
            if args.ugr_path is None:
                args.ugr_path = os.getenv("UGR_PATH")
        except ImportError:
            logger.warning("python-dotenv not available, using command line args only")

    # Validate inputs
    if not args.skip_ctu and not args.ctu_path:
        logger.error(
            "CTU-13 path not provided (use --ctu-path or set CTU_PATH env var)"
        )
        sys.exit(1)

    if not args.skip_ugr and not args.ugr_path:
        logger.error(
            "UGR'16 path not provided (use --ugr-path or set UGR_PATH env var)"
        )
        sys.exit(1)

    # Initialize sampler
    sampler = DatasetSampler(
        sample_size=args.sample_size,
        min_malicious_ratio=args.min_malicious_ratio,
        random_seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Sample CTU-13
    if not args.skip_ctu:
        ctu_input = Path(args.ctu_path)
        if not ctu_input.exists():
            logger.error(f"CTU-13 file not found: {ctu_input}")
            sys.exit(1)

        ctu_output = output_dir / "ctu13_sample.parquet"
        try:
            ctu_df, ctu_metadata = sampler.sample_ctu13(ctu_input, ctu_output)
            results["ctu13"] = {
                "success": True,
                "output_path": str(ctu_output),
                "metadata": ctu_metadata,
            }
        except Exception as e:
            logger.error(f"CTU-13 sampling failed: {e}", exc_info=True)
            results["ctu13"] = {"success": False, "error": str(e)}

    # Sample UGR'16
    if not args.skip_ugr:
        ugr_input = Path(args.ugr_path)
        if not ugr_input.exists():
            logger.error(f"UGR'16 file not found: {ugr_input}")
            sys.exit(1)

        ugr_output = output_dir / "ugr16_sample.parquet"
        try:
            ugr_df, ugr_metadata = sampler.sample_ugr16(ugr_input, ugr_output)
            results["ugr16"] = {
                "success": True,
                "output_path": str(ugr_output),
                "metadata": ugr_metadata,
            }
        except Exception as e:
            logger.error(f"UGR'16 sampling failed: {e}", exc_info=True)
            results["ugr16"] = {"success": False, "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLING COMPLETE")
    logger.info("=" * 80)

    for dataset, result in results.items():
        if result["success"]:
            logger.info(f"✓ {dataset.upper()}: {result['output_path']}")
        else:
            logger.error(f"✗ {dataset.upper()}: {result['error']}")

    logger.info("=" * 80)

    # Exit with error if any failed
    if any(not r["success"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
