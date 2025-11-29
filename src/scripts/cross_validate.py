"""K-Fold Cross-validation script for anomaly detection."""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # noqa: E402

from src.evaluation import compute_metrics  # noqa: E402
from src.models import NeuralNetworkModel, RandomForestModel, XGBoostModel  # noqa: E402
from src.utils.data_loading import (  # noqa: E402
    dataframe_to_arrays,
    load_splits,
    normalize_data_section,
)
from src.utils.mlflow_logger import MLflowLogger  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SplitterConfig:
    """Normalized configuration for CV split strategy."""

    strategy: str
    n_folds: int
    random_state: int
    shuffle: bool
    time_test_size: Optional[int]
    time_gap: int
    max_train_size: Optional[int]


def normalize_splitter_config(
    config: dict, group_column: Optional[str]
) -> SplitterConfig:
    """Normalize splitter configuration with sensible defaults."""

    splitter_cfg = config.get("splitter", {})
    strategy = splitter_cfg.get("strategy")

    if strategy is None:
        strategy = "stratified_group_kfold" if group_column else "stratified_kfold"

    n_folds = splitter_cfg.get("n_folds") or config.get("n_folds", 5)
    random_state = splitter_cfg.get("random_state") or config.get("random_state", 42)
    shuffle = splitter_cfg.get("shuffle", True)

    time_fraction = splitter_cfg.get("time_test_size_fraction")
    time_gap = splitter_cfg.get("time_gap", 0)
    max_train_size = splitter_cfg.get("max_train_size")

    return SplitterConfig(
        strategy=strategy,
        n_folds=n_folds,
        random_state=random_state,
        shuffle=shuffle,
        time_test_size=time_fraction,
        time_gap=time_gap,
        max_train_size=max_train_size,
    )


def build_splitter(
    splitter_cfg: SplitterConfig,
    y: np.ndarray,
    groups: Optional[np.ndarray],
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Instantiate appropriate splitter iterator."""

    if splitter_cfg.strategy == "stratified_group_kfold":
        if groups is None:
            raise ValueError("group_column must be provided for stratified_group_kfold")
        splitter = StratifiedGroupKFold(
            n_splits=splitter_cfg.n_folds,
            shuffle=splitter_cfg.shuffle,
            random_state=splitter_cfg.random_state,
        )
        return splitter.split(np.zeros(len(y)), y, groups)

    if splitter_cfg.strategy == "temporal":
        test_size = splitter_cfg.time_test_size
        if test_size is not None and test_size < 1:
            test_size = max(1, int(len(y) * test_size))
        splitter = TimeSeriesSplit(
            n_splits=splitter_cfg.n_folds,
            test_size=test_size,
            gap=splitter_cfg.time_gap,
            max_train_size=splitter_cfg.max_train_size,
        )
        indices = np.arange(len(y))
        return splitter.split(indices)

    splitter = StratifiedKFold(
        n_splits=splitter_cfg.n_folds,
        shuffle=splitter_cfg.shuffle,
        random_state=splitter_cfg.random_state,
    )
    return splitter.split(np.zeros(len(y)), y)


def setup_logging(level: str = "INFO", format_str: str = None):
    """Set up logging configuration."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler()],
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, model_config: dict, input_dim: int = None):
    """Create model instance."""
    if model_type == "random_forest":
        return RandomForestModel(**model_config["random_forest"])
    elif model_type == "xgboost":
        return XGBoostModel(**model_config["xgboost"])
    elif model_type == "neural_network":
        if input_dim is None:
            raise ValueError("input_dim required for neural_network")
        return NeuralNetworkModel(input_dim=input_dim, **model_config["neural_network"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_cross_validation(config: dict):
    """Run k-fold cross-validation."""
    setup_logging(config.get("logging_level", "INFO"))
    logger.info("Starting k-fold cross-validation")

    data_section = dict(config.get("data", {}))
    data_section.setdefault("dataset", config.get("dataset", "ctu13"))
    data_section.setdefault("data_path", config.get("data_path", "data/processed"))
    splits = data_section.get("splits") or ["train", "val"]

    data_schema = normalize_data_section(
        data_section,
        fallback_dataset=config.get("dataset", "ctu13"),
        fallback_path=config.get("data_path", "data/processed"),
    )

    splitter_cfg = normalize_splitter_config(config, data_schema.group_column)

    logger.info(
        "Loading %s splits (%s) from %s",
        data_schema.dataset,
        ", ".join(splits),
        data_schema.data_path,
    )

    df = load_splits(data_schema, splits)

    # Temporal strategy requires deterministic ordering before feature extraction
    if splitter_cfg.strategy == "temporal":
        if not data_schema.time_column:
            raise ValueError(
                "time_column must be set in data config for temporal splitting"
            )
        if data_schema.time_column not in df.columns:
            raise KeyError(
                f"Time column '{data_schema.time_column}' not present in dataframe"
            )
        df = df.sort_values(data_schema.time_column).reset_index(drop=True)

    X_full, y_full, feature_names, groups = dataframe_to_arrays(
        df, data_schema, return_groups=True
    )

    # Apply sampling if specified
    sample_size = config.get("sample_size")
    if sample_size is not None and sample_size < len(X_full):
        if splitter_cfg.strategy == "temporal":
            logger.info(
                "Temporal split selected; taking first %d samples to preserve order",
                sample_size,
            )
            X_full = X_full[:sample_size]
            y_full = y_full[:sample_size]
            if groups is not None:
                groups = groups[:sample_size]
        else:
            logger.info(f"Sampling {sample_size} from {len(X_full)} total samples")
            rng = np.random.RandomState(splitter_cfg.random_state)
            sample_idx = rng.choice(len(X_full), size=sample_size, replace=False)
            X_full = X_full[sample_idx]
            y_full = y_full[sample_idx]
            if groups is not None:
                groups = groups[sample_idx]

    logger.info(f"Combined dataset shape: {X_full.shape}")
    logger.info(f"Class distribution: {np.bincount(y_full.astype(int))}")

    split_iterator = build_splitter(splitter_cfg, y_full, groups)

    logger.info(
        "Using %s with %d folds",
        splitter_cfg.strategy.replace("_", " ").title(),
        splitter_cfg.n_folds,
    )

    # Initialize MLflow logger
    mlflow_logger = None
    if config.get("use_mlflow", False):
        mlflow_logger = MLflowLogger(
            tracking_uri=config.get("mlflow_tracking_uri", "file:./mlflow"),
            experiment_name=config.get("experiment_name", "cross-validation"),
        )
        mlflow_logger.start_run()

    # Store results for each fold
    fold_results = []

    # Run cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
        logger.info(f"{'=' * 60}")

        # Split data
        X_fold_train, X_fold_val = X_full[train_idx], X_full[val_idx]
        y_fold_train, y_fold_val = y_full[train_idx], y_full[val_idx]

        logger.info(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")

        # Calculate class distribution
        train_neg_count = np.sum(y_fold_train == 0)
        train_pos_count = np.sum(y_fold_train == 1)
        val_neg_count = np.sum(y_fold_val == 0)
        val_pos_count = np.sum(y_fold_val == 1)

        logger.info(f"Train - Negative: {train_neg_count}, Positive: {train_pos_count}")
        logger.info(f"Val - Negative: {val_neg_count}, Positive: {val_pos_count}")

        # Create model for this fold
        model = create_model(
            config["model_type"],
            config["model"],
            input_dim=(
                X_fold_train.shape[1]
                if config["model_type"] == "neural_network"
                else None
            ),
        )

        # Train model
        logger.info(f"Training {config['model_type']} model...")
        model.fit(X_fold_train, y_fold_train)

        # Evaluate on validation fold
        logger.info("Evaluating on validation fold...")
        y_pred = model.predict(X_fold_val)
        y_pred_proba = (
            model.predict_proba(X_fold_val)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        # Compute metrics
        metrics = compute_metrics(y_fold_val, y_pred, y_pred_proba)

        # Log fold results
        logger.info(f"\nFold {fold_idx + 1} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Store results
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "train_size": len(X_fold_train),
                "val_size": len(X_fold_val),
                "val_pos_count": val_pos_count,
                "val_neg_count": val_neg_count,
            }
        )

    # Calculate summary statistics
    df_results = pd.DataFrame(fold_results)

    logger.info(f"\n{'=' * 60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'=' * 60}\n")

    metrics_summary = {
        "accuracy_mean": df_results["accuracy"].mean(),
        "accuracy_std": df_results["accuracy"].std(),
        "precision_mean": df_results["precision"].mean(),
        "precision_std": df_results["precision"].std(),
        "recall_mean": df_results["recall"].mean(),
        "recall_std": df_results["recall"].std(),
        "f1_mean": df_results["f1"].mean(),
        "f1_std": df_results["f1"].std(),
        "roc_auc_mean": df_results["roc_auc"].mean(),
        "roc_auc_std": df_results["roc_auc"].std(),
    }

    logger.info("Mean ± Std across folds:")
    logger.info(
        f"  Accuracy:  {metrics_summary['accuracy_mean']:.4f} ± {metrics_summary['accuracy_std']:.4f}"
    )
    logger.info(
        f"  Precision: {metrics_summary['precision_mean']:.4f} ± {metrics_summary['precision_std']:.4f}"
    )
    logger.info(
        f"  Recall:    {metrics_summary['recall_mean']:.4f} ± {metrics_summary['recall_std']:.4f}"
    )
    logger.info(
        f"  F1 Score:  {metrics_summary['f1_mean']:.4f} ± {metrics_summary['f1_std']:.4f}"
    )
    logger.info(
        f"  ROC-AUC:   {metrics_summary['roc_auc_mean']:.4f} ± {metrics_summary['roc_auc_std']:.4f}"
    )

    # Calculate 95% confidence intervals
    logger.info("\n95% Confidence Intervals:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        mean = df_results[metric].mean()
        std = df_results[metric].std()
        ci_lower = mean - 1.96 * std / np.sqrt(splitter_cfg.n_folds)
        ci_upper = mean + 1.96 * std / np.sqrt(splitter_cfg.n_folds)
        logger.info(f"  {metric.capitalize():10s}: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Log to MLflow
    if mlflow_logger:
        # Log all fold results
        for fold_result in fold_results:
            for key, value in fold_result.items():
                if key != "fold":
                    mlflow_logger.log_metric(f"fold_{fold_result['fold']}_{key}", value)

        # Log summary statistics
        for key, value in metrics_summary.items():
            mlflow_logger.log_metric(key, value)

        # Log config
        mlflow_logger.log_params(config)

        mlflow_logger.end_run()

    # Save detailed results
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{config['model_type']}_cv_results.csv"
    df_results.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")

    # Save summary
    summary_file = output_dir / f"{config['model_type']}_cv_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Cross-Validation Summary - {config['model_type']}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Dataset: {data_schema.dataset}\n")
        f.write(f"Splits used: {', '.join(splits)}\n")
        f.write(f"Number of folds: {splitter_cfg.n_folds}\n")
        f.write(f"Total samples: {len(X_full)}\n")
        f.write(f"Class distribution: {np.bincount(y_full.astype(int))}\n\n")
        f.write("Mean ± Std across folds:\n")
        f.write(
            f"  Accuracy:  {metrics_summary['accuracy_mean']:.4f} ± {metrics_summary['accuracy_std']:.4f}\n"
        )
        f.write(
            f"  Precision: {metrics_summary['precision_mean']:.4f} ± {metrics_summary['precision_std']:.4f}\n"
        )
        f.write(
            f"  Recall:    {metrics_summary['recall_mean']:.4f} ± {metrics_summary['recall_std']:.4f}\n"
        )
        f.write(
            f"  F1 Score:  {metrics_summary['f1_mean']:.4f} ± {metrics_summary['f1_std']:.4f}\n"
        )
        f.write(
            f"  ROC-AUC:   {metrics_summary['roc_auc_mean']:.4f} ± {metrics_summary['roc_auc_std']:.4f}\n"
        )

    logger.info(f"Summary saved to: {summary_file}")

    return metrics_summary, df_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-validation for anomaly detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_cross_validation(config)


if __name__ == "__main__":
    main()
