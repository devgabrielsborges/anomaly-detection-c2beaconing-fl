"""Centralized training script for anomaly detection."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # noqa: E402

from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)  # noqa: E402, E501
from src.federated.data_loader import load_preprocessed_data  # noqa: E402
from src.models import NeuralNetworkModel, RandomForestModel, XGBoostModel  # noqa: E402
from src.utils.mlflow_logger import MLflowLogger  # noqa: E402

logger = logging.getLogger(__name__)


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
        # Calculate scale_pos_weight if not provided
        config = model_config["xgboost"].copy()
        return XGBoostModel(**config)
    elif model_type == "neural_network":
        config = model_config["neural_network"].copy()
        config["input_dim"] = input_dim
        return NeuralNetworkModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_centralized(config: dict):
    """Train model in centralized setting."""
    logger.info("=" * 80)
    logger.info("Starting Centralized Training")
    logger.info("=" * 80)

    # Create output directories
    output_dir = Path(config["output"]["model_dir"])
    results_dir = Path(config["output"]["results_dir"])
    plots_dir = Path(config["output"]["plots_dir"])

    output_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("Loading preprocessed data...")
    X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data(
        data_path=config["data"]["data_path"],
        dataset=config["data"]["dataset"],
        features_to_drop=config["data"].get("features_to_drop"),
    )

    # Sample data if requested (for quick testing)
    if "sample_size" in config.get("training", {}):
        sample_size = config["training"]["sample_size"]
        if sample_size < len(X_train):
            logger.info(f"Sampling {sample_size} examples for quick testing...")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            logger.info(f"Training set reduced to {len(X_train)} samples")

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    logger.info(f"Class distribution - Normal: {neg_count:,}, Anomaly: {pos_count:,}")
    logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

    # Apply class weights if needed
    if config["model"]["type"] == "xgboost":
        if config["model"]["xgboost"]["scale_pos_weight"] is None:
            config["model"]["xgboost"]["scale_pos_weight"] = scale_pos_weight
            logger.info(f"Applied scale_pos_weight: {scale_pos_weight:.2f}")

    elif config["model"]["type"] == "neural_network":
        if config["model"]["neural_network"]["class_weight"] is None:
            config["model"]["neural_network"]["class_weight"] = [
                1.0,
                scale_pos_weight,
            ]
            logger.info(f"Applied class_weight: [1.0, {scale_pos_weight:.2f}]")

    # Create model
    logger.info(f"Creating {config['model']['type']} model...")
    model = create_model(
        config["model"]["type"],
        config["model"],
        input_dim=X_train.shape[1],
    )

    # Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name=config["experiment"]["name"],
        run_name=f"{config['data']['dataset']}_{config['model']['type']}_centralized",
        tracking_uri=config["experiment"].get("tracking_uri"),
    )

    try:
        # Log experiment parameters
        mlflow_logger.log_params(
            {
                "dataset": config["data"]["dataset"],
                "model_type": config["model"]["type"],
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "class_imbalance_ratio": float(neg_count / pos_count),
                **model.get_params(),
            }
        )

        # Train model
        logger.info("Training model...")
        history = model.fit(X_train, y_train)

        # Save model
        model_path = (
            output_dir / f"{config['data']['dataset']}_{config['model']['type']}.pkl"
        )
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log metrics
        mlflow_logger.log_metrics(metrics)

        # Print metrics
        logger.info("=" * 80)
        logger.info("Test Set Results:")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"MCC: {metrics['mcc']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"G-Mean: {metrics['g_mean']:.4f}")
        logger.info("=" * 80)

        # Generate plots
        logger.info("Generating plots...")

        # Confusion matrix
        cm_fig = plot_confusion_matrix(y_test, y_pred, normalize=True)
        cm_path = (
            plots_dir
            / f"{config['data']['dataset']}_{config['model']['type']}_confusion_matrix.png"
        )
        cm_fig.savefig(cm_path, dpi=300, bbox_inches="tight")
        mlflow_logger.log_artifact(str(cm_path))

        # ROC curve
        roc_fig = plot_roc_curve(y_test, y_proba)
        roc_path = (
            plots_dir
            / f"{config['data']['dataset']}_{config['model']['type']}_roc_curve.png"
        )
        roc_fig.savefig(roc_path, dpi=300, bbox_inches="tight")
        mlflow_logger.log_artifact(str(roc_path))

        # PR curve
        pr_fig = plot_pr_curve(y_test, y_proba)
        pr_path = (
            plots_dir
            / f"{config['data']['dataset']}_{config['model']['type']}_pr_curve.png"
        )
        pr_fig.savefig(pr_path, dpi=300, bbox_inches="tight")
        mlflow_logger.log_artifact(str(pr_path))

        logger.info("Plots saved")

        # Log model to MLflow
        mlflow_logger.log_model(
            model.model,
            f"{config['model']['type']}_model",
            model_type="sklearn"
            if config["model"]["type"] in ["random_forest", "xgboost"]
            else "pytorch",
        )

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        mlflow_logger.end_run()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Centralized training for anomaly detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/centralized_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        level=config["logging"]["level"],
        format_str=config["logging"]["format"],
    )

    # Train
    train_centralized(config)


if __name__ == "__main__":
    main()
