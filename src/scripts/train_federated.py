"""Leakage-aware federated training runner backed by Flower simulation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # noqa: E402

from src.federated.client import create_client_fn  # noqa: E402
from src.federated.data_loader import DataPartitioner  # noqa: E402
from src.federated.strategy import FedAvgStrategy, FedProxStrategy  # noqa: E402
from src.models import (  # noqa: E402
    AutoencoderModel,
    NeuralNetworkModel,
    RandomForestModel,
    XGBoostModel,
)
from src.utils.data_loading import (
    dataframe_to_arrays,
    load_splits,
    normalize_data_section,
)  # noqa: E402
from src.utils.mlflow_logger import MLflowLogger  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "neural_network": NeuralNetworkModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "autoencoder": AutoencoderModel,
}


def setup_logging(level: str = "INFO", format_str: str = None) -> None:
    """Configure logging formatting."""

    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler()],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_data_schema(config: dict) -> Tuple[dict, dict]:
    """Prepare normalized data schema and expose split selections."""

    data_section = dict(config.get("data", {}))
    default_dataset = data_section.get("dataset", "ctu13")
    default_path = data_section.get("data_path", "data/processed")

    data_section.setdefault("dataset", default_dataset)
    data_section.setdefault("data_path", default_path)

    train_splits = data_section.get("train_splits") or ["train"]

    schema = normalize_data_section(
        data_section,
        fallback_dataset=default_dataset,
        fallback_path=default_path,
    )

    return schema, {"train_splits": train_splits}


def compute_class_stats(labels: np.ndarray) -> Tuple[int, int, float]:
    """Return majority/minority counts and imbalance ratio."""

    neg = int(np.sum(labels == 0))
    pos = int(np.sum(labels == 1))
    pos = max(pos, 1)
    imbalance = neg / pos
    return neg, pos, imbalance


def prepare_model_params(
    config_model: Dict[str, dict],
    model_type: str,
    input_dim: int,
    imbalance_ratio: float,
) -> Tuple[type, Dict[str, any], int]:
    """Return model class, initialization kwargs, and epochs per round."""

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_cfg = config_model.get(model_type, {}).copy()

    if model_type == "neural_network":
        model_cfg = config_model.get("neural_network", {}).copy()
        model_cfg["input_dim"] = input_dim
        if model_cfg.get("class_weight") in (None, "auto"):
            model_cfg["class_weight"] = [1.0, imbalance_ratio]
        epochs_per_round = model_cfg.pop("epochs_per_round", 1)
    elif model_type == "autoencoder":
        model_cfg = config_model.get("autoencoder", {}).copy()
        model_cfg["input_dim"] = input_dim
        epochs_per_round = model_cfg.pop("epochs_per_round", 1)
    elif model_type == "xgboost":
        model_cfg = config_model.get("xgboost", {}).copy()
        if model_cfg.get("scale_pos_weight") in (None, "auto"):
            model_cfg["scale_pos_weight"] = imbalance_ratio
        epochs_per_round = 1
    else:
        model_cfg = config_model.get("random_forest", {}).copy()
        epochs_per_round = 1

    return MODEL_MAP[model_type], model_cfg, epochs_per_round


def build_strategy(strategy_name: str, fed_cfg: dict) -> fl.server.strategy.Strategy:
    """Instantiate configured Flower strategy."""

    common_kwargs = dict(
        fraction_fit=fed_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fed_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fed_cfg.get("min_fit_clients", 2),
        min_evaluate_clients=fed_cfg.get("min_evaluate_clients", 2),
        min_available_clients=fed_cfg.get("min_available_clients", 2),
    )

    if strategy_name.lower() == "fedprox":
        return FedProxStrategy(
            proximal_mu=fed_cfg.get("proximal_mu", 0.1),
            **common_kwargs,
        )

    return FedAvgStrategy(**common_kwargs)


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def run_simulation(config: dict) -> None:
    """Execute federated simulation using the provided configuration."""

    data_schema, split_cfg = resolve_data_schema(config)
    train_splits = split_cfg["train_splits"]

    logger.info(
        "Loading federated dataset %s (splits=%s) from %s",
        data_schema.dataset,
        ",".join(train_splits),
        data_schema.data_path,
    )

    train_df = load_splits(data_schema, train_splits)
    X, y, feature_names, _ = dataframe_to_arrays(train_df, data_schema)

    # Normalize features
    logger.info("Normalizing features using MinMaxScaler")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    neg, pos, imbalance = compute_class_stats(y)
    logger.info(
        "Global class distribution -> Normal: %d, Anomaly: %d (%.2f:1)",
        neg,
        pos,
        imbalance,
    )

    model_type = config["model"]["type"]
    model_class, model_params, epochs_per_round = prepare_model_params(
        config["model"], model_type, X.shape[1], imbalance
    )

    fed_cfg = config.get("federated", {})
    if not fed_cfg.get("use_simulation", True):
        raise NotImplementedError(
            "Process-based Flower deployments are not wired yet. Set use_simulation=true."
        )

    partitioner = DataPartitioner(
        num_clients=fed_cfg.get("num_clients", 5),
        partition_strategy=fed_cfg.get("partition_strategy", "non-iid-dirichlet"),
        alpha=fed_cfg.get("dirichlet_alpha", 0.5),
        random_state=config.get("training", {}).get("random_state", 42),
    )

    partitions = partitioner.partition(
        X,
        y,
        val_size=fed_cfg.get("client_val_size", 0.0),
    )

    logger.info("Prepared %d client partitions", len(partitions))

    client_fn = create_client_fn(
        model_class=model_class,
        model_params=model_params,
        data_partitions=partitions,
        epochs_per_round=epochs_per_round,
    )

    strategy = build_strategy(fed_cfg.get("strategy", "fedavg"), fed_cfg)
    server_config = fl.server.ServerConfig(num_rounds=fed_cfg.get("num_rounds", 10))

    # Setup MLflow
    mlflow_cfg = config.get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "federated_anomaly_detection")
    run_name = mlflow_cfg.get("run_name", f"fed_{model_type}")

    with MLflowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=mlflow_cfg.get("tracking_uri"),
    ) as mlflow_logger:
        # Log configuration
        mlflow_logger.log_params(flatten_dict(config))
        mlflow_logger.log_dict(config, "config.yaml")

        logger.info("Starting simulation...")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(partitions),
            config=server_config,
            strategy=strategy,
        )

        # Log distributed metrics (aggregated from clients)
        for metric_name, metric_values in history.metrics_distributed.items():
            for round_num, value in metric_values:
                mlflow_logger.log_metric(
                    f"distributed_{metric_name}", value, step=round_num
                )

        # Log distributed training metrics (aggregated from clients during fit)
        metrics_distributed_fit = getattr(history, "metrics_distributed_fit", {})
        for metric_name, metric_values in metrics_distributed_fit.items():
            for round_num, value in metric_values:
                mlflow_logger.log_metric(
                    f"distributed_{metric_name}", value, step=round_num
                )

        # Log distributed losses
        for round_num, loss in history.losses_distributed:
            mlflow_logger.log_metric("distributed_loss", loss, step=round_num)

        # Log centralized metrics (if any)
        for metric_name, metric_values in history.metrics_centralized.items():
            for round_num, value in metric_values:
                mlflow_logger.log_metric(
                    f"centralized_{metric_name}", value, step=round_num
                )

        # Log centralized losses
        for round_num, loss in history.losses_centralized:
            mlflow_logger.log_metric("centralized_loss", loss, step=round_num)

        centralized_metrics = getattr(history, "metrics_centralized", {})
        logger.info(
            "Federated run finished. Metrics keys: %s",
            list(centralized_metrics.keys()),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated training runner for anomaly detection",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to federated configuration YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logging(log_cfg.get("level", "INFO"), log_cfg.get("format"))

    run_simulation(config)


if __name__ == "__main__":
    main()
