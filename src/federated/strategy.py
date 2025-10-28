"""Federated learning strategies (FedAvg, FedProx)."""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Compute weighted average of metrics.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples

    Returns:
        Weighted average metrics
    """
    # Get total number of examples
    total_examples = sum([num_examples for num_examples, _ in metrics])

    # Weighted average for each metric
    aggregated = {}

    if total_examples == 0:
        return aggregated

    # Get all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())

    # Compute weighted average for each metric
    for key in all_keys:
        weighted_sum = sum(
            [num_examples * m.get(key, 0.0) for num_examples, m in metrics]
        )
        aggregated[key] = weighted_sum / total_examples

    return aggregated


class FedAvgStrategy(FedAvg):
    """Federated Averaging (FedAvg) strategy."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        """
        Initialize FedAvg strategy.

        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            evaluate_fn: Optional server-side evaluation function
            on_fit_config_fn: Function to generate fit config for each round
            on_evaluate_config_fn: Function to generate eval config
            accept_failures: Whether to accept failures from clients
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Metrics aggregation for training
            evaluate_metrics_aggregation_fn: Metrics aggregation for evaluation
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn or weighted_average,
            evaluate_metrics_aggregation_fn=(
                evaluate_metrics_aggregation_fn or weighted_average
            ),
        )

        logger.info("Initialized FedAvg strategy")


class FedProxStrategy(FedAvg):
    """Federated Proximal (FedProx) strategy with proximal term."""

    def __init__(
        self,
        proximal_mu: float = 0.1,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        """
        Initialize FedProx strategy.

        Args:
            proximal_mu: Proximal term coefficient (controls heterogeneity)
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            evaluate_fn: Optional server-side evaluation function
            on_fit_config_fn: Function to generate fit config for each round
            on_evaluate_config_fn: Function to generate eval config
            accept_failures: Whether to accept failures from clients
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Metrics aggregation for training
            evaluate_metrics_aggregation_fn: Metrics aggregation for evaluation
        """
        self.proximal_mu = proximal_mu

        # Create config function that includes proximal_mu
        original_on_fit_config_fn = on_fit_config_fn

        def on_fit_config_fn_with_mu(server_round: int) -> Dict[str, Scalar]:
            config = {}
            if original_on_fit_config_fn is not None:
                config = original_on_fit_config_fn(server_round)
            config["proximal_mu"] = self.proximal_mu
            return config

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn_with_mu,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn or weighted_average,
            evaluate_metrics_aggregation_fn=(
                evaluate_metrics_aggregation_fn or weighted_average
            ),
        )

        logger.info(f"Initialized FedProx strategy with mu={proximal_mu}")
