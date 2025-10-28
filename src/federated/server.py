"""Flower server implementation for federated learning."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class FlowerServer:
    """Flower server for federated learning."""

    def __init__(
        self,
        strategy: Strategy,
        num_rounds: int = 10,
        server_address: str = "[::]:8080",
    ):
        """
        Initialize Flower server.

        Args:
            strategy: Federated learning strategy
            num_rounds: Number of federated learning rounds
            server_address: Server address
        """
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.server_address = server_address

    def start(self):
        """Start the Flower server."""
        logger.info(f"Starting Flower server at {self.server_address}")
        logger.info(f"Number of rounds: {self.num_rounds}")

        # Configure server
        config = ServerConfig(num_rounds=self.num_rounds)

        # Start server
        fl.server.start_server(
            server_address=self.server_address,
            config=config,
            strategy=self.strategy,
        )

        logger.info("Flower server stopped")


def start_server(
    strategy: Strategy,
    num_rounds: int = 10,
    server_address: str = "[::]:8080",
) -> None:
    """
    Start Flower server.

    Args:
        strategy: Federated learning strategy
        num_rounds: Number of federated learning rounds
        server_address: Server address
    """
    server = FlowerServer(
        strategy=strategy,
        num_rounds=num_rounds,
        server_address=server_address,
    )
    server.start()
