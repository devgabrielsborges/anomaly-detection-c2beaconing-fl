"""MLflow experiment tracking utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """
    Set up MLflow tracking.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking server URI. If None, uses local file store
        artifact_location: Path to store artifacts. If None, uses default

    Returns:
        Experiment ID
    """
    # Set tracking URI
    if tracking_uri is None:
        # Use local file store
        mlflow_dir = Path("mlflow")
        mlflow_dir.mkdir(exist_ok=True)
        tracking_uri = f"file://{mlflow_dir.absolute()}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Set experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
            logger.info(
                f"Created new experiment: {experiment_name} (ID: {experiment_id})"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment: {experiment_name} (ID: {experiment_id})"
            )
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        raise

    mlflow.set_experiment(experiment_name)
    return experiment_id


class MLflowLogger:
    """MLflow logging wrapper for federated learning experiments."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        nested: bool = False,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run. If None, MLflow generates one
            tracking_uri: MLflow tracking server URI
            nested: Whether this is a nested run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.nested = nested
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Set up experiment
        self.experiment_id = setup_mlflow(experiment_name, tracking_uri)

        # Start run
        self.run = None
        self.start_run()

    def start_run(self):
        """Start a new MLflow run."""
        if self.run is not None:
            logger.warning("Run already active. Ending previous run.")
            self.end_run()

        self.run = mlflow.start_run(
            run_name=self.run_name,
            experiment_id=self.experiment_id,
            nested=self.nested,
        )
        logger.info(f"Started MLflow run: {self.run.info.run_id}")
        return self.run

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.run is not None:
            mlflow.end_run(status=status)
            logger.info(
                f"Ended MLflow run: {self.run.info.run_id} with status {status}"
            )
            self.run = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Error logging parameter {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number for tracking metric evolution
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics at step {step}")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Error logging metric {key}: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file).

        Args:
            local_path: Path to the artifact
            artifact_path: Artifact path in MLflow
        """
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact {local_path}: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log a directory of artifacts.

        Args:
            local_dir: Path to the directory
            artifact_path: Artifact path in MLflow
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts from {local_dir}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = "sklearn",
        **kwargs,
    ):
        """
        Log a model.

        Args:
            model: The model to log
            artifact_path: Path to store the model
            model_type: Type of model (sklearn, pytorch, xgboost)
            **kwargs: Additional arguments for model logging
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path, **kwargs)
            else:
                # Generic logging
                mlflow.log_artifact(artifact_path)

            logger.info(f"Logged {model_type} model to {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
        """
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dictionary to {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging dictionary: {e}")

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Name of the artifact file
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure to {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")

    def log_text(self, text: str, artifact_file: str):
        """
        Log text as an artifact.

        Args:
            text: Text to log
            artifact_file: Name of the artifact file
        """
        try:
            mlflow.log_text(text, artifact_file)
            logger.debug(f"Logged text to {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging text: {e}")

    def set_tags(self, tags: Dict[str, Any]):
        """
        Set tags for the run.

        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Error setting tags: {e}")

    def set_tag(self, key: str, value: Any):
        """
        Set a single tag.

        Args:
            key: Tag name
            value: Tag value
        """
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Error setting tag {key}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
