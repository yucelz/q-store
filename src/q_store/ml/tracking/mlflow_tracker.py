"""
MLflow Integration for Quantum ML Experiment Tracking.

This module provides a comprehensive interface to MLflow for tracking quantum ML experiments,
including parameters, metrics, models, and artifacts.

Key Components:
    - MLflowTracker: Main interface for MLflow experiment tracking
    - MLflowConfig: Configuration for MLflow connection

Example:
    >>> from q_store.ml.tracking import MLflowTracker, MLflowConfig
    >>>
    >>> # Configure MLflow
    >>> config = MLflowConfig(
    ...     tracking_uri='http://localhost:5000',
    ...     experiment_name='quantum_experiments',
    ...     registry_uri=None
    ... )
    >>>
    >>> # Create tracker
    >>> tracker = MLflowTracker(config)
    >>> tracker.start_run(run_name='experiment_001')
    >>>
    >>> # Log parameters
    >>> tracker.log_params({
    ...     'n_qubits': 4,
    ...     'circuit_depth': 3,
    ...     'learning_rate': 0.01
    ... })
    >>>
    >>> # Log metrics
    >>> for epoch in range(10):
    ...     tracker.log_metric('loss', loss_value, step=epoch)
    ...     tracker.log_metric('accuracy', acc_value, step=epoch)
    >>>
    >>> # End run
    >>> tracker.end_run()
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """
    Configuration for MLflow tracking.

    Args:
        tracking_uri: MLflow tracking server URI (default: file-based local)
        experiment_name: Name of the experiment (default: 'quantum_ml')
        registry_uri: Model registry URI (default: None, uses tracking_uri)
        artifact_location: Location to store artifacts (default: None)
        run_name_prefix: Prefix for run names (default: 'run')
        auto_log_models: Automatically log models (default: False)
        log_system_metrics: Log system metrics (CPU, memory) (default: True)
        tags: Default tags for all runs (default: {})

    Example:
        >>> config = MLflowConfig(
        ...     tracking_uri='http://mlflow-server:5000',
        ...     experiment_name='quantum_cifar10',
        ...     run_name_prefix='qnn'
        ... )
    """
    tracking_uri: str = './mlruns'
    experiment_name: str = 'quantum_ml'
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    run_name_prefix: str = 'run'
    auto_log_models: bool = False
    log_system_metrics: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


class MLflowTracker:
    """
    MLflow experiment tracker for quantum ML.

    Provides a high-level interface to MLflow for:
    - Experiment and run management
    - Parameter and metric logging
    - Model and artifact tracking
    - Tag and metadata management

    Args:
        config: MLflowConfig instance or None for defaults

    Example:
        >>> tracker = MLflowTracker()
        >>> tracker.start_run(run_name='baseline_experiment')
        >>> tracker.log_params({'n_qubits': 4, 'lr': 0.01})
        >>> tracker.log_metric('train_loss', 0.5, step=1)
        >>> tracker.log_artifact('model.pkl')
        >>> tracker.end_run()
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow tracker.

        Args:
            config: MLflowConfig instance
        """
        self.config = config or MLflowConfig()

        # Try to import MLflow
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            logger.warning(
                "MLflow not available. Install with: pip install mlflow\n"
                "Tracker will run in mock mode."
            )
            self.mlflow = None
            self.mlflow_available = False
            return

        # Set tracking URI
        self.mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set registry URI if specified
        if self.config.registry_uri:
            self.mlflow.set_registry_uri(self.config.registry_uri)

        # Get or create experiment
        self.experiment = self._get_or_create_experiment()

        # Current run
        self.active_run = None
        self.run_id = None

        logger.info(
            f"MLflow tracker initialized: experiment='{self.config.experiment_name}', "
            f"tracking_uri='{self.config.tracking_uri}'"
        )

    def _get_or_create_experiment(self):
        """Get or create MLflow experiment."""
        if not self.mlflow_available:
            return None

        try:
            experiment = self.mlflow.get_experiment_by_name(self.config.experiment_name)

            if experiment is None:
                # Create new experiment
                experiment_id = self.mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags=self.config.tags
                )
                experiment = self.mlflow.get_experiment(experiment_id)
                logger.info(f"Created new MLflow experiment: {self.config.experiment_name}")
            else:
                logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")

            return experiment

        except Exception as e:
            logger.error(f"Failed to get/create experiment: {e}")
            return None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> Optional[str]:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run (default: auto-generated)
            tags: Additional tags for the run
            nested: Whether this is a nested run

        Returns:
            Run ID if successful, None otherwise

        Example:
            >>> run_id = tracker.start_run(
            ...     run_name='experiment_001',
            ...     tags={'model_type': 'qnn', 'dataset': 'mnist'}
            ... )
        """
        if not self.mlflow_available:
            logger.warning("MLflow not available, cannot start run")
            return None

        try:
            # Merge tags
            run_tags = {**self.config.tags, **(tags or {})}

            # Start run
            self.active_run = self.mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=run_name,
                tags=run_tags,
                nested=nested
            )

            self.run_id = self.active_run.info.run_id

            logger.info(f"Started MLflow run: {run_name or self.run_id}")

            # Log system metrics if enabled
            if self.config.log_system_metrics:
                self._log_system_info()

            return self.run_id

        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            return None

    def end_run(self, status: str = 'FINISHED'):
        """
        End the current MLflow run.

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')

        Example:
            >>> tracker.end_run(status='FINISHED')
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id} with status {status}")

            self.active_run = None
            self.run_id = None

        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Example:
            >>> tracker.log_param('learning_rate', 0.01)
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameters

        Example:
            >>> tracker.log_params({
            ...     'n_qubits': 4,
            ...     'circuit_depth': 3,
            ...     'learning_rate': 0.01
            ... })
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number (e.g., epoch, iteration)

        Example:
            >>> tracker.log_metric('train_loss', 0.5, step=10)
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number

        Example:
            >>> tracker.log_metrics({
            ...     'train_loss': 0.5,
            ...     'val_loss': 0.6,
            ...     'accuracy': 0.85
            ... }, step=10)
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics at step {step}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file).

        Args:
            local_path: Local file path
            artifact_path: Path within artifact store

        Example:
            >>> tracker.log_artifact('model.pkl', 'models')
            >>> tracker.log_artifact('training_plot.png', 'plots')
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Local directory path
            artifact_path: Path within artifact store

        Example:
            >>> tracker.log_artifacts('outputs/', 'experiment_outputs')
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from directory: {local_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts from {local_dir}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = 'model',
        registered_model_name: Optional[str] = None
    ):
        """
        Log a model.

        Args:
            model: Model object to log
            artifact_path: Path within artifact store
            registered_model_name: Name for model registry

        Example:
            >>> tracker.log_model(
            ...     quantum_model,
            ...     artifact_path='quantum_model',
            ...     registered_model_name='qnn_v1'
            ... )
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            # Log as generic Python model
            import mlflow.pyfunc

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name
            )

            logger.info(f"Logged model to {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_figure(self, figure: Any, artifact_file: str):
        """
        Log a matplotlib/plotly figure.

        Args:
            figure: Figure object
            artifact_file: Filename for the saved figure

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [1, 4, 9])
            >>> tracker.log_figure(fig, 'training_curve.png')
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_figure(figure, artifact_file)
            logger.info(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the saved JSON

        Example:
            >>> tracker.log_dict(
            ...     {'config': {'lr': 0.01}, 'results': {'acc': 0.9}},
            ...     'experiment_config.json'
            ... )
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.log_dict(dictionary, artifact_file)
            logger.info(f"Logged dictionary: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log dictionary: {e}")

    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.

        Args:
            key: Tag key
            value: Tag value

        Example:
            >>> tracker.set_tag('model_version', 'v2.1')
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag {key}: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set multiple tags for the current run.

        Args:
            tags: Dictionary of tags

        Example:
            >>> tracker.set_tags({
            ...     'model_type': 'quantum_nn',
            ...     'dataset': 'mnist',
            ...     'optimizer': 'adam'
            ... })
        """
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            self.mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def _log_system_info(self):
        """Log system information as tags."""
        if not self.mlflow_available or self.active_run is None:
            return

        try:
            import platform
            import psutil

            system_info = {
                'system': platform.system(),
                'python_version': platform.python_version(),
                'cpu_count': str(psutil.cpu_count()),
                'memory_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
            }

            self.set_tags(system_info)

        except Exception as e:
            logger.debug(f"Could not log system info: {e}")

    def get_run_id(self) -> Optional[str]:
        """
        Get current run ID.

        Returns:
            Run ID or None if no active run
        """
        return self.run_id

    def get_experiment_id(self) -> Optional[str]:
        """
        Get experiment ID.

        Returns:
            Experiment ID or None if not available
        """
        return self.experiment.experiment_id if self.experiment else None

    @property
    def is_active(self) -> bool:
        """Check if there's an active run."""
        return self.active_run is not None

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status='FAILED')
        else:
            self.end_run(status='FINISHED')
