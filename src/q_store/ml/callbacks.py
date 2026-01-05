"""
Training Callbacks for Quantum ML.

This module provides a comprehensive callback system for training,
including model checkpointing, logging, and experiment tracking.

Key Components:
    - Callback: Base callback class
    - ModelCheckpoint: Save best models during training
    - CSVLogger: Log metrics to CSV files
    - TensorBoardCallback: TensorBoard integration
    - MLflowCallback: MLflow experiment tracking
    - WandBCallback: Weights & Biases integration
    - ProgressCallback: Training progress tracking
    - LearningRateLogger: Track learning rate changes

Example:
    >>> from q_store.ml.callbacks import ModelCheckpoint, CSVLogger, CallbackList
    >>>
    >>> # Create callbacks
    >>> checkpoint = ModelCheckpoint(
    ...     filepath='best_model.pkl',
    ...     monitor='val_loss',
    ...     mode='min',
    ...     save_best_only=True
    ... )
    >>> csv_logger = CSVLogger('training_log.csv')
    >>>
    >>> # Use in training
    >>> callbacks = CallbackList([checkpoint, csv_logger])
    >>> callbacks.on_train_begin()
    >>> for epoch in range(100):
    ...     callbacks.on_epoch_begin(epoch)
    ...     # ... training ...
    ...     callbacks.on_epoch_end(epoch, logs={'loss': 0.5, 'val_loss': 0.6})
"""

import logging
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for callbacks.

    Callbacks can hook into various stages of training:
    - on_train_begin/end
    - on_epoch_begin/end
    - on_batch_begin/end

    All callbacks should inherit from this class and override
    relevant methods.
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a batch."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.

    Args:
        callbacks: List of Callback instances

    Example:
        >>> callbacks = CallbackList([checkpoint, csv_logger, tensorboard])
        >>> callbacks.on_train_begin()
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_batch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_batch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Args:
        filepath: Path to save model (can include formatting like 'model_{epoch:02d}.pkl')
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min' or 'max' (default: 'min')
        save_best_only: Only save when monitored metric improves (default: False)
        save_weights_only: Only save model weights, not full model (default: False)
        verbose: Print messages (default: True)

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='models/best_model.pkl',
        ...     monitor='val_accuracy',
        ...     mode='max',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Save path
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Save only best models
            save_weights_only: Save only weights
            verbose: Print status
        """
        super().__init__()

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        # Create directory if needed
        filepath_dir = os.path.dirname(filepath)
        if filepath_dir and not os.path.exists(filepath_dir):
            os.makedirs(filepath_dir, exist_ok=True)

        # Initialize best value
        if mode == 'min':
            self.best_value = np.inf
            self.monitor_op = np.less
        else:
            self.best_value = -np.inf
            self.monitor_op = np.greater

        logger.info(
            f"Initialized ModelCheckpoint: filepath={filepath}, "
            f"monitor={monitor}, mode={mode}"
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Save model at end of epoch if conditions are met.

        Args:
            epoch: Current epoch
            logs: Metrics dictionary
        """
        logs = logs or {}

        # Format filepath with epoch
        filepath = self.filepath.format(epoch=epoch, **logs)

        # Get current metric value
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose:
                logger.warning(
                    f"ModelCheckpoint: {self.monitor} not found in logs. "
                    f"Available metrics: {list(logs.keys())}"
                )
            return

        # Check if we should save
        if self.save_best_only:
            if self.monitor_op(current, self.best_value):
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch}: {self.monitor} improved from "
                        f"{self.best_value:.5f} to {current:.5f}, saving model to {filepath}"
                    )
                self.best_value = current
                self._save_model(filepath, logs)
            else:
                if self.verbose:
                    logger.debug(
                        f"Epoch {epoch}: {self.monitor} did not improve from {self.best_value:.5f}"
                    )
        else:
            if self.verbose:
                logger.info(f"Epoch {epoch}: saving model to {filepath}")
            self._save_model(filepath, logs)

    def _save_model(self, filepath: str, logs: Dict[str, Any]):
        """
        Save model to file.

        Args:
            filepath: Path to save to
            logs: Metrics to save with model
        """
        # Get model from logs if available
        model = logs.get('model')

        if model is None:
            logger.warning("No model found in logs to save")
            return

        try:
            # Save based on save_weights_only flag
            if self.save_weights_only:
                if hasattr(model, 'get_weights'):
                    weights = model.get_weights()
                    self._save_weights(filepath, weights, logs)
                else:
                    logger.warning("Model does not have get_weights method")
            else:
                self._save_full_model(filepath, model, logs)

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _save_full_model(self, filepath: str, model: Any, logs: Dict[str, Any]):
        """Save full model."""
        import pickle

        checkpoint = {
            'model': model,
            'metrics': {k: v for k, v in logs.items() if k != 'model'},
            'timestamp': time.time()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def _save_weights(self, filepath: str, weights: Any, logs: Dict[str, Any]):
        """Save only model weights."""
        import pickle

        checkpoint = {
            'weights': weights,
            'metrics': {k: v for k, v in logs.items() if k != 'model'},
            'timestamp': time.time()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)


class CSVLogger(Callback):
    """
    Log training metrics to CSV file.

    Args:
        filename: CSV file path
        separator: CSV separator (default: ',')
        append: Append to existing file (default: False)

    Example:
        >>> csv_logger = CSVLogger('training_log.csv')
        >>> # Logs will be written to CSV after each epoch
    """

    def __init__(
        self,
        filename: str,
        separator: str = ',',
        append: bool = False
    ):
        """
        Initialize CSV logger.

        Args:
            filename: CSV file path
            separator: CSV separator
            append: Append mode
        """
        super().__init__()

        self.filename = filename
        self.separator = separator
        self.append = append
        self.writer = None
        self.csv_file = None
        self.keys = None

        # Create directory if needed
        filepath_dir = os.path.dirname(filename)
        if filepath_dir and not os.path.exists(filepath_dir):
            os.makedirs(filepath_dir, exist_ok=True)

        logger.info(f"Initialized CSVLogger: filename={filename}")

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Open CSV file at start of training."""
        mode = 'a' if self.append else 'w'
        self.csv_file = open(self.filename, mode, newline='')

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Write metrics to CSV at end of epoch."""
        logs = logs or {}

        # Filter out non-serializable values (like model)
        row_dict = {'epoch': epoch}
        for key, value in logs.items():
            if key != 'model' and isinstance(value, (int, float, str, bool)):
                row_dict[key] = value

        if self.keys is None:
            # First epoch - initialize writer with headers
            self.keys = list(row_dict.keys())
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=self.keys,
                delimiter=self.separator
            )
            self.writer.writeheader()

        # Write row
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Close CSV file at end of training."""
        if self.csv_file is not None:
            self.csv_file.close()
            logger.info(f"Training log saved to {self.filename}")


class ProgressCallback(Callback):
    """
    Display training progress.

    Args:
        print_freq: Print frequency in epochs (default: 1)
        metrics_to_display: List of metrics to display (default: None = all)
        verbose: Verbosity level (default: 1)

    Example:
        >>> progress = ProgressCallback(print_freq=5, verbose=1)
    """

    def __init__(
        self,
        print_freq: int = 1,
        metrics_to_display: Optional[List[str]] = None,
        verbose: int = 1
    ):
        """
        Initialize progress callback.

        Args:
            print_freq: Print frequency
            metrics_to_display: Metrics to display
            verbose: Verbosity level
        """
        super().__init__()

        self.print_freq = print_freq
        self.metrics_to_display = metrics_to_display
        self.verbose = verbose
        self.start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Record training start time."""
        self.start_time = time.time()
        if self.verbose:
            logger.info("Training started")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Record epoch start time."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Display progress at end of epoch."""
        logs = logs or {}

        if epoch % self.print_freq != 0:
            return

        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        # Filter metrics to display
        if self.metrics_to_display:
            metrics = {k: v for k, v in logs.items()
                      if k in self.metrics_to_display and k != 'model'}
        else:
            metrics = {k: v for k, v in logs.items() if k != 'model'}

        # Format metrics string
        metrics_str = ' - '.join([f'{k}: {v:.6f}' if isinstance(v, float)
                                 else f'{k}: {v}'
                                 for k, v in metrics.items()])

        if self.verbose:
            logger.info(
                f"Epoch {epoch} - {epoch_time:.2f}s - {metrics_str}"
            )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Display total training time."""
        if self.verbose and self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time:.2f}s")


class LearningRateLogger(Callback):
    """
    Log learning rate changes.

    Args:
        verbose: Print LR changes (default: True)

    Example:
        >>> lr_logger = LearningRateLogger(verbose=True)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize learning rate logger.

        Args:
            verbose: Print status
        """
        super().__init__()
        self.verbose = verbose
        self.lr_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log learning rate at end of epoch."""
        logs = logs or {}

        lr = logs.get('lr') or logs.get('learning_rate')

        if lr is not None:
            self.lr_history.append({'epoch': epoch, 'lr': lr})

            if self.verbose:
                logger.info(f"Epoch {epoch}: learning_rate = {lr:.6e}")

    def get_lr_history(self) -> List[Dict[str, float]]:
        """
        Get learning rate history.

        Returns:
            List of {epoch, lr} dictionaries
        """
        return self.lr_history


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback.

    Args:
        log_dir: TensorBoard log directory (default: './logs')
        write_graph: Write computational graph (default: False)
        update_freq: Update frequency - 'epoch' or batch number (default: 'epoch')

    Example:
        >>> tensorboard = TensorBoardCallback(log_dir='./runs/experiment1')
    """

    def __init__(
        self,
        log_dir: str = './logs',
        write_graph: bool = False,
        update_freq: Union[str, int] = 'epoch'
    ):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Log directory
            write_graph: Write graph
            update_freq: Update frequency
        """
        super().__init__()

        self.log_dir = log_dir
        self.write_graph = write_graph
        self.update_freq = update_freq
        self.writer = None

        # Try to import tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_available = True
            self.SummaryWriter = SummaryWriter
        except ImportError:
            logger.warning(
                "TensorBoard not available. "
                "Install with: pip install tensorboard"
            )
            self.tensorboard_available = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize TensorBoard writer."""
        if not self.tensorboard_available:
            return

        self.writer = self.SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard logging to {self.log_dir}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log metrics to TensorBoard."""
        if not self.tensorboard_available or self.writer is None:
            return

        logs = logs or {}

        for key, value in logs.items():
            if key != 'model' and isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")


class MLflowCallback(Callback):
    """
    MLflow experiment tracking callback.

    Args:
        tracking_uri: MLflow tracking server URI (default: None)
        experiment_name: Experiment name (default: 'quantum_ml')
        run_name: Run name (default: None)
        log_models: Log models to MLflow (default: False)

    Example:
        >>> mlflow_cb = MLflowCallback(
        ...     experiment_name='my_experiment',
        ...     run_name='run_001'
        ... )
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = 'quantum_ml',
        run_name: Optional[str] = None,
        log_models: bool = False
    ):
        """
        Initialize MLflow callback.

        Args:
            tracking_uri: Tracking URI
            experiment_name: Experiment name
            run_name: Run name
            log_models: Log models
        """
        super().__init__()

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_models = log_models
        self.run = None

        # Try to import mlflow
        try:
            import mlflow
            self.mlflow_available = True
            self.mlflow = mlflow
        except ImportError:
            logger.warning(
                "MLflow not available. "
                "Install with: pip install mlflow"
            )
            self.mlflow_available = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Start MLflow run."""
        if not self.mlflow_available:
            return

        logs = logs or {}

        # Set tracking URI
        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)

        # Set experiment
        self.mlflow.set_experiment(self.experiment_name)

        # Start run
        self.run = self.mlflow.start_run(run_name=self.run_name)

        # Log hyperparameters
        params = logs.get('params', {})
        if params:
            self.mlflow.log_params(params)

        logger.info(f"MLflow run started: {self.run.info.run_id}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log metrics to MLflow."""
        if not self.mlflow_available or self.run is None:
            return

        logs = logs or {}

        # Log metrics
        metrics = {k: v for k, v in logs.items()
                  if k != 'model' and isinstance(v, (int, float))}

        if metrics:
            self.mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """End MLflow run."""
        if not self.mlflow_available or self.run is None:
            return

        # Log final model if requested
        if self.log_models:
            model = logs.get('model') if logs else None
            if model:
                self.mlflow.log_artifact(model)

        self.mlflow.end_run()
        logger.info("MLflow run ended")


class WandBCallback(Callback):
    """
    Weights & Biases tracking callback.

    Args:
        project: W&B project name
        entity: W&B entity (username or team) (default: None)
        name: Run name (default: None)
        config: Configuration dictionary (default: None)

    Example:
        >>> wandb_cb = WandBCallback(
        ...     project='quantum-ml',
        ...     name='experiment_1',
        ...     config={'lr': 0.001, 'n_qubits': 4}
        ... )
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize W&B callback.

        Args:
            project: Project name
            entity: Entity name
            name: Run name
            config: Configuration
        """
        super().__init__()

        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}

        # Try to import wandb
        try:
            import wandb
            self.wandb_available = True
            self.wandb = wandb
        except ImportError:
            logger.warning(
                "Weights & Biases not available. "
                "Install with: pip install wandb"
            )
            self.wandb_available = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize W&B run."""
        if not self.wandb_available:
            return

        logs = logs or {}

        # Merge config from logs
        config = {**self.config, **logs.get('params', {})}

        # Initialize run
        self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=config
        )

        logger.info(f"W&B run initialized: {self.wandb.run.name}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log metrics to W&B."""
        if not self.wandb_available:
            return

        logs = logs or {}

        # Log metrics
        metrics = {k: v for k, v in logs.items()
                  if k != 'model' and isinstance(v, (int, float))}

        if metrics:
            self.wandb.log({**metrics, 'epoch': epoch})

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Finish W&B run."""
        if not self.wandb_available:
            return

        self.wandb.finish()
        logger.info("W&B run finished")


def create_callback(callback_type: str, **kwargs) -> Callback:
    """
    Convenience function to create a callback.

    Args:
        callback_type: Type of callback ('checkpoint', 'csv', 'tensorboard',
                      'mlflow', 'wandb', 'progress', 'lr_logger')
        **kwargs: Configuration parameters

    Returns:
        Callback instance

    Example:
        >>> checkpoint = create_callback('checkpoint', filepath='model.pkl')
        >>> csv = create_callback('csv', filename='log.csv')
        >>> mlflow = create_callback('mlflow', experiment_name='exp1')
    """
    callback_map = {
        'checkpoint': ModelCheckpoint,
        'csv': CSVLogger,
        'tensorboard': TensorBoardCallback,
        'mlflow': MLflowCallback,
        'wandb': WandBCallback,
        'progress': ProgressCallback,
        'lr_logger': LearningRateLogger,
    }

    if callback_type not in callback_map:
        raise ValueError(
            f"Unsupported callback type: {callback_type}. "
            f"Available types: {list(callback_map.keys())}"
        )

    return callback_map[callback_type](**kwargs)
