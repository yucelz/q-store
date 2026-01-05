"""
Structured Logging for Quantum ML Training (v4.1.1).

This module provides enhanced logging capabilities specifically designed for
quantum machine learning workflows, including structured logging, log levels,
and integration with tracking systems.

Key Components:
    - QuantumMLLogger: Structured logger for quantum ML
    - LogLevel: Log level enumeration
    - LogFormatter: Custom log formatting

Example:
    >>> from q_store.ml.logger import QuantumMLLogger, LogLevel
    >>>
    >>> # Create logger
    >>> logger = QuantumMLLogger(
    ...     name='quantum_training',
    ...     log_level=LogLevel.INFO,
    ...     log_file='training.log'
    ... )
    >>>
    >>> # Log messages
    >>> logger.info("Training started")
    >>> logger.metric("epoch_loss", 0.5, epoch=10)
    >>> logger.param("learning_rate", 0.01)
"""

import logging
import sys
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np


class LogLevel(Enum):
    """Log levels for quantum ML logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logs.

    Formats log records as JSON for easy parsing and analysis.
    """

    def __init__(self, include_timestamp: bool = True):
        """
        Initialize structured formatter.

        Args:
            include_timestamp: Include ISO timestamp in logs
        """
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        if self.include_timestamp:
            log_data['timestamp'] = datetime.utcnow().isoformat()

        # Add custom fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class QuantumMLLogger:
    """
    Structured logger for quantum ML experiments.

    Provides enhanced logging capabilities including:
    - Structured JSON logging
    - Metric and parameter tracking
    - Multiple output handlers (file, console)
    - Custom log levels

    Args:
        name: Logger name (default: 'quantum_ml')
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
        log_dir: Optional log directory
        structured: Use structured JSON logging (default: False)
        console_output: Enable console output (default: True)
        file_output: Enable file output (default: True)

    Example:
        >>> logger = QuantumMLLogger(
        ...     name='qnn_training',
        ...     log_level=LogLevel.INFO,
        ...     log_file='training.log',
        ...     structured=True
        ... )
        >>> logger.info("Starting training")
        >>> logger.metric("loss", 0.5, step=1)
        >>> logger.param("n_qubits", 4)
    """

    def __init__(
        self,
        name: str = 'quantum_ml',
        log_level: Union[LogLevel, int] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        structured: bool = False,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize quantum ML logger.

        Args:
            name: Logger name
            log_level: Logging level
            log_file: Log file path
            log_dir: Log directory
            structured: Use JSON formatting
            console_output: Enable console
            file_output: Enable file output
        """
        self.name = name
        self.structured = structured

        # Get or create logger
        self.logger = logging.getLogger(name)

        # Set log level
        if isinstance(log_level, LogLevel):
            self.logger.setLevel(log_level.value)
        else:
            self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers = []

        # Create formatter
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file_output:
            # Determine log file path
            if log_file:
                log_path = Path(log_file)
            elif log_dir:
                log_dir_path = Path(log_dir)
                log_dir_path.mkdir(parents=True, exist_ok=True)
                log_path = log_dir_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            else:
                log_path = Path(f"{name}.log")

            # Create file handler
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = str(log_path)
        else:
            self.log_file = None

        self.logger.info(f"QuantumMLLogger initialized: {name}")

    def debug(self, message: str, **kwargs):
        """
        Log debug message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """
        Log info message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """
        Log warning message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """
        Log error message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """
        Log critical message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(logging.CRITICAL, message, **kwargs)

    def metric(self, name: str, value: float, step: Optional[int] = None, **kwargs):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Step number (epoch, iteration)
            **kwargs: Additional metadata

        Example:
            >>> logger.metric("train_loss", 0.5, step=10, phase="training")
        """
        metric_data = {
            'type': 'metric',
            'metric_name': name,
            'value': float(value),
        }

        if step is not None:
            metric_data['step'] = step

        metric_data.update(kwargs)

        self._log(logging.INFO, f"Metric: {name}={value:.6f}", **metric_data)

    def param(self, name: str, value: Any, **kwargs):
        """
        Log a parameter.

        Args:
            name: Parameter name
            value: Parameter value
            **kwargs: Additional metadata

        Example:
            >>> logger.param("learning_rate", 0.01)
            >>> logger.param("n_qubits", 4, circuit_type="variational")
        """
        param_data = {
            'type': 'parameter',
            'param_name': name,
            'value': value,
        }
        param_data.update(kwargs)

        self._log(logging.INFO, f"Parameter: {name}={value}", **param_data)

    def params(self, params: Dict[str, Any], **kwargs):
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameters
            **kwargs: Additional metadata

        Example:
            >>> logger.params({
            ...     'n_qubits': 4,
            ...     'circuit_depth': 3,
            ...     'learning_rate': 0.01
            ... })
        """
        for name, value in params.items():
            self.param(name, value, **kwargs)

    def epoch_start(self, epoch: int, **kwargs):
        """
        Log epoch start.

        Args:
            epoch: Epoch number
            **kwargs: Additional metadata
        """
        epoch_data = {'type': 'epoch_start', 'epoch': epoch}
        epoch_data.update(kwargs)

        self._log(logging.INFO, f"Starting epoch {epoch}", **epoch_data)

    def epoch_end(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Log epoch end with metrics.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            **kwargs: Additional metadata

        Example:
            >>> logger.epoch_end(10, {
            ...     'train_loss': 0.5,
            ...     'val_loss': 0.6,
            ...     'accuracy': 0.85
            ... })
        """
        epoch_data = {'type': 'epoch_end', 'epoch': epoch}

        if metrics:
            epoch_data['metrics'] = metrics
            metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
            message = f"Epoch {epoch} complete: {metrics_str}"
        else:
            message = f"Epoch {epoch} complete"

        epoch_data.update(kwargs)

        self._log(logging.INFO, message, **epoch_data)

    def training_start(self, **kwargs):
        """
        Log training start.

        Args:
            **kwargs: Training metadata (epochs, batch_size, etc.)
        """
        train_data = {'type': 'training_start'}
        train_data.update(kwargs)

        self._log(logging.INFO, "Training started", **train_data)

    def training_end(self, **kwargs):
        """
        Log training end.

        Args:
            **kwargs: Final training statistics
        """
        train_data = {'type': 'training_end'}
        train_data.update(kwargs)

        self._log(logging.INFO, "Training completed", **train_data)

    def experiment_info(self, info: Dict[str, Any]):
        """
        Log experiment information.

        Args:
            info: Experiment metadata

        Example:
            >>> logger.experiment_info({
            ...     'experiment_name': 'qnn_mnist',
            ...     'model_type': 'variational_quantum_circuit',
            ...     'dataset': 'mnist',
            ...     'timestamp': '2025-01-05T10:30:00'
            ... })
        """
        exp_data = {'type': 'experiment_info'}
        exp_data.update(info)

        self._log(logging.INFO, "Experiment info", **exp_data)

    def system_info(self, info: Dict[str, Any]):
        """
        Log system information.

        Args:
            info: System metadata (CPU, memory, etc.)
        """
        sys_data = {'type': 'system_info'}
        sys_data.update(info)

        self._log(logging.INFO, "System info", **sys_data)

    def _log(self, level: int, message: str, **kwargs):
        """
        Internal logging method.

        Args:
            level: Log level
            message: Log message
            **kwargs: Extra fields
        """
        if self.structured and kwargs:
            # Add extra fields to log record
            extra = {'extra': kwargs}
            self.logger.log(level, message, extra=extra)
        else:
            self.logger.log(level, message)

    def get_log_file(self) -> Optional[str]:
        """
        Get log file path.

        Returns:
            Log file path or None
        """
        return self.log_file

    def set_level(self, level: Union[LogLevel, int]):
        """
        Set log level.

        Args:
            level: New log level

        Example:
            >>> logger.set_level(LogLevel.DEBUG)
        """
        if isinstance(level, LogLevel):
            self.logger.setLevel(level.value)
        else:
            self.logger.setLevel(level)


def create_logger(
    name: str = 'quantum_ml',
    log_level: Union[LogLevel, int, str] = LogLevel.INFO,
    **kwargs
) -> QuantumMLLogger:
    """
    Convenience function to create a QuantumMLLogger.

    Args:
        name: Logger name
        log_level: Log level (LogLevel enum, int, or string)
        **kwargs: Additional QuantumMLLogger parameters

    Returns:
        QuantumMLLogger instance

    Example:
        >>> logger = create_logger('my_experiment', 'DEBUG', log_file='exp.log')
    """
    # Convert string to LogLevel if needed
    if isinstance(log_level, str):
        log_level = LogLevel[log_level.upper()]

    return QuantumMLLogger(name=name, log_level=log_level, **kwargs)
