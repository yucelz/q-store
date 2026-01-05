"""
Experiment Tracking and Monitoring for Quantum ML (v4.1.1).

This module provides comprehensive experiment tracking capabilities including:
- MLflow integration for experiment management
- Structured logging with QuantumMLLogger
- Enhanced metrics tracking and visualization

Example:
    >>> from q_store.ml.tracking import MLflowTracker, QuantumMLLogger
    >>>
    >>> # Initialize MLflow tracker
    >>> tracker = MLflowTracker(experiment_name='quantum_mnist')
    >>> tracker.start_run(run_name='baseline')
    >>>
    >>> # Log parameters and metrics
    >>> tracker.log_params({'n_qubits': 4, 'lr': 0.01})
    >>> tracker.log_metric('loss', 0.5, step=1)
"""

from .mlflow_tracker import (
    MLflowTracker,
    MLflowConfig,
)

__all__ = [
    'MLflowTracker',
    'MLflowConfig',
]
