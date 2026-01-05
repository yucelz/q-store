"""
Early Stopping and Convergence Detection for Quantum ML Training.

This module provides early stopping mechanisms and convergence detection
to prevent overfitting and identify training completion.

Key Components:
    - EarlyStopping: Stop training when monitored metric stops improving
    - ConvergenceDetector: Detect when training has converged or plateaued

Example:
    >>> from q_store.ml.early_stopping import EarlyStopping, ConvergenceDetector
    >>>
    >>> # Early stopping
    >>> early_stop = EarlyStopping(
    ...     patience=10,
    ...     min_delta=0.001,
    ...     mode='min',
    ...     restore_best_weights=True
    ... )
    >>>
    >>> for epoch in range(100):
    ...     loss = train_epoch()
    ...     if early_stop.should_stop(epoch, loss):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
"""

import logging
from typing import Optional, Union, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training when it stops improving
    for a specified number of epochs (patience).

    Args:
        patience: Number of epochs with no improvement to wait before stopping (default: 10)
        min_delta: Minimum change to qualify as an improvement (default: 0.0)
        mode: 'min' for metrics that should decrease (loss), 'max' for metrics
              that should increase (accuracy) (default: 'min')
        restore_best_weights: Whether to restore weights from best epoch (default: True)
        baseline: Baseline value for the monitored metric (default: None)
        verbose: Whether to print messages (default: True)

    Example:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop.should_stop(epoch, val_loss):
        ...         print("Training stopped early")
        ...         break
        >>> print(f"Best epoch: {early_stop.best_epoch}")
        >>> print(f"Best value: {early_stop.best_value}")
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        baseline: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum improvement threshold
            mode: 'min' or 'max'
            restore_best_weights: Restore best weights on stop
            baseline: Initial baseline value
            verbose: Print status messages
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

        if patience < 0:
            raise ValueError(f"Patience must be non-negative, got: {patience}")

        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got: {min_delta}")

        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.verbose = verbose

        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_value = None
        self.best_weights = None

        # Set comparison operators based on mode
        if mode == 'min':
            self.monitor_op = np.less
            self.best_value = np.inf if baseline is None else baseline
        else:  # mode == 'max'
            self.monitor_op = np.greater
            self.best_value = -np.inf if baseline is None else baseline

        logger.info(
            f"Initialized EarlyStopping: patience={patience}, "
            f"min_delta={min_delta}, mode={mode}"
        )

    def should_stop(
        self,
        epoch: int,
        current_value: float,
        weights: Optional[Any] = None
    ) -> bool:
        """
        Check if training should stop.

        Args:
            epoch: Current epoch number
            current_value: Current metric value
            weights: Current model weights (optional, for restoration)

        Returns:
            True if training should stop, False otherwise
        """
        # Check if current value is an improvement
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0

            # Store weights if restoration is enabled
            if self.restore_best_weights and weights is not None:
                self.best_weights = weights

            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.mode} improved to {current_value:.6f}"
                )

        else:
            self.wait += 1

            if self.verbose and self.wait > 0:
                logger.info(
                    f"Epoch {epoch}: No improvement for {self.wait} epochs "
                    f"(patience: {self.patience})"
                )

            # Check if patience exceeded
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch} with value: {self.best_value:.6f}"
                    )

                return True

        return False

    def _is_improvement(self, current_value: float) -> bool:
        """
        Check if current value is an improvement over best value.

        Args:
            current_value: Current metric value

        Returns:
            True if improved, False otherwise
        """
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            return current_value > (self.best_value + self.min_delta)

    def get_best_weights(self) -> Optional[Any]:
        """
        Get the best weights stored during training.

        Returns:
            Best weights or None if not available
        """
        return self.best_weights

    def reset(self):
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

        if self.mode == 'min':
            self.best_value = np.inf if self.baseline is None else self.baseline
        else:
            self.best_value = -np.inf if self.baseline is None else self.baseline

        self.best_weights = None

        if self.verbose:
            logger.info("Early stopping state reset")

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state as dictionary.

        Returns:
            State dictionary
        """
        return {
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
        }


class ConvergenceDetector:
    """
    Detect convergence and plateau in training metrics.

    Provides advanced detection methods including:
    - Plateau detection (metric stops changing)
    - Divergence detection (metric getting worse)
    - Oscillation detection (metric fluctuating)
    - Trend analysis

    Args:
        window_size: Number of epochs to analyze (default: 10)
        plateau_threshold: Threshold for plateau detection (default: 1e-4)
        divergence_threshold: Threshold for divergence detection (default: 0.1)
        mode: 'min' or 'max' (default: 'min')
        verbose: Whether to print messages (default: True)

    Example:
        >>> detector = ConvergenceDetector(window_size=10, plateau_threshold=1e-4)
        >>> for epoch in range(100):
        ...     loss = train_epoch()
        ...     detector.update(epoch, loss)
        ...     if detector.has_plateaued():
        ...         print("Training has plateaued")
        ...     if detector.is_diverging():
        ...         print("Training is diverging")
    """

    def __init__(
        self,
        window_size: int = 10,
        plateau_threshold: float = 1e-4,
        divergence_threshold: float = 0.1,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize convergence detector.

        Args:
            window_size: Size of analysis window
            plateau_threshold: Plateau detection threshold
            divergence_threshold: Divergence detection threshold
            mode: 'min' or 'max'
            verbose: Print status messages
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got: {window_size}")

        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.divergence_threshold = divergence_threshold
        self.mode = mode
        self.verbose = verbose

        # History of metric values
        self.history: List[float] = []
        self.epochs: List[int] = []

        logger.info(
            f"Initialized ConvergenceDetector: window_size={window_size}, "
            f"plateau_threshold={plateau_threshold}, mode={mode}"
        )

    def update(self, epoch: int, value: float):
        """
        Update with new metric value.

        Args:
            epoch: Current epoch
            value: Metric value
        """
        self.history.append(value)
        self.epochs.append(epoch)

        # Keep only last window_size * 2 values for efficiency
        max_history = self.window_size * 3
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            self.epochs = self.epochs[-max_history:]

    def has_plateaued(self) -> bool:
        """
        Check if metric has plateaued.

        Returns:
            True if plateaued, False otherwise
        """
        if len(self.history) < self.window_size:
            return False

        # Get recent window
        recent_values = self.history[-self.window_size:]

        # Calculate coefficient of variation (std / mean)
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)

        # Avoid division by zero
        if abs(mean_val) < 1e-10:
            cv = std_val
        else:
            cv = std_val / abs(mean_val)

        # Check if variation is below threshold
        is_plateau = cv < self.plateau_threshold

        if is_plateau and self.verbose:
            logger.info(
                f"Plateau detected: CV={cv:.6f} < threshold={self.plateau_threshold}"
            )

        return is_plateau

    def is_diverging(self) -> bool:
        """
        Check if metric is diverging (getting worse).

        Returns:
            True if diverging, False otherwise
        """
        if len(self.history) < self.window_size:
            return False

        # Get recent window
        recent_values = self.history[-self.window_size:]

        # Calculate trend
        trend = self._calculate_trend(recent_values)

        # Check divergence based on mode
        if self.mode == 'min':
            # For minimization, positive trend is bad
            is_diverging = trend > self.divergence_threshold
        else:  # mode == 'max'
            # For maximization, negative trend is bad
            is_diverging = trend < -self.divergence_threshold

        if is_diverging and self.verbose:
            logger.warning(
                f"Divergence detected: trend={trend:.6f}, "
                f"threshold={self.divergence_threshold}"
            )

        return is_diverging

    def is_oscillating(self, oscillation_threshold: float = 0.5) -> bool:
        """
        Check if metric is oscillating.

        Args:
            oscillation_threshold: Threshold for oscillation detection

        Returns:
            True if oscillating, False otherwise
        """
        if len(self.history) < self.window_size:
            return False

        # Get recent window
        recent_values = np.array(self.history[-self.window_size:])

        # Count direction changes
        diffs = np.diff(recent_values)
        direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # Calculate oscillation ratio
        max_changes = len(diffs) - 1
        oscillation_ratio = direction_changes / max_changes if max_changes > 0 else 0

        is_oscillating = oscillation_ratio > oscillation_threshold

        if is_oscillating and self.verbose:
            logger.info(
                f"Oscillation detected: ratio={oscillation_ratio:.2f}, "
                f"changes={direction_changes}/{max_changes}"
            )

        return is_oscillating

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend using linear regression.

        Args:
            values: List of values

        Returns:
            Slope of the trend line
        """
        n = len(values)
        if n < 2:
            return 0.0

        x = np.arange(n)
        y = np.array(values)

        # Linear regression: y = mx + b
        # m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        xy = x * y
        x_squared = x * x

        numerator = n * np.sum(xy) - np.sum(x) * np.sum(y)
        denominator = n * np.sum(x_squared) - np.sum(x) ** 2

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        return slope

    def get_trend_direction(self) -> str:
        """
        Get current trend direction.

        Returns:
            'improving', 'degrading', 'stable', or 'insufficient_data'
        """
        if len(self.history) < self.window_size:
            return 'insufficient_data'

        recent_values = self.history[-self.window_size:]
        trend = self._calculate_trend(recent_values)

        # Determine direction based on mode
        if abs(trend) < self.plateau_threshold:
            return 'stable'
        elif self.mode == 'min':
            return 'improving' if trend < 0 else 'degrading'
        else:  # mode == 'max'
            return 'improving' if trend > 0 else 'degrading'

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about convergence.

        Returns:
            Dictionary with statistics
        """
        if len(self.history) < self.window_size:
            return {
                'status': 'insufficient_data',
                'samples': len(self.history),
                'required': self.window_size
            }

        recent_values = np.array(self.history[-self.window_size:])

        return {
            'current_value': self.history[-1],
            'best_value': min(self.history) if self.mode == 'min' else max(self.history),
            'mean': float(np.mean(recent_values)),
            'std': float(np.std(recent_values)),
            'min': float(np.min(recent_values)),
            'max': float(np.max(recent_values)),
            'trend': float(self._calculate_trend(recent_values)),
            'trend_direction': self.get_trend_direction(),
            'has_plateaued': self.has_plateaued(),
            'is_diverging': self.is_diverging(),
            'is_oscillating': self.is_oscillating(),
            'samples_analyzed': len(recent_values)
        }

    def reset(self):
        """Reset convergence detector state."""
        self.history = []
        self.epochs = []

        if self.verbose:
            logger.info("Convergence detector state reset")


def create_early_stopping(
    strategy: str = 'patience',
    **kwargs
) -> Union[EarlyStopping, ConvergenceDetector]:
    """
    Convenience function to create early stopping or convergence detector.

    Args:
        strategy: 'patience' for EarlyStopping, 'convergence' for ConvergenceDetector
        **kwargs: Configuration parameters

    Returns:
        EarlyStopping or ConvergenceDetector instance

    Example:
        >>> early_stop = create_early_stopping('patience', patience=10, mode='min')
        >>> detector = create_early_stopping('convergence', window_size=15)
    """
    if strategy == 'patience':
        return EarlyStopping(**kwargs)
    elif strategy == 'convergence':
        return ConvergenceDetector(**kwargs)
    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            "Use 'patience' or 'convergence'"
        )
