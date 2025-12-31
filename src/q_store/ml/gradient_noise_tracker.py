"""
Gradient Noise Tracker - v4.1 Enhanced
Monitors gradient statistics for training stability

Key Features:
- Rolling window gradient history
- Gradient norm tracking
- Gradient variance monitoring
- Signal-to-noise ratio (SNR) computation
- Automatic adaptation signals

Use Cases:
- Detect training instability
- Trigger adaptive measurement policies
- Inform SPSA sample count adjustments
- Identify barren plateaus
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradientStatistics:
    """
    Gradient statistics at a given step.

    Attributes
    ----------
    step : int
        Global training step
    gradient_norm : float
        L2 norm of gradient
    gradient_variance : float
        Variance across gradient dimensions
    gradient_snr : float
        Signal-to-noise ratio
    gradient_max : float
        Maximum gradient component
    gradient_min : float
        Minimum gradient component
    mean_gradient_magnitude : float
        Mean absolute gradient value
    """
    step: int
    gradient_norm: float
    gradient_variance: float
    gradient_snr: float
    gradient_max: float
    gradient_min: float
    mean_gradient_magnitude: float


class GradientNoiseTracker:
    """
    Track gradient statistics for training stability monitoring.

    Maintains a rolling window of gradient history and computes
    statistics useful for adaptive training.

    Parameters
    ----------
    window_size : int, default=100
        Size of rolling window for statistics
    compute_snr_window : int, default=10
        Window for SNR computation (must be <= window_size)

    Examples
    --------
    >>> tracker = GradientNoiseTracker(window_size=100)
    >>>
    >>> # During training
    >>> for step in range(1000):
    ...     gradient = compute_gradient(...)
    ...     stats = tracker.update(gradient, step)
    ...
    ...     if stats.gradient_snr < 1.0:
    ...         print("Warning: Low SNR, training may be unstable")
    ...
    ...     if stats.gradient_variance > 0.5:
    ...         print("High variance, consider increasing SPSA samples")
    """

    def __init__(
        self,
        window_size: int = 100,
        compute_snr_window: int = 10
    ):
        if compute_snr_window > window_size:
            raise ValueError(
                f"compute_snr_window ({compute_snr_window}) must be "
                f"<= window_size ({window_size})"
            )

        self.window_size = window_size
        self.compute_snr_window = compute_snr_window

        # Gradient history (deque for efficient rolling window)
        self.gradient_history: deque = deque(maxlen=window_size)

        # Statistics history
        self.stats_history: List[GradientStatistics] = []

        logger.info(
            f"Initialized GradientNoiseTracker: "
            f"window_size={window_size}, snr_window={compute_snr_window}"
        )

    def update(
        self,
        gradient: np.ndarray,
        step: int
    ) -> GradientStatistics:
        """
        Update tracker with new gradient and compute statistics.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient vector
        step : int
            Global training step

        Returns
        -------
        GradientStatistics
            Computed statistics for this step
        """
        # Add to history
        self.gradient_history.append(gradient.copy())

        # Compute statistics
        stats = self._compute_statistics(gradient, step)

        # Store statistics
        self.stats_history.append(stats)

        # Log warnings if needed
        self._check_stability(stats)

        return stats

    def _compute_statistics(
        self,
        gradient: np.ndarray,
        step: int
    ) -> GradientStatistics:
        """Compute gradient statistics."""

        # Basic statistics
        grad_norm = np.linalg.norm(gradient)
        grad_variance = np.var(gradient)
        grad_max = np.max(gradient)
        grad_min = np.min(gradient)
        mean_grad_magnitude = np.mean(np.abs(gradient))

        # Signal-to-noise ratio (needs history)
        if len(self.gradient_history) >= self.compute_snr_window:
            grad_snr = self._compute_snr()
        else:
            grad_snr = 0.0  # Not enough history

        return GradientStatistics(
            step=step,
            gradient_norm=grad_norm,
            gradient_variance=grad_variance,
            gradient_snr=grad_snr,
            gradient_max=grad_max,
            gradient_min=grad_min,
            mean_gradient_magnitude=mean_grad_magnitude
        )

    def _compute_snr(self) -> float:
        """
        Compute signal-to-noise ratio.

        SNR = |mean(gradient)| / std(gradient)

        Higher SNR indicates more reliable gradient direction.
        """
        # Get recent gradients
        recent_grads = np.array(list(self.gradient_history)[-self.compute_snr_window:])

        # Compute mean and std across time dimension
        grad_mean = np.mean(recent_grads, axis=0)
        grad_std = np.std(recent_grads, axis=0)

        # SNR per parameter
        snr_per_param = np.abs(grad_mean) / (grad_std + 1e-8)

        # Average SNR across parameters
        avg_snr = np.mean(snr_per_param)

        return float(avg_snr)

    def _check_stability(self, stats: GradientStatistics):
        """Check for training stability issues and log warnings."""

        # Low SNR warning
        if stats.gradient_snr > 0 and stats.gradient_snr < 0.5:
            logger.warning(
                f"Step {stats.step}: Low gradient SNR ({stats.gradient_snr:.3f}). "
                f"Training may be unstable. Consider increasing SPSA samples."
            )

        # High variance warning
        if stats.gradient_variance > 1.0:
            logger.warning(
                f"Step {stats.step}: High gradient variance "
                f"({stats.gradient_variance:.3f}). "
                f"Consider adaptive measurement or more shots."
            )

        # Vanishing gradients warning (potential barren plateau)
        if stats.gradient_norm < 1e-6:
            logger.warning(
                f"Step {stats.step}: Very small gradient norm "
                f"({stats.gradient_norm:.2e}). "
                f"Possible barren plateau!"
            )

    def get_recent_statistics(
        self,
        n_steps: int = 10
    ) -> Dict[str, float]:
        """
        Get statistics averaged over recent steps.

        Parameters
        ----------
        n_steps : int, default=10
            Number of recent steps to average

        Returns
        -------
        dict
            Averaged statistics
        """
        if not self.stats_history:
            return {}

        # Get recent stats
        recent_stats = self.stats_history[-n_steps:]

        if not recent_stats:
            return {}

        # Compute averages
        avg_stats = {
            'avg_gradient_norm': np.mean([s.gradient_norm for s in recent_stats]),
            'avg_gradient_variance': np.mean([s.gradient_variance for s in recent_stats]),
            'avg_gradient_snr': np.mean([s.gradient_snr for s in recent_stats]),
            'avg_gradient_magnitude': np.mean([s.mean_gradient_magnitude for s in recent_stats]),
            'n_steps': len(recent_stats)
        }

        return avg_stats

    def should_increase_samples(
        self,
        variance_threshold: float = 0.5,
        snr_threshold: float = 1.0
    ) -> bool:
        """
        Check if SPSA samples should be increased.

        Parameters
        ----------
        variance_threshold : float, default=0.5
            Variance threshold for triggering increase
        snr_threshold : float, default=1.0
            SNR threshold (increase if below)

        Returns
        -------
        bool
            True if samples should be increased
        """
        if not self.stats_history:
            return False

        recent = self.get_recent_statistics(n_steps=10)
        if not recent:
            return False

        # High variance or low SNR → increase samples
        high_variance = recent['avg_gradient_variance'] > variance_threshold
        low_snr = recent['avg_gradient_snr'] < snr_threshold and recent['avg_gradient_snr'] > 0

        return high_variance or low_snr

    def should_decrease_samples(
        self,
        variance_threshold: float = 0.1,
        snr_threshold: float = 3.0
    ) -> bool:
        """
        Check if SPSA samples can be decreased.

        Parameters
        ----------
        variance_threshold : float, default=0.1
            Variance threshold (decrease if below)
        snr_threshold : float, default=3.0
            SNR threshold (decrease if above)

        Returns
        -------
        bool
            True if samples can be decreased
        """
        if not self.stats_history:
            return False

        recent = self.get_recent_statistics(n_steps=10)
        if not recent:
            return False

        # Low variance and high SNR → can reduce samples
        low_variance = recent['avg_gradient_variance'] < variance_threshold
        high_snr = recent['avg_gradient_snr'] > snr_threshold

        return low_variance and high_snr

    def detect_plateau(
        self,
        window: int = 20,
        threshold: float = 1e-4
    ) -> bool:
        """
        Detect if training has plateaued (stagnant gradients).

        Parameters
        ----------
        window : int, default=20
            Window size for plateau detection
        threshold : float, default=1e-4
            Gradient norm threshold

        Returns
        -------
        bool
            True if plateau detected
        """
        if len(self.stats_history) < window:
            return False

        # Get recent gradient norms
        recent_norms = [s.gradient_norm for s in self.stats_history[-window:]]

        # Check if all norms are below threshold
        plateau = all(norm < threshold for norm in recent_norms)

        if plateau:
            logger.warning(
                f"Training plateau detected: gradient norm < {threshold} "
                f"for {window} consecutive steps"
            )

        return plateau

    def reset(self):
        """Reset all tracking history."""
        self.gradient_history.clear()
        self.stats_history.clear()
        logger.info("Reset GradientNoiseTracker")

    def get_summary(self) -> Dict[str, any]:
        """Get summary of tracked gradients."""
        if not self.stats_history:
            return {'total_steps': 0}

        all_norms = [s.gradient_norm for s in self.stats_history]
        all_variances = [s.gradient_variance for s in self.stats_history]
        all_snrs = [s.gradient_snr for s in self.stats_history if s.gradient_snr > 0]

        summary = {
            'total_steps': len(self.stats_history),
            'mean_gradient_norm': np.mean(all_norms),
            'std_gradient_norm': np.std(all_norms),
            'mean_variance': np.mean(all_variances),
            'mean_snr': np.mean(all_snrs) if all_snrs else 0.0,
            'min_gradient_norm': np.min(all_norms),
            'max_gradient_norm': np.max(all_norms),
        }

        return summary
