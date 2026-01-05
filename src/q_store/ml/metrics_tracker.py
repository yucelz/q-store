"""
Enhanced Metrics Tracking for Quantum ML (v4.1.1).

This module provides comprehensive metrics tracking, aggregation, and analysis
for quantum machine learning experiments.

Key Components:
    - MetricsTracker: Track and aggregate training metrics
    - MetricHistory: Store and query metric history
    - MetricsAnalyzer: Analyze and visualize metrics

Example:
    >>> from q_store.ml.metrics_tracker import MetricsTracker
    >>>
    >>> # Create tracker
    >>> tracker = MetricsTracker()
    >>>
    >>> # Track metrics
    >>> tracker.add_metric('train_loss', 0.5, step=1)
    >>> tracker.add_metric('val_loss', 0.6, step=1)
    >>>
    >>> # Get statistics
    >>> stats = tracker.get_statistics('train_loss')
    >>> print(f"Mean: {stats['mean']}, Min: {stats['min']}")
"""

import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """
    Single metric entry.

    Attributes:
        name: Metric name
        value: Metric value
        step: Step number (epoch, iteration)
        timestamp: Unix timestamp
        metadata: Additional metadata
    """
    name: str
    value: float
    step: Optional[int] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricHistory:
    """
    Store and manage metric history.

    Provides efficient storage and retrieval of metric values over time.

    Example:
        >>> history = MetricHistory('train_loss')
        >>> history.add(0.5, step=1)
        >>> history.add(0.4, step=2)
        >>> print(history.get_latest())  # 0.4
        >>> print(history.get_all())  # [0.5, 0.4]
    """

    def __init__(self, name: str):
        """
        Initialize metric history.

        Args:
            name: Metric name
        """
        self.name = name
        self.values: List[float] = []
        self.steps: List[int] = []
        self.timestamps: List[float] = []

    def add(self, value: float, step: Optional[int] = None, timestamp: Optional[float] = None):
        """
        Add a metric value.

        Args:
            value: Metric value
            step: Step number
            timestamp: Unix timestamp
        """
        self.values.append(value)

        if step is not None:
            self.steps.append(step)
        else:
            self.steps.append(len(self.values) - 1)

        if timestamp is not None:
            self.timestamps.append(timestamp)
        else:
            import time
            self.timestamps.append(time.time())

    def get_latest(self) -> Optional[float]:
        """
        Get the latest value.

        Returns:
            Latest value or None if empty
        """
        return self.values[-1] if self.values else None

    def get_all(self) -> List[float]:
        """
        Get all values.

        Returns:
            List of all values
        """
        return self.values.copy()

    def get_at_step(self, step: int) -> Optional[float]:
        """
        Get value at specific step.

        Args:
            step: Step number

        Returns:
            Value at step or None if not found
        """
        try:
            idx = self.steps.index(step)
            return self.values[idx]
        except ValueError:
            return None

    def get_range(self, start_step: int, end_step: int) -> List[Tuple[int, float]]:
        """
        Get values in step range.

        Args:
            start_step: Start step (inclusive)
            end_step: End step (inclusive)

        Returns:
            List of (step, value) tuples
        """
        result = []
        for step, value in zip(self.steps, self.values):
            if start_step <= step <= end_step:
                result.append((step, value))
        return result

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary.

        Returns:
            Dictionary with mean, std, min, max, etc.
        """
        if not self.values:
            return {}

        values_array = np.array(self.values)

        return {
            'count': len(self.values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'latest': self.get_latest(),
        }

    def clear(self):
        """Clear all history."""
        self.values = []
        self.steps = []
        self.timestamps = []

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self.values)


class MetricsTracker:
    """
    Comprehensive metrics tracker for quantum ML.

    Tracks multiple metrics over time, provides aggregation,
    statistics, and analysis capabilities.

    Args:
        auto_compute_stats: Automatically compute statistics (default: True)
        history_limit: Maximum history entries per metric (0 = unlimited) (default: 0)

    Example:
        >>> tracker = MetricsTracker()
        >>>
        >>> # Add metrics
        >>> for epoch in range(100):
        ...     tracker.add_metric('train_loss', loss, step=epoch)
        ...     tracker.add_metric('val_loss', val_loss, step=epoch)
        >>>
        >>> # Get latest values
        >>> print(tracker.get_latest('train_loss'))
        >>>
        >>> # Get statistics
        >>> stats = tracker.get_statistics('train_loss')
        >>> print(f"Mean loss: {stats['mean']:.4f}")
        >>>
        >>> # Get all metrics
        >>> all_metrics = tracker.get_all_metrics()
    """

    def __init__(
        self,
        auto_compute_stats: bool = True,
        history_limit: int = 0
    ):
        """
        Initialize metrics tracker.

        Args:
            auto_compute_stats: Auto-compute statistics
            history_limit: Max history entries (0 = unlimited)
        """
        self.auto_compute_stats = auto_compute_stats
        self.history_limit = history_limit

        # Metric histories
        self.histories: Dict[str, MetricHistory] = {}

        # Cached statistics
        self._stats_cache: Dict[str, Dict[str, float]] = {}
        self._cache_valid: Dict[str, bool] = {}

        logger.info("MetricsTracker initialized")

    def add_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Step number
            timestamp: Unix timestamp
            metadata: Additional metadata

        Example:
            >>> tracker.add_metric('train_loss', 0.5, step=10)
            >>> tracker.add_metric('accuracy', 0.85, step=10, metadata={'phase': 'validation'})
        """
        # Create history if doesn't exist
        if name not in self.histories:
            self.histories[name] = MetricHistory(name)

        # Add to history
        self.histories[name].add(value, step, timestamp)

        # Invalidate cache
        self._cache_valid[name] = False

        # Enforce history limit
        if self.history_limit > 0:
            history = self.histories[name]
            if len(history) > self.history_limit:
                # Keep only last N entries
                history.values = history.values[-self.history_limit:]
                history.steps = history.steps[-self.history_limit:]
                history.timestamps = history.timestamps[-self.history_limit:]

    def add_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Add multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number for all metrics

        Example:
            >>> tracker.add_metrics({
            ...     'train_loss': 0.5,
            ...     'val_loss': 0.6,
            ...     'accuracy': 0.85
            ... }, step=10)
        """
        for name, value in metrics.items():
            self.add_metric(name, value, step=step)

    def get_latest(self, name: str) -> Optional[float]:
        """
        Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest value or None if metric not found

        Example:
            >>> latest_loss = tracker.get_latest('train_loss')
        """
        if name not in self.histories:
            return None
        return self.histories[name].get_latest()

    def get_history(self, name: str) -> Optional[MetricHistory]:
        """
        Get full history for a metric.

        Args:
            name: Metric name

        Returns:
            MetricHistory or None if not found
        """
        return self.histories.get(name)

    def get_all_values(self, name: str) -> List[float]:
        """
        Get all values for a metric.

        Args:
            name: Metric name

        Returns:
            List of values
        """
        if name not in self.histories:
            return []
        return self.histories[name].get_all()

    def get_statistics(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = tracker.get_statistics('train_loss')
            >>> print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        """
        if name not in self.histories:
            return {}

        # Check cache
        if self.auto_compute_stats and self._cache_valid.get(name, False):
            return self._stats_cache[name]

        # Compute statistics
        stats = self.histories[name].get_statistics()

        # Update cache
        if self.auto_compute_stats:
            self._stats_cache[name] = stats
            self._cache_valid[name] = True

        return stats

    def get_all_metrics(self) -> List[str]:
        """
        Get list of all tracked metric names.

        Returns:
            List of metric names
        """
        return list(self.histories.keys())

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary mapping metric names to statistics

        Example:
            >>> summary = tracker.get_metrics_summary()
            >>> for metric, stats in summary.items():
            ...     print(f"{metric}: mean={stats['mean']:.4f}")
        """
        summary = {}
        for name in self.histories.keys():
            summary[name] = self.get_statistics(name)
        return summary

    def compare_metrics(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """
        Compare two metrics.

        Args:
            metric1: First metric name
            metric2: Second metric name

        Returns:
            Comparison results

        Example:
            >>> comparison = tracker.compare_metrics('train_loss', 'val_loss')
            >>> print(f"Gap: {comparison['mean_diff']:.4f}")
        """
        if metric1 not in self.histories or metric2 not in self.histories:
            return {}

        stats1 = self.get_statistics(metric1)
        stats2 = self.get_statistics(metric2)

        if not stats1 or not stats2:
            return {}

        return {
            'mean_diff': stats1['mean'] - stats2['mean'],
            'std_diff': stats1['std'] - stats2['std'],
            'correlation': self._compute_correlation(metric1, metric2),
        }

    def _compute_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """Compute correlation between two metrics."""
        values1 = self.get_all_values(metric1)
        values2 = self.get_all_values(metric2)

        if len(values1) != len(values2) or len(values1) == 0:
            return None

        # Compute Pearson correlation
        return float(np.corrcoef(values1, values2)[0, 1])

    def get_best_value(self, name: str, mode: str = 'min') -> Optional[float]:
        """
        Get best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best value or None

        Example:
            >>> best_loss = tracker.get_best_value('train_loss', mode='min')
            >>> best_acc = tracker.get_best_value('accuracy', mode='max')
        """
        if name not in self.histories:
            return None

        values = self.histories[name].get_all()
        if not values:
            return None

        if mode == 'min':
            return float(np.min(values))
        else:
            return float(np.max(values))

    def get_improvement(self, name: str, window: int = 10) -> Optional[float]:
        """
        Get improvement over last N steps.

        Args:
            name: Metric name
            window: Window size for comparison

        Returns:
            Improvement value (negative = worse)

        Example:
            >>> improvement = tracker.get_improvement('train_loss', window=10)
            >>> if improvement < 0:
            ...     print("Performance degraded")
        """
        if name not in self.histories:
            return None

        values = self.histories[name].get_all()
        if len(values) < window:
            return None

        old_mean = np.mean(values[-2*window:-window])
        new_mean = np.mean(values[-window:])

        # For losses (lower is better), improvement is old - new
        # For accuracy (higher is better), improvement is new - old
        # Returning old - new as default (loss-like metric)
        return float(old_mean - new_mean)

    def reset_metric(self, name: str):
        """
        Reset a specific metric.

        Args:
            name: Metric name
        """
        if name in self.histories:
            self.histories[name].clear()
            self._cache_valid[name] = False
            logger.info(f"Reset metric: {name}")

    def reset_all(self):
        """Reset all metrics."""
        self.histories.clear()
        self._stats_cache.clear()
        self._cache_valid.clear()
        logger.info("Reset all metrics")

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all metrics to dictionary.

        Returns:
            Dictionary with all metric data

        Example:
            >>> data = tracker.export_to_dict()
            >>> import json
            >>> with open('metrics.json', 'w') as f:
            ...     json.dump(data, f)
        """
        export = {}
        for name, history in self.histories.items():
            export[name] = {
                'values': history.values,
                'steps': history.steps,
                'timestamps': history.timestamps,
                'statistics': self.get_statistics(name),
            }
        return export

    def import_from_dict(self, data: Dict[str, Any]):
        """
        Import metrics from dictionary.

        Args:
            data: Dictionary with metric data

        Example:
            >>> import json
            >>> with open('metrics.json', 'r') as f:
            ...     data = json.load(f)
            >>> tracker.import_from_dict(data)
        """
        for name, metric_data in data.items():
            history = MetricHistory(name)
            history.values = metric_data['values']
            history.steps = metric_data['steps']
            history.timestamps = metric_data['timestamps']
            self.histories[name] = history

        logger.info(f"Imported {len(data)} metrics")

    def __repr__(self) -> str:
        """String representation."""
        n_metrics = len(self.histories)
        total_entries = sum(len(h) for h in self.histories.values())
        return f"MetricsTracker({n_metrics} metrics, {total_entries} entries)"


class MetricsAnalyzer:
    """
    Analyze and visualize metrics.

    Provides analysis tools for metric trends, outliers, and patterns.

    Args:
        tracker: MetricsTracker instance

    Example:
        >>> analyzer = MetricsAnalyzer(tracker)
        >>> trend = analyzer.detect_trend('train_loss')
        >>> print(f"Trend: {trend}")  # 'improving', 'degrading', or 'stable'
    """

    def __init__(self, tracker: MetricsTracker):
        """
        Initialize metrics analyzer.

        Args:
            tracker: MetricsTracker to analyze
        """
        self.tracker = tracker

    def detect_trend(self, name: str, window: int = 10) -> str:
        """
        Detect trend in metric.

        Args:
            name: Metric name
            window: Window size for trend detection

        Returns:
            'improving', 'degrading', or 'stable'

        Example:
            >>> trend = analyzer.detect_trend('train_loss')
        """
        values = self.tracker.get_all_values(name)

        if len(values) < window:
            return 'insufficient_data'

        recent = values[-window:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        # For loss metrics (lower is better)
        if abs(slope) < 0.001:
            return 'stable'
        elif slope < 0:
            return 'improving'
        else:
            return 'degrading'

    def detect_overfitting(
        self,
        train_metric: str,
        val_metric: str,
        threshold: float = 0.1
    ) -> bool:
        """
        Detect potential overfitting.

        Args:
            train_metric: Training metric name
            val_metric: Validation metric name
            threshold: Gap threshold for overfitting

        Returns:
            True if overfitting detected

        Example:
            >>> is_overfitting = analyzer.detect_overfitting('train_loss', 'val_loss')
        """
        train_stats = self.tracker.get_statistics(train_metric)
        val_stats = self.tracker.get_statistics(val_metric)

        if not train_stats or not val_stats:
            return False

        # Check if validation is significantly worse than training
        gap = val_stats['latest'] - train_stats['latest']

        return gap > threshold

    def get_convergence_status(self, name: str, window: int = 20) -> Dict[str, Any]:
        """
        Get convergence status for a metric.

        Args:
            name: Metric name
            window: Window size for analysis

        Returns:
            Convergence status information

        Example:
            >>> status = analyzer.get_convergence_status('train_loss')
            >>> print(f"Converged: {status['converged']}")
        """
        values = self.tracker.get_all_values(name)

        if len(values) < window:
            return {'converged': False, 'reason': 'insufficient_data'}

        recent = values[-window:]
        variance = np.var(recent)

        # Check if variance is low (converged)
        converged = variance < 0.001

        return {
            'converged': converged,
            'variance': float(variance),
            'trend': self.detect_trend(name, window),
            'recent_mean': float(np.mean(recent)),
        }


def create_metrics_tracker(**kwargs) -> MetricsTracker:
    """
    Convenience function to create a MetricsTracker.

    Args:
        **kwargs: MetricsTracker parameters

    Returns:
        MetricsTracker instance

    Example:
        >>> tracker = create_metrics_tracker(history_limit=1000)
    """
    return MetricsTracker(**kwargs)
