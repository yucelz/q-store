"""
Performance Monitoring System for Quantum Backends.

Tracks execution metrics, costs, throughput, latency, and provides
visualization and reporting capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import time
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""
    backend_name: str
    circuit_qubits: int
    circuit_gates: int
    circuit_depth: int
    shots: int
    execution_time: float  # seconds
    queue_time: float = 0.0  # seconds
    cost: float = 0.0  # dollars or credits
    success: bool = True
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class BackendMetrics:
    """Aggregated metrics for a backend."""
    backend_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_shots: int = 0
    total_cost: float = 0.0
    total_execution_time: float = 0.0
    total_queue_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_queue_time: float = 0.0
    avg_cost_per_execution: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    p50_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0  # executions per second
    first_execution: Optional[datetime] = None
    last_execution: Optional[datetime] = None

    def update(self, metric: ExecutionMetrics):
        """Update aggregated metrics with new execution."""
        self.total_executions += 1

        if metric.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.total_shots += metric.shots
        self.total_cost += metric.cost
        self.total_execution_time += metric.execution_time
        self.total_queue_time += metric.queue_time

        # Update min/max
        self.min_execution_time = min(self.min_execution_time, metric.execution_time)
        self.max_execution_time = max(self.max_execution_time, metric.execution_time)

        # Update averages
        if self.total_executions > 0:
            self.avg_execution_time = self.total_execution_time / self.total_executions
            self.avg_queue_time = self.total_queue_time / self.total_executions
            self.avg_cost_per_execution = self.total_cost / self.total_executions
            self.success_rate = self.successful_executions / self.total_executions

        # Update timestamps
        if self.first_execution is None:
            self.first_execution = metric.timestamp
        self.last_execution = metric.timestamp

        # Calculate throughput
        if self.first_execution and self.last_execution:
            duration = (self.last_execution - self.first_execution).total_seconds()
            if duration > 0:
                self.throughput = self.total_executions / duration


@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    backend_name: str
    total_cost: float = 0.0
    cost_by_circuit_size: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    cost_by_shots: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    cost_over_time: List[Tuple[datetime, float]] = field(default_factory=list)

    def add_cost(self, cost: float, qubits: int, shots: int, timestamp: datetime):
        """Add a cost entry."""
        self.total_cost += cost
        self.cost_by_circuit_size[qubits] += cost
        self.cost_by_shots[shots] += cost
        self.cost_over_time.append((timestamp, cost))


@dataclass
class MetricsSummary:
    """Summary of all metrics."""
    total_backends: int
    total_executions: int
    total_successful: int
    total_failed: int
    total_cost: float
    total_shots: int
    overall_success_rate: float
    avg_execution_time: float
    total_execution_time: float
    backends: Dict[str, BackendMetrics]
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        backends_dict = {}
        for name, metrics in self.backends.items():
            metrics_dict = asdict(metrics)
            # Convert datetime objects to ISO strings
            if metrics_dict.get('first_execution'):
                metrics_dict['first_execution'] = metrics_dict['first_execution'].isoformat()
            if metrics_dict.get('last_execution'):
                metrics_dict['last_execution'] = metrics_dict['last_execution'].isoformat()
            backends_dict[name] = metrics_dict

        return {
            'total_backends': self.total_backends,
            'total_executions': self.total_executions,
            'total_successful': self.total_successful,
            'total_failed': self.total_failed,
            'total_cost': self.total_cost,
            'total_shots': self.total_shots,
            'overall_success_rate': self.overall_success_rate,
            'avg_execution_time': self.avg_execution_time,
            'total_execution_time': self.total_execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'backends': backends_dict
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for quantum backends.

    Features:
    - Per-backend execution metrics (throughput, latency, success rate)
    - Cost tracking and analysis
    - Execution history with configurable retention
    - Percentile calculations for latency
    - Time-series data for visualization
    - Summary statistics and reporting

    Example:
        >>> monitor = PerformanceMonitor(history_size=1000)
        >>> monitor.record_execution(
        ...     backend_name='qsim',
        ...     circuit_qubits=4,
        ...     circuit_gates=10,
        ...     circuit_depth=5,
        ...     shots=1000,
        ...     execution_time=0.5,
        ...     cost=0.01
        ... )
        >>> summary = monitor.get_summary()
        >>> print(f"Success rate: {summary.overall_success_rate:.2%}")
    """

    def __init__(self, history_size: int = 10000, enable_detailed_logging: bool = False):
        """
        Initialize performance monitor.

        Args:
            history_size: Maximum number of executions to keep in history per backend
            enable_detailed_logging: Whether to log detailed execution info
        """
        self.history_size = history_size
        self.enable_detailed_logging = enable_detailed_logging

        # Per-backend metrics
        self.backend_metrics: Dict[str, BackendMetrics] = {}
        self.cost_metrics: Dict[str, CostMetrics] = {}

        # Execution history (rolling window)
        self.execution_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Latency history for percentile calculations
        self.latency_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Global stats
        self.start_time = datetime.now()
        self.total_executions = 0

        logger.info(f"PerformanceMonitor initialized (history_size={history_size})")

    def record_execution(
        self,
        backend_name: str,
        circuit_qubits: int,
        circuit_gates: int,
        circuit_depth: int,
        shots: int,
        execution_time: float,
        queue_time: float = 0.0,
        cost: float = 0.0,
        success: bool = True,
        error_message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a circuit execution.

        Args:
            backend_name: Name of backend used
            circuit_qubits: Number of qubits in circuit
            circuit_gates: Number of gates in circuit
            circuit_depth: Circuit depth
            shots: Number of measurement shots
            execution_time: Execution time in seconds
            queue_time: Time spent in queue (seconds)
            cost: Cost of execution (dollars/credits)
            success: Whether execution succeeded
            error_message: Error message if failed
            metadata: Additional metadata
        """
        # Create execution metrics
        metric = ExecutionMetrics(
            backend_name=backend_name,
            circuit_qubits=circuit_qubits,
            circuit_gates=circuit_gates,
            circuit_depth=circuit_depth,
            shots=shots,
            execution_time=execution_time,
            queue_time=queue_time,
            cost=cost,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        # Initialize backend metrics if needed
        if backend_name not in self.backend_metrics:
            self.backend_metrics[backend_name] = BackendMetrics(backend_name=backend_name)
            self.cost_metrics[backend_name] = CostMetrics(backend_name=backend_name)

        # Update backend metrics
        self.backend_metrics[backend_name].update(metric)

        # Update cost metrics
        if cost > 0:
            self.cost_metrics[backend_name].add_cost(
                cost, circuit_qubits, shots, metric.timestamp
            )

        # Add to history
        self.execution_history[backend_name].append(metric)
        self.latency_history[backend_name].append(execution_time)

        # Update percentiles
        self._update_percentiles(backend_name)

        # Update global stats
        self.total_executions += 1

        # Detailed logging
        if self.enable_detailed_logging:
            status = "SUCCESS" if success else "FAILED"
            logger.info(
                f"[{backend_name}] {status} - "
                f"qubits={circuit_qubits}, gates={circuit_gates}, "
                f"shots={shots}, time={execution_time:.3f}s, cost=${cost:.4f}"
            )

    def _update_percentiles(self, backend_name: str):
        """Update percentile calculations for a backend."""
        latencies = list(self.latency_history[backend_name])

        if len(latencies) > 0:
            metrics = self.backend_metrics[backend_name]
            metrics.p50_execution_time = np.percentile(latencies, 50)
            metrics.p95_execution_time = np.percentile(latencies, 95)
            metrics.p99_execution_time = np.percentile(latencies, 99)

    def get_backend_metrics(self, backend_name: str) -> Optional[BackendMetrics]:
        """
        Get metrics for a specific backend.

        Args:
            backend_name: Backend name

        Returns:
            BackendMetrics or None if not found
        """
        return self.backend_metrics.get(backend_name)

    def get_cost_metrics(self, backend_name: str) -> Optional[CostMetrics]:
        """
        Get cost metrics for a specific backend.

        Args:
            backend_name: Backend name

        Returns:
            CostMetrics or None if not found
        """
        return self.cost_metrics.get(backend_name)

    def get_execution_history(
        self,
        backend_name: str,
        limit: Optional[int] = None
    ) -> List[ExecutionMetrics]:
        """
        Get execution history for a backend.

        Args:
            backend_name: Backend name
            limit: Maximum number of entries to return

        Returns:
            List of ExecutionMetrics (most recent first)
        """
        history = list(self.execution_history.get(backend_name, []))
        history.reverse()  # Most recent first

        if limit:
            history = history[:limit]

        return history

    def get_summary(
        self,
        backend_names: Optional[List[str]] = None
    ) -> MetricsSummary:
        """
        Get summary statistics.

        Args:
            backend_names: Specific backends to include (None = all)

        Returns:
            MetricsSummary with aggregated statistics
        """
        backends_to_include = backend_names or list(self.backend_metrics.keys())

        total_executions = 0
        total_successful = 0
        total_failed = 0
        total_cost = 0.0
        total_shots = 0
        total_execution_time = 0.0

        filtered_backends = {}

        for name in backends_to_include:
            if name in self.backend_metrics:
                metrics = self.backend_metrics[name]
                filtered_backends[name] = metrics

                total_executions += metrics.total_executions
                total_successful += metrics.successful_executions
                total_failed += metrics.failed_executions
                total_cost += metrics.total_cost
                total_shots += metrics.total_shots
                total_execution_time += metrics.total_execution_time

        overall_success_rate = (
            total_successful / total_executions if total_executions > 0 else 0.0
        )

        avg_execution_time = (
            total_execution_time / total_executions if total_executions > 0 else 0.0
        )

        return MetricsSummary(
            total_backends=len(filtered_backends),
            total_executions=total_executions,
            total_successful=total_successful,
            total_failed=total_failed,
            total_cost=total_cost,
            total_shots=total_shots,
            overall_success_rate=overall_success_rate,
            avg_execution_time=avg_execution_time,
            total_execution_time=total_execution_time,
            backends=filtered_backends,
            start_time=self.start_time,
            end_time=datetime.now()
        )

    def get_throughput_over_time(
        self,
        backend_name: str,
        window_seconds: int = 60
    ) -> List[Tuple[datetime, float]]:
        """
        Calculate throughput over time using sliding windows.

        Args:
            backend_name: Backend name
            window_seconds: Window size in seconds

        Returns:
            List of (timestamp, throughput) tuples
        """
        history = self.execution_history.get(backend_name, [])

        if not history:
            return []

        throughput_data = []
        window = timedelta(seconds=window_seconds)

        # Sample every minute
        times = [metric.timestamp for metric in history]
        if not times:
            return []

        current_time = times[0]
        end_time = times[-1]

        while current_time <= end_time:
            # Count executions in window
            window_start = current_time
            window_end = current_time + window

            count = sum(
                1 for metric in history
                if window_start <= metric.timestamp < window_end
            )

            throughput = count / window_seconds if window_seconds > 0 else 0
            throughput_data.append((current_time, throughput))

            current_time += timedelta(seconds=60)  # Sample every minute

        return throughput_data

    def get_cost_over_time(
        self,
        backend_name: Optional[str] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get cumulative cost over time.

        Args:
            backend_name: Specific backend (None = all backends)

        Returns:
            List of (timestamp, cumulative_cost) tuples
        """
        if backend_name:
            cost_metrics = self.cost_metrics.get(backend_name)
            if not cost_metrics:
                return []

            # Calculate cumulative
            cumulative = 0.0
            result = []
            for timestamp, cost in cost_metrics.cost_over_time:
                cumulative += cost
                result.append((timestamp, cumulative))

            return result
        else:
            # Combine all backends
            all_costs = []
            for backend in self.cost_metrics.values():
                all_costs.extend(backend.cost_over_time)

            # Sort by timestamp
            all_costs.sort(key=lambda x: x[0])

            # Calculate cumulative
            cumulative = 0.0
            result = []
            for timestamp, cost in all_costs:
                cumulative += cost
                result.append((timestamp, cumulative))

            return result

    def export_to_json(
        self,
        filepath: str,
        include_history: bool = False
    ):
        """
        Export metrics to JSON file.

        Args:
            filepath: Output file path
            include_history: Whether to include full execution history
        """
        summary = self.get_summary()

        data = summary.to_dict()

        if include_history:
            data['history'] = {}
            for backend_name, history in self.execution_history.items():
                data['history'][backend_name] = [
                    metric.to_dict() for metric in history
                ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")

    def print_summary(self, backend_names: Optional[List[str]] = None):
        """
        Print formatted summary to console.

        Args:
            backend_names: Specific backends to include (None = all)
        """
        summary = self.get_summary(backend_names)

        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Time Period: {summary.start_time} to {summary.end_time}")
        print(f"Total Backends: {summary.total_backends}")
        print(f"Total Executions: {summary.total_executions}")
        print(f"  Successful: {summary.total_successful}")
        print(f"  Failed: {summary.total_failed}")
        print(f"Success Rate: {summary.overall_success_rate:.2%}")
        print(f"Total Cost: ${summary.total_cost:.4f}")
        print(f"Total Shots: {summary.total_shots:,}")
        print(f"Avg Execution Time: {summary.avg_execution_time:.3f}s")
        print(f"Total Execution Time: {summary.total_execution_time:.2f}s")

        print("\n" + "-"*80)
        print("PER-BACKEND METRICS")
        print("-"*80)

        for backend_name, metrics in summary.backends.items():
            print(f"\n{backend_name}:")
            print(f"  Executions: {metrics.total_executions}")
            print(f"  Success Rate: {metrics.success_rate:.2%}")
            print(f"  Throughput: {metrics.throughput:.2f} exec/s")
            print(f"  Avg Execution Time: {metrics.avg_execution_time:.3f}s")
            print(f"  P50/P95/P99: {metrics.p50_execution_time:.3f}s / "
                  f"{metrics.p95_execution_time:.3f}s / {metrics.p99_execution_time:.3f}s")
            print(f"  Total Cost: ${metrics.total_cost:.4f}")
            print(f"  Avg Cost: ${metrics.avg_cost_per_execution:.6f}")

        print("\n" + "="*80 + "\n")

    def reset(self, backend_name: Optional[str] = None):
        """
        Reset metrics.

        Args:
            backend_name: Specific backend to reset (None = reset all)
        """
        if backend_name:
            # Reset specific backend
            if backend_name in self.backend_metrics:
                del self.backend_metrics[backend_name]
            if backend_name in self.cost_metrics:
                del self.cost_metrics[backend_name]
            if backend_name in self.execution_history:
                del self.execution_history[backend_name]
            if backend_name in self.latency_history:
                del self.latency_history[backend_name]

            logger.info(f"Reset metrics for backend: {backend_name}")
        else:
            # Reset all
            self.backend_metrics.clear()
            self.cost_metrics.clear()
            self.execution_history.clear()
            self.latency_history.clear()
            self.start_time = datetime.now()
            self.total_executions = 0

            logger.info("Reset all metrics")


def create_performance_monitor(
    history_size: int = 10000,
    enable_detailed_logging: bool = False
) -> PerformanceMonitor:
    """
    Factory function to create a performance monitor.

    Args:
        history_size: Maximum execution history per backend
        enable_detailed_logging: Enable detailed execution logging

    Returns:
        Configured PerformanceMonitor

    Example:
        >>> monitor = create_performance_monitor(history_size=5000)
    """
    return PerformanceMonitor(
        history_size=history_size,
        enable_detailed_logging=enable_detailed_logging
    )
