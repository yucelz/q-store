"""
Tests for Performance Monitor.
"""

import pytest
import time
import json
import tempfile
from datetime import datetime, timedelta

from q_store.monitoring import (
    PerformanceMonitor,
    ExecutionMetrics,
    BackendMetrics,
    CostMetrics,
    MetricsSummary,
    create_performance_monitor
)


class TestExecutionMetrics:
    """Test ExecutionMetrics dataclass."""

    def test_creation(self):
        """Test creating execution metrics."""
        metric = ExecutionMetrics(
            backend_name='qsim',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=0.5,
            cost=0.01
        )

        assert metric.backend_name == 'qsim'
        assert metric.circuit_qubits == 4
        assert metric.shots == 1000
        assert metric.success == True

    def test_to_dict(self):
        """Test converting to dictionary."""
        metric = ExecutionMetrics(
            backend_name='test',
            circuit_qubits=2,
            circuit_gates=5,
            circuit_depth=3,
            shots=100,
            execution_time=0.1
        )

        data = metric.to_dict()

        assert data['backend_name'] == 'test'
        assert data['circuit_qubits'] == 2
        assert 'timestamp' in data


class TestBackendMetrics:
    """Test BackendMetrics aggregation."""

    def test_initialization(self):
        """Test backend metrics initialization."""
        metrics = BackendMetrics(backend_name='test')

        assert metrics.backend_name == 'test'
        assert metrics.total_executions == 0
        assert metrics.success_rate == 1.0

    def test_update(self):
        """Test updating metrics."""
        metrics = BackendMetrics(backend_name='test')

        exec1 = ExecutionMetrics(
            backend_name='test',
            circuit_qubits=2,
            circuit_gates=5,
            circuit_depth=3,
            shots=1000,
            execution_time=0.5,
            cost=0.01
        )

        metrics.update(exec1)

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.total_shots == 1000
        assert metrics.total_cost == 0.01
        assert metrics.avg_execution_time == 0.5
        assert metrics.success_rate == 1.0

    def test_multiple_updates(self):
        """Test multiple updates."""
        metrics = BackendMetrics(backend_name='test')

        # Add 3 successful executions
        for i in range(3):
            exec_metric = ExecutionMetrics(
                backend_name='test',
                circuit_qubits=2,
                circuit_gates=5,
                circuit_depth=3,
                shots=1000,
                execution_time=0.5 + i * 0.1,
                cost=0.01
            )
            metrics.update(exec_metric)

        # Add 1 failed execution
        failed = ExecutionMetrics(
            backend_name='test',
            circuit_qubits=2,
            circuit_gates=5,
            circuit_depth=3,
            shots=1000,
            execution_time=0.2,
            success=False
        )
        metrics.update(failed)

        assert metrics.total_executions == 4
        assert metrics.successful_executions == 3
        assert metrics.failed_executions == 1
        assert metrics.success_rate == 0.75
        assert metrics.total_cost == 0.03


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(history_size=100)

        assert monitor.history_size == 100
        assert len(monitor.backend_metrics) == 0
        assert monitor.total_executions == 0

    def test_record_execution(self):
        """Test recording an execution."""
        monitor = PerformanceMonitor()

        monitor.record_execution(
            backend_name='qsim',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=0.5,
            cost=0.01
        )

        assert monitor.total_executions == 1
        assert 'qsim' in monitor.backend_metrics

        metrics = monitor.get_backend_metrics('qsim')
        assert metrics.total_executions == 1
        assert metrics.success_rate == 1.0

    def test_multiple_backends(self):
        """Test recording executions from multiple backends."""
        monitor = PerformanceMonitor()

        monitor.record_execution(
            backend_name='qsim',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=0.5
        )

        monitor.record_execution(
            backend_name='lightning',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=0.3
        )

        assert len(monitor.backend_metrics) == 2
        assert 'qsim' in monitor.backend_metrics
        assert 'lightning' in monitor.backend_metrics

    def test_execution_history(self):
        """Test execution history tracking."""
        monitor = PerformanceMonitor(history_size=5)

        # Add 10 executions
        for i in range(10):
            monitor.record_execution(
                backend_name='test',
                circuit_qubits=2,
                circuit_gates=5,
                circuit_depth=3,
                shots=1000,
                execution_time=0.1 * i
            )

        # Should keep only last 5
        history = monitor.get_execution_history('test')
        assert len(history) == 5

        # Most recent first
        assert history[0].execution_time == 0.9
        assert history[4].execution_time == 0.5

    def test_percentile_calculation(self):
        """Test percentile calculations."""
        monitor = PerformanceMonitor()

        # Add executions with known distribution
        times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for t in times:
            monitor.record_execution(
                backend_name='test',
                circuit_qubits=2,
                circuit_gates=5,
                circuit_depth=3,
                shots=1000,
                execution_time=t
            )

        metrics = monitor.get_backend_metrics('test')

        assert metrics.p50_execution_time == pytest.approx(0.55, abs=0.1)
        assert metrics.p95_execution_time == pytest.approx(0.95, abs=0.1)
        assert metrics.p99_execution_time == pytest.approx(0.99, abs=0.1)

    def test_cost_tracking(self):
        """Test cost metrics tracking."""
        monitor = PerformanceMonitor()

        monitor.record_execution(
            backend_name='qpu',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=1.0,
            cost=0.5
        )

        monitor.record_execution(
            backend_name='qpu',
            circuit_qubits=8,
            circuit_gates=20,
            circuit_depth=10,
            shots=2000,
            execution_time=2.0,
            cost=1.0
        )

        cost_metrics = monitor.get_cost_metrics('qpu')

        assert cost_metrics.total_cost == 1.5
        assert 4 in cost_metrics.cost_by_circuit_size
        assert 8 in cost_metrics.cost_by_circuit_size

    def test_success_rate(self):
        """Test success rate calculation."""
        monitor = PerformanceMonitor()

        # 7 successful, 3 failed
        for i in range(10):
            success = i < 7
            monitor.record_execution(
                backend_name='test',
                circuit_qubits=2,
                circuit_gates=5,
                circuit_depth=3,
                shots=1000,
                execution_time=0.1,
                success=success
            )

        metrics = monitor.get_backend_metrics('test')
        assert metrics.success_rate == 0.7
        assert metrics.successful_executions == 7
        assert metrics.failed_executions == 3

    def test_get_summary(self):
        """Test getting summary statistics."""
        monitor = PerformanceMonitor()

        # Add data for two backends
        for _ in range(5):
            monitor.record_execution(
                backend_name='backend1',
                circuit_qubits=4,
                circuit_gates=10,
                circuit_depth=5,
                shots=1000,
                execution_time=0.5,
                cost=0.1
            )

        for _ in range(3):
            monitor.record_execution(
                backend_name='backend2',
                circuit_qubits=4,
                circuit_gates=10,
                circuit_depth=5,
                shots=1000,
                execution_time=0.3,
                cost=0.05,
                success=False
            )

        summary = monitor.get_summary()

        assert summary.total_backends == 2
        assert summary.total_executions == 8
        assert summary.total_successful == 5
        assert summary.total_failed == 3
        assert summary.overall_success_rate == pytest.approx(5/8)
        assert summary.total_cost == pytest.approx(0.5 + 0.15)

    def test_get_summary_filtered(self):
        """Test getting summary for specific backends."""
        monitor = PerformanceMonitor()

        monitor.record_execution('backend1', 2, 5, 3, 1000, 0.5)
        monitor.record_execution('backend2', 2, 5, 3, 1000, 0.3)
        monitor.record_execution('backend3', 2, 5, 3, 1000, 0.4)

        summary = monitor.get_summary(backend_names=['backend1', 'backend2'])

        assert summary.total_backends == 2
        assert summary.total_executions == 2
        assert 'backend3' not in summary.backends

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        monitor = PerformanceMonitor()

        # Add executions over time
        base_time = datetime.now()

        for i in range(10):
            # Simulate executions 1 second apart
            exec_metric = ExecutionMetrics(
                backend_name='test',
                circuit_qubits=2,
                circuit_gates=5,
                circuit_depth=3,
                shots=1000,
                execution_time=0.1,
                timestamp=base_time + timedelta(seconds=i)
            )
            monitor.backend_metrics.setdefault('test', BackendMetrics('test')).update(exec_metric)
            monitor.backend_metrics['test'].first_execution = base_time
            monitor.backend_metrics['test'].last_execution = base_time + timedelta(seconds=9)

        metrics = monitor.get_backend_metrics('test')

        # 10 executions over 9 seconds ≈ 1.11 exec/s
        assert metrics.throughput > 1.0
        assert metrics.throughput < 1.2

    def test_export_to_json(self):
        """Test exporting metrics to JSON."""
        monitor = PerformanceMonitor()

        monitor.record_execution(
            backend_name='test',
            circuit_qubits=4,
            circuit_gates=10,
            circuit_depth=5,
            shots=1000,
            execution_time=0.5,
            cost=0.1
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        monitor.export_to_json(filepath)

        # Read and verify
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert data['total_backends'] == 1
        assert data['total_executions'] == 1
        assert 'test' in data['backends']

    def test_export_with_history(self):
        """Test exporting with execution history."""
        monitor = PerformanceMonitor()

        monitor.record_execution('test', 2, 5, 3, 1000, 0.1)
        monitor.record_execution('test', 2, 5, 3, 1000, 0.2)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        monitor.export_to_json(filepath, include_history=True)

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'history' in data
        assert 'test' in data['history']
        assert len(data['history']['test']) == 2

    def test_reset_all(self):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()

        monitor.record_execution('backend1', 2, 5, 3, 1000, 0.5)
        monitor.record_execution('backend2', 2, 5, 3, 1000, 0.3)

        assert len(monitor.backend_metrics) == 2

        monitor.reset()

        assert len(monitor.backend_metrics) == 0
        assert monitor.total_executions == 0

    def test_reset_specific_backend(self):
        """Test resetting specific backend."""
        monitor = PerformanceMonitor()

        monitor.record_execution('backend1', 2, 5, 3, 1000, 0.5)
        monitor.record_execution('backend2', 2, 5, 3, 1000, 0.3)

        monitor.reset('backend1')

        assert 'backend1' not in monitor.backend_metrics
        assert 'backend2' in monitor.backend_metrics


class TestCreatePerformanceMonitor:
    """Test factory function."""

    def test_create(self):
        """Test creating monitor via factory."""
        monitor = create_performance_monitor(history_size=5000)

        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.history_size == 5000

    def test_create_with_logging(self):
        """Test creating with detailed logging enabled."""
        monitor = create_performance_monitor(enable_detailed_logging=True)

        assert monitor.enable_detailed_logging == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
