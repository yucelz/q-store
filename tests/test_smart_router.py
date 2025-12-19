"""
Tests for Smart Backend Router.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from q_store.core import UnifiedCircuit, GateType
from q_store.backends import BackendType
from q_store.routing import (
    SmartBackendRouter,
    BackendScore,
    RoutingStrategy,
    create_smart_router
)
from q_store.routing.smart_router import CircuitComplexity, PerformanceTracker


@dataclass
class BackendCapabilities:
    """Mock backend capabilities for testing."""
    backend_type: BackendType
    max_qubits: int
    supports_batching: bool = True
    supports_state_vector: bool = True
    supports_expectations: bool = True


class MockBackend:
    """Mock backend for testing (simple non-abstract class)."""

    def __init__(self, name, max_qubits=32, supports_state_vector=True):
        self.name = name
        self._max_qubits = max_qubits
        self._supports_state_vector = supports_state_vector
        self.execution_count = 0

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_type=BackendType.SIMULATOR,
            max_qubits=self._max_qubits,
            supports_batching=True,
            supports_state_vector=self._supports_state_vector,
            supports_expectations=True
        )

    def execute(self, circuit, shots=1000, **kwargs):
        self.execution_count += 1
        # Return mock result
        n_qubits = circuit.n_qubits
        return {
            'counts': {'0' * n_qubits: shots},
            'backend': self.name
        }

    def get_state_vector(self, circuit, **kwargs):
        n_qubits = circuit.n_qubits
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        return state


class TestCircuitComplexity:
    """Test circuit complexity analysis."""

    def test_simple_circuit(self):
        """Test complexity analysis of simple circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        complexity = CircuitComplexity.analyze(circuit)

        assert complexity.n_qubits == 2
        assert complexity.n_gates == 2
        assert complexity.n_two_qubit_gates == 1
        assert complexity.estimated_runtime > 0

    def test_complex_circuit(self):
        """Test complexity analysis of larger circuit."""
        circuit = UnifiedCircuit(4)
        for i in range(4):
            circuit.add_gate(GateType.H, targets=[i])
        for i in range(3):
            circuit.add_gate(GateType.CNOT, targets=[i, i+1])
        circuit.add_gate(GateType.RZ, targets=[0], parameters={'angle': 'theta_0'})  # Symbolic parameter
        
        complexity = CircuitComplexity.analyze(circuit)
        
        assert complexity.n_qubits == 4
        assert complexity.n_gates == 8
        assert complexity.n_two_qubit_gates == 3
        assert complexity.n_parameters == 1
class TestPerformanceTracker:
    """Test performance tracking."""

    def test_record_execution(self):
        """Test recording execution metrics."""
        tracker = PerformanceTracker(history_size=10)

        tracker.record_execution('backend1', 0.5, True, 0.01)
        tracker.record_execution('backend1', 0.6, True, 0.01)
        tracker.record_execution('backend1', 0.4, False, 0.0)

        avg_time = tracker.get_avg_execution_time('backend1')
        success_rate = tracker.get_success_rate('backend1')
        avg_cost = tracker.get_avg_cost('backend1')

        assert abs(avg_time - 0.5) < 0.1
        assert abs(success_rate - 0.667) < 0.01
        assert abs(avg_cost - 0.0067) < 0.01

    def test_history_limit(self):
        """Test history size limit."""
        tracker = PerformanceTracker(history_size=5)

        for i in range(10):
            tracker.record_execution('backend1', float(i), True, 0.0)

        # Should only keep last 5
        avg_time = tracker.get_avg_execution_time('backend1')
        assert avg_time >= 5.0  # Average of 5,6,7,8,9

    def test_no_history(self):
        """Test behavior with no history."""
        tracker = PerformanceTracker()

        # Should return defaults
        assert tracker.get_avg_execution_time('unknown') == 1.0
        assert tracker.get_success_rate('unknown') == 1.0
        assert tracker.get_avg_cost('unknown') == 0.0


class TestSmartBackendRouter:
    """Test smart backend router."""

    def test_initialization(self):
        """Test router initialization."""
        router = SmartBackendRouter(strategy=RoutingStrategy.BALANCED)

        assert router.strategy == RoutingStrategy.BALANCED
        assert len(router.backends) == 0
        assert isinstance(router.performance_tracker, PerformanceTracker)

    def test_register_backend(self):
        """Test backend registration."""
        router = SmartBackendRouter()
        backend1 = MockBackend('backend1')
        backend2 = MockBackend('backend2', max_qubits=16)

        router.register_backend('backend1', backend1, cost_per_shot=0.01)
        router.register_backend('backend2', backend2, cost_per_shot=0.0)

        assert len(router.backends) == 2
        assert 'backend1' in router.backends
        assert 'backend2' in router.backends
        assert router.cost_per_shot['backend1'] == 0.01
        assert router.cost_per_shot['backend2'] == 0.0

    def test_unregister_backend(self):
        """Test backend unregistration."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1')

        router.register_backend('backend1', backend)
        assert 'backend1' in router.backends

        router.unregister_backend('backend1')
        assert 'backend1' not in router.backends

    def test_capability_score(self):
        """Test capability scoring."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1', max_qubits=4)
        router.register_backend('backend1', backend)

        # Circuit within limits
        circuit_ok = UnifiedCircuit(3)
        complexity_ok = CircuitComplexity.analyze(circuit_ok)
        score_ok = router._calculate_capability_score('backend1', complexity_ok)
        assert score_ok > 0

        # Circuit exceeds limits
        circuit_too_big = UnifiedCircuit(10)
        complexity_too_big = CircuitComplexity.analyze(circuit_too_big)
        score_too_big = router._calculate_capability_score('backend1', complexity_too_big)
        assert score_too_big == 0.0

    def test_score_backend(self):
        """Test backend scoring."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1')
        router.register_backend('backend1', backend, cost_per_shot=0.0)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        score = router.score_backend('backend1', circuit, shots=1000)

        assert isinstance(score, BackendScore)
        assert score.backend_name == 'backend1'
        assert 0 <= score.total_score <= 1
        assert 0 <= score.speed_score <= 1
        assert 0 <= score.cost_score <= 1
        assert 0 <= score.precision_score <= 1

    def test_select_backend_single(self):
        """Test backend selection with single backend."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1')
        router.register_backend('backend1', backend)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        selected, score = router.select_backend(circuit, shots=1000)

        assert selected == backend
        assert score.backend_name == 'backend1'
        assert score.total_score > 0

    def test_select_backend_multiple(self):
        """Test backend selection with multiple backends."""
        router = SmartBackendRouter(strategy=RoutingStrategy.SPEED)

        # Fast backend (fewer qubits)
        fast_backend = MockBackend('fast', max_qubits=8)
        # Slow backend (more qubits)
        slow_backend = MockBackend('slow', max_qubits=32)

        router.register_backend('fast', fast_backend, cost_per_shot=0.0)
        router.register_backend('slow', slow_backend, cost_per_shot=0.0)

        # Record performance history
        router.performance_tracker.record_execution('fast', 0.1, True)
        router.performance_tracker.record_execution('slow', 1.0, True)

        circuit = UnifiedCircuit(4)
        circuit.add_gate(GateType.H, targets=[0])

        selected, score = router.select_backend(circuit, shots=1000)

        # Should select based on strategy and capabilities
        assert selected in [fast_backend, slow_backend]
        assert score.total_score > 0

    def test_cost_optimization(self):
        """Test cost-based backend selection."""
        router = SmartBackendRouter(strategy=RoutingStrategy.COST)

        free_backend = MockBackend('free')
        paid_backend = MockBackend('paid')

        router.register_backend('free', free_backend, cost_per_shot=0.0)
        router.register_backend('paid', paid_backend, cost_per_shot=1.0)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        selected, score = router.select_backend(circuit, shots=1000)

        # Should prefer free backend
        assert selected == free_backend
        assert score.backend_name == 'free'

    def test_execute_with_fallback_success(self):
        """Test successful execution with fallback."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1')
        router.register_backend('backend1', backend)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        result = router.execute_with_fallback(circuit, shots=1000)

        assert result is not None
        assert 'counts' in result
        assert backend.execution_count == 1

    def test_execute_with_fallback_retry(self):
        """Test fallback to secondary backend on failure."""
        router = SmartBackendRouter()

        # Failing backend
        failing_backend = MockBackend('failing')
        failing_backend.execute = Mock(side_effect=RuntimeError("Simulated failure"))

        # Working backup
        working_backend = MockBackend('working')

        router.register_backend('failing', failing_backend, cost_per_shot=0.0)
        router.register_backend('working', working_backend, cost_per_shot=0.1)

        # Make failing backend score higher initially
        router.performance_tracker.record_execution('failing', 0.1, True)
        router.performance_tracker.record_execution('working', 1.0, True)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        result = router.execute_with_fallback(circuit, shots=1000, max_retries=3)

        # Should fallback to working backend
        assert result is not None
        assert result['backend'] == 'working'

    def test_execute_with_fallback_all_fail(self):
        """Test all backends failing."""
        router = SmartBackendRouter()

        failing_backend = MockBackend('failing')
        failing_backend.execute = Mock(side_effect=RuntimeError("Simulated failure"))

        router.register_backend('failing', failing_backend)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        with pytest.raises(RuntimeError, match="All backends failed"):
            router.execute_with_fallback(circuit, shots=1000, max_retries=1)

    def test_no_backends_registered(self):
        """Test error when no backends registered."""
        router = SmartBackendRouter()

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        with pytest.raises(ValueError, match="No backends registered"):
            router.select_backend(circuit, shots=1000)

    def test_get_backend_statistics(self):
        """Test getting backend statistics."""
        router = SmartBackendRouter()
        backend = MockBackend('backend1')
        router.register_backend('backend1', backend)

        # Record some executions
        router.performance_tracker.record_execution('backend1', 0.5, True, 0.01)
        router.performance_tracker.record_execution('backend1', 0.6, True, 0.01)

        stats = router.get_backend_statistics()

        assert 'backend1' in stats
        assert 'avg_execution_time' in stats['backend1']
        assert 'success_rate' in stats['backend1']
        assert 'avg_cost' in stats['backend1']
        assert stats['backend1']['success_rate'] == 1.0

    def test_strategy_override(self):
        """Test temporary strategy override."""
        router = SmartBackendRouter(strategy=RoutingStrategy.BALANCED)
        backend = MockBackend('backend1')
        router.register_backend('backend1', backend)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        # Use different strategy for this selection
        selected, score = router.select_backend(
            circuit,
            shots=1000,
            strategy=RoutingStrategy.SPEED
        )

        # Original strategy should be restored
        assert router.strategy == RoutingStrategy.BALANCED
        assert score.metadata['strategy'] == 'speed'


class TestCreateSmartRouter:
    """Test smart router factory function."""

    def test_create_empty(self):
        """Test creating router with no backends."""
        router = create_smart_router()

        assert isinstance(router, SmartBackendRouter)
        assert len(router.backends) == 0
        assert router.strategy == RoutingStrategy.BALANCED

    def test_create_with_backends(self):
        """Test creating router with pre-registered backends."""
        backend1 = MockBackend('backend1')
        backend2 = MockBackend('backend2')

        router = create_smart_router({
            'backend1': backend1,
            'backend2': backend2
        })

        assert len(router.backends) == 2
        assert 'backend1' in router.backends
        assert 'backend2' in router.backends

    def test_create_with_strategy(self):
        """Test creating router with specific strategy."""
        router = create_smart_router(strategy=RoutingStrategy.COST)

        assert router.strategy == RoutingStrategy.COST


class TestRoutingStrategies:
    """Test different routing strategies."""

    def test_speed_strategy(self):
        """Test speed-optimized routing."""
        router = SmartBackendRouter(strategy=RoutingStrategy.SPEED)

        fast_backend = MockBackend('fast')
        slow_backend = MockBackend('slow')

        router.register_backend('fast', fast_backend)
        router.register_backend('slow', slow_backend)

        # Simulate performance history
        router.performance_tracker.record_execution('fast', 0.1, True)
        router.performance_tracker.record_execution('slow', 1.0, True)

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, targets=[0])

        selected, score = router.select_backend(circuit)

        # Should prefer faster backend
        assert selected == fast_backend

    def test_adaptive_strategy(self):
        """Test adaptive routing based on circuit."""
        router = SmartBackendRouter(strategy=RoutingStrategy.ADAPTIVE)

        backend = MockBackend('backend1')
        router.register_backend('backend1', backend)

        # Small circuit
        small_circuit = UnifiedCircuit(2)
        small_circuit.add_gate(GateType.H, targets=[0])

        # Large circuit
        large_circuit = UnifiedCircuit(25)
        for i in range(25):
            large_circuit.add_gate(GateType.H, targets=[i])

        score_small = router.score_backend('backend1', small_circuit)
        score_large = router.score_backend('backend1', large_circuit)

        # Both should score, but potentially with different weighting
        assert score_small.total_score > 0
        assert score_large.total_score > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
