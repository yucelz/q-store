"""
Test Suite for v3.5 Advanced Optimizations
Tests multi-backend orchestration, adaptive optimizations, and natural gradient
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import v3.5 components
from q_store.ml.multi_backend_orchestrator import (
    MultiBackendOrchestrator,
    BackendConfig,
    BackendStats,
)
from q_store.ml.adaptive_circuit_optimizer import (
    AdaptiveCircuitOptimizer,
    CircuitOptimizationResult,
)
from q_store.ml.adaptive_shot_allocator import AdaptiveShotAllocator
from q_store.ml.natural_gradient_estimator import (
    NaturalGradientEstimator,
    QFIMResult,
)
from q_store.backends.quantum_backend_interface import (
    QuantumBackend,
    ExecutionResult,
)


# ============================================================================
# Test MultiBackendOrchestrator
# ============================================================================


class MockBackend:
    """Mock quantum backend for testing"""

    def __init__(self, name: str, execution_time_ms: float = 100.0, fail: bool = False):
        self.name = name
        self.execution_time_ms = execution_time_ms
        self.fail = fail
        self.call_count = 0

    async def execute_circuit(self, circuit, shots):
        """Execute single circuit"""
        self.call_count += 1
        await asyncio.sleep(self.execution_time_ms / 1000.0)

        if self.fail:
            raise RuntimeError(f"Backend {self.name} failed")

        counts = {"00": shots // 2, "11": shots // 2}
        probabilities = {"00": 0.5, "11": 0.5}

        return ExecutionResult(
            counts=counts,
            probabilities=probabilities,
            total_shots=shots,
            metadata={"backend": self.name},
        )

    async def execute_batch(self, circuits, shots):
        """Execute batch of circuits"""
        results = []
        for circuit in circuits:
            result = await self.execute_circuit(circuit, shots)
            results.append(result)
        return results


@pytest.mark.asyncio
async def test_multi_backend_orchestrator_basic():
    """Test basic multi-backend distribution"""
    # Create mock backends
    backend1 = MockBackend("backend1", execution_time_ms=100)
    backend2 = MockBackend("backend2", execution_time_ms=100)
    backend3 = MockBackend("backend3", execution_time_ms=100)

    configs = [
        BackendConfig(backend=backend1, name="backend1", priority=0),
        BackendConfig(backend=backend2, name="backend2", priority=0),
        BackendConfig(backend=backend3, name="backend3", priority=0),
    ]

    orchestrator = MultiBackendOrchestrator(configs)

    # Create test circuits
    circuits = [{"qubits": 2, "circuit": []} for _ in range(30)]

    # Execute distributed
    results = await orchestrator.execute_distributed(circuits, shots=1000)

    # Verify results
    assert len(results) == 30
    assert all(isinstance(r, ExecutionResult) for r in results)

    # Verify load distribution (should be roughly equal)
    assert backend1.call_count > 0
    assert backend2.call_count > 0
    assert backend3.call_count > 0

    # Total calls should equal number of circuits
    total_calls = backend1.call_count + backend2.call_count + backend3.call_count
    assert total_calls == 30

    # Check statistics
    stats = orchestrator.get_statistics()
    assert stats["total_circuits_executed"] == 30
    assert len(stats["backends"]) == 3


@pytest.mark.asyncio
async def test_multi_backend_failover():
    """Test automatic failover when backend fails"""
    # Create backends (one will fail)
    backend1 = MockBackend("backend1", fail=True)
    backend2 = MockBackend("backend2", execution_time_ms=50)

    configs = [
        BackendConfig(backend=backend1, name="backend1", priority=1),
        BackendConfig(backend=backend2, name="backend2", priority=0),
    ]

    orchestrator = MultiBackendOrchestrator(configs)

    # Create test circuits
    circuits = [{"qubits": 2, "circuit": []} for _ in range(10)]

    # Execute - should fallback to backend2
    results = await orchestrator.execute_distributed(circuits, shots=1000)

    # All circuits should complete on backend2
    assert len(results) == 10
    assert backend2.call_count == 10  # All circuits retried on backend2

    # Check statistics show failures
    stats = orchestrator.get_statistics()
    assert stats["backends"]["backend1"]["failures"] > 0


@pytest.mark.asyncio
async def test_multi_backend_empty_circuits():
    """Test handling of empty circuit list"""
    backend = MockBackend("backend1")
    configs = [BackendConfig(backend=backend, name="backend1")]

    orchestrator = MultiBackendOrchestrator(configs)
    results = await orchestrator.execute_distributed([], shots=1000)

    assert len(results) == 0


# ============================================================================
# Test AdaptiveCircuitOptimizer
# ============================================================================


def test_adaptive_circuit_optimizer_depth_schedule():
    """Test depth scheduling across epochs"""
    optimizer = AdaptiveCircuitOptimizer(
        initial_depth=6, min_depth=2, adaptation_schedule="linear"
    )

    # Early epoch: should be near initial depth
    depth_early = optimizer.get_depth_for_epoch(0, 100)
    assert depth_early == 6

    # Mid epoch: should be between min and max
    depth_mid = optimizer.get_depth_for_epoch(50, 100)
    assert 2 <= depth_mid <= 6

    # Late epoch: should be near min depth
    depth_late = optimizer.get_depth_for_epoch(99, 100)
    assert depth_late >= 2  # Allow for rounding


def test_adaptive_circuit_optimizer_gate_merging():
    """Test merging of consecutive rotation gates"""
    optimizer = AdaptiveCircuitOptimizer(enable_gate_merging=True)

    # Circuit with consecutive RY gates on same qubit
    circuit = {
        "qubits": 2,
        "circuit": [
            {"gate": "ry", "target": 0, "rotation": np.pi / 4},
            {"gate": "ry", "target": 0, "rotation": np.pi / 4},
            {"gate": "ry", "target": 1, "rotation": np.pi / 2},
        ],
    }

    optimized = optimizer.optimize_circuit(circuit, target_depth=3)

    # Should merge first two RY gates
    assert len(optimized["circuit"]) == 2  # Merged from 3 to 2

    # First gate should have combined rotation
    assert optimized["circuit"][0]["rotation"] == pytest.approx(np.pi / 2)


def test_adaptive_circuit_optimizer_identity_removal():
    """Test removal of near-zero rotation gates"""
    optimizer = AdaptiveCircuitOptimizer(
        enable_identity_removal=True, identity_threshold=1e-6
    )

    # Circuit with near-zero rotation
    circuit = {
        "qubits": 2,
        "circuit": [
            {"gate": "ry", "target": 0, "rotation": 1e-8},  # Should be removed
            {"gate": "ry", "target": 1, "rotation": np.pi / 2},  # Keep
            {"gate": "rz", "target": 0, "rotation": 0.0},  # Should be removed
        ],
    }

    optimized = optimizer.optimize_circuit(circuit, target_depth=3)

    # Should remove identity gates
    assert len(optimized["circuit"]) == 1
    assert optimized["circuit"][0]["target"] == 1


def test_adaptive_circuit_optimizer_exponential_schedule():
    """Test exponential depth schedule"""
    optimizer = AdaptiveCircuitOptimizer(
        initial_depth=6, min_depth=2, adaptation_schedule="exponential"
    )

    # Exponential should keep high depth longer
    depth_early = optimizer.get_depth_for_epoch(10, 100)
    depth_mid = optimizer.get_depth_for_epoch(50, 100)
    depth_late = optimizer.get_depth_for_epoch(90, 100)

    # Note: values may be same due to rounding, check general trend
    assert depth_early >= depth_mid >= depth_late
    assert depth_late >= 2  # Should reach or be close to minimum


# ============================================================================
# Test AdaptiveShotAllocator
# ============================================================================


def test_adaptive_shot_allocator_phase_based():
    """Test shot allocation based on training phase"""
    allocator = AdaptiveShotAllocator(
        min_shots=500, max_shots=2000, base_shots=1000
    )

    # Early training: should use min shots
    shots_early = allocator.get_shots_for_batch(0, 100)
    assert shots_early == 500

    # Mid training: should use base shots
    shots_mid = allocator.get_shots_for_batch(50, 100)
    assert shots_mid == 1000

    # Late training: should use max shots
    shots_late = allocator.get_shots_for_batch(90, 100)
    assert shots_late == 2000


def test_adaptive_shot_allocator_variance_adjustment():
    """Test shot adjustment based on gradient variance"""
    allocator = AdaptiveShotAllocator(
        min_shots=500,
        max_shots=2000,
        base_shots=1000,
        high_variance_threshold=0.1,
        low_variance_threshold=0.01,
    )

    # High variance gradients
    high_var_grads = [
        np.array([1.0, 0.5, 0.2]),
        np.array([0.2, 1.5, 0.1]),
        np.array([0.8, 0.3, 0.9]),
    ]

    shots_high_var = allocator.get_shots_for_batch(50, 100, high_var_grads)

    # Should increase shots for high variance
    assert shots_high_var > 1000

    # Low variance gradients
    low_var_grads = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.1, 0.1, 0.1]),
        np.array([0.1, 0.1, 0.1]),
    ]

    shots_low_var = allocator.get_shots_for_batch(50, 100, low_var_grads)

    # Should decrease shots for low variance
    assert shots_low_var < 1000


def test_adaptive_shot_allocator_bounds():
    """Test shots are clamped to min/max bounds"""
    allocator = AdaptiveShotAllocator(min_shots=500, max_shots=2000)

    # Even with extreme variance, should respect bounds
    extreme_grads = [np.array([100.0, 200.0, 300.0]) for _ in range(5)]

    shots = allocator.get_shots_for_batch(50, 100, extreme_grads)

    assert 500 <= shots <= 2000


def test_adaptive_shot_allocator_history():
    """Test gradient history tracking"""
    allocator = AdaptiveShotAllocator()

    # Update history
    for i in range(10):
        gradient = np.random.randn(5)
        allocator.update_gradient_history(gradient)

    # Check statistics
    stats = allocator.get_statistics()
    assert "gradient_variance" in stats
    assert "gradient_mean_norm" in stats


# ============================================================================
# Test NaturalGradientEstimator
# ============================================================================


@pytest.mark.asyncio
async def test_natural_gradient_estimator_basic():
    """Test basic natural gradient computation"""
    # Mock backend
    backend = MockBackend("test_backend")

    estimator = NaturalGradientEstimator(
        backend=backend, regularization=0.01, use_qfim_cache=False
    )

    # Mock circuit function
    def circuit_fn(params):
        return {"qubits": 2, "circuit": [{"gate": "ry", "target": 0, "rotation": params[0]}]}

    # Mock loss function
    def loss_fn(params):
        return np.sum(params**2)

    # Parameters
    params = np.array([0.1, 0.2, 0.3])

    # Estimate natural gradient
    result = await estimator.estimate_natural_gradient(
        circuit_fn=circuit_fn,
        parameters=params,
        loss_fn=loss_fn,
        shots=1000,
    )

    # Check result
    assert len(result.gradients) == 3
    assert result.n_circuit_executions > 0
    assert result.computation_time_ms > 0


@pytest.mark.asyncio
async def test_natural_gradient_estimator_qfim_cache():
    """Test QFIM caching"""
    backend = MockBackend("test_backend")

    estimator = NaturalGradientEstimator(
        backend=backend, use_qfim_cache=True, cache_size=10
    )

    def circuit_fn(params):
        return {"qubits": 2, "circuit": []}

    def loss_fn(params):
        return np.sum(params**2)

    params = np.array([0.1, 0.2])

    # First call - cache miss
    result1 = await estimator.estimate_natural_gradient(
        circuit_fn, params, loss_fn, shots=1000
    )

    # Second call - should hit cache
    result2 = await estimator.estimate_natural_gradient(
        circuit_fn, params, loss_fn, shots=1000
    )

    # Check cache statistics
    stats = estimator.get_statistics()
    assert stats["cache_hits"] >= 1
    assert stats["cache_size"] > 0


def test_natural_gradient_estimator_qfim_inversion():
    """Test QFIM matrix inversion with regularization"""
    backend = MockBackend("test_backend")
    estimator = NaturalGradientEstimator(backend=backend, regularization=0.01)

    # Create a test QFIM
    qfim = np.array([[1.0, 0.1], [0.1, 1.0]])

    # Invert
    qfim_inv = estimator._invert_qfim(qfim)

    # Check shape
    assert qfim_inv.shape == (2, 2)

    # Check that it's approximately inverse
    product = qfim @ qfim_inv
    expected = np.eye(2)
    assert np.allclose(product, expected, atol=0.1)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_v35_integration():
    """Integration test combining all v3.5 components"""
    # Setup backends
    backend1 = MockBackend("backend1", execution_time_ms=50)
    backend2 = MockBackend("backend2", execution_time_ms=50)

    configs = [
        BackendConfig(backend=backend1, name="backend1"),
        BackendConfig(backend=backend2, name="backend2"),
    ]

    # Create orchestrator
    orchestrator = MultiBackendOrchestrator(configs)

    # Create circuit optimizer
    circuit_optimizer = AdaptiveCircuitOptimizer(
        initial_depth=4, min_depth=2, adaptation_schedule="linear"
    )

    # Create shot allocator
    shot_allocator = AdaptiveShotAllocator(
        min_shots=500, max_shots=2000, base_shots=1000
    )

    # Simulate training loop
    n_epochs = 10
    circuits_per_epoch = 20

    for epoch in range(n_epochs):
        # Get adaptive depth
        target_depth = circuit_optimizer.get_depth_for_epoch(epoch, n_epochs)

        # Get adaptive shots
        shots = shot_allocator.get_shots_for_batch(epoch, n_epochs)

        # Create circuits
        circuits = []
        for _ in range(circuits_per_epoch):
            circuit = {
                "qubits": 2,
                "circuit": [
                    {"gate": "ry", "target": 0, "rotation": np.random.rand()},
                    {"gate": "cnot", "control": 0, "target": 1},
                    {"gate": "ry", "target": 1, "rotation": np.random.rand()},
                ],
            }
            # Optimize circuit
            circuit = circuit_optimizer.optimize_circuit(circuit, target_depth)
            circuits.append(circuit)

        # Execute distributed
        results = await orchestrator.execute_distributed(circuits, shots=shots)

        assert len(results) == circuits_per_epoch

    # Check final statistics
    orchestrator_stats = orchestrator.get_statistics()
    assert orchestrator_stats["total_circuits_executed"] == n_epochs * circuits_per_epoch

    circuit_stats = circuit_optimizer.get_statistics()
    assert circuit_stats["current_depth"] <= circuit_stats["initial_depth"]

    shot_stats = shot_allocator.get_statistics()
    assert shot_stats["total_allocations"] == n_epochs


def test_configuration_v35():
    """Test v3.5 configuration options"""
    from q_store.ml import TrainingConfig

    # Test v3.5 configuration
    config = TrainingConfig(
        pinecone_api_key="test",
        # v3.5 features
        enable_all_v35_features=True,
        enable_multi_backend=True,
        adaptive_circuit_depth=True,
        adaptive_shot_allocation=True,
        use_natural_gradient=True,
        enable_circuit_optimization=True,
    )

    # Should enable all v3.5 features
    assert config.enable_multi_backend is True
    assert config.adaptive_circuit_depth is True
    assert config.adaptive_shot_allocation is True
    assert config.use_natural_gradient is True
    assert config.enable_circuit_optimization is True

    # Should also enable v3.4 features
    assert config.use_concurrent_submission is True
    assert config.enable_smart_caching is True


def test_backward_compatibility():
    """Test backward compatibility with v3.4 config"""
    from q_store.ml import TrainingConfig

    # Old v3.4 style config (deprecated use_batch_api)
    config = TrainingConfig(
        pinecone_api_key="test",
        use_batch_api=True,
        enable_all_v34_features=True,
    )

    # Should map to new name
    assert config.use_concurrent_submission is True

    # v3.4 features should still work
    assert config.enable_smart_caching is True
    assert config.use_native_gates is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
