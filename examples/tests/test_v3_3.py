#!/usr/bin/env python3
"""
Test script for Q-Store v3.3 components
Tests SPSA, circuit batching, caching, hardware-efficient layers, and adaptive optimization
"""

import sys
import asyncio
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        logger.info(f"✅ PASS: {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        logger.error(f"❌ FAIL: {test_name} - {error}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        logger.warning(f"⚠️  WARNING: {test_name} - {warning}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests: {total}")
        print(f"Passed: {len(self.passed)} ✅")
        print(f"Failed: {len(self.failed)} ❌")
        print(f"Warnings: {len(self.warnings)} ⚠️")

        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")

        if self.warnings:
            print("\nWarnings:")
            for name, warning in self.warnings:
                print(f"  - {name}: {warning}")

        print("=" * 70)
        return len(self.failed) == 0


async def test_imports(results: TestResults):
    """Test that all v3.3 components can be imported"""
    test_name = "Import v3.3 components"
    try:
        from q_store.ml import (
            SPSAGradientEstimator,
            CircuitBatchManager,
            QuantumCircuitCache,
            HardwareEfficientQuantumLayer,
            AdaptiveGradientOptimizer,
            PerformanceTracker
        )
        results.add_pass(test_name)
        return True
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


async def test_spsa_gradient_estimator(results: TestResults):
    """Test SPSA gradient estimator"""
    test_name = "SPSA Gradient Estimator"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import SPSAGradientEstimator

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_spsa", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()
        backend = backend_manager.get_backend()

        # Create estimator
        estimator = SPSAGradientEstimator(backend)

        # Test gradient computation
        def circuit_builder(params):
            from q_store.backends.quantum_backend_interface import CircuitBuilder
            builder = CircuitBuilder(2)
            builder.ry(0, params[0])
            builder.ry(1, params[1])
            builder.cnot(0, 1)
            builder.measure_all()
            return builder.build()

        def loss_function(result):
            # Simple loss: maximize |00> probability
            counts = result.counts
            count_00 = counts.get('00', 0)
            total = sum(counts.values())
            return 1.0 - (count_00 / total if total > 0 else 0)

        params = np.array([0.1, 0.2])

        result = await estimator.estimate_gradient(
            circuit_builder,
            loss_function,
            params,
            shots=1000
        )

        # Validate result
        assert result.n_circuit_executions == 2, f"Expected 2 circuits, got {result.n_circuit_executions}"
        assert len(result.gradients) == len(params), "Gradient length mismatch"
        assert result.function_value >= 0, "Loss should be non-negative"
        assert result.method == 'spsa', "Method should be SPSA"

        results.add_pass(test_name)
        logger.info(f"  Circuits executed: {result.n_circuit_executions}")
        logger.info(f"  Gradient norm: {np.linalg.norm(result.gradients):.4f}")
        logger.info(f"  Loss: {result.function_value:.4f}")

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_circuit_batch_manager(results: TestResults):
    """Test circuit batch manager"""
    test_name = "Circuit Batch Manager"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import CircuitBatchManager
        from q_store.backends.quantum_backend_interface import CircuitBuilder

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_batch", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()
        backend = backend_manager.get_backend()

        # Create batch manager
        batch_manager = CircuitBatchManager(backend)

        # Create multiple circuits
        circuits = []
        for i in range(5):
            builder = CircuitBuilder(2)
            builder.h(0)
            builder.cnot(0, 1)
            builder.measure_all()
            circuits.append(builder.build())

        # Execute batch
        results_list = await batch_manager.execute_batch(circuits, shots=100)

        # Validate
        assert len(results_list) == len(circuits), "Result count mismatch"
        for result in results_list:
            assert result is not None, "Null result"
            assert len(result.counts) > 0, "Empty counts"
            assert result.total_shots > 0, "No shots recorded"

        stats = batch_manager.get_statistics()
        logger.info(f"  Circuits submitted: {stats['total_circuits_submitted']}")
        logger.info(f"  Circuits completed: {stats['total_circuits_completed']}")
        logger.info(f"  Success rate: {stats['success_rate']*100:.1f}%")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_circuit_cache(results: TestResults):
    """Test quantum circuit cache"""
    test_name = "Quantum Circuit Cache"
    try:
        from q_store.ml import QuantumCircuitCache
        from q_store.backends.quantum_backend_interface import CircuitBuilder, ExecutionResult

        # Create cache
        cache = QuantumCircuitCache(max_compiled_circuits=10, max_results=50)

        # Create test circuit
        builder = CircuitBuilder(2)
        builder.h(0)
        builder.cnot(0, 1)
        circuit = builder.build()

        # Test compiled circuit caching
        compiled_circuit = "compiled_mock"
        cache.cache_compiled_circuit(circuit, "mock", compiled_circuit)

        retrieved = cache.get_compiled_circuit(circuit, "mock")
        assert retrieved == compiled_circuit, "Compiled circuit cache failed"

        # Test execution result caching
        params = np.array([0.1, 0.2])
        result = ExecutionResult(
            counts={'00': 50, '11': 50},
            probabilities={'00': 0.5, '11': 0.5},
            total_shots=100
        )

        cache.cache_execution_result(circuit, params, 100, result)
        cached_result = cache.get_execution_result(circuit, params, 100)

        assert cached_result is not None, "Result cache failed"
        assert cached_result.counts == result.counts, "Cached result mismatch"

        # Check stats
        stats = cache.get_stats()
        assert stats['hits'] > 0, "No cache hits recorded"
        assert stats['hit_rate'] > 0, "Zero hit rate"

        logger.info(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        logger.info(f"  Cached circuits: {stats['compiled_circuits']}")
        logger.info(f"  Cached results: {stats['cached_results']}")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_hardware_efficient_layer(results: TestResults):
    """Test hardware-efficient quantum layer"""
    test_name = "Hardware-Efficient Quantum Layer"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import HardwareEfficientQuantumLayer

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_he_layer", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()
        backend = backend_manager.get_backend()

        # Create layer
        layer = HardwareEfficientQuantumLayer(
            n_qubits=4,
            depth=2,
            backend=backend
        )

        # Check parameter count (should be n_qubits * depth * 2)
        expected_params = 4 * 2 * 2  # 16 parameters
        assert layer.n_parameters == expected_params, \
            f"Expected {expected_params} params, got {layer.n_parameters}"

        # Build circuit
        input_data = np.random.randn(4)
        circuit = layer.build_circuit(input_data)

        assert circuit.n_qubits == 4, "Qubit count mismatch"
        assert len(circuit.gates) > 0, "Empty circuit"

        # Execute forward pass
        output = await layer.forward(input_data, shots=100)
        assert len(output) > 0, "Empty output"

        logger.info(f"  Parameters: {layer.n_parameters} (33% reduction from standard)")
        logger.info(f"  Circuit gates: {len(circuit.gates)}")
        logger.info(f"  Output dimension: {len(output)}")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_adaptive_optimizer(results: TestResults):
    """Test adaptive gradient optimizer"""
    test_name = "Adaptive Gradient Optimizer"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import AdaptiveGradientOptimizer
        from q_store.backends.quantum_backend_interface import CircuitBuilder

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_adaptive", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()
        backend = backend_manager.get_backend()

        # Create optimizer
        optimizer = AdaptiveGradientOptimizer(backend, enable_adaptation=True)

        # Test gradient computation
        def circuit_builder(params):
            builder = CircuitBuilder(2)
            builder.ry(0, params[0])
            builder.ry(1, params[1])
            builder.cnot(0, 1)
            builder.measure_all()
            return builder.build()

        def loss_function(result):
            counts = result.counts
            count_00 = counts.get('00', 0)
            total = sum(counts.values())
            return 1.0 - (count_00 / total if total > 0 else 0)

        params = np.array([0.1, 0.2])

        # Compute gradients (should start with SPSA)
        result = await optimizer.compute_gradients(
            circuit_builder,
            loss_function,
            params,
            shots=100
        )

        assert result.gradients is not None, "No gradients computed"
        assert len(result.gradients) == len(params), "Gradient length mismatch"

        # Check stats
        stats = optimizer.get_statistics()
        assert stats['iteration'] > 0, "No iterations recorded"
        assert stats['current_method'] in ['spsa', 'parameter_shift'], "Invalid method"

        logger.info(f"  Current method: {stats['current_method']}")
        logger.info(f"  Iterations: {stats['iteration']}")
        logger.info(f"  Method usage: {stats['method_counts']}")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_performance_tracker(results: TestResults):
    """Test performance tracker"""
    test_name = "Performance Tracker"
    try:
        from q_store.ml import PerformanceTracker

        # Create tracker
        tracker = PerformanceTracker()

        # Log some batches
        for i in range(5):
            tracker.log_batch(
                batch_idx=i,
                epoch=0,
                loss=1.0 - i * 0.1,
                gradient_norm=0.5,
                n_circuits=2,
                time_ms=100.0,
                learning_rate=0.01,
                method_used='spsa'
            )

        # Log epoch
        tracker.log_epoch(0)

        # Get statistics
        stats = tracker.get_statistics()

        assert stats['total_batches'] == 5, "Batch count mismatch"
        assert stats['total_epochs'] == 1, "Epoch count mismatch"
        assert stats['total_circuits'] == 10, "Circuit count mismatch"
        assert stats['final_loss'] is not None, "No final loss"

        # Test speedup estimation
        speedup = tracker.estimate_speedup(baseline_circuits_per_batch=96)
        assert speedup['circuit_reduction_factor'] > 1, "No speedup detected"

        logger.info(f"  Total batches: {stats['total_batches']}")
        logger.info(f"  Total circuits: {stats['total_circuits']}")
        logger.info(f"  Estimated speedup: {speedup['circuit_reduction_factor']:.1f}x")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_training_config(results: TestResults):
    """Test v3.3 training configuration"""
    test_name = "Training Config v3.3"
    try:
        from q_store.ml import TrainingConfig

        # Create config with v3.3 parameters
        config = TrainingConfig(
            n_qubits=8,
            circuit_depth=2,
            gradient_method='spsa',
            hardware_efficient_ansatz=True,
            enable_circuit_cache=True,
            enable_batch_execution=True,
            enable_performance_tracking=True,
            spsa_c_initial=0.1,
            spsa_a_initial=0.01,
            performance_log_dir='./test_logs',
            pinecone_api_key='mock-key',
            pinecone_environment='us-east-1'
        )

        # Validate parameters
        assert config.gradient_method == 'spsa', "Gradient method not set"
        assert config.hardware_efficient_ansatz == True, "HE ansatz not enabled"
        assert config.enable_circuit_cache == True, "Cache not enabled"
        assert config.enable_batch_execution == True, "Batch execution not enabled"
        assert config.enable_performance_tracking == True, "Performance tracking not enabled"

        logger.info(f"  Gradient method: {config.gradient_method}")
        logger.info(f"  HE ansatz: {config.hardware_efficient_ansatz}")
        logger.info(f"  Cache: {config.enable_circuit_cache}")
        logger.info(f"  Batch execution: {config.enable_batch_execution}")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def test_integration(results: TestResults):
    """Test v3.3 integration with trainer"""
    test_name = "Integration Test - Full Training"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import (
            QuantumTrainer,
            QuantumModel,
            TrainingConfig
        )

        # Create config
        config = TrainingConfig(
            n_qubits=4,
            circuit_depth=2,
            gradient_method='spsa',
            hardware_efficient_ansatz=True,
            enable_circuit_cache=True,
            enable_batch_execution=True,
            enable_performance_tracking=True,
            learning_rate=0.01,
            batch_size=5,
            epochs=2,
            pinecone_api_key='mock-key',
            pinecone_environment='us-east-1'
        )

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_integration", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()

        # Create trainer
        trainer = QuantumTrainer(config, backend_manager)

        # Verify v3.3 components initialized
        assert trainer.circuit_cache is not None, "Circuit cache not initialized"
        assert trainer.batch_manager is not None, "Batch manager not initialized"
        assert trainer.performance_tracker is not None, "Performance tracker not initialized"

        # Create model
        model = QuantumModel(
            input_dim=4,
            n_qubits=4,
            output_dim=2,
            backend=backend_manager.get_backend(),
            hardware_efficient=True
        )

        # Check model uses HE layer
        from q_store.ml import HardwareEfficientQuantumLayer
        assert isinstance(model.quantum_layer, HardwareEfficientQuantumLayer), \
            "Model not using hardware-efficient layer"

        # Create simple dataset
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 2, size=(10, 2))

        # Simple data loader
        class SimpleDataLoader:
            def __init__(self, X, y, batch_size):
                self.X = X
                self.y = y
                self.batch_size = batch_size

            async def __aiter__(self):
                for i in range(0, len(self.X), self.batch_size):
                    yield self.X[i:i + self.batch_size], self.y[i:i + self.batch_size]

        # Train for one epoch
        data_loader = SimpleDataLoader(X, y, config.batch_size)
        metrics = await trainer.train_epoch(model, data_loader, epoch=0)

        # Validate metrics
        assert metrics.loss >= 0, "Invalid loss"
        assert metrics.n_circuit_executions > 0, "No circuits executed"

        # Check performance tracker
        stats = trainer.performance_tracker.get_statistics()
        assert stats['total_batches'] > 0, "No batches tracked"

        logger.info(f"  Loss: {metrics.loss:.4f}")
        logger.info(f"  Circuits executed: {metrics.n_circuit_executions}")
        logger.info(f"  Time: {metrics.epoch_time_ms/1000:.2f}s")

        # Estimate speedup
        speedup = trainer.performance_tracker.estimate_speedup(baseline_circuits_per_batch=96)
        logger.info(f"  Estimated speedup: {speedup['circuit_reduction_factor']:.1f}x")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))
        import traceback
        logger.error(traceback.format_exc())


async def test_backward_compatibility(results: TestResults):
    """Test v3.2 code still works in v3.3"""
    test_name = "Backward Compatibility (v3.2 code)"
    try:
        from q_store.backends import BackendManager, MockQuantumBackend
        from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig

        # v3.2 style config (no v3.3 parameters)
        config = TrainingConfig(
            n_qubits=4,
            circuit_depth=2,
            learning_rate=0.01,
            batch_size=5,
            epochs=1,
            pinecone_api_key='mock-key',
            pinecone_environment='us-east-1'
        )

        # Initialize backend
        backend_manager = BackendManager()
        mock_backend = MockQuantumBackend("test_backward", max_qubits=4, noise_level=0.0)
        backend_manager.register_backend("mock", mock_backend, set_as_default=True)
        await mock_backend.initialize()

        # Create trainer (should use v3.2 defaults)
        trainer = QuantumTrainer(config, backend_manager)

        # Create v3.2 style model
        model = QuantumModel(
            input_dim=4,
            n_qubits=4,
            output_dim=2,
            backend=backend_manager.get_backend()
        )

        # Verify it works
        assert trainer is not None, "Trainer creation failed"
        assert model is not None, "Model creation failed"

        logger.info("  v3.2 code works without changes ✓")

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, str(e))


async def main():
    """Run all tests"""
    print("=" * 70)
    print("Q-Store v3.3 Test Suite")
    print("=" * 70)
    print()

    results = TestResults()

    # Run tests
    print("Running tests...\n")

    await test_imports(results)
    await test_spsa_gradient_estimator(results)
    await test_circuit_batch_manager(results)
    await test_circuit_cache(results)
    await test_hardware_efficient_layer(results)
    await test_adaptive_optimizer(results)
    await test_performance_tracker(results)
    await test_training_config(results)
    await test_backward_compatibility(results)
    await test_integration(results)

    # Print summary
    success = results.summary()

    if success:
        print("\n🎉 All tests passed! v3.3 is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
