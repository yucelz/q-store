#!/usr/bin/env python3
"""
Verification script for Q-Store v3.2 ML Training Components
Tests that all new components are properly installed and functional
"""

import sys
import asyncio
import numpy as np


def test_imports():
    """Test that all v3.2 components can be imported"""
    print("Testing imports...")

    try:
        # Core ML components
        from q_store.ml import (
            QuantumLayer,
            QuantumConvolutionalLayer,
            QuantumPoolingLayer,
            LayerConfig,
            QuantumGradientComputer,
            FiniteDifferenceGradient,
            NaturalGradientComputer,
            GradientResult,
            QuantumDataEncoder,
            QuantumFeatureMap,
            QuantumTrainer,
            QuantumModel,
            TrainingConfig,
            TrainingMetrics,
        )
        print("✓ All ML components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


async def test_quantum_layer():
    """Test QuantumLayer functionality"""
    print("\nTesting QuantumLayer...")

    try:
        from q_store.ml import QuantumLayer
        from q_store.backends import create_default_backend_manager

        backend_manager = create_default_backend_manager()
        backend = backend_manager.get_backend()

        layer = QuantumLayer(
            n_qubits=4,
            depth=2,
            backend=backend,
            entanglement='linear'
        )

        # Test forward pass
        input_data = np.random.randn(4)
        output = await layer.forward(input_data, shots=100)

        assert len(output) == 4, "Output dimension mismatch"
        assert not np.isnan(output).any(), "NaN in output"

        print(f"✓ QuantumLayer test passed (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ QuantumLayer test failed: {e}")
        return False


async def test_data_encoder():
    """Test QuantumDataEncoder functionality"""
    print("\nTesting QuantumDataEncoder...")

    try:
        from q_store.ml import QuantumDataEncoder, QuantumFeatureMap

        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Test amplitude encoding
        encoder = QuantumDataEncoder('amplitude')
        circuit = encoder.encode(data)
        assert circuit.n_qubits == 3, "Amplitude encoding qubit count wrong"

        # Test angle encoding
        encoder = QuantumDataEncoder('angle')
        circuit = encoder.encode(data, n_qubits=8)
        assert circuit.n_qubits == 8, "Angle encoding qubit count wrong"

        # Test feature map
        feature_map = QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
        circuit = feature_map.map_features(data)
        assert circuit.n_qubits == 8, "Feature map qubit count wrong"

        print("✓ QuantumDataEncoder test passed")
        return True
    except Exception as e:
        print(f"✗ QuantumDataEncoder test failed: {e}")
        return False


async def test_gradient_computer():
    """Test QuantumGradientComputer functionality"""
    print("\nTesting QuantumGradientComputer...")

    try:
        from q_store.ml import QuantumGradientComputer, QuantumLayer
        from q_store.backends import create_default_backend_manager

        backend_manager = create_default_backend_manager()
        backend = backend_manager.get_backend()

        layer = QuantumLayer(
            n_qubits=2,
            depth=1,
            backend=backend
        )

        grad_computer = QuantumGradientComputer(backend)

        # Simple loss function
        def loss_fn(result):
            return sum(result.probabilities.values())

        # Compute gradients
        result = await grad_computer.compute_gradients(
            circuit_builder=lambda p: layer.build_circuit(np.array([0.5, 0.5])),
            loss_function=loss_fn,
            parameters=layer.parameters
        )

        assert len(result.gradients) == len(layer.parameters), "Gradient count mismatch"
        assert result.n_circuit_executions > 0, "No circuits executed"

        print(f"✓ QuantumGradientComputer test passed ({result.n_circuit_executions} circuits executed)")
        return True
    except Exception as e:
        print(f"✗ QuantumGradientComputer test failed: {e}")
        return False


async def test_quantum_trainer():
    """Test QuantumTrainer functionality"""
    print("\nTesting QuantumTrainer...")

    try:
        from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig
        from q_store.backends import create_default_backend_manager

        # Minimal config
        config = TrainingConfig(
            pinecone_api_key="mock-key",
            quantum_sdk="mock",
            learning_rate=0.01,
            batch_size=2,
            epochs=1,
            n_qubits=2,
            circuit_depth=1
        )

        backend_manager = create_default_backend_manager()
        trainer = QuantumTrainer(config, backend_manager)

        # Create simple model
        model = QuantumModel(
            input_dim=2,
            n_qubits=2,
            output_dim=2,
            backend=backend_manager.get_backend(),
            depth=1
        )

        # Simple data loader
        class TestDataLoader:
            async def __aiter__(self):
                X = np.random.randn(4, 2)
                y = np.eye(2)[np.random.randint(0, 2, 4)]
                yield X[:2], y[:2]
                yield X[2:], y[2:]

        # Train for 1 epoch
        await trainer.train(model, TestDataLoader(), epochs=1)

        assert len(trainer.training_history) == 1, "Training history wrong length"
        assert trainer.training_history[0].epoch == 0, "Epoch number wrong"

        print(f"✓ QuantumTrainer test passed (loss: {trainer.training_history[0].loss:.4f})")
        return True
    except Exception as e:
        print(f"✗ QuantumTrainer test failed: {e}")
        return False


async def test_examples():
    """Test that examples can be imported"""
    print("\nTesting examples...")

    try:
        from q_store_examples import examples_v3_2

        # Check that main functions exist
        assert hasattr(examples_v3_2, 'example_1_basic_training'), "Example 1 missing"
        assert hasattr(examples_v3_2, 'example_2_data_encoding'), "Example 2 missing"
        assert hasattr(examples_v3_2, 'main'), "Main function missing"

        print("✓ Examples import test passed")
        return True
    except Exception as e:
        print(f"✗ Examples test failed: {e}")
        return False


async def main():
    """Run all verification tests"""
    print("="*70)
    print("Q-Store v3.2 ML Training Components Verification")
    print("="*70)

    results = []

    # Test imports
    results.append(test_imports())

    # Test components (async)
    results.append(await test_quantum_layer())
    results.append(await test_data_encoder())
    results.append(await test_gradient_computer())
    results.append(await test_quantum_trainer())
    results.append(await test_examples())

    # Summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("\n✓ All v3.2 components verified successfully!")
        print("\nYou can now run examples with:")
        print("  python -m q_store_examples.examples_v3_2")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
