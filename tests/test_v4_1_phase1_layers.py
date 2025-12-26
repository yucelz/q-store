"""
Comprehensive Tests for Q-Store v4.1 Phase 1: Quantum-First Layers

Tests all Phase 1 components:
- QuantumFeatureExtractor with async execution
- QuantumNonlinearity (damping, phase, parametric)
- QuantumPooling (partial trace, measurement-based)
- QuantumReadout (Born rule measurements)
- EncodingLayer (minimal preprocessing)
- DecodingLayer (minimal postprocessing)
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import cirq

from q_store.layers import (
    QuantumFeatureExtractor,
    QuantumNonlinearity,
    QuantumPooling,
    QuantumReadout,
    EncodingLayer,
    DecodingLayer,
)


# ============================================================================
# Test QuantumFeatureExtractor
# ============================================================================

class TestQuantumFeatureExtractor:
    """Test QuantumFeatureExtractor with async execution."""

    def test_initialization(self):
        """Test layer initialization with various configurations."""
        # Basic initialization
        layer = QuantumFeatureExtractor(
            n_qubits=8,
            depth=3,
            entanglement='linear'
        )

        assert layer.n_qubits == 8
        assert layer.depth == 3
        assert layer.entanglement == 'linear'
        assert layer.n_parameters > 0

    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        patterns = ['linear', 'full', 'circular']

        for pattern in patterns:
            layer = QuantumFeatureExtractor(
                n_qubits=4,
                depth=2,
                entanglement=pattern
            )
            assert layer.entanglement == pattern

            # Build circuit to verify structure
            circuit = layer.build_circuit(np.random.randn(4))
            assert circuit is not None
            assert len(list(circuit.all_operations())) > 0

    def test_amplitude_encoding(self):
        """Test amplitude encoding of input data."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        # Test with normalized input
        input_data = np.array([0.5, 0.5, 0.5, 0.5])
        circuit = layer.build_circuit(input_data)

        assert circuit is not None
        # Verify circuit has encoding gates
        gates = list(circuit.all_operations())
        assert len(gates) > 0

    def test_multi_basis_measurements(self):
        """Test measurement in multiple bases (X, Y, Z)."""
        layer = QuantumFeatureExtractor(
            n_qubits=4,
            depth=2,
            measurement_bases=['X', 'Y', 'Z']
        )

        assert len(layer.measurement_bases) == 3

        # Output dimension should account for all bases
        expected_dim = 4 * 3  # n_qubits * n_bases
        assert layer.output_dim == expected_dim

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test async quantum execution."""
        layer = QuantumFeatureExtractor(
            n_qubits=4,
            depth=2,
            backend='cirq_simulator',
            shots=100
        )

        input_data = np.random.randn(4)

        # Execute asynchronously
        result = await layer.execute_async(input_data)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_parameter_updates(self):
        """Test parameter gradient updates."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        initial_params = layer.get_parameters().copy()

        # Simulate gradient update
        gradients = np.random.randn(len(initial_params)) * 0.01
        layer.update_parameters(gradients)

        updated_params = layer.get_parameters()

        # Verify parameters changed
        assert not np.allclose(initial_params, updated_params)

    def test_variational_layers(self):
        """Test variational quantum layers (RY, RZ gates)."""
        layer = QuantumFeatureExtractor(
            n_qubits=4,
            depth=3,
            use_ry=True,
            use_rz=True
        )

        circuit = layer.build_circuit(np.random.randn(4))
        gates = list(circuit.all_operations())

        # Check for rotation gates
        gate_types = [type(op.gate).__name__ for op in gates if hasattr(op, 'gate')]
        assert any('Ry' in gt or 'ry' in gt.lower() for gt in gate_types)
        assert any('Rz' in gt or 'rz' in gt.lower() for gt in gate_types)


# ============================================================================
# Test QuantumNonlinearity
# ============================================================================

class TestQuantumNonlinearity:
    """Test QuantumNonlinearity layer."""

    def test_amplitude_damping(self):
        """Test amplitude damping activation."""
        layer = QuantumNonlinearity(
            n_qubits=4,
            activation_type='amplitude_damping',
            damping_rate=0.1
        )

        assert layer.activation_type == 'amplitude_damping'
        assert layer.damping_rate == 0.1

        # Test forward pass
        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        assert output is not None
        assert len(output) == len(input_state)

    def test_phase_damping(self):
        """Test phase damping activation."""
        layer = QuantumNonlinearity(
            n_qubits=4,
            activation_type='phase_damping',
            damping_rate=0.15
        )

        assert layer.activation_type == 'phase_damping'

        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        assert output is not None

    def test_parametric_evolution(self):
        """Test parametric evolution activation."""
        layer = QuantumNonlinearity(
            n_qubits=4,
            activation_type='parametric',
            evolution_time=0.5
        )

        assert layer.activation_type == 'parametric'
        assert layer.evolution_time == 0.5

        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        assert output is not None

    def test_comparison_with_classical(self):
        """Compare quantum activation with classical ReLU/Tanh."""
        n_qubits = 4
        input_data = np.array([-1.0, -0.5, 0.5, 1.0])

        # Quantum activation
        quantum_layer = QuantumNonlinearity(
            n_qubits=n_qubits,
            activation_type='amplitude_damping'
        )
        quantum_output = quantum_layer.forward(input_data)

        # Classical activations
        relu_output = np.maximum(0, input_data)
        tanh_output = np.tanh(input_data)

        # Quantum should be different from classical
        assert not np.allclose(quantum_output, relu_output)
        assert not np.allclose(quantum_output, tanh_output)


# ============================================================================
# Test QuantumPooling
# ============================================================================

class TestQuantumPooling:
    """Test QuantumPooling layer."""

    def test_partial_trace_pooling(self):
        """Test partial trace pooling strategy."""
        layer = QuantumPooling(
            n_qubits=8,
            pooling_type='partial_trace',
            reduction_factor=2
        )

        assert layer.pooling_type == 'partial_trace'
        assert layer.output_qubits == 4  # 8 / 2

        input_state = np.random.randn(8)
        output = layer.forward(input_state)

        assert len(output) == 4

    def test_measurement_based_pooling(self):
        """Test measurement-based pooling."""
        layer = QuantumPooling(
            n_qubits=8,
            pooling_type='measurement',
            reduction_factor=2
        )

        assert layer.pooling_type == 'measurement'

        input_state = np.random.randn(8)
        output = layer.forward(input_state)

        assert len(output) < len(input_state)

    def test_pooling_dimension_reduction(self):
        """Test dimension reduction through pooling."""
        initial_dim = 16
        reduction_factor = 4

        layer = QuantumPooling(
            n_qubits=initial_dim,
            pooling_type='partial_trace',
            reduction_factor=reduction_factor
        )

        input_state = np.random.randn(initial_dim)
        output = layer.forward(input_state)

        expected_dim = initial_dim // reduction_factor
        assert len(output) == expected_dim


# ============================================================================
# Test QuantumReadout
# ============================================================================

class TestQuantumReadout:
    """Test QuantumReadout layer."""

    def test_computational_basis_measurement(self):
        """Test measurement in computational basis."""
        layer = QuantumReadout(
            n_qubits=4,
            n_classes=10,
            shots=1000
        )

        assert layer.n_qubits == 4
        assert layer.n_classes == 10
        assert layer.shots == 1000

        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        # Output should be probability distribution
        assert len(output) == 10
        assert np.isclose(np.sum(output), 1.0, atol=0.01)

    def test_born_rule_probabilities(self):
        """Test Born rule probability extraction."""
        layer = QuantumReadout(
            n_qubits=4,
            n_classes=10
        )

        input_state = np.random.randn(4)
        probs = layer.extract_probabilities(input_state)

        # Probabilities should sum to 1
        assert np.isclose(np.sum(probs), 1.0, atol=0.01)

        # All probabilities should be non-negative
        assert np.all(probs >= 0)

    def test_classification_output(self):
        """Test output for classification tasks."""
        n_classes = 5
        layer = QuantumReadout(
            n_qubits=4,
            n_classes=n_classes
        )

        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        assert len(output) == n_classes

        # Get predicted class
        predicted_class = np.argmax(output)
        assert 0 <= predicted_class < n_classes

    def test_regression_output(self):
        """Test output for regression tasks."""
        layer = QuantumReadout(
            n_qubits=4,
            task_type='regression',
            output_dim=3
        )

        input_state = np.random.randn(4)
        output = layer.forward(input_state)

        assert len(output) == 3
        # Regression output can be any real values


# ============================================================================
# Test EncodingLayer
# ============================================================================

class TestEncodingLayer:
    """Test minimal classical EncodingLayer."""

    def test_l2_normalization(self):
        """Test L2 normalization."""
        layer = EncodingLayer(
            output_dim=10,
            normalize=True
        )

        input_data = np.random.randn(10) * 5  # Large values
        output = layer.forward(input_data)

        # Check L2 norm is close to 1
        l2_norm = np.linalg.norm(output)
        assert np.isclose(l2_norm, 1.0, atol=0.01)

    def test_padding(self):
        """Test padding to target dimension."""
        target_dim = 16
        layer = EncodingLayer(
            output_dim=target_dim,
            normalize=False
        )

        input_data = np.random.randn(10)  # Smaller than target
        output = layer.forward(input_data)

        assert len(output) == target_dim

    def test_truncation(self):
        """Test truncation to target dimension."""
        target_dim = 8
        layer = EncodingLayer(
            output_dim=target_dim,
            normalize=False
        )

        input_data = np.random.randn(12)  # Larger than target
        output = layer.forward(input_data)

        assert len(output) == target_dim

    def test_minimal_processing(self):
        """Test that encoding is minimal (no heavy computation)."""
        layer = EncodingLayer(output_dim=10, normalize=True)

        input_data = np.random.randn(10)

        # Time the operation
        import time
        start = time.time()
        output = layer.forward(input_data)
        duration = time.time() - start

        # Should be very fast (<1ms)
        assert duration < 0.001


# ============================================================================
# Test DecodingLayer
# ============================================================================

class TestDecodingLayer:
    """Test minimal classical DecodingLayer."""

    def test_scaling(self):
        """Test output scaling."""
        scale = 2.0
        layer = DecodingLayer(
            output_dim=10,
            scale=scale
        )

        input_data = np.ones(10)
        output = layer.forward(input_data)

        # Output should be scaled
        assert np.allclose(output, input_data * scale)

    def test_optional_projection(self):
        """Test optional linear projection."""
        layer = DecodingLayer(
            output_dim=5,
            use_projection=True
        )

        input_data = np.random.randn(10)
        output = layer.forward(input_data)

        assert len(output) == 5

    def test_no_projection(self):
        """Test pass-through without projection."""
        layer = DecodingLayer(
            output_dim=10,
            use_projection=False,
            scale=1.0
        )

        input_data = np.random.randn(10)
        output = layer.forward(input_data)

        # Should be identical to input
        assert np.allclose(output, input_data)

    def test_minimal_processing(self):
        """Test that decoding is minimal."""
        layer = DecodingLayer(output_dim=10, use_projection=False)

        input_data = np.random.randn(10)

        # Time the operation
        import time
        start = time.time()
        output = layer.forward(input_data)
        duration = time.time() - start

        # Should be very fast
        assert duration < 0.001


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 layers."""

    def test_full_quantum_pipeline(self):
        """Test complete quantum-first pipeline."""
        input_dim = 28 * 28  # MNIST-like
        n_qubits = 8
        n_classes = 10

        # Build pipeline
        encoding = EncodingLayer(output_dim=n_qubits, normalize=True)
        feature_extractor = QuantumFeatureExtractor(n_qubits=n_qubits, depth=2)
        nonlinearity = QuantumNonlinearity(n_qubits=n_qubits, activation_type='amplitude_damping')
        pooling = QuantumPooling(n_qubits=n_qubits, pooling_type='partial_trace', reduction_factor=2)
        readout = QuantumReadout(n_qubits=n_qubits//2, n_classes=n_classes)
        decoding = DecodingLayer(output_dim=n_classes)

        # Forward pass
        x = np.random.randn(input_dim)

        x = encoding.forward(x[:n_qubits])  # Take first n_qubits
        x = feature_extractor.forward(x)
        x = nonlinearity.forward(x)
        x = pooling.forward(x)
        x = readout.forward(x)
        x = decoding.forward(x)

        # Final output should be class probabilities
        assert len(x) == n_classes
        assert np.isclose(np.sum(x), 1.0, atol=0.1)

    def test_layer_composability(self):
        """Test that layers can be composed in different orders."""
        n_qubits = 4

        layers = [
            EncodingLayer(output_dim=n_qubits),
            QuantumFeatureExtractor(n_qubits=n_qubits, depth=1),
            QuantumNonlinearity(n_qubits=n_qubits),
            DecodingLayer(output_dim=n_qubits),
        ]

        x = np.random.randn(n_qubits)

        # Forward through all layers
        for layer in layers:
            x = layer.forward(x)

        assert x is not None
        assert len(x) == n_qubits


# ============================================================================
# Performance Tests
# ============================================================================

class TestPhase1Performance:
    """Performance tests for Phase 1 layers."""

    def test_encoding_speed(self):
        """Test encoding layer is fast."""
        layer = EncodingLayer(output_dim=1000, normalize=True)

        import time
        start = time.time()

        for _ in range(1000):
            x = np.random.randn(1000)
            layer.forward(x)

        duration = time.time() - start

        # Should process 1000 samples in <0.1s
        assert duration < 0.1

    def test_quantum_layer_batching(self):
        """Test quantum layers can process batches efficiently."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        batch_size = 10
        batch = [np.random.randn(4) for _ in range(batch_size)]

        import time
        start = time.time()

        results = [layer.forward(x) for x in batch]

        duration = time.time() - start

        assert len(results) == batch_size
        # Should be reasonably fast
        assert duration < 5.0  # 5 seconds for 10 samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
