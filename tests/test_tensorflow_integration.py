"""
Tests for TensorFlow Integration
"""

import pytest
import numpy as np

# Check if TensorFlow is available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from q_store.core import UnifiedCircuit, GateType

# Only run tests if TensorFlow is installed
pytestmark = pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")


class TestQuantumLayer:
    """Test QuantumLayer Keras layer"""

    def test_layer_creation(self):
        """Test basic layer creation"""
        from q_store.tensorflow import QuantumLayer

        layer = QuantumLayer(n_qubits=4, depth=2, backend='qsim')

        assert layer.n_qubits == 4
        assert layer.depth == 2
        assert layer.backend_name == 'qsim'

    def test_layer_build(self):
        """Test layer build creates parameters"""
        from q_store.tensorflow import QuantumLayer

        layer = QuantumLayer(n_qubits=2, depth=1)

        # Build layer with dummy input shape
        layer.build(input_shape=(None, 4))

        # Should have trainable parameters
        assert layer.theta is not None
        assert len(layer.trainable_weights) > 0

    def test_layer_in_model(self):
        """Test quantum layer in Keras model"""
        from q_store.tensorflow import QuantumLayer

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            QuantumLayer(n_qubits=2, depth=1, backend='qsim'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        # Check model structure
        assert len(model.layers) == 2  # QuantumLayer + Dense

        # Check output shape
        output_shape = model.compute_output_shape((None, 4))
        assert output_shape == (None, 2)

    def test_layer_get_config(self):
        """Test layer serialization"""
        from q_store.tensorflow import QuantumLayer

        layer = QuantumLayer(
            n_qubits=3,
            depth=2,
            backend='qsim',
            entanglement='circular'
        )

        config = layer.get_config()

        assert config['n_qubits'] == 3
        assert config['depth'] == 2
        assert config['backend'] == 'qsim'
        assert config['entanglement'] == 'circular'

        # Test from_config
        restored_layer = QuantumLayer.from_config(config)
        assert restored_layer.n_qubits == layer.n_qubits
        assert restored_layer.depth == layer.depth


class TestAmplitudeEncoding:
    """Test AmplitudeEncoding layer"""

    def test_encoding_creation(self):
        """Test amplitude encoding layer creation"""
        from q_store.tensorflow import AmplitudeEncoding

        layer = AmplitudeEncoding(n_qubits=3)

        assert layer.n_qubits == 3
        assert layer.n_features == 8  # 2^3

    def test_encoding_normalization(self):
        """Test that output is normalized"""
        from q_store.tensorflow import AmplitudeEncoding

        layer = AmplitudeEncoding(n_qubits=2, normalize=True)

        # Input data
        inputs = tf.constant([[1.0, 2.0, 3.0, 4.0]])

        # Encode
        encoded = layer(inputs)

        # Check shape
        assert encoded.shape == (1, 4)

        # Check normalization (should be unit vector)
        norm = tf.norm(encoded[0])
        assert abs(norm.numpy() - 1.0) < 1e-5

    def test_encoding_padding(self):
        """Test padding when input is smaller than 2^n_qubits"""
        from q_store.tensorflow import AmplitudeEncoding

        layer = AmplitudeEncoding(n_qubits=3)  # Expects 8 features

        # Input with only 4 features
        inputs = tf.constant([[1.0, 2.0, 3.0, 4.0]])

        encoded = layer(inputs)

        # Should be padded to 8
        assert encoded.shape == (1, 8)


class TestAngleEncoding:
    """Test AngleEncoding layer"""

    def test_angle_encoding(self):
        """Test angle encoding layer"""
        from q_store.tensorflow import AngleEncoding

        layer = AngleEncoding(n_qubits=4, scaling=np.pi)

        inputs = tf.constant([[0.5, 0.25, 0.75, 1.0]])
        encoded = layer(inputs)

        # Check shape
        assert encoded.shape == (1, 4)

        # Check scaling
        expected = inputs * np.pi
        np.testing.assert_allclose(encoded.numpy(), expected.numpy(), rtol=1e-5)

    def test_angle_encoding_truncation(self):
        """Test truncation when input is larger than n_qubits"""
        from q_store.tensorflow import AngleEncoding

        layer = AngleEncoding(n_qubits=2)

        # Input with 4 features
        inputs = tf.constant([[1.0, 2.0, 3.0, 4.0]])

        encoded = layer(inputs)

        # Should be truncated to 2
        assert encoded.shape == (1, 2)


class TestCircuitExecutor:
    """Test TensorFlowCircuitExecutor"""

    def test_executor_creation(self):
        """Test executor creation"""
        from q_store.tensorflow import TensorFlowCircuitExecutor

        executor = TensorFlowCircuitExecutor(backend='qsim', shots=1000)

        assert executor.backend_name == 'qsim'
        assert executor.shots == 1000

    def test_circuit_caching(self):
        """Test circuit caching"""
        from q_store.tensorflow import TensorFlowCircuitExecutor

        executor = TensorFlowCircuitExecutor(cache_circuits=True)

        # Get initial cache stats
        stats = executor.get_cache_stats()
        assert stats['cached_circuits'] == 0
        assert stats['cache_enabled'] is True

        # Clear cache
        executor.clear_cache()
        assert executor.get_cache_stats()['cached_circuits'] == 0


class TestGradients:
    """Test gradient computation methods"""

    def test_parameter_shift_gradient_creation(self):
        """Test parameter shift gradient computer creation"""
        from q_store.tensorflow.gradients import ParameterShiftGradient

        grad_computer = ParameterShiftGradient(backend='qsim')

        assert grad_computer.backend_name == 'qsim'
        assert grad_computer.shift_amount == np.pi / 2

    def test_adjoint_gradient_creation(self):
        """Test adjoint gradient computer creation"""
        from q_store.tensorflow.gradients import AdjointGradient

        grad_computer = AdjointGradient(backend='lightning')

        assert grad_computer.backend_name == 'lightning'

    def test_spsa_gradient_creation(self):
        """Test SPSA gradient estimator creation"""
        from q_store.tensorflow.gradients import SPSAGradient

        grad_computer = SPSAGradient(backend='qsim', epsilon=0.1)

        assert grad_computer.backend_name == 'qsim'
        assert grad_computer.epsilon == 0.1

    def test_gradient_method_selection(self):
        """Test automatic gradient method selection"""
        from q_store.tensorflow.gradients import select_gradient_method

        # Small circuit - should use parameter shift
        small_circuit = UnifiedCircuit(n_qubits=2)
        small_circuit.add_parameterized_layer(GateType.RY, 'theta')

        method = select_gradient_method(small_circuit, backend='qsim')
        assert method == 'parameter_shift'

        # Large circuit - should use SPSA
        large_circuit = UnifiedCircuit(n_qubits=6)
        for i in range(5):
            large_circuit.add_parameterized_layer(GateType.RY, f'theta_{i}')

        method = select_gradient_method(large_circuit, backend='qsim')
        assert method == 'spsa'


class TestTensorConversion:
    """Test circuit-tensor conversion"""

    def test_circuit_to_tensor(self):
        """Test converting circuit to tensor"""
        from q_store.tensorflow.circuit_executor import circuit_to_tensor

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        tensor = circuit_to_tensor(circuit)

        assert tensor.dtype == tf.string
        assert tensor.shape == ()

    def test_tensor_to_circuit(self):
        """Test converting tensor back to circuit"""
        from q_store.tensorflow.circuit_executor import (
            circuit_to_tensor,
            tensor_to_circuit
        )

        original = UnifiedCircuit(n_qubits=2)
        original.add_gate(GateType.H, targets=[0])
        original.add_gate(GateType.CNOT, targets=[0, 1])

        # Convert to tensor and back
        tensor = circuit_to_tensor(original)
        restored = tensor_to_circuit(tensor)

        assert restored.n_qubits == original.n_qubits
        assert len(restored.gates) == len(original.gates)


class TestIntegration:
    """Integration tests for TensorFlow module"""

    def test_end_to_end_model(self):
        """Test complete model creation and compilation"""
        from q_store.tensorflow import QuantumLayer

        # Create hybrid quantum-classical model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(4),
            QuantumLayer(n_qubits=2, depth=1, backend='qsim'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Check model summary works
        assert model.built
        assert len(model.trainable_weights) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
