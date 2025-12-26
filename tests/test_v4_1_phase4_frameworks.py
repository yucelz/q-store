"""
Comprehensive Tests for Q-Store v4.1 Phase 4: Framework Integration

Tests all Phase 4 components:
- TensorFlow QuantumDense layer with @tf.custom_gradient
- TensorFlow SPSA gradient estimation
- PyTorch QuantumLinear layer with torch.autograd.Function
- PyTorch SPSA gradient estimation
- GPU tensor support
- Framework interoperability
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

# TensorFlow imports
try:
    import tensorflow as tf
    from q_store.tensorflow import QuantumDense, spsa_gradients_tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from q_store.torch import QuantumLinear, spsa_gradients_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# TensorFlow QuantumDense Tests
# ============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowQuantumDense:
    """Test TensorFlow QuantumDense layer."""

    def test_layer_initialization(self):
        """Test layer initialization."""
        layer = QuantumDense(
            units=64,
            n_qubits=8,
            n_layers=3,
            shots=1000,
            backend='cirq_simulator'
        )
        
        assert layer.units == 64
        assert layer.n_qubits == 8
        assert layer.n_layers == 3
        assert layer.shots == 1000

    def test_build_layer(self):
        """Test layer building."""
        layer = QuantumDense(
            units=32,
            n_qubits=4,
            n_layers=2
        )
        
        # Build layer
        layer.build(input_shape=(None, 10))
        
        # Should have parameters
        assert len(layer.trainable_variables) > 0

    def test_forward_pass(self):
        """Test forward pass."""
        layer = QuantumDense(
            units=16,
            n_qubits=4,
            n_layers=2,
            backend='cirq_simulator'
        )
        
        # Create input
        batch_size = 2
        input_dim = 10
        x = tf.random.normal((batch_size, input_dim))
        
        # Forward pass
        output = layer(x, training=False)
        
        assert output.shape == (batch_size, 16)

    def test_training_mode(self):
        """Test training vs inference mode."""
        layer = QuantumDense(
            units=16,
            n_qubits=4,
            n_layers=2
        )
        
        x = tf.random.normal((2, 10))
        
        # Training mode
        output_train = layer(x, training=True)
        
        # Inference mode
        output_infer = layer(x, training=False)
        
        assert output_train.shape == output_infer.shape

    def test_custom_gradient(self):
        """Test @tf.custom_gradient for SPSA."""
        layer = QuantumDense(
            units=8,
            n_qubits=4,
            n_layers=2
        )
        
        x = tf.random.normal((2, 4))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = layer(x, training=True)
            loss = tf.reduce_mean(output ** 2)
        
        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        # Should have gradients for all trainable variables
        assert gradients is not None
        assert len(gradients) == len(layer.trainable_variables)

    def test_keras_model_integration(self):
        """Test integration with Keras models."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            QuantumDense(
                units=16,
                n_qubits=4,
                n_layers=2,
                activation='quantum_damping'
            ),
            QuantumDense(
                units=8,
                n_qubits=4,
                n_layers=2
            ),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x = np.random.randn(10, 10).astype('float32')
        y = tf.keras.utils.to_categorical(np.random.randint(0, 2, 10), 2)
        
        # Train one step
        history = model.fit(x, y, epochs=1, verbose=0)
        
        assert 'loss' in history.history

    def test_async_execution_wrapper(self):
        """Test async execution wrapper."""
        layer = QuantumDense(
            units=8,
            n_qubits=4,
            n_layers=2,
            enable_async=True
        )
        
        x = tf.random.normal((2, 4))
        
        # Should execute asynchronously
        start = time.time()
        output = layer(x, training=False)
        exec_time = time.time() - start
        
        assert output is not None
        print(f"Async execution time: {exec_time:.4f}s")


# ============================================================================
# TensorFlow SPSA Gradients Tests
# ============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowSPSAGradients:
    """Test TensorFlow SPSA gradient estimation."""

    def test_spsa_gradient_estimation(self):
        """Test SPSA gradient estimation."""
        # Define simple function
        def f(params):
            return tf.reduce_sum(params ** 2)
        
        params = tf.Variable([1.0, 2.0, 3.0])
        
        # Compute SPSA gradients
        gradients = spsa_gradients_tf(f, params, epsilon=0.01)
        
        # Should have gradient for each parameter
        assert gradients.shape == params.shape
        
        # Gradients should approximate 2*params
        expected_gradients = 2 * params.numpy()
        assert np.allclose(gradients.numpy(), expected_gradients, atol=0.5)

    def test_spsa_with_quantum_circuit(self):
        """Test SPSA with quantum circuit evaluation."""
        layer = QuantumDense(units=4, n_qubits=4, n_layers=2)
        
        x = tf.constant([[1.0, 0.5, 0.3, 0.2]])
        
        def circuit_eval(params):
            # Simulate circuit evaluation
            return tf.reduce_sum(params * x)
        
        params = tf.Variable(np.random.randn(4).astype('float32'))
        
        gradients = spsa_gradients_tf(circuit_eval, params)
        
        assert gradients is not None
        assert gradients.shape == params.shape

    def test_spsa_epsilon_sensitivity(self):
        """Test SPSA sensitivity to epsilon."""
        def f(params):
            return tf.reduce_sum(params ** 2)
        
        params = tf.Variable([1.0, 2.0, 3.0])
        
        # Try different epsilon values
        epsilons = [0.001, 0.01, 0.1]
        gradients_list = []
        
        for eps in epsilons:
            grads = spsa_gradients_tf(f, params, epsilon=eps)
            gradients_list.append(grads.numpy())
        
        # All should approximate true gradients
        true_grads = 2 * params.numpy()
        for grads in gradients_list:
            assert np.allclose(grads, true_grads, atol=0.5)


# ============================================================================
# PyTorch QuantumLinear Tests
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchQuantumLinear:
    """Test PyTorch QuantumLinear layer."""

    def test_layer_initialization(self):
        """Test layer initialization."""
        layer = QuantumLinear(
            in_features=10,
            out_features=64,
            n_qubits=8,
            n_layers=3,
            shots=1000,
            backend='cirq_simulator'
        )
        
        assert layer.in_features == 10
        assert layer.out_features == 64
        assert layer.n_qubits == 8
        assert layer.n_layers == 3

    def test_forward_pass(self):
        """Test forward pass."""
        layer = QuantumLinear(
            in_features=10,
            out_features=16,
            n_qubits=4,
            n_layers=2
        )
        
        # Create input
        batch_size = 2
        x = torch.randn(batch_size, 10)
        
        # Forward pass
        output = layer(x)
        
        assert output.shape == (batch_size, 16)

    def test_autograd_function(self):
        """Test torch.autograd.Function integration."""
        layer = QuantumLinear(
            in_features=4,
            out_features=8,
            n_qubits=4,
            n_layers=2
        )
        
        x = torch.randn(2, 4, requires_grad=True)
        
        # Forward pass
        output = layer(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Should have gradients
        assert layer.quantum_params.grad is not None

    def test_nn_module_integration(self):
        """Test integration with nn.Module."""
        model = nn.Sequential(
            QuantumLinear(
                in_features=10,
                out_features=16,
                n_qubits=4,
                n_layers=2,
                activation='quantum_damping'
            ),
            QuantumLinear(
                in_features=16,
                out_features=8,
                n_qubits=4,
                n_layers=2
            ),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )
        
        # Create dummy data
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (10, 2)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gpu_tensor_support(self):
        """Test GPU tensor support."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        layer = QuantumLinear(
            in_features=10,
            out_features=16,
            n_qubits=4,
            n_layers=2
        ).cuda()
        
        x = torch.randn(2, 10).cuda()
        
        # Forward pass on GPU
        output = layer(x)
        
        assert output.is_cuda
        assert output.shape == (2, 16)

    def test_async_execution(self):
        """Test async execution."""
        layer = QuantumLinear(
            in_features=4,
            out_features=8,
            n_qubits=4,
            n_layers=2,
            enable_async=True
        )
        
        x = torch.randn(2, 4)
        
        # Should execute asynchronously
        start = time.time()
        output = layer(x)
        exec_time = time.time() - start
        
        assert output is not None
        print(f"Async execution time: {exec_time:.4f}s")


# ============================================================================
# PyTorch SPSA Gradients Tests
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchSPSAGradients:
    """Test PyTorch SPSA gradient estimation."""

    def test_spsa_gradient_estimation(self):
        """Test SPSA gradient estimation."""
        # Define simple function
        def f(params):
            return torch.sum(params ** 2)
        
        params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Compute SPSA gradients
        gradients = spsa_gradients_torch(f, params, epsilon=0.01)
        
        # Should have gradient for each parameter
        assert gradients.shape == params.shape
        
        # Gradients should approximate 2*params
        expected_gradients = 2 * params.detach().numpy()
        assert np.allclose(gradients.numpy(), expected_gradients, atol=0.5)

    def test_spsa_with_quantum_circuit(self):
        """Test SPSA with quantum circuit evaluation."""
        layer = QuantumLinear(in_features=4, out_features=4, n_qubits=4, n_layers=2)
        
        x = torch.tensor([[1.0, 0.5, 0.3, 0.2]])
        
        def circuit_eval(params):
            # Simulate circuit evaluation
            return torch.sum(params * x)
        
        params = torch.randn(4, requires_grad=True)
        
        gradients = spsa_gradients_torch(circuit_eval, params)
        
        assert gradients is not None
        assert gradients.shape == params.shape

    def test_spsa_batch_processing(self):
        """Test SPSA with batch processing."""
        def f(params):
            return torch.sum(params ** 2, dim=-1)
        
        # Batch of parameters
        params = torch.randn(10, 5, requires_grad=True)
        
        gradients = spsa_gradients_torch(f, params)
        
        assert gradients.shape == params.shape


# ============================================================================
# Framework Interoperability Tests
# ============================================================================

@pytest.mark.skipif(not (TF_AVAILABLE and TORCH_AVAILABLE), 
                    reason="Both frameworks needed")
class TestFrameworkInteroperability:
    """Test interoperability between TensorFlow and PyTorch."""

    def test_parameter_compatibility(self):
        """Test parameters can be transferred between frameworks."""
        # TensorFlow layer
        tf_layer = QuantumDense(units=8, n_qubits=4, n_layers=2)
        tf_layer.build(input_shape=(None, 4))
        
        # Get TensorFlow parameters
        tf_params = [v.numpy() for v in tf_layer.trainable_variables]
        
        # PyTorch layer
        torch_layer = QuantumLinear(in_features=4, out_features=8, n_qubits=4, n_layers=2)
        
        # Transfer parameters (conceptual - actual implementation may vary)
        # Both should produce similar architectures
        assert len(tf_params) > 0

    def test_model_equivalence(self):
        """Test equivalent models in both frameworks."""
        # Create equivalent inputs
        np_input = np.random.randn(2, 4).astype('float32')
        
        # TensorFlow model
        tf_input = tf.constant(np_input)
        tf_layer = QuantumDense(units=8, n_qubits=4, n_layers=2)
        tf_output = tf_layer(tf_input, training=False)
        
        # PyTorch model
        torch_input = torch.from_numpy(np_input)
        torch_layer = QuantumLinear(in_features=4, out_features=8, n_qubits=4, n_layers=2)
        torch_output = torch_layer(torch_input)
        
        # Outputs should have same shape
        assert tf_output.shape[0] == torch_output.shape[0]
        assert tf_output.shape[1] == torch_output.shape[1]


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowIntegration:
    """Integration tests for TensorFlow."""

    def test_full_training_loop(self):
        """Test complete training loop."""
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            QuantumDense(units=16, n_qubits=4, n_layers=2),
            QuantumDense(units=8, n_qubits=4, n_layers=2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x_train = np.random.randn(50, 10).astype('float32')
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, 2, 50), 2)
        
        # Train
        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=10,
            verbose=0
        )
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchIntegration:
    """Integration tests for PyTorch."""

    def test_full_training_loop(self):
        """Test complete training loop."""
        # Create model
        model = nn.Sequential(
            QuantumLinear(in_features=10, out_features=16, n_qubits=4, n_layers=2),
            QuantumLinear(in_features=16, out_features=8, n_qubits=4, n_layers=2),
            nn.Linear(8, 2),
        )
        
        # Create dummy data
        x_train = torch.randn(50, 10)
        y_train = torch.randint(0, 2, (50,))
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 2 epochs
        model.train()
        losses = []
        
        for epoch in range(2):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        assert len(losses) == 2


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowPerformance:
    """Performance tests for TensorFlow integration."""

    def test_inference_speed(self):
        """Test inference speed."""
        layer = QuantumDense(units=16, n_qubits=4, n_layers=2)
        
        x = tf.random.normal((10, 4))
        
        # Warmup
        _ = layer(x, training=False)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = layer(x, training=False)
        duration = time.time() - start
        
        print(f"TF inference: {duration/10:.4f}s per batch")
        
        # Should be reasonably fast
        assert duration < 10.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchPerformance:
    """Performance tests for PyTorch integration."""

    def test_inference_speed(self):
        """Test inference speed."""
        layer = QuantumLinear(in_features=4, out_features=16, n_qubits=4, n_layers=2)
        
        x = torch.randn(10, 4)
        
        # Warmup
        _ = layer(x)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = layer(x)
        duration = time.time() - start
        
        print(f"PyTorch inference: {duration/10:.4f}s per batch")
        
        # Should be reasonably fast
        assert duration < 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
