"""
Tests for PyTorch integration components.

This module tests the PyTorch nn.Module layers, autograd integration,
and circuit execution within PyTorch computation graphs.
"""

import pytest
import sys

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not HAS_TORCH,
    reason="PyTorch not installed"
)

if HAS_TORCH:
    from q_store.torch import (
        QuantumLayer,
        AmplitudeEncoding,
        AngleEncoding,
        PyTorchCircuitExecutor,
        ParameterShiftGradient,
        AdjointGradient,
    )
    from q_store.torch.gradients import QuantumExecution, select_gradient_method
    from q_store.torch.circuit_executor import circuit_to_tensor, tensor_to_circuit
    from q_store.core import UnifiedCircuit, GateType


class TestQuantumLayer:
    """Test QuantumLayer nn.Module."""

    def test_creation(self):
        """Test creating a QuantumLayer."""
        layer = QuantumLayer(n_qubits=4, depth=2)

        assert layer.n_qubits == 4
        assert layer.depth == 2
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, 'quantum_weights')
        assert isinstance(layer.quantum_weights, nn.Parameter)

    def test_forward_pass(self):
        """Test forward pass through QuantumLayer."""
        layer = QuantumLayer(n_qubits=4, depth=1)

        # Forward pass without input
        output = layer()
        assert isinstance(output, torch.Tensor)
        assert output.shape[-1] == 4  # n_qubits outputs

    def test_forward_with_input(self):
        """Test forward pass with input encoding."""
        layer = QuantumLayer(n_qubits=4, depth=1, input_encoding='angle')

        # Create batch of inputs
        x = torch.randn(8, 4)  # batch_size=8, features=4
        output = layer(x)

        assert output.shape == (8, 4)  # [batch_size, n_qubits]

    def test_trainable_parameters(self):
        """Test that quantum weights are trainable."""
        layer = QuantumLayer(n_qubits=4, depth=2)

        # Check parameters
        params = list(layer.parameters())
        assert len(params) == 1
        assert params[0] is layer.quantum_weights
        assert params[0].requires_grad

    def test_in_sequential_model(self):
        """Test QuantumLayer in a sequential model."""
        model = nn.Sequential(
            AngleEncoding(n_qubits=4),
            QuantumLayer(n_qubits=4, depth=2, input_encoding='angle'),
            nn.Linear(4, 2)
        )

        x = torch.randn(8, 4)
        output = model(x)

        assert output.shape == (8, 2)

    def test_gradient_flow(self):
        """Test that gradients flow through QuantumLayer."""
        layer = QuantumLayer(n_qubits=2, depth=1)

        # Forward pass
        x = torch.randn(4, 2)
        output = layer(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert layer.quantum_weights.grad is not None

    def test_state_dict(self):
        """Test state dict serialization."""
        layer = QuantumLayer(n_qubits=4, depth=2)

        # Get state dict
        state = layer.state_dict()
        assert 'quantum_weights' in state

        # Create new layer and load state
        new_layer = QuantumLayer(n_qubits=4, depth=2)
        new_layer.load_state_dict(state)

        # Check weights match
        assert torch.allclose(layer.quantum_weights, new_layer.quantum_weights)

    def test_device_movement(self):
        """Test moving layer to different devices."""
        layer = QuantumLayer(n_qubits=4, depth=2)

        # Should be on CPU initially
        assert layer.quantum_weights.device.type == 'cpu'

        # Move to CPU explicitly (no-op)
        layer = layer.cpu()
        assert layer.quantum_weights.device.type == 'cpu'

        # Test CUDA only if available
        if torch.cuda.is_available():
            layer = layer.cuda()
            assert layer.quantum_weights.device.type == 'cuda'


class TestAmplitudeEncoding:
    """Test AmplitudeEncoding layer."""

    def test_creation(self):
        """Test creating AmplitudeEncoding layer."""
        encoder = AmplitudeEncoding(n_qubits=3)

        assert encoder.n_qubits == 3
        assert encoder.n_features == 8  # 2^3
        assert isinstance(encoder, nn.Module)

    def test_normalization(self):
        """Test amplitude normalization."""
        encoder = AmplitudeEncoding(n_qubits=3, normalize=True)

        x = torch.randn(4, 8)
        output = encoder(x)

        # Check normalization
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_padding(self):
        """Test padding when input has fewer features."""
        encoder = AmplitudeEncoding(n_qubits=3)  # Expects 8 features

        x = torch.randn(4, 5)  # Only 5 features
        output = encoder(x)

        assert output.shape == (4, 8)
        # Last 3 elements should be zero (padded)
        assert torch.allclose(output[:, 5:], torch.zeros(4, 3))

    def test_truncation(self):
        """Test truncation when input has more features."""
        encoder = AmplitudeEncoding(n_qubits=3)  # Expects 8 features

        x = torch.randn(4, 12)  # 12 features
        output = encoder(x)

        assert output.shape == (4, 8)
        # Should use first 8 features
        assert torch.allclose(output[:, :8], encoder(x[:, :8]))


class TestAngleEncoding:
    """Test AngleEncoding layer."""

    def test_creation(self):
        """Test creating AngleEncoding layer."""
        encoder = AngleEncoding(n_qubits=4)

        assert encoder.n_qubits == 4
        assert isinstance(encoder, nn.Module)

    def test_scaling(self):
        """Test angle scaling."""
        import math
        encoder = AngleEncoding(n_qubits=4, scaling=math.pi)

        x = torch.ones(4, 4)
        output = encoder(x)

        # Should be scaled by π
        assert torch.allclose(output, torch.ones(4, 4) * math.pi)

    def test_truncation(self):
        """Test truncation to n_qubits."""
        encoder = AngleEncoding(n_qubits=4)

        x = torch.randn(4, 8)  # More features than qubits
        output = encoder(x)

        assert output.shape == (4, 4)  # Truncated to n_qubits


class TestCircuitExecutor:
    """Test PyTorchCircuitExecutor."""

    def test_creation(self):
        """Test creating circuit executor."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal')

        assert executor.backend_name == 'mock_ideal'
        assert executor.cache_circuits

    def test_execute_simple_circuit(self):
        """Test executing a simple circuit."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal')

        # Create simple circuit
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        # Execute
        result = executor.execute(circuit)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)  # n_qubits outputs

    def test_execute_with_parameters(self):
        """Test executing circuit with parameters."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal')

        # Create parameterized circuit
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.RY, [0], parameters={'theta': 'theta_0'})

        # Execute with parameters
        params = {'theta_0': torch.tensor(0.5)}
        result = executor.execute(circuit, params)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    def test_circuit_caching(self):
        """Test circuit caching."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal', cache_circuits=True)

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, [0])

        # Execute twice
        result1 = executor.execute(circuit)
        result2 = executor.execute(circuit)

        # Results should be identical
        assert torch.allclose(result1, result2)

        # Clear cache and verify
        executor.clear_cache()
        assert len(executor._circuit_cache) == 0


class TestGradients:
    """Test gradient computation methods."""

    def test_parameter_shift_gradient(self):
        """Test ParameterShiftGradient."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal')
        gradient = ParameterShiftGradient(executor)

        # Create parameterized circuit
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.RY, [0], parameters={'theta': 'theta_0'})

        # Compute with gradient
        params = {'theta_0': torch.tensor(0.5)}
        output, grads = gradient.compute_with_gradient(circuit, params)

        assert isinstance(output, torch.Tensor)
        assert 'theta_0' in grads
        assert isinstance(grads['theta_0'], torch.Tensor)

    def test_adjoint_gradient(self):
        """Test AdjointGradient (may fall back to parameter shift)."""
        executor = PyTorchCircuitExecutor(backend='mock_ideal')
        gradient = AdjointGradient(executor)

        # Create parameterized circuit
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.RY, [0], parameters={'theta': 'theta_0'})

        # Compute with gradient
        params = {'theta_0': torch.tensor(0.5)}
        output, grads = gradient.compute_with_gradient(circuit, params)

        assert isinstance(output, torch.Tensor)
        assert 'theta_0' in grads

    def test_gradient_method_selection(self):
        """Test automatic gradient method selection."""
        # Small circuit should use parameter shift
        small_circuit = UnifiedCircuit(n_qubits=2)
        small_circuit.add_gate(GateType.RY, [0], parameters={'theta': 'theta_0'})

        method = select_gradient_method(small_circuit, 'mock_ideal')
        assert method in ['parameter_shift', 'adjoint']


class TestTensorConversion:
    """Test circuit to tensor conversion."""

    def test_circuit_to_tensor(self):
        """Test converting circuit to tensor."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        tensor = circuit_to_tensor(circuit)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.uint8

    def test_tensor_to_circuit(self):
        """Test converting tensor back to circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        # Convert to tensor and back
        tensor = circuit_to_tensor(circuit)
        reconstructed = tensor_to_circuit(tensor)

        assert reconstructed.n_qubits == circuit.n_qubits
        assert len(reconstructed.gates) == len(circuit.gates)


class TestIntegration:
    """Integration tests for PyTorch components."""

    def test_end_to_end_training(self):
        """Test end-to-end model creation and training step."""
        # Create model
        model = nn.Sequential(
            AngleEncoding(n_qubits=4),
            QuantumLayer(n_qubits=4, depth=1, input_encoding='angle'),
            nn.Linear(4, 2)
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training step
        x = torch.randn(8, 4)
        target = torch.randn(8, 2)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_multiple_quantum_layers(self):
        """Test model with multiple quantum layers."""
        model = nn.Sequential(
            QuantumLayer(n_qubits=4, depth=1),
            nn.ReLU(),
            QuantumLayer(n_qubits=4, depth=1),
            nn.Linear(4, 2)
        )

        x = torch.randn(4, 4)
        output = model(x)

        assert output.shape == (4, 2)

    def test_batch_processing(self):
        """Test batch processing through quantum layer."""
        layer = QuantumLayer(n_qubits=4, depth=1, input_encoding='angle')

        # Different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 4)
            output = layer(x)
            assert output.shape == (batch_size, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
