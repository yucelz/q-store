"""
PyTorch nn.Module implementations for quantum layers.

This module provides PyTorch-compatible quantum layers that can be used in
standard PyTorch models. All layers inherit from nn.Module and support:
- Trainable quantum parameters via nn.Parameter
- Module registration and state management
- Device movement (to(), cuda(), cpu())
- State dict serialization
- Integration with PyTorch autograd
"""

from typing import Optional, List, Literal, Union
import math

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from ..core import UnifiedCircuit, GateType
    from ..backends import BackendManager
    from .circuit_executor import PyTorchCircuitExecutor
    from .gradients import QuantumExecution


class QuantumLayer(nn.Module):
    """
    PyTorch nn.Module for parameterized quantum circuits.

    This layer executes a parameterized quantum circuit with trainable parameters.
    It integrates seamlessly with PyTorch's autograd system for gradient computation.

    Args:
        n_qubits: Number of qubits in the quantum circuit
        depth: Number of repeated circuit layers
        backend: Backend name for circuit execution ('mock_ideal', 'cirq_simulator', etc.)
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
        measurement: Measurement type ('expectation' or 'sample')
        input_encoding: Input encoding method ('angle', 'amplitude', None)
        backend_manager: Optional BackendManager instance (creates default if None)

    Attributes:
        quantum_weights: nn.Parameter containing trainable quantum parameters

    Example:
        >>> layer = QuantumLayer(n_qubits=4, depth=2)
        >>> x = torch.randn(32, 4)  # Batch of 32 samples
        >>> output = layer(x)  # Shape: (32, 4)
        >>> loss = torch.nn.functional.mse_loss(output, target)
        >>> loss.backward()  # Gradients computed via parameter shift
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 1,
        backend: str = 'mock_ideal',
        entanglement: Literal['linear', 'circular', 'full'] = 'linear',
        measurement: Literal['expectation', 'sample'] = 'expectation',
        input_encoding: Optional[Literal['angle', 'amplitude']] = None,
        backend_manager: Optional['BackendManager'] = None,
    ):
        super().__init__()

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for QuantumLayer")

        self.n_qubits = n_qubits
        self.depth = depth
        self.backend_name = backend
        self.entanglement = entanglement
        self.measurement = measurement
        self.input_encoding = input_encoding

        # Initialize circuit executor
        self.executor = PyTorchCircuitExecutor(
            backend=backend,
            backend_manager=backend_manager
        )

        # Calculate number of parameters needed
        # Each layer has: n_qubits rotation gates + entangling gates
        self.n_params = n_qubits * depth * 3  # 3 rotations per qubit per layer

        # Create trainable parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )

    def _build_circuit(self, input_data: Optional[torch.Tensor] = None) -> UnifiedCircuit:
        """
        Build the quantum circuit for this layer.

        Args:
            input_data: Optional input tensor for encoding (shape: [batch, features])

        Returns:
            UnifiedCircuit ready for execution
        """
        circuit = UnifiedCircuit(self.n_qubits)
        param_idx = 0

        # Input encoding layer
        if input_data is not None and self.input_encoding == 'angle':
            for i in range(min(self.n_qubits, input_data.shape[-1])):
                circuit.add_gate(GateType.RY, [i], parameters={'theta': f'input_{i}'})
        elif input_data is not None and self.input_encoding == 'amplitude':
            # Amplitude encoding requires preparing state
            # For now, use angle encoding as approximation
            for i in range(min(self.n_qubits, input_data.shape[-1])):
                circuit.add_gate(GateType.RY, [i], parameters={'theta': f'input_{i}'})

        # Parameterized layers
        for d in range(self.depth):
            # Rotation layer
            for i in range(self.n_qubits):
                circuit.add_gate(GateType.RX, [i], parameters={'theta': f'param_{param_idx}'})
                param_idx += 1
                circuit.add_gate(GateType.RY, [i], parameters={'theta': f'param_{param_idx}'})
                param_idx += 1
                circuit.add_gate(GateType.RZ, [i], parameters={'theta': f'param_{param_idx}'})
                param_idx += 1

            # Entangling layer
            circuit.add_entangling_layer(GateType.CNOT, pattern=self.entanglement)

        return circuit

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x: Input tensor (optional), shape [batch_size, features] or None

        Returns:
            Quantum measurement results, shape [batch_size, n_qubits]
        """
        # Build base circuit
        circuit = self._build_circuit(x)

        # Bind input data if provided
        if x is not None and self.input_encoding is not None:
            batch_size = x.shape[0]
            n_features = min(self.n_qubits, x.shape[-1])

            # Execute circuit for each sample in batch using autograd function
            results = []
            for b in range(batch_size):
                # Prepare all parameters including input encoding
                all_params = torch.cat([
                    x[b, :n_features],  # Input features
                    self.quantum_weights  # Trainable weights
                ])

                # Create parameter name mapping
                param_names = [f'input_{i}' for i in range(n_features)] + \
                             [f'param_{i}' for i in range(self.n_params)]

                # Execute with autograd support
                result = QuantumExecution.apply(circuit, all_params, self.executor, param_names)
                results.append(result)

            return torch.stack(results)
        else:
            # Execute without input encoding using autograd function
            param_names = [f'param_{i}' for i in range(self.n_params)]
            result = QuantumExecution.apply(circuit, self.quantum_weights, self.executor, param_names)

            # If input is batched, repeat result for each batch element
            if x is not None:
                batch_size = x.shape[0]
                return result.unsqueeze(0).expand(batch_size, -1)

            return result

    def extra_repr(self) -> str:
        """Return extra representation string for this module."""
        return (
            f'n_qubits={self.n_qubits}, depth={self.depth}, '
            f'backend={self.backend_name}, entanglement={self.entanglement}'
        )


class AmplitudeEncoding(nn.Module):
    """
    Encode classical data as quantum amplitudes.

    This layer normalizes input data and encodes it as quantum state amplitudes.
    Requires 2^n_qubits features in the input.

    Args:
        n_qubits: Number of qubits (determines state space size 2^n_qubits)
        normalize: Whether to normalize input to unit vector

    Example:
        >>> encoder = AmplitudeEncoding(n_qubits=3)  # Expects 8 features
        >>> x = torch.randn(32, 8)
        >>> encoded = encoder(x)  # Shape: (32, 8) normalized
    """

    def __init__(self, n_qubits: int, normalize: bool = True):
        super().__init__()

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AmplitudeEncoding")

        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input as quantum amplitudes.

        Args:
            x: Input tensor, shape [batch_size, features]

        Returns:
            Normalized amplitudes, shape [batch_size, 2^n_qubits]
        """
        batch_size = x.shape[0]
        n_input_features = x.shape[-1]

        # Pad or truncate to match required size
        if n_input_features < self.n_features:
            padding = torch.zeros(
                batch_size, self.n_features - n_input_features,
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=-1)
        elif n_input_features > self.n_features:
            x = x[..., :self.n_features]

        # Normalize to unit vector
        if self.normalize:
            norm = torch.norm(x, dim=-1, keepdim=True)
            # Avoid division by zero
            norm = torch.where(norm > 1e-10, norm, torch.ones_like(norm))
            x = x / norm

        return x

    def extra_repr(self) -> str:
        """Return extra representation string for this module."""
        return f'n_qubits={self.n_qubits}, n_features={self.n_features}, normalize={self.normalize}'


class AngleEncoding(nn.Module):
    """
    Encode classical data as rotation angles.

    This layer scales input features to appropriate ranges for rotation gates
    and can be used before a QuantumLayer with input_encoding='angle'.

    Args:
        n_qubits: Number of qubits (maximum features to encode)
        scaling: Scaling factor for input features (default: π)

    Example:
        >>> encoder = AngleEncoding(n_qubits=4)
        >>> x = torch.randn(32, 4)
        >>> angles = encoder(x)  # Shape: (32, 4), scaled to [-π, π]
    """

    def __init__(self, n_qubits: int, scaling: float = math.pi):
        super().__init__()

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AngleEncoding")

        self.n_qubits = n_qubits
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input as rotation angles.

        Args:
            x: Input tensor, shape [batch_size, features]

        Returns:
            Scaled angles, shape [batch_size, min(features, n_qubits)]
        """
        # Truncate to n_qubits if needed
        if x.shape[-1] > self.n_qubits:
            x = x[..., :self.n_qubits]

        # Scale to rotation angle range
        return x * self.scaling

    def extra_repr(self) -> str:
        """Return extra representation string for this module."""
        return f'n_qubits={self.n_qubits}, scaling={self.scaling:.4f}'


# Export classes only if PyTorch is available
if not HAS_TORCH:
    QuantumLayer = None
    AmplitudeEncoding = None
    AngleEncoding = None
