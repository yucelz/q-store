"""
Quantum Neural Network Layer
Hardware-agnostic quantum layer for ML models
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..backends.quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    CircuitBuilder,
    GateType,
    ExecutionResult
)

logger = logging.getLogger(__name__)


@dataclass
class LayerConfig:
    """Configuration for quantum layer"""
    n_qubits: int
    depth: int
    entanglement: str = 'linear'  # 'linear', 'circular', 'full'
    measurement_basis: str = 'computational'  # 'computational', 'hadamard'
    trainable: bool = True


class QuantumLayer:
    """
    Quantum neural network layer with variational circuits

    Implements a parametrized quantum circuit that can be trained
    using gradient-based optimization.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        backend: QuantumBackend,
        entanglement: str = 'linear',
        measurement_basis: str = 'computational'
    ):
        """
        Initialize quantum layer

        Args:
            n_qubits: Number of qubits
            depth: Circuit depth (number of variational layers)
            backend: Quantum backend for execution
            entanglement: Entanglement pattern ('linear', 'circular', 'full')
            measurement_basis: Measurement basis ('computational', 'hadamard')
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.entanglement = entanglement
        self.measurement_basis = measurement_basis

        # Trainable parameters: 3 rotations per qubit per layer
        self.n_parameters = depth * n_qubits * 3
        self.parameters = self._initialize_parameters()

        # Parameter freezing for transfer learning
        self._frozen_params = set()

        # Track parameter history for debugging
        self._param_history = []

    def _initialize_parameters(self) -> np.ndarray:
        """Initialize parameters with small random values"""
        return np.random.randn(self.n_parameters) * 0.1

    def build_circuit(self, input_data: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Build the complete quantum circuit

        Args:
            input_data: Optional input data to encode

        Returns:
            QuantumCircuit ready for execution
        """
        builder = CircuitBuilder(self.n_qubits)

        # 1. Encoding layer (if input provided)
        if input_data is not None:
            self._add_encoding_layer(builder, input_data)

        # 2. Variational layers
        param_idx = 0
        for layer in range(self.depth):
            # Rotation layer
            param_idx = self._add_rotation_layer(builder, param_idx)
            # Entanglement layer
            self._add_entanglement_layer(builder)

        # 3. Measurement layer
        if self.measurement_basis == 'hadamard':
            # Apply Hadamard before measurement for X-basis
            for i in range(self.n_qubits):
                builder.h(i)

        builder.measure_all()

        return builder.build()

    def _add_encoding_layer(
        self,
        builder: CircuitBuilder,
        data: np.ndarray
    ) -> None:
        """
        Encode classical data into quantum state
        Uses angle encoding: R_y(θ_i)|0⟩
        """
        for i in range(min(self.n_qubits, len(data))):
            angle = data[i] * np.pi
            builder.ry(i, angle)

    def _add_rotation_layer(
        self,
        builder: CircuitBuilder,
        start_idx: int
    ) -> int:
        """
        Add trainable rotation gates
        Returns updated parameter index
        """
        for i in range(self.n_qubits):
            # RX, RY, RZ rotations for each qubit
            builder.rx(i, self.parameters[start_idx])
            builder.ry(i, self.parameters[start_idx + 1])
            builder.rz(i, self.parameters[start_idx + 2])
            start_idx += 3

        return start_idx

    def _add_entanglement_layer(self, builder: CircuitBuilder) -> None:
        """Add entanglement gates based on pattern"""
        if self.entanglement == 'linear':
            # Linear chain: 0-1, 1-2, 2-3, ...
            for i in range(self.n_qubits - 1):
                builder.cnot(i, i + 1)

        elif self.entanglement == 'circular':
            # Circular: linear + last-first connection
            for i in range(self.n_qubits - 1):
                builder.cnot(i, i + 1)
            builder.cnot(self.n_qubits - 1, 0)

        elif self.entanglement == 'full':
            # Full connectivity
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    builder.cnot(i, j)

        else:
            raise ValueError(f"Unknown entanglement pattern: {self.entanglement}")

    async def forward(
        self,
        x: np.ndarray,
        shots: int = 1000
    ) -> np.ndarray:
        """
        Forward pass through quantum layer

        Args:
            x: Input data (classical vector)
            shots: Number of measurement shots

        Returns:
            Output vector from measurements
        """
        # Build circuit with input encoding
        circuit = self.build_circuit(x)

        # Execute on quantum backend
        result = await self.backend.execute_circuit(circuit, shots=shots)

        # Process measurements to output vector
        output = self._process_measurements(result)

        return output

    def _process_measurements(self, result: ExecutionResult) -> np.ndarray:
        """
        Convert measurement results to output vector

        Uses expectation values of Pauli-Z on each qubit
        """
        output = np.zeros(self.n_qubits)

        for bitstring, count in result.counts.items():
            # Convert bitstring to string if it's an integer (from some backends)
            if isinstance(bitstring, int):
                bitstring = format(bitstring, 'b')
            # Pad bitstring to n_qubits
            bitstring = bitstring.zfill(self.n_qubits)
            # Compute expectation: +1 for |0⟩, -1 for |1⟩
            for i, bit in enumerate(bitstring):
                if bit == '0':
                    output[i] += count
                else:
                    output[i] -= count

        # Normalize by total shots
        output /= result.total_shots

        return output

    def freeze_parameters(self, indices: List[int]) -> None:
        """
        Freeze specific parameters (for transfer learning)

        Args:
            indices: List of parameter indices to freeze
        """
        self._frozen_params.update(indices)
        logger.info(f"Frozen {len(indices)} parameters")

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters"""
        self._frozen_params.clear()
        logger.info("Unfrozen all parameters")

    def update_parameters(self, new_params: np.ndarray) -> None:
        """
        Update trainable parameters

        Args:
            new_params: New parameter values
        """
        if len(new_params) != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, got {len(new_params)}"
            )

        # Update only non-frozen parameters
        for i in range(self.n_parameters):
            if i not in self._frozen_params:
                self.parameters[i] = new_params[i]

        # Track history
        self._param_history.append(self.parameters.copy())

    def get_parameters(self) -> np.ndarray:
        """Get current parameters"""
        return self.parameters.copy()

    def compute_entanglement_entropy(self) -> float:
        """
        Compute entanglement entropy of the current state

        This is a simplified version - full implementation would
        require state tomography
        """
        # Placeholder: estimate based on entanglement pattern
        if self.entanglement == 'linear':
            return 0.5 * self.depth * (self.n_qubits - 1)
        elif self.entanglement == 'circular':
            return 0.5 * self.depth * self.n_qubits
        elif self.entanglement == 'full':
            return 0.5 * self.depth * self.n_qubits * (self.n_qubits - 1) / 2
        return 0.0

    def get_config(self) -> LayerConfig:
        """Get layer configuration"""
        return LayerConfig(
            n_qubits=self.n_qubits,
            depth=self.depth,
            entanglement=self.entanglement,
            measurement_basis=self.measurement_basis,
            trainable=len(self._frozen_params) < self.n_parameters
        )

    def save_state(self) -> Dict[str, Any]:
        """Save layer state for checkpointing"""
        return {
            'parameters': self.parameters.tolist(),
            'frozen_params': list(self._frozen_params),
            'config': {
                'n_qubits': self.n_qubits,
                'depth': self.depth,
                'entanglement': self.entanglement,
                'measurement_basis': self.measurement_basis
            }
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load layer state from checkpoint"""
        self.parameters = np.array(state['parameters'])
        self._frozen_params = set(state['frozen_params'])
        logger.info(f"Loaded layer state with {self.n_parameters} parameters")


class QuantumConvolutionalLayer(QuantumLayer):
    """
    Quantum convolutional layer
    Applies quantum circuit as a sliding window
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        backend: QuantumBackend,
        kernel_size: int = 2,
        stride: int = 1,
        **kwargs
    ):
        super().__init__(n_qubits, depth, backend, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

        # Recalculate parameters for convolutional structure
        self.n_parameters = depth * kernel_size * 3
        self.parameters = self._initialize_parameters()

    async def forward(
        self,
        x: np.ndarray,
        shots: int = 1000
    ) -> np.ndarray:
        """
        Convolutional forward pass
        Slides kernel across input qubits
        """
        outputs = []

        # Slide window across qubits
        for i in range(0, self.n_qubits - self.kernel_size + 1, self.stride):
            # Create sub-circuit for this window
            window_data = x[i:i+self.kernel_size] if len(x) > i else x[i:]

            # Build and execute circuit for this window
            builder = CircuitBuilder(self.kernel_size)
            self._add_encoding_layer(builder, window_data)

            param_idx = 0
            for layer in range(self.depth):
                param_idx = self._add_rotation_layer(builder, param_idx)

            builder.measure_all()
            circuit = builder.build()

            result = await self.backend.execute_circuit(circuit, shots=shots)
            output = self._process_measurements(result)
            outputs.append(output.mean())  # Pool the window output

        return np.array(outputs)


class QuantumPoolingLayer:
    """
    Quantum pooling layer
    Reduces dimensionality via quantum measurements
    """

    def __init__(
        self,
        pool_size: int = 2,
        method: str = 'max'
    ):
        """
        Initialize pooling layer

        Args:
            pool_size: Size of pooling window
            method: 'max' or 'average'
        """
        self.pool_size = pool_size
        self.method = method

    async def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Pool input by taking max or average over windows

        Args:
            x: Input vector

        Returns:
            Pooled output
        """
        pooled = []

        for i in range(0, len(x), self.pool_size):
            window = x[i:i+self.pool_size]

            if self.method == 'max':
                pooled.append(np.max(window))
            elif self.method == 'average':
                pooled.append(np.mean(window))
            else:
                raise ValueError(f"Unknown pooling method: {self.method}")

        return np.array(pooled)
