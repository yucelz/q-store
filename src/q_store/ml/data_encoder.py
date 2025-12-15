"""
Quantum Data Encoder
Encodes classical data into quantum states
"""

import numpy as np
import logging
from typing import Optional

from ..backends.quantum_backend_interface import (
    QuantumCircuit,
    CircuitBuilder,
    GateType
)

logger = logging.getLogger(__name__)


class QuantumDataEncoder:
    """
    Encodes classical data into quantum states

    Supports multiple encoding strategies:
    - Amplitude encoding
    - Angle encoding
    - Basis encoding
    """

    def __init__(self, encoding_type: str = 'amplitude'):
        """
        Initialize encoder

        Args:
            encoding_type: Type of encoding ('amplitude', 'angle', 'basis')
        """
        self.encoding_type = encoding_type

    def amplitude_encode(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> QuantumCircuit:
        """
        Encode data as quantum amplitudes

        Maps N-dimensional vector to quantum state:
        |ψ⟩ = Σ_i α_i |i⟩

        where α_i are the data values (normalized)

        Args:
            data: Classical data vector
            normalize: Whether to normalize the data

        Returns:
            QuantumCircuit with encoded state
        """
        # Normalize data
        if normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm
            else:
                logger.warning("Cannot normalize zero vector")

        # Pad to power of 2
        n_qubits = int(np.ceil(np.log2(len(data))))
        padded_data = np.pad(data, (0, 2**n_qubits - len(data)))

        # Build state preparation circuit
        circuit = self._build_amplitude_circuit(padded_data, n_qubits)

        return circuit

    def _build_amplitude_circuit(
        self,
        amplitudes: np.ndarray,
        n_qubits: int
    ) -> QuantumCircuit:
        """
        Build circuit for amplitude encoding

        Uses recursive decomposition into rotation and CNOT gates
        This is a simplified version - full implementation would use
        more sophisticated state preparation algorithms
        """
        builder = CircuitBuilder(n_qubits)

        # Simplified amplitude encoding using rotations
        for i in range(min(n_qubits, len(amplitudes))):
            if abs(amplitudes[i]) > 1e-10:
                # Compute rotation angle
                angle = 2 * np.arcsin(min(abs(amplitudes[i]), 1.0))
                builder.ry(i, angle)

        # Add entanglement to create superposition
        for i in range(n_qubits - 1):
            builder.cnot(i, i + 1)

        return builder.build()

    def angle_encode(
        self,
        data: np.ndarray,
        n_qubits: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Encode data as rotation angles

        Each feature x_i maps to rotation:
        R_y(θ_i)|0⟩ where θ_i = x_i * π

        Args:
            data: Classical data vector
            n_qubits: Number of qubits (uses len(data) if None)

        Returns:
            QuantumCircuit with angle-encoded state
        """
        if n_qubits is None:
            n_qubits = len(data)

        builder = CircuitBuilder(n_qubits)

        # Encode each feature as rotation angle
        for i in range(min(n_qubits, len(data))):
            angle = data[i] * np.pi
            builder.ry(i, angle)

        # Optional: Add entanglement
        for i in range(n_qubits - 1):
            builder.cnot(i, i + 1)

        return builder.build()

    def basis_encode(
        self,
        data: np.ndarray,
        threshold: float = 0.5
    ) -> QuantumCircuit:
        """
        Encode binary data in computational basis

        Each bit maps to |0⟩ or |1⟩

        Args:
            data: Classical data vector
            threshold: Threshold for binarization

        Returns:
            QuantumCircuit with basis-encoded state
        """
        # Binarize data
        binary_data = (data > threshold).astype(int)

        n_qubits = len(binary_data)
        builder = CircuitBuilder(n_qubits)

        # Apply X gate where bit is 1
        for i, bit in enumerate(binary_data):
            if bit == 1:
                builder.x(i)

        return builder.build()

    def encode(
        self,
        data: np.ndarray,
        n_qubits: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Encode data using configured encoding type

        Args:
            data: Classical data to encode
            n_qubits: Number of qubits (optional)

        Returns:
            QuantumCircuit with encoded data
        """
        if self.encoding_type == 'amplitude':
            return self.amplitude_encode(data)
        elif self.encoding_type == 'angle':
            return self.angle_encode(data, n_qubits)
        elif self.encoding_type == 'basis':
            return self.basis_encode(data)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def encode_batch(
        self,
        batch: np.ndarray,
        n_qubits: Optional[int] = None
    ) -> list:
        """
        Encode a batch of data samples

        Args:
            batch: Batch of data samples (shape: [batch_size, features])
            n_qubits: Number of qubits per sample

        Returns:
            List of QuantumCircuits
        """
        circuits = []

        for sample in batch:
            circuit = self.encode(sample, n_qubits)
            circuits.append(circuit)

        return circuits


class QuantumFeatureMap:
    """
    Advanced quantum feature mapping
    Creates high-dimensional feature spaces
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map_type: str = 'ZZFeatureMap'
    ):
        """
        Initialize feature map

        Args:
            n_qubits: Number of qubits
            feature_map_type: Type of feature map
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type

    def zz_feature_map(
        self,
        data: np.ndarray,
        reps: int = 2
    ) -> QuantumCircuit:
        """
        ZZ feature map (second-order Pauli feature map)

        Creates entanglement via ZZ interactions

        Args:
            data: Input features
            reps: Number of repetitions

        Returns:
            QuantumCircuit implementing feature map
        """
        builder = CircuitBuilder(self.n_qubits)

        for rep in range(reps):
            # Hadamard layer
            for i in range(self.n_qubits):
                builder.h(i)

            # Encode data
            for i in range(min(self.n_qubits, len(data))):
                builder.rz(i, data[i])

            # Entanglement layer (ZZ interactions)
            for i in range(self.n_qubits - 1):
                # ZZ(θ) = exp(-i θ Z⊗Z/2) implemented via CNOTs
                builder.cnot(i, i + 1)
                angle = (data[i] if i < len(data) else 0) * (
                    data[i + 1] if i + 1 < len(data) else 0
                )
                builder.rz(i + 1, angle)
                builder.cnot(i, i + 1)

        return builder.build()

    def pauli_feature_map(
        self,
        data: np.ndarray,
        paulis: str = 'ZZ'
    ) -> QuantumCircuit:
        """
        General Pauli feature map

        Args:
            data: Input features
            paulis: Pauli string ('Z', 'ZZ', 'ZZZ', etc.)

        Returns:
            QuantumCircuit implementing Pauli feature map
        """
        if paulis == 'ZZ':
            return self.zz_feature_map(data)

        builder = CircuitBuilder(self.n_qubits)

        # Hadamard initialization
        for i in range(self.n_qubits):
            builder.h(i)

        # Encode with Pauli rotations
        for i in range(min(self.n_qubits, len(data))):
            builder.rz(i, data[i])

        return builder.build()

    def map_features(self, data: np.ndarray) -> QuantumCircuit:
        """
        Map features using configured feature map

        Args:
            data: Input features

        Returns:
            QuantumCircuit with mapped features
        """
        if self.feature_map_type == 'ZZFeatureMap':
            return self.zz_feature_map(data)
        elif self.feature_map_type == 'PauliFeatureMap':
            return self.pauli_feature_map(data)
        else:
            # Default to simple angle encoding
            encoder = QuantumDataEncoder('angle')
            return encoder.encode(data, self.n_qubits)
