"""
Basis encoding for quantum machine learning.

Basis encoding represents classical data in computational basis states.
"""

import numpy as np

from q_store.core import UnifiedCircuit, GateType


class BasisEncoding:
    """
    Basis Encoding (Computational Basis Encoding).

    Encodes classical binary data directly into computational basis:
    data = [1, 0, 1] → |101⟩

    For non-binary data, uses binary representation.
    """

    def __init__(self):
        pass

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode binary data into computational basis.

        Args:
            data: Binary data (0s and 1s)

        Returns:
            Quantum circuit preparing the basis state
        """
        # Convert to binary if needed
        binary_data = np.array(data, dtype=int)

        n_qubits = len(binary_data)
        circuit = UnifiedCircuit(n_qubits=n_qubits)

        # Apply X gates where data is 1
        for q, bit in enumerate(binary_data):
            if bit == 1:
                circuit.add_gate(GateType.X, targets=[q])

        return circuit

    def encode_integer(self, value: int, n_qubits: int) -> UnifiedCircuit:
        """
        Encode integer as binary in computational basis.

        Args:
            value: Integer value to encode
            n_qubits: Number of qubits (bits)

        Returns:
            Circuit encoding the integer
        """
        if value >= 2**n_qubits:
            raise ValueError(f"Value {value} too large for {n_qubits} qubits")

        # Convert to binary
        binary = format(value, f'0{n_qubits}b')
        data = [int(b) for b in binary]

        return self.encode(np.array(data))


def basis_encode(data: np.ndarray) -> UnifiedCircuit:
    """
    Convenience function for basis encoding.

    Args:
        data: Binary data vector

    Returns:
        Circuit encoding the data in computational basis
    """
    encoder = BasisEncoding()
    return encoder.encode(data)


def basis_encode_integer(value: int, n_qubits: int) -> UnifiedCircuit:
    """
    Encode integer value as binary quantum state.

    Args:
        value: Integer to encode
        n_qubits: Number of qubits

    Returns:
        Circuit encoding the integer
    """
    encoder = BasisEncoding()
    return encoder.encode_integer(value, n_qubits)
