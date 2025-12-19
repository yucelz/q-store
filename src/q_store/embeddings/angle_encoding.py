"""
Angle encoding for quantum machine learning.

Angle encoding maps classical data to rotation angles.
"""

import numpy as np

from q_store.core import UnifiedCircuit, GateType


class AngleEncoding:
    """
    Angle Encoding (also called Basis Rotation Encoding).

    Encodes each feature as a rotation angle on a qubit:
    |ψ⟩ = ⊗_i R_i(x_i) |0⟩

    Args:
        rotation: Rotation type ('X', 'Y', 'Z')
    """

    def __init__(self, rotation: str = 'Y'):
        if rotation not in ('X', 'Y', 'Z'):
            raise ValueError(f"Rotation must be X, Y, or Z, got {rotation}")
        self.rotation = rotation

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode data as rotation angles.

        Args:
            data: Classical data vector (one value per qubit)

        Returns:
            Quantum circuit with angle-encoded data
        """
        n_qubits = len(data)
        circuit = UnifiedCircuit(n_qubits=n_qubits)

        # Map rotation type to gate type
        gate_map = {
            'X': GateType.RX,
            'Y': GateType.RY,
            'Z': GateType.RZ,
        }
        gate_type = gate_map[self.rotation]

        # Apply rotations
        for q, value in enumerate(data):
            circuit.add_gate(gate_type, targets=[q], parameters={'angle': value})

        return circuit


def angle_encode(data: np.ndarray, rotation: str = 'Y') -> UnifiedCircuit:
    """
    Convenience function for angle encoding.

    Args:
        data: Classical data vector
        rotation: Rotation axis ('X', 'Y', 'Z')

    Returns:
        Circuit with angle-encoded data
    """
    encoder = AngleEncoding(rotation=rotation)
    return encoder.encode(data)
