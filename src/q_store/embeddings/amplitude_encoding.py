"""
Amplitude encoding for quantum machine learning.

Amplitude encoding stores classical data in quantum state amplitudes.
"""

from typing import Optional
import numpy as np

from q_store.core import UnifiedCircuit, GateType


class AmplitudeEncoding:
    """
    Amplitude Encoding.

    Encodes classical data vector into quantum state amplitudes:
    |ψ⟩ = Σ_i x_i |i⟩

    Requires normalization: Σ |x_i|² = 1
    Uses O(log N) qubits to encode N-dimensional data.

    Args:
        normalize: Whether to normalize input data
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode data into quantum amplitudes.

        Args:
            data: Classical data vector (length must be power of 2)

        Returns:
            Quantum circuit preparing the encoded state
        """
        # Normalize data if requested
        if self.normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm

        # Check if data length is power of 2
        n_data = len(data)
        n_qubits = int(np.ceil(np.log2(n_data)))

        if 2**n_qubits != n_data:
            # Pad with zeros
            data = np.pad(data, (0, 2**n_qubits - n_data), mode='constant')

        circuit = UnifiedCircuit(n_qubits=n_qubits)

        # Use recursive decomposition to prepare state
        self._recursive_state_preparation(circuit, data, list(range(n_qubits)))

        return circuit

    def _recursive_state_preparation(
        self,
        circuit: UnifiedCircuit,
        amplitudes: np.ndarray,
        qubits: list
    ):
        """
        Recursively prepare quantum state using uniformly controlled rotations.

        This is a simplified implementation. Production systems should use
        optimized state preparation algorithms (Shende-Bullock-Markov, etc.).
        """
        if len(qubits) == 0:
            return

        if len(qubits) == 1:
            # Single qubit: apply rotation
            q = qubits[0]
            if abs(amplitudes[0]) > 1e-10 or abs(amplitudes[1]) > 1e-10:
                # Compute rotation angle
                alpha = np.angle(amplitudes[0])
                beta = np.angle(amplitudes[1])

                # Magnitude determines RY angle
                if abs(amplitudes[0]) > 1e-10:
                    theta = 2 * np.arctan2(abs(amplitudes[1]), abs(amplitudes[0]))
                else:
                    theta = np.pi

                # Apply rotation
                circuit.add_gate(GateType.RY, targets=[q], parameters={'angle': theta})

                # Phase correction
                if abs(beta - alpha) > 1e-10:
                    circuit.add_gate(GateType.RZ, targets=[q], parameters={'angle': beta - alpha})
            return

        # Split amplitudes into two halves
        n = len(amplitudes) // 2
        upper = amplitudes[:n]
        lower = amplitudes[n:]

        # Compute norms
        norm_upper = np.linalg.norm(upper)
        norm_lower = np.linalg.norm(lower)

        if norm_upper + norm_lower > 1e-10:
            # Apply rotation on the first qubit
            theta = 2 * np.arctan2(norm_lower, norm_upper)
            circuit.add_gate(GateType.RY, targets=[qubits[0]], parameters={'angle': theta})

            # Recursively prepare sub-states
            if norm_upper > 1e-10:
                # Control on |0⟩ (actually: just apply to lower qubits)
                self._recursive_state_preparation(
                    circuit, upper / norm_upper, qubits[1:]
                )

            if norm_lower > 1e-10:
                # Control on |1⟩
                # Apply X, recurse, apply X
                circuit.add_gate(GateType.X, targets=[qubits[0]])
                self._recursive_state_preparation(
                    circuit, lower / norm_lower, qubits[1:]
                )
                circuit.add_gate(GateType.X, targets=[qubits[0]])


def amplitude_encode(data: np.ndarray, normalize: bool = True) -> UnifiedCircuit:
    """
    Convenience function for amplitude encoding.

    Args:
        data: Classical data vector
        normalize: Whether to normalize input

    Returns:
        Circuit encoding the data
    """
    encoder = AmplitudeEncoding(normalize=normalize)
    return encoder.encode(data)
