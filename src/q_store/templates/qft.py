"""
Quantum Fourier Transform (QFT) implementation.

The QFT is a quantum analogue of the discrete Fourier transform and is
a key component of many quantum algorithms including Shor's algorithm
and quantum phase estimation.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType


def qft(n_qubits: int, approximation_degree: int = None) -> UnifiedCircuit:
    """
    Create Quantum Fourier Transform circuit.

    The QFT transforms the computational basis states as:
    |j⟩ → (1/√N) Σ_k exp(2πijk/N) |k⟩

    Args:
        n_qubits: Number of qubits
        approximation_degree: If set, use approximate QFT (drop small rotations)
                             to reduce circuit depth

    Returns:
        QFT circuit
    """
    circuit = UnifiedCircuit(n_qubits=n_qubits)

    for j in range(n_qubits):
        # Apply Hadamard to current qubit
        circuit.add_gate(GateType.H, targets=[j])

        # Apply controlled rotations
        for k in range(j + 1, n_qubits):
            # Skip small rotations if approximation is requested
            if approximation_degree is not None:
                if k - j > approximation_degree:
                    continue

            # Controlled phase rotation
            angle = 2 * np.pi / (2 ** (k - j + 1))
            _add_controlled_phase(circuit, control=k, target=j, angle=angle)

    # Swap qubits to reverse order (optional depending on convention)
    for i in range(n_qubits // 2):
        circuit.add_gate(GateType.SWAP, targets=[i, n_qubits - 1 - i])

    return circuit


def inverse_qft(n_qubits: int, approximation_degree: int = None) -> UnifiedCircuit:
    """
    Create inverse Quantum Fourier Transform circuit.

    Args:
        n_qubits: Number of qubits
        approximation_degree: Approximation degree for QFT

    Returns:
        Inverse QFT circuit
    """
    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Swap qubits first (reverse of QFT)
    for i in range(n_qubits // 2):
        circuit.add_gate(GateType.SWAP, targets=[i, n_qubits - 1 - i])

    # Apply inverse rotations in reverse order
    for j in range(n_qubits - 1, -1, -1):
        # Inverse controlled rotations (negative angles)
        for k in range(n_qubits - 1, j, -1):
            if approximation_degree is not None:
                if k - j > approximation_degree:
                    continue

            angle = -2 * np.pi / (2 ** (k - j + 1))
            _add_controlled_phase(circuit, control=k, target=j, angle=angle)

        # Apply Hadamard
        circuit.add_gate(GateType.H, targets=[j])

    return circuit


def _add_controlled_phase(circuit: UnifiedCircuit, control: int, target: int, angle: float):
    """
    Add controlled phase rotation.

    Implements: |0⟩|ψ⟩ → |0⟩|ψ⟩, |1⟩|ψ⟩ → |1⟩exp(iθ)|ψ⟩
    """
    # Controlled-P gate can be implemented as:
    # CP(θ) = I ⊗ P(θ/2) @ CNOT @ I ⊗ P(-θ/2) @ CNOT @ I ⊗ P(θ/2)
    # Simplified: RZ on target, CNOT, RZ on target, CNOT

    # Apply phase to control and target
    circuit.add_gate(GateType.RZ, targets=[control], parameters={'angle': angle / 2})
    circuit.add_gate(GateType.RZ, targets=[target], parameters={'angle': angle / 2})
    circuit.add_gate(GateType.CNOT, targets=[control, target])
    circuit.add_gate(GateType.RZ, targets=[target], parameters={'angle': -angle / 2})
    circuit.add_gate(GateType.CNOT, targets=[control, target])


def qft_rotations(n_qubits: int, start_qubit: int = 0, approximation_degree: int = None) -> UnifiedCircuit:
    """
    Create QFT without final swaps (useful for subroutines).

    Args:
        n_qubits: Number of qubits
        start_qubit: Starting qubit index
        approximation_degree: Approximation degree

    Returns:
        QFT rotation circuit (without swaps)
    """
    circuit = UnifiedCircuit(n_qubits=start_qubit + n_qubits)

    for j in range(start_qubit, start_qubit + n_qubits):
        circuit.add_gate(GateType.H, targets=[j])

        for k in range(j + 1, start_qubit + n_qubits):
            if approximation_degree is not None:
                if k - j > approximation_degree:
                    continue

            angle = 2 * np.pi / (2 ** (k - j + 1))
            _add_controlled_phase(circuit, control=k, target=j, angle=angle)

    return circuit
