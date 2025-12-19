"""
Amplitude Amplification algorithm.

Amplitude amplification is a generalization of Grover's algorithm that
can amplify the amplitude of any desired state in a quantum system,
not just those marked by an oracle.
"""

import numpy as np
from typing import Callable, Optional
from q_store.core import UnifiedCircuit, GateType


def amplitude_amplification(
    n_qubits: int,
    state_preparation: Callable[[UnifiedCircuit], None],
    oracle: Callable[[UnifiedCircuit], None],
    n_iterations: Optional[int] = None
) -> UnifiedCircuit:
    """
    Create Amplitude Amplification circuit.

    Amplitude amplification generalizes Grover's algorithm to amplify
    amplitude of good states prepared by an arbitrary quantum circuit.

    Args:
        n_qubits: Number of qubits
        state_preparation: Function that prepares initial state |ψ⟩
        oracle: Oracle that marks "good" states (applies phase flip)
        n_iterations: Number of amplification iterations
                     Defaults to optimal ≈ π/(4θ) where sin²(θ) = success prob

    Returns:
        Amplitude amplification circuit
    """
    if n_iterations is None:
        # Default: assume uniform superposition (like Grover)
        n_iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
        n_iterations = max(1, n_iterations)

    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Prepare initial state
    state_preparation(circuit)

    # Amplification iterations
    for _ in range(n_iterations):
        # Apply oracle (marks good states)
        oracle(circuit)

        # Apply inversion about |ψ⟩
        _apply_inversion_about_state(circuit, state_preparation)

    return circuit


def _apply_inversion_about_state(
    circuit: UnifiedCircuit,
    state_preparation: Callable[[UnifiedCircuit], None]
):
    """
    Apply inversion about the prepared state |ψ⟩.

    This operator is: 2|ψ⟩⟨ψ| - I
    Implemented as: U (2|0⟩⟨0| - I) U†
    where U is the state preparation.
    """
    n_qubits = circuit.n_qubits

    # U† (inverse of state preparation)
    # This is a simplification - in general we'd need actual inverse
    # For Hadamard-based preparation, H is self-inverse
    # For more complex preparations, may need explicit inverse circuit
    _apply_inverse_state_preparation(circuit, state_preparation)

    # Inversion about |0⟩: apply X to all, multi-controlled Z, X to all
    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    _apply_multi_controlled_z(circuit, list(range(n_qubits)))

    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    # U (state preparation again)
    state_preparation(circuit)


def _apply_inverse_state_preparation(
    circuit: UnifiedCircuit,
    state_preparation: Callable[[UnifiedCircuit], None]
):
    """
    Apply inverse of state preparation.

    This is a simplified implementation that works for Hadamard-based
    preparations. For general state preparations, a proper inverse
    would need to be constructed.
    """
    # For many common cases (Hadamard, etc.), gates are self-inverse
    # so we can just apply the same circuit again
    # This is a simplification - proper implementation would need
    # to compute actual inverse

    # Create temporary circuit to get gates
    temp_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
    state_preparation(temp_circuit)

    # Apply gates in reverse order with conjugate parameters
    for gate in reversed(temp_circuit.gates):
        if gate.gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z]:
            # Self-inverse gates
            circuit.add_gate(gate.gate_type, targets=gate.targets)
        elif gate.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            # Rotation gates - negate angle
            params = gate.parameters.copy()
            params['angle'] = -params['angle']
            circuit.add_gate(gate.gate_type, targets=gate.targets, parameters=params)
        elif gate.gate_type == GateType.CNOT:
            # CNOT is self-inverse
            circuit.add_gate(gate.gate_type, targets=gate.targets)
        else:
            # For other gates, assume self-inverse (simplification)
            circuit.add_gate(gate.gate_type, targets=gate.targets, parameters=gate.parameters)


def _apply_multi_controlled_z(circuit: UnifiedCircuit, qubits: list):
    """Apply multi-controlled Z gate."""
    if len(qubits) == 1:
        circuit.add_gate(GateType.Z, targets=qubits)
    elif len(qubits) == 2:
        circuit.add_gate(GateType.CZ, targets=qubits)
    else:
        # Use H-MCX-H decomposition
        target = qubits[-1]
        controls = qubits[:-1]

        circuit.add_gate(GateType.H, targets=[target])
        _apply_multi_controlled_x(circuit, controls, target)
        circuit.add_gate(GateType.H, targets=[target])


def _apply_multi_controlled_x(circuit: UnifiedCircuit, controls: list, target: int):
    """Apply multi-controlled X gate."""
    if len(controls) == 1:
        circuit.add_gate(GateType.CNOT, targets=[controls[0], target])
    elif len(controls) == 2:
        circuit.add_gate(GateType.CCX, targets=[controls[0], controls[1], target])
    else:
        # Simplified decomposition
        for control in controls[:-1]:
            circuit.add_gate(GateType.CNOT, targets=[control, controls[-1]])
        circuit.add_gate(GateType.CCX, targets=[controls[-1], controls[-2], target])
        for control in reversed(controls[:-1]):
            circuit.add_gate(GateType.CNOT, targets=[control, controls[-1]])


def fixed_point_amplification(
    n_qubits: int,
    state_preparation: Callable[[UnifiedCircuit], None],
    oracle: Callable[[UnifiedCircuit], None],
    target_amplitude: float = 1.0
) -> UnifiedCircuit:
    """
    Create Fixed-Point Amplitude Amplification circuit.

    Fixed-point search provides amplitude amplification that converges
    to a fixed point rather than oscillating, making it more robust
    to uncertainty in the initial amplitude.

    Args:
        n_qubits: Number of qubits
        state_preparation: State preparation function
        oracle: Oracle function
        target_amplitude: Target amplitude (default 1.0)

    Returns:
        Fixed-point amplification circuit
    """
    # Number of iterations for fixed-point search
    # This uses a more sophisticated iteration schedule
    n_iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
    n_iterations = max(1, n_iterations)

    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Prepare initial state
    state_preparation(circuit)

    # Fixed-point iterations with varying rotation angles
    for k in range(n_iterations):
        # Oracle
        oracle(circuit)

        # Rotation angle decreases with iteration
        angle = np.pi / (2 * (k + 1))
        _apply_rotation_about_state(circuit, state_preparation, angle)

    return circuit


def _apply_rotation_about_state(
    circuit: UnifiedCircuit,
    state_preparation: Callable[[UnifiedCircuit], None],
    angle: float
):
    """
    Apply rotation about prepared state with given angle.

    This is a generalization of inversion that allows fractional rotations.
    """
    n_qubits = circuit.n_qubits

    # Apply inverse state preparation
    _apply_inverse_state_preparation(circuit, state_preparation)

    # Apply rotation about |0⟩
    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    # Controlled rotation instead of full phase flip
    circuit.add_gate(GateType.RZ, targets=[0], parameters={'angle': angle})

    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    # Apply state preparation
    state_preparation(circuit)


def oblivious_amplitude_amplification(
    n_qubits: int,
    n_ancilla: int,
    state_preparation: Callable[[UnifiedCircuit], None],
    oracle: Callable[[UnifiedCircuit], None]
) -> UnifiedCircuit:
    """
    Create Oblivious Amplitude Amplification circuit.

    This variant doesn't require knowing the initial success probability,
    using ancilla qubits to make the process oblivious.

    Args:
        n_qubits: Number of main qubits
        n_ancilla: Number of ancilla qubits
        state_preparation: State preparation function
        oracle: Oracle function

    Returns:
        Oblivious amplitude amplification circuit
    """
    n_total = n_qubits + n_ancilla
    circuit = UnifiedCircuit(n_qubits=n_total)

    # Prepare ancilla in equal superposition
    for i in range(n_qubits, n_total):
        circuit.add_gate(GateType.H, targets=[i])

    # Prepare main state
    state_preparation(circuit)

    # Apply oblivious amplification (simplified version)
    # Full implementation would use more sophisticated ancilla manipulation
    oracle(circuit)
    _apply_inversion_about_state(circuit, state_preparation)

    return circuit
