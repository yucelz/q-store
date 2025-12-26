"""
Quantum Phase Estimation (QPE) algorithm.

QPE estimates the eigenvalue of a unitary operator U given an eigenstate |ψ⟩:
U|ψ⟩ = exp(2πiθ)|ψ⟩

The algorithm estimates θ with precision determined by the number of
counting qubits used.
"""

import numpy as np
from typing import Callable
from q_store.core import UnifiedCircuit, GateType
from q_store.templates.qft import inverse_qft


def phase_estimation(
    n_counting_qubits: int,
    n_target_qubits: int,
    unitary: Callable[[UnifiedCircuit, int], None],
    state_preparation: Callable[[UnifiedCircuit], None] = None
) -> UnifiedCircuit:
    """
    Create Quantum Phase Estimation circuit.

    Args:
        n_counting_qubits: Number of qubits for phase precision
                          (precision ≈ 2^-n_counting_qubits)
        n_target_qubits: Number of qubits in eigenstate
        unitary: Function that applies controlled-U^(2^k) operation
                 Takes (circuit, power) and applies controlled unitary
        state_preparation: Optional function to prepare eigenstate
                          If None, assumes |ψ⟩ is already prepared

    Returns:
        Phase estimation circuit
    """
    n_total = n_counting_qubits + n_target_qubits
    circuit = UnifiedCircuit(n_qubits=n_total)

    counting_qubits = list(range(n_counting_qubits))
    target_qubits = list(range(n_counting_qubits, n_total))

    # Prepare eigenstate if function provided
    if state_preparation is not None:
        state_preparation(circuit)

    # Initialize counting qubits in superposition
    for qubit in counting_qubits:
        circuit.add_gate(GateType.H, targets=[qubit])

    # Apply controlled unitaries
    for idx, control_qubit in enumerate(counting_qubits):
        power = 2 ** (n_counting_qubits - 1 - idx)
        # Apply controlled-U^power
        _apply_controlled_unitary_power(
            circuit, control_qubit, target_qubits, unitary, power
        )

    # Apply inverse QFT to counting qubits
    iqft_circuit = inverse_qft(n_counting_qubits)
    for gate in iqft_circuit.gates:
        circuit.add_gate(gate.gate_type, targets=gate.targets, parameters=gate.parameters)

    return circuit


def _apply_controlled_unitary_power(
    circuit: UnifiedCircuit,
    control: int,
    targets: list,
    unitary: Callable,
    power: int
):
    """
    Apply controlled-U^power operation.

    This calls the unitary function which should implement the controlled
    version of U^power.
    """
    # The unitary function is responsible for implementing the controlled
    # version with the given power
    unitary(circuit, power)


def iterative_phase_estimation(
    n_target_qubits: int,
    unitary: Callable[[UnifiedCircuit, int], None],
    precision_bits: int = 8,
    state_preparation: Callable[[UnifiedCircuit], None] = None
) -> UnifiedCircuit:
    """
    Create iterative Phase Estimation circuit (uses only 1 counting qubit).

    This variant uses only one counting qubit and performs multiple
    measurements, reducing qubit requirements at the cost of more
    circuit executions.

    Args:
        n_target_qubits: Number of qubits in eigenstate
        unitary: Unitary operator function
        precision_bits: Desired precision
        state_preparation: State preparation function

    Returns:
        Iterative phase estimation circuit for one iteration
    """
    n_total = 1 + n_target_qubits
    circuit = UnifiedCircuit(n_qubits=n_total)

    # This is a single iteration - full IPE requires multiple runs
    if state_preparation is not None:
        state_preparation(circuit)

    # Hadamard on counting qubit
    circuit.add_gate(GateType.H, targets=[0])

    # Controlled unitary (power = 1 for first iteration)
    target_qubits = list(range(1, n_total))
    unitary(circuit, 1)

    # Phase correction (would be based on previous measurements)
    # In full IPE, this would include feedback from earlier measurements

    # Hadamard on counting qubit
    circuit.add_gate(GateType.H, targets=[0])

    return circuit


def create_phase_estimation_unitary(
    gate_type: GateType,
    target_qubit: int,
    angle: float = None
) -> Callable[[UnifiedCircuit, int], None]:
    """
    Create a simple unitary function for phase estimation.

    This is a helper for common cases like estimating phase of rotation gates.

    Args:
        gate_type: Type of gate (RZ, RX, RY, etc.)
        target_qubit: Which qubit the gate acts on
        angle: Gate angle parameter

    Returns:
        Unitary function compatible with phase_estimation
    """
    def unitary(circuit: UnifiedCircuit, power: int):
        # Apply gate 'power' times
        for _ in range(power):
            if angle is not None:
                circuit.add_gate(
                    gate_type,
                    targets=[target_qubit],
                    parameters={'angle': angle}
                )
            else:
                circuit.add_gate(gate_type, targets=[target_qubit])

    return unitary


def phase_kickback_circuit(
    n_qubits: int,
    phase_angle: float,
    target_qubit: int = 0
) -> UnifiedCircuit:
    """
    Create circuit demonstrating phase kickback.

    Phase kickback is the key mechanism in QPE where phase information
    is transferred from target to control qubits.

    Args:
        n_qubits: Total qubits
        phase_angle: Phase to apply
        target_qubit: Target qubit

    Returns:
        Phase kickback demonstration circuit
    """
    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Put target in |-> state (eigenstate of X)
    circuit.add_gate(GateType.X, targets=[target_qubit])
    circuit.add_gate(GateType.H, targets=[target_qubit])

    # Put control in superposition
    for i in range(n_qubits):
        if i != target_qubit:
            circuit.add_gate(GateType.H, targets=[i])

    # Apply controlled phase rotation
    # Phase kicks back to control qubit
    for i in range(n_qubits):
        if i != target_qubit:
            circuit.add_gate(GateType.RZ, targets=[target_qubit], parameters={'angle': phase_angle})
            circuit.add_gate(GateType.CNOT, targets=[i, target_qubit])

    return circuit
