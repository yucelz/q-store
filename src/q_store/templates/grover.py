"""
Grover's algorithm implementation for quantum search.

Grover's algorithm provides quadratic speedup for unstructured search,
finding a marked item in O(√N) queries compared to O(N) classically.
"""

import numpy as np
from typing import Callable, List, Optional
from q_store.core import UnifiedCircuit, GateType


def grover_circuit(
    n_qubits: int,
    oracle: Callable[[UnifiedCircuit], None],
    n_iterations: Optional[int] = None
) -> UnifiedCircuit:
    """
    Create complete Grover's search circuit.

    Args:
        n_qubits: Number of qubits
        oracle: Oracle function that marks the target state
                Takes a circuit and applies phase flip to target
        n_iterations: Number of Grover iterations (defaults to optimal √N)

    Returns:
        Grover search circuit
    """
    if n_iterations is None:
        # Optimal iterations ≈ π/4 √N
        n_iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
        n_iterations = max(1, n_iterations)

    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Initialize in equal superposition
    for i in range(n_qubits):
        circuit.add_gate(GateType.H, targets=[i])

    # Grover iterations
    for _ in range(n_iterations):
        # Apply oracle
        oracle(circuit)

        # Apply diffusion operator
        _apply_diffusion(circuit)

    return circuit


def grover_diffusion(n_qubits: int) -> UnifiedCircuit:
    """
    Create Grover diffusion operator (inversion about average).

    The diffusion operator is: D = 2|ψ⟩⟨ψ| - I
    where |ψ⟩ = H^⊗n|0⟩ is the equal superposition state.

    Args:
        n_qubits: Number of qubits

    Returns:
        Diffusion operator circuit
    """
    circuit = UnifiedCircuit(n_qubits=n_qubits)
    _apply_diffusion(circuit)
    return circuit


def _apply_diffusion(circuit: UnifiedCircuit):
    """Apply Grover diffusion operator to circuit."""
    n_qubits = circuit.n_qubits

    # H⊗n
    for i in range(n_qubits):
        circuit.add_gate(GateType.H, targets=[i])

    # X⊗n
    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    # Multi-controlled Z (phase flip on |111...1⟩)
    _apply_multi_controlled_z(circuit, list(range(n_qubits)))

    # X⊗n
    for i in range(n_qubits):
        circuit.add_gate(GateType.X, targets=[i])

    # H⊗n
    for i in range(n_qubits):
        circuit.add_gate(GateType.H, targets=[i])


def _apply_multi_controlled_z(circuit: UnifiedCircuit, qubits: List[int]):
    """
    Apply multi-controlled Z gate (Z on all qubits controlled).

    This is equivalent to a phase flip when all qubits are |1⟩.
    """
    if len(qubits) == 1:
        circuit.add_gate(GateType.Z, targets=qubits)
    elif len(qubits) == 2:
        circuit.add_gate(GateType.CZ, targets=qubits)
    else:
        # Multi-controlled Z using decomposition
        # MCZ = H(target) @ MCX @ H(target)
        target = qubits[-1]
        controls = qubits[:-1]

        circuit.add_gate(GateType.H, targets=[target])
        _apply_multi_controlled_x(circuit, controls, target)
        circuit.add_gate(GateType.H, targets=[target])


def _apply_multi_controlled_x(circuit: UnifiedCircuit, controls: List[int], target: int):
    """
    Apply multi-controlled X (Toffoli generalization).

    Uses recursive decomposition with ancilla qubits.
    """
    if len(controls) == 1:
        circuit.add_gate(GateType.CNOT, targets=[controls[0], target])
    elif len(controls) == 2:
        # Toffoli gate (CCX)
        circuit.add_gate(GateType.CCX, targets=[controls[0], controls[1], target])
    else:
        # Decompose using V gates where V^2 = X
        # This is a simplified decomposition
        for control in controls[:-1]:
            circuit.add_gate(GateType.CNOT, targets=[control, controls[-1]])
        circuit.add_gate(GateType.CCX, targets=[controls[-1], controls[-2], target])
        for control in reversed(controls[:-1]):
            circuit.add_gate(GateType.CNOT, targets=[control, controls[-1]])


def create_oracle_from_bitstring(bitstring: str) -> Callable[[UnifiedCircuit], None]:
    """
    Create oracle that marks a specific bitstring.

    Args:
        bitstring: Target state as binary string (e.g., "101")

    Returns:
        Oracle function
    """
    def oracle(circuit: UnifiedCircuit):
        n_qubits = circuit.n_qubits
        if len(bitstring) != n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != n_qubits {n_qubits}")

        # Flip qubits that should be 0
        for i, bit in enumerate(bitstring):
            if bit == '0':
                circuit.add_gate(GateType.X, targets=[i])

        # Multi-controlled Z
        _apply_multi_controlled_z(circuit, list(range(n_qubits)))

        # Flip back
        for i, bit in enumerate(bitstring):
            if bit == '0':
                circuit.add_gate(GateType.X, targets=[i])

    return oracle


def grover_search(
    n_qubits: int,
    marked_states: List[str],
    n_iterations: Optional[int] = None
) -> UnifiedCircuit:
    """
    Create Grover circuit to search for multiple marked states.

    Args:
        n_qubits: Number of qubits
        marked_states: List of marked states as bitstrings
        n_iterations: Number of iterations

    Returns:
        Grover search circuit
    """
    def multi_oracle(circuit: UnifiedCircuit):
        for state in marked_states:
            oracle = create_oracle_from_bitstring(state)
            oracle(circuit)

    return grover_circuit(n_qubits, multi_oracle, n_iterations)
