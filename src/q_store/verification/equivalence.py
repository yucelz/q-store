"""
Circuit and operator equivalence checking.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from q_store.core import UnifiedCircuit, GateType


def _circuit_to_matrix(circuit: UnifiedCircuit) -> np.ndarray:
    """
    Convert a circuit to its unitary matrix representation.

    Args:
        circuit: Circuit to convert

    Returns:
        Unitary matrix
    """
    n_qubits = circuit.n_qubits
    dim = 2 ** n_qubits

    # Start with identity
    U = np.eye(dim, dtype=complex)

    # Gate matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    gate_matrices = {
        GateType.X: X,
        GateType.Y: Y,
        GateType.Z: Z,
        GateType.H: H,
        GateType.S: S,
        GateType.T: T
    }

    # Apply each gate
    for gate in circuit.gates:
        if gate.gate_type == GateType.CNOT:
            # CNOT gate - both control and target are in targets list
            control, target = gate.targets[0], gate.targets[1]
            gate_matrix = _cnot_matrix(n_qubits, control, target)
        elif gate.gate_type in gate_matrices:
            # Single qubit gate
            target = gate.targets[0]
            gate_matrix = _single_qubit_gate_matrix(gate_matrices[gate.gate_type], target, n_qubits)
        else:
            # Unknown gate, use identity
            gate_matrix = np.eye(dim)

        U = gate_matrix @ U

    return U


def _single_qubit_gate_matrix(gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Apply single qubit gate to target qubit."""
    I = np.eye(2)
    matrix = np.array([[1.0]], dtype=complex)

    for i in range(n_qubits):
        if i == target:
            matrix = np.kron(matrix, gate)
        else:
            matrix = np.kron(matrix, I)

    return matrix


def _cnot_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    """Create CNOT matrix."""
    dim = 2 ** n_qubits
    matrix = np.eye(dim, dtype=complex)

    # CNOT flips target if control is 1
    for i in range(dim):
        if (i >> (n_qubits - 1 - control)) & 1:  # Control is 1
            # Flip target bit
            j = i ^ (1 << (n_qubits - 1 - target))
            matrix[j, i] = 1
            matrix[i, i] = 0

    return matrix


def check_unitary_equivalence(U1: np.ndarray, U2: np.ndarray,
                              tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Check if two unitary matrices are equivalent.

    Two unitaries are equivalent if U1 = e^(iφ) U2 for some global phase φ.

    Args:
        U1: First unitary matrix
        U2: Second unitary matrix
        tolerance: Numerical tolerance

    Returns:
        Tuple of (are_equivalent, distance)
    """
    if U1.shape != U2.shape:
        return False, float('inf')

    # Compute inner product: Tr(U1† U2)
    inner_product = np.trace(np.conj(U1.T) @ U2)

    # Normalize by dimension
    dim = U1.shape[0]
    normalized_overlap = inner_product / dim

    # Check if overlap has magnitude 1 (equivalent up to global phase)
    overlap_magnitude = np.abs(normalized_overlap)

    # Compute distance
    distance = 1.0 - overlap_magnitude

    is_equivalent = distance < tolerance

    return is_equivalent, float(distance)


def check_state_equivalence(state1: np.ndarray, state2: np.ndarray,
                           tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Check if two quantum states are equivalent (up to global phase).

    Args:
        state1: First state vector
        state2: Second state vector
        tolerance: Numerical tolerance

    Returns:
        Tuple of (are_equivalent, distance)
    """
    # Normalize states
    state1_norm = state1 / np.linalg.norm(state1)
    state2_norm = state2 / np.linalg.norm(state2)

    # Compute overlap
    overlap = np.abs(np.vdot(state1_norm, state2_norm))

    # Distance from 1
    distance = 1.0 - overlap

    is_equivalent = distance < tolerance

    return is_equivalent, float(distance)


def check_circuit_equivalence(circuit1: UnifiedCircuit, circuit2: UnifiedCircuit,
                             tolerance: float = 1e-10) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if two circuits are functionally equivalent.

    Args:
        circuit1: First circuit
        circuit2: Second circuit
        tolerance: Numerical tolerance

    Returns:
        Tuple of (are_equivalent, details_dict)
    """
    if circuit1.n_qubits != circuit2.n_qubits:
        return False, {'reason': 'Different number of qubits'}

    n_qubits = circuit1.n_qubits

    # Get unitary matrices
    U1 = _circuit_to_matrix(circuit1)
    U2 = _circuit_to_matrix(circuit2)

    # Check unitary equivalence
    is_equiv, distance = check_unitary_equivalence(U1, U2, tolerance)

    details = {
        'unitary_distance': distance,
        'n_qubits': n_qubits,
        'circuit1_depth': circuit1.depth,
        'circuit2_depth': circuit2.depth,
        'circuit1_gates': len(circuit1.gates),
        'circuit2_gates': len(circuit2.gates)
    }

    return is_equiv, details


def circuits_equal_up_to_phase(circuit1: UnifiedCircuit, circuit2: UnifiedCircuit,
                               tolerance: float = 1e-10) -> bool:
    """
    Check if two circuits are equal up to a global phase.

    Args:
        circuit1: First circuit
        circuit2: Second circuit
        tolerance: Numerical tolerance

    Returns:
        True if circuits are equivalent up to global phase
    """
    is_equiv, _ = check_circuit_equivalence(circuit1, circuit2, tolerance)
    return is_equiv


class EquivalenceChecker:
    """
    Class for checking various types of equivalence.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize equivalence checker.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance

    def check_unitary(self, U1: np.ndarray, U2: np.ndarray) -> Dict[str, Any]:
        """
        Check unitary equivalence with detailed results.

        Args:
            U1: First unitary
            U2: Second unitary

        Returns:
            Dictionary with equivalence results
        """
        is_equiv, distance = check_unitary_equivalence(U1, U2, self.tolerance)

        # Compute global phase if equivalent
        global_phase = None
        if is_equiv:
            inner_product = np.trace(np.conj(U1.T) @ U2) / U1.shape[0]
            global_phase = np.angle(inner_product)

        return {
            'equivalent': is_equiv,
            'distance': distance,
            'global_phase': global_phase,
            'matrix_norm_diff': np.linalg.norm(U1 - U2, 'fro')
        }

    def check_circuit(self, circuit1: UnifiedCircuit, circuit2: UnifiedCircuit) -> Dict[str, Any]:
        """
        Check circuit equivalence with detailed analysis.

        Args:
            circuit1: First circuit
            circuit2: Second circuit

        Returns:
            Dictionary with equivalence results and analysis
        """
        is_equiv, details = check_circuit_equivalence(circuit1, circuit2, self.tolerance)

        result = {
            'equivalent': is_equiv,
            'details': details
        }

        # Additional checks
        if circuit1.n_qubits == circuit2.n_qubits:
            # Test on standard basis states
            n_qubits = circuit1.n_qubits
            basis_state_results = []

            U1 = _circuit_to_matrix(circuit1)
            U2 = _circuit_to_matrix(circuit2)

            for i in range(min(4, 2**n_qubits)):  # Test first few basis states
                state = np.zeros(2**n_qubits)
                state[i] = 1.0

                output1 = U1 @ state
                output2 = U2 @ state

                is_state_equiv, state_dist = check_state_equivalence(output1, output2, self.tolerance)
                basis_state_results.append({
                    'input_state': i,
                    'equivalent': is_state_equiv,
                    'distance': state_dist
                })

            result['basis_state_tests'] = basis_state_results

        return result

    def compare_multiple_circuits(self, circuits: list) -> Dict[str, Any]:
        """
        Compare multiple circuits for equivalence.

        Args:
            circuits: List of circuits to compare

        Returns:
            Dictionary with pairwise comparisons
        """
        n = len(circuits)
        equivalence_matrix = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    equivalence_matrix[i, j] = True
                else:
                    is_equiv = circuits_equal_up_to_phase(
                        circuits[i], circuits[j], self.tolerance
                    )
                    equivalence_matrix[i, j] = is_equiv
                    equivalence_matrix[j, i] = is_equiv

        return {
            'n_circuits': n,
            'equivalence_matrix': equivalence_matrix,
            'all_equivalent': np.all(equivalence_matrix)
        }
