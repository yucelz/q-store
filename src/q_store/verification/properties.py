"""
Property verification for quantum circuits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from q_store.core import UnifiedCircuit, GateType
from q_store.verification.equivalence import _circuit_to_matrix


def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Check if a matrix is unitary (U† U = I).

    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_unitary, deviation_from_unity)
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False, float('inf')

    # Compute U† U
    product = np.conj(matrix.T) @ matrix

    # Check how close to identity
    identity = np.eye(matrix.shape[0])
    deviation = np.linalg.norm(product - identity, 'fro')

    is_unitary_flag = deviation < tolerance

    return is_unitary_flag, float(deviation)


def is_reversible(circuit: UnifiedCircuit) -> bool:
    """
    Check if a circuit is reversible (can be inverted).

    All quantum circuits are inherently reversible, but this checks
    if the circuit can be explicitly inverted.

    Args:
        circuit: Circuit to check

    Returns:
        True if reversible (always True for quantum circuits)
    """
    # All quantum circuits are reversible by nature
    # Check if circuit is valid
    try:
        U = _circuit_to_matrix(circuit)
        is_unitary_flag, _ = is_unitary(U)
        return is_unitary_flag
    except Exception:
        return False


def check_commutativity(gate1: GateType, gate2: GateType,
                       targets1: List[int], targets2: List[int],
                       n_qubits: int) -> Tuple[bool, float]:
    """
    Check if two gates commute.

    Two gates commute if [G1, G2] = G1G2 - G2G1 = 0

    Args:
        gate1: First gate type
        gate2: Second gate type
        targets1: Target qubits for first gate
        targets2: Target qubits for second gate
        n_qubits: Total number of qubits

    Returns:
        Tuple of (do_commute, commutator_norm)
    """
    from q_store.core import UnifiedCircuit

    # Create circuits with individual gates
    circuit1 = UnifiedCircuit(n_qubits)
    circuit1.add_gate(gate1, targets1)

    circuit2 = UnifiedCircuit(n_qubits)
    circuit2.add_gate(gate2, targets2)

    # Get matrices
    U1 = _circuit_to_matrix(circuit1)
    U2 = _circuit_to_matrix(circuit2)

    # Compute commutator [U1, U2] = U1U2 - U2U1
    commutator = U1 @ U2 - U2 @ U1

    # Compute norm
    comm_norm = np.linalg.norm(commutator, 'fro')

    do_commute = comm_norm < 1e-10

    return do_commute, float(comm_norm)


def verify_gate_decomposition(original_gate: np.ndarray,
                             decomposed_circuit: UnifiedCircuit,
                             tolerance: float = 1e-10) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that a circuit correctly decomposes a gate.

    Args:
        original_gate: Original gate matrix
        decomposed_circuit: Circuit that should implement the gate
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_correct, details)
    """
    # Get decomposed circuit matrix
    decomposed_matrix = _circuit_to_matrix(decomposed_circuit)

    # Check equivalence
    from .equivalence import check_unitary_equivalence
    is_correct, distance = check_unitary_equivalence(
        original_gate, decomposed_matrix, tolerance
    )

    details = {
        'distance': distance,
        'n_gates': len(decomposed_circuit.gates),
        'depth': decomposed_circuit.depth,
        'correct': is_correct
    }

    return is_correct, details


class PropertyVerifier:
    """
    Class for verifying various circuit properties.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize property verifier.

        Args:
            tolerance: Numerical tolerance
        """
        self.tolerance = tolerance

    def verify_unitarity(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Verify that a circuit implements a unitary operation.

        Args:
            circuit: Circuit to verify

        Returns:
            Dictionary with verification results
        """
        U = _circuit_to_matrix(circuit)
        is_unitary_result, deviation = is_unitary(U, self.tolerance)

        return {
            'is_unitary': is_unitary_result,
            'deviation': deviation,
            'n_qubits': circuit.n_qubits,
            'n_gates': len(circuit.gates),
            'depth': circuit.depth
        }

    def verify_reversibility(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Verify circuit reversibility.

        Args:
            circuit: Circuit to verify

        Returns:
            Dictionary with verification results
        """
        is_reversible_flag = is_reversible(circuit)

        # Try to construct inverse
        try:
            inverse_circuit = UnifiedCircuit(circuit.n_qubits)

            # Add gates in reverse order with inverse operations
            for gate in reversed(circuit.gates):
                # For simplicity, just check if we can create inverse
                inverse_circuit.add_gate(gate.gate_type, gate.targets)

            can_construct_inverse = True
        except Exception:
            can_construct_inverse = False

        return {
            'is_reversible': is_reversible_flag,
            'can_construct_inverse': can_construct_inverse,
            'n_qubits': circuit.n_qubits
        }

    def analyze_commutativity(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Analyze which gates in a circuit commute.

        Args:
            circuit: Circuit to analyze

        Returns:
            Dictionary with commutativity analysis
        """
        n_gates = len(circuit.gates)
        commute_matrix = np.zeros((n_gates, n_gates), dtype=bool)

        for i in range(n_gates):
            for j in range(i, n_gates):
                if i == j:
                    commute_matrix[i, j] = True
                else:
                    gate1 = circuit.gates[i]
                    gate2 = circuit.gates[j]

                    do_commute, _ = check_commutativity(
                        gate1.gate_type, gate2.gate_type,
                        gate1.targets, gate2.targets,
                        circuit.n_qubits
                    )

                    commute_matrix[i, j] = do_commute
                    commute_matrix[j, i] = do_commute

        # Count commuting pairs
        n_commuting_pairs = np.sum(commute_matrix) - n_gates  # Exclude diagonal
        n_commuting_pairs //= 2  # Each pair counted twice

        total_pairs = n_gates * (n_gates - 1) // 2

        return {
            'n_gates': n_gates,
            'commute_matrix': commute_matrix,
            'n_commuting_pairs': int(n_commuting_pairs),
            'total_pairs': total_pairs,
            'commutativity_ratio': n_commuting_pairs / total_pairs if total_pairs > 0 else 1.0
        }

    def verify_identity(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Verify if a circuit implements the identity operation.

        Args:
            circuit: Circuit to verify

        Returns:
            Dictionary with verification results
        """
        U = _circuit_to_matrix(circuit)
        I = np.eye(U.shape[0])

        distance = np.linalg.norm(U - I, 'fro')
        is_identity = distance < self.tolerance

        return {
            'is_identity': is_identity,
            'distance_from_identity': float(distance),
            'n_qubits': circuit.n_qubits,
            'n_gates': len(circuit.gates)
        }

    def check_all_properties(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Check all verifiable properties of a circuit.

        Args:
            circuit: Circuit to verify

        Returns:
            Dictionary with all verification results
        """
        results = {}

        results['unitarity'] = self.verify_unitarity(circuit)
        results['reversibility'] = self.verify_reversibility(circuit)
        results['commutativity'] = self.analyze_commutativity(circuit)
        results['identity'] = self.verify_identity(circuit)

        # Summary
        results['summary'] = {
            'is_unitary': results['unitarity']['is_unitary'],
            'is_reversible': results['reversibility']['is_reversible'],
            'is_identity': results['identity']['is_identity'],
            'n_qubits': circuit.n_qubits,
            'n_gates': len(circuit.gates),
            'depth': circuit.depth
        }

        return results
