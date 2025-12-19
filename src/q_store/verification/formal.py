"""
Formal verification for quantum circuits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from q_store.core import UnifiedCircuit, GateType
from q_store.verification.equivalence import _circuit_to_matrix


def verify_circuit_identity(circuit1: UnifiedCircuit, circuit2: UnifiedCircuit,
                           property_name: str = "inverse") -> Dict[str, Any]:
    """
    Verify that two circuits satisfy a specific relationship.

    Common properties:
    - "inverse": circuit2 is the inverse of circuit1 (circuit1 * circuit2 = I)
    - "equivalent": circuit1 and circuit2 are equivalent
    - "commute": circuit1 and circuit2 commute

    Args:
        circuit1: First circuit
        circuit2: Second circuit
        property_name: Name of property to verify

    Returns:
        Dictionary with verification results
    """
    if circuit1.n_qubits != circuit2.n_qubits:
        return {
            'verified': False,
            'reason': 'Different number of qubits'
        }

    U1 = _circuit_to_matrix(circuit1)
    U2 = _circuit_to_matrix(circuit2)

    if property_name == "inverse":
        # Check if U1 * U2 = I
        product = U1 @ U2
        identity = np.eye(product.shape[0])
        distance = np.linalg.norm(product - identity, 'fro')

        verified = distance < 1e-10

        return {
            'verified': verified,
            'property': 'inverse',
            'distance_from_identity': float(distance)
        }

    elif property_name == "equivalent":
        # Check if U1 = e^(iφ) U2
        from .equivalence import check_unitary_equivalence
        is_equiv, distance = check_unitary_equivalence(U1, U2)

        return {
            'verified': is_equiv,
            'property': 'equivalent',
            'distance': distance
        }

    elif property_name == "commute":
        # Check if [U1, U2] = 0
        commutator = U1 @ U2 - U2 @ U1
        comm_norm = np.linalg.norm(commutator, 'fro')

        verified = comm_norm < 1e-10

        return {
            'verified': verified,
            'property': 'commute',
            'commutator_norm': float(comm_norm)
        }

    else:
        return {
            'verified': False,
            'reason': f'Unknown property: {property_name}'
        }


def check_algebraic_property(circuit: UnifiedCircuit, property_check: Callable,
                            property_name: str = "") -> Dict[str, Any]:
    """
    Check if a circuit satisfies an algebraic property.

    Args:
        circuit: Circuit to check
        property_check: Function that takes circuit matrix and returns bool
        property_name: Name of the property being checked

    Returns:
        Dictionary with verification results
    """
    U = _circuit_to_matrix(circuit)

    try:
        satisfies = property_check(U)

        return {
            'verified': bool(satisfies),
            'property': property_name,
            'n_qubits': circuit.n_qubits,
            'n_gates': len(circuit.gates)
        }
    except Exception as e:
        return {
            'verified': False,
            'property': property_name,
            'error': str(e)
        }


def symbolic_circuit_analysis(circuit: UnifiedCircuit) -> Dict[str, Any]:
    """
    Perform symbolic analysis of a circuit.

    Analyzes circuit structure, gate patterns, and mathematical properties.

    Args:
        circuit: Circuit to analyze

    Returns:
        Dictionary with symbolic analysis results
    """
    analysis = {
        'n_qubits': circuit.n_qubits,
        'n_gates': len(circuit.gates),
        'depth': circuit.depth
    }

    # Count gate types
    gate_counts = {}
    for gate in circuit.gates:
        gate_type = gate.gate_type
        gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1

    analysis['gate_counts'] = {str(k): v for k, v in gate_counts.items()}

    # Identify patterns
    patterns = []

    # Check for repeated gates
    if len(circuit.gates) > 1:
        for i in range(len(circuit.gates) - 1):
            if circuit.gates[i].gate_type == circuit.gates[i + 1].gate_type:
                if circuit.gates[i].targets == circuit.gates[i + 1].targets:
                    patterns.append(f"Repeated {circuit.gates[i].gate_type} at position {i}")

    # Check for inverse pairs (simplified)
    # Note: S and T are their own triple inverses (S^3 = S†, T^3 = T†)
    inverse_pairs = {
        GateType.X: GateType.X,
        GateType.Y: GateType.Y,
        GateType.Z: GateType.Z,
        GateType.H: GateType.H,
        # Note: S and T don't have explicit dagger gates in this GateType enum
    }

    for i in range(len(circuit.gates) - 1):
        gate1 = circuit.gates[i]
        gate2 = circuit.gates[i + 1]

        if gate1.targets == gate2.targets:
            if inverse_pairs.get(gate1.gate_type) == gate2.gate_type:
                patterns.append(f"Inverse pair at position {i}: {gate1.gate_type}-{gate2.gate_type}")

    analysis['patterns'] = patterns

    # Mathematical properties
    U = _circuit_to_matrix(circuit)

    # Check if hermitian
    is_hermitian = np.allclose(U, U.conj().T)
    analysis['is_hermitian'] = is_hermitian

    # Check if real
    is_real = np.allclose(U.imag, 0)
    analysis['is_real'] = is_real

    # Compute determinant
    det = np.linalg.det(U)
    analysis['determinant'] = complex(det)
    analysis['determinant_magnitude'] = float(np.abs(det))

    return analysis


class FormalVerifier:
    """
    Class for formal verification of quantum circuits.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize formal verifier.

        Args:
            tolerance: Numerical tolerance
        """
        self.tolerance = tolerance

    def verify_relationship(self, circuit1: UnifiedCircuit, circuit2: UnifiedCircuit,
                          relationship: str) -> Dict[str, Any]:
        """
        Verify a specific relationship between two circuits.

        Args:
            circuit1: First circuit
            circuit2: Second circuit
            relationship: Relationship to verify

        Returns:
            Dictionary with verification results
        """
        return verify_circuit_identity(circuit1, circuit2, relationship)

    def verify_transformation(self, input_circuit: UnifiedCircuit,
                            output_circuit: UnifiedCircuit,
                            transformation_name: str) -> Dict[str, Any]:
        """
        Verify that a transformation was correctly applied.

        Args:
            input_circuit: Original circuit
            output_circuit: Transformed circuit
            transformation_name: Name of transformation

        Returns:
            Dictionary with verification results
        """
        from .equivalence import check_circuit_equivalence

        is_equiv, details = check_circuit_equivalence(input_circuit, output_circuit, self.tolerance)

        return {
            'transformation': transformation_name,
            'preserves_functionality': is_equiv,
            'input_gates': len(input_circuit.gates),
            'output_gates': len(output_circuit.gates),
            'gate_reduction': len(input_circuit.gates) - len(output_circuit.gates),
            'details': details
        }

    def verify_optimization(self, original: UnifiedCircuit, optimized: UnifiedCircuit) -> Dict[str, Any]:
        """
        Verify that an optimization preserves circuit functionality.

        Args:
            original: Original circuit
            optimized: Optimized circuit

        Returns:
            Dictionary with verification and improvement metrics
        """
        result = self.verify_transformation(original, optimized, "optimization")

        # Add optimization metrics
        result['improvements'] = {
            'gate_count_reduction': len(original.gates) - len(optimized.gates),
            'depth_reduction': original.depth - optimized.depth,
            'original_depth': original.depth,
            'optimized_depth': optimized.depth
        }

        return result

    def verify_decomposition(self, target_gate: np.ndarray,
                           decomposition: UnifiedCircuit) -> Dict[str, Any]:
        """
        Verify that a circuit correctly decomposes a target gate.

        Args:
            target_gate: Target gate matrix
            decomposition: Circuit implementing the decomposition

        Returns:
            Dictionary with verification results
        """
        from .properties import verify_gate_decomposition

        is_correct, details = verify_gate_decomposition(
            target_gate, decomposition, self.tolerance
        )

        return {
            'decomposition_correct': is_correct,
            'details': details
        }

    def comprehensive_verification(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Perform comprehensive verification of a circuit.

        Args:
            circuit: Circuit to verify

        Returns:
            Dictionary with complete verification results
        """
        from .properties import PropertyVerifier
        from .equivalence import EquivalenceChecker

        results = {}

        # Property verification
        prop_verifier = PropertyVerifier(self.tolerance)
        results['properties'] = prop_verifier.check_all_properties(circuit)

        # Symbolic analysis
        results['symbolic_analysis'] = symbolic_circuit_analysis(circuit)

        # Check against identity
        identity_circuit = UnifiedCircuit(circuit.n_qubits)
        results['is_identity'] = self.verify_relationship(
            circuit, identity_circuit, "equivalent"
        )

        return results
