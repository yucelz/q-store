"""
Advanced verification tools for quantum circuits.

This module provides tools for verifying quantum circuit properties including:
- Circuit equivalence checking (unitary, state, up to global phase)
- Property verification (unitarity, reversibility, commutativity)
- Formal verification and symbolic reasoning
"""

from .equivalence import (
    check_unitary_equivalence,
    check_state_equivalence,
    check_circuit_equivalence,
    circuits_equal_up_to_phase,
    EquivalenceChecker
)

from .properties import (
    is_unitary,
    is_reversible,
    check_commutativity,
    verify_gate_decomposition,
    PropertyVerifier
)

from .formal import (
    verify_circuit_identity,
    check_algebraic_property,
    symbolic_circuit_analysis,
    FormalVerifier
)

__all__ = [
    # Equivalence checking
    'check_unitary_equivalence',
    'check_state_equivalence',
    'check_circuit_equivalence',
    'circuits_equal_up_to_phase',
    'EquivalenceChecker',

    # Property verification
    'is_unitary',
    'is_reversible',
    'check_commutativity',
    'verify_gate_decomposition',
    'PropertyVerifier',

    # Formal verification
    'verify_circuit_identity',
    'check_algebraic_property',
    'symbolic_circuit_analysis',
    'FormalVerifier'
]
