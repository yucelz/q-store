"""
Quantum chemistry module for molecular simulations.

This module provides tools for quantum chemistry simulations including:
- Molecular Hamiltonians (H2, LiH, BeH2, etc.)
- Fermionic operators and Jordan-Wigner transformation
- VQE for molecular ground state calculations
- Bond length optimization and molecular properties
"""

from .hamiltonian import (
    MolecularHamiltonian,
    jordan_wigner_transform,
    fermion_to_qubit_operator,
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_beh2_hamiltonian
)

from .vqe import (
    MolecularVQE,
    estimate_ground_state,
    optimize_bond_length,
    compute_molecular_properties
)

from .operators import (
    FermionOperator,
    QubitOperator,
    commutator,
    anticommutator,
    normal_ordered
)

__all__ = [
    # Hamiltonian generation
    'MolecularHamiltonian',
    'jordan_wigner_transform',
    'fermion_to_qubit_operator',
    'create_h2_hamiltonian',
    'create_lih_hamiltonian',
    'create_beh2_hamiltonian',

    # VQE for chemistry
    'MolecularVQE',
    'estimate_ground_state',
    'optimize_bond_length',
    'compute_molecular_properties',

    # Operators
    'FermionOperator',
    'QubitOperator',
    'commutator',
    'anticommutator',
    'normal_ordered'
]
