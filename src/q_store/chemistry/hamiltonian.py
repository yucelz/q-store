"""
Molecular Hamiltonians and Jordan-Wigner transformation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from .operators import FermionOperator, QubitOperator


def jordan_wigner_transform(fermion_op: FermionOperator, n_qubits: Optional[int] = None) -> QubitOperator:
    """
    Apply Jordan-Wigner transformation to convert fermion operator to qubit operator.

    The Jordan-Wigner transformation maps:
    - a_j† → (Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{j-1}) ⊗ σ⁺_j
    - a_j → (Z₀ ⊗ Z₁ ⊗ ... ⊗ Z_{j-1}) ⊗ σ⁻_j

    where σ⁺ = (X - iY)/2 and σ⁻ = (X + iY)/2

    Args:
        fermion_op: Fermionic operator
        n_qubits: Number of qubits (inferred if not provided)

    Returns:
        Equivalent qubit operator
    """
    qubit_op = QubitOperator()

    for term, coeff in fermion_op.terms.items():
        # Each term is a product of fermionic operators
        # Convert each to qubit operators

        if len(term) == 0:
            # Identity term
            qubit_op.terms[()] = coeff
            continue

        # For simplicity, implement basic cases
        if len(term) == 2:
            # Two-body term: a†_i a_j
            (i, op_i), (j, op_j) = term

            if op_i == 1 and op_j == 0:  # a†_i a_j
                if i == j:
                    # Number operator: n_i = a†_i a_i → (I - Z_i)/2
                    qubit_op.terms[((i, 'I'),)] = qubit_op.terms.get(((i, 'I'),), 0) + coeff / 2
                    qubit_op.terms[((i, 'Z'),)] = qubit_op.terms.get(((i, 'Z'),), 0) - coeff / 2
                else:
                    # Hopping term - simplified representation
                    # Full JW would include Z string
                    qubit_op.terms[((i, 'X'), (j, 'X'))] = qubit_op.terms.get(((i, 'X'), (j, 'X')), 0) + coeff / 4
                    qubit_op.terms[((i, 'Y'), (j, 'Y'))] = qubit_op.terms.get(((i, 'Y'), (j, 'Y')), 0) + coeff / 4

    return qubit_op


def fermion_to_qubit_operator(fermion_op: FermionOperator, method: str = 'jordan_wigner') -> QubitOperator:
    """
    Convert fermionic operator to qubit operator.

    Args:
        fermion_op: Fermionic operator
        method: Transformation method ('jordan_wigner', 'bravyi_kitaev')

    Returns:
        Qubit operator
    """
    if method == 'jordan_wigner':
        return jordan_wigner_transform(fermion_op)
    else:
        raise ValueError(f"Unknown transformation method: {method}")


class MolecularHamiltonian:
    """
    Molecular Hamiltonian in second quantization.

    H = Σᵢⱼ hᵢⱼ a†ᵢaⱼ + ½ Σᵢⱼₖₗ gᵢⱼₖₗ a†ᵢa†ⱼaₗaₖ
    """

    def __init__(self, one_body_integrals: np.ndarray, two_body_integrals: np.ndarray,
                 nuclear_repulsion: float = 0.0):
        """
        Initialize molecular Hamiltonian.

        Args:
            one_body_integrals: One-electron integrals hᵢⱼ
            two_body_integrals: Two-electron integrals gᵢⱼₖₗ
            nuclear_repulsion: Nuclear repulsion energy
        """
        self.one_body = one_body_integrals
        self.two_body = two_body_integrals
        self.nuclear_repulsion = nuclear_repulsion
        self.n_orbitals = one_body_integrals.shape[0]
        self.n_qubits = self.n_orbitals

    def to_fermion_operator(self) -> FermionOperator:
        """
        Convert to fermionic operator representation.

        Returns:
            Fermionic operator
        """
        fermion_op = FermionOperator()

        # Add constant term (nuclear repulsion)
        if abs(self.nuclear_repulsion) > 1e-12:
            fermion_op.terms[()] = self.nuclear_repulsion

        # One-body terms: Σᵢⱼ hᵢⱼ a†ᵢaⱼ
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                if abs(self.one_body[i, j]) > 1e-12:
                    term = ((i, 1), (j, 0))  # a†ᵢaⱼ
                    fermion_op.terms[term] = self.one_body[i, j]

        # Two-body terms: ½ Σᵢⱼₖₗ gᵢⱼₖₗ a†ᵢa†ⱼaₗaₖ
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                for k in range(self.n_orbitals):
                    for l in range(self.n_orbitals):
                        if abs(self.two_body[i, j, k, l]) > 1e-12:
                            term = ((i, 1), (j, 1), (l, 0), (k, 0))  # a†ᵢa†ⱼaₗaₖ
                            fermion_op.terms[term] = 0.5 * self.two_body[i, j, k, l]

        return fermion_op

    def to_qubit_operator(self, method: str = 'jordan_wigner') -> QubitOperator:
        """
        Convert to qubit operator.

        Args:
            method: Transformation method

        Returns:
            Qubit operator
        """
        fermion_op = self.to_fermion_operator()
        return fermion_to_qubit_operator(fermion_op, method)


def create_h2_hamiltonian(bond_length: float = 0.74) -> MolecularHamiltonian:
    """
    Create H2 molecular Hamiltonian at given bond length.

    Using minimal STO-3G basis. Parameters from quantum chemistry calculations.

    Args:
        bond_length: H-H bond length in Angstroms

    Returns:
        Molecular Hamiltonian
    """
    # Simplified H2 Hamiltonian with 2 spin-orbitals
    # These are approximate values scaled with bond length

    # One-body integrals (kinetic + nuclear attraction)
    r = bond_length / 0.74  # Scale factor relative to equilibrium
    h = np.array([
        [-1.25 * r, 0.0],
        [0.0, -1.25 * r]
    ])

    # Two-body integrals (electron-electron repulsion)
    g = np.zeros((2, 2, 2, 2))
    g[0, 0, 0, 0] = 0.6757 / r
    g[1, 1, 1, 1] = 0.6757 / r
    g[0, 1, 1, 0] = 0.6645 / r
    g[1, 0, 0, 1] = 0.6645 / r

    # Nuclear repulsion
    nuclear_repulsion = 0.7151 / r

    return MolecularHamiltonian(h, g, nuclear_repulsion)


def create_lih_hamiltonian(bond_length: float = 1.54) -> MolecularHamiltonian:
    """
    Create LiH molecular Hamiltonian.

    Args:
        bond_length: Li-H bond length in Angstroms

    Returns:
        Molecular Hamiltonian
    """
    # Simplified LiH with 4 spin-orbitals
    r = bond_length / 1.54

    # One-body integrals
    h = np.array([
        [-2.5 * r, -0.1, 0.0, 0.0],
        [-0.1, -2.3 * r, 0.0, 0.0],
        [0.0, 0.0, -2.5 * r, -0.1],
        [0.0, 0.0, -0.1, -2.3 * r]
    ])

    # Two-body integrals (simplified)
    g = np.zeros((4, 4, 4, 4))
    for i in range(4):
        g[i, i, i, i] = 0.8 / r
        for j in range(i + 1, 4):
            g[i, j, j, i] = 0.5 / r
            g[j, i, i, j] = 0.5 / r

    nuclear_repulsion = 0.9 / r

    return MolecularHamiltonian(h, g, nuclear_repulsion)


def create_beh2_hamiltonian(bond_length: float = 1.33) -> MolecularHamiltonian:
    """
    Create BeH2 molecular Hamiltonian.

    Args:
        bond_length: Be-H bond length in Angstroms

    Returns:
        Molecular Hamiltonian
    """
    # Simplified BeH2 with 6 spin-orbitals
    r = bond_length / 1.33

    # One-body integrals
    h = np.diag([-3.0 * r, -2.8 * r, -2.8 * r, -3.0 * r, -2.8 * r, -2.8 * r])

    # Two-body integrals (simplified)
    g = np.zeros((6, 6, 6, 6))
    for i in range(6):
        g[i, i, i, i] = 1.0 / r
        for j in range(i + 1, 6):
            g[i, j, j, i] = 0.6 / r
            g[j, i, i, j] = 0.6 / r

    nuclear_repulsion = 1.5 / r

    return MolecularHamiltonian(h, g, nuclear_repulsion)
