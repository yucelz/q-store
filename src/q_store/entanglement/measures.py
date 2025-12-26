"""
Entanglement measures for quantum states.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.linalg import sqrtm, logm


def concurrence(state: np.ndarray) -> float:
    """
    Calculate the concurrence of a two-qubit state.

    For a two-qubit density matrix ρ, concurrence C is defined as:
    C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    where λᵢ are the square roots of eigenvalues of ρ(σy⊗σy)ρ*(σy⊗σy) in decreasing order.

    Args:
        state: 4x4 density matrix of two-qubit system

    Returns:
        Concurrence value between 0 (separable) and 1 (maximally entangled)
    """
    if state.shape != (4, 4):
        raise ValueError("Concurrence requires a 4x4 two-qubit density matrix")

    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sigma_y, sigma_y)

    # Compute R = ρ(σy⊗σy)ρ*(σy⊗σy)
    R = state @ sy_sy @ state.conj() @ sy_sy

    # Get eigenvalues and sort in decreasing order
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate concurrence
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    return float(C)


def negativity(state: np.ndarray, subsystem: int = 0) -> float:
    """
    Calculate the negativity of a bipartite quantum state.

    Negativity is defined as N = (||ρ^Tₐ||₁ - 1)/2, where ρ^Tₐ is the
    partial transpose with respect to subsystem A.

    Args:
        state: Density matrix of bipartite system
        subsystem: Which subsystem to transpose (0 or 1)

    Returns:
        Negativity value (0 for separable states, > 0 for entangled)
    """
    dim = state.shape[0]

    # Determine subsystem dimensions (assume equal for simplicity)
    dim_a = int(np.sqrt(dim))
    dim_b = dim // dim_a

    if dim_a * dim_b != dim:
        raise ValueError("Cannot determine subsystem dimensions")

    # Partial transpose
    rho_pt = partial_transpose(state, dim_a, dim_b, subsystem)

    # Calculate trace norm (sum of absolute eigenvalues)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    trace_norm = np.sum(np.abs(eigenvalues))

    # Negativity
    N = (trace_norm - 1) / 2

    return float(N)


def partial_transpose(state: np.ndarray, dim_a: int, dim_b: int, subsystem: int = 0) -> np.ndarray:
    """
    Compute partial transpose of a bipartite state.

    Args:
        state: Density matrix
        dim_a: Dimension of subsystem A
        dim_b: Dimension of subsystem B
        subsystem: Which subsystem to transpose (0 for A, 1 for B)

    Returns:
        Partially transposed density matrix
    """
    rho_pt = np.zeros_like(state)

    if subsystem == 0:
        # Transpose subsystem A
        for i in range(dim_a):
            for j in range(dim_a):
                for k in range(dim_b):
                    for l in range(dim_b):
                        rho_pt[i * dim_b + k, j * dim_b + l] = state[j * dim_b + k, i * dim_b + l]
    else:
        # Transpose subsystem B
        for i in range(dim_a):
            for j in range(dim_a):
                for k in range(dim_b):
                    for l in range(dim_b):
                        rho_pt[i * dim_b + k, j * dim_b + l] = state[i * dim_b + l, j * dim_b + k]

    return rho_pt


def entropy_of_entanglement(state: np.ndarray) -> float:
    """
    Calculate the entropy of entanglement for a pure bipartite state.

    For a pure state |ψ⟩, the entropy of entanglement is the von Neumann
    entropy of the reduced density matrix: E = -Tr(ρₐ log ρₐ)

    Args:
        state: Pure state density matrix (or state vector)

    Returns:
        Entropy of entanglement
    """
    # If state vector provided, convert to density matrix
    if state.ndim == 1:
        state = np.outer(state, state.conj())

    dim = state.shape[0]
    dim_a = int(np.sqrt(dim))
    dim_b = dim // dim_a

    # Compute reduced density matrix for subsystem A
    rho_a = partial_trace(state, dim_a, dim_b, keep=0)

    # Calculate von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_a)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues

    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return float(entropy)


def partial_trace(state: np.ndarray, dim_a: int, dim_b: int, keep: int = 0) -> np.ndarray:
    """
    Compute partial trace over one subsystem.

    Args:
        state: Density matrix
        dim_a: Dimension of subsystem A
        dim_b: Dimension of subsystem B
        keep: Which subsystem to keep (0 for A, 1 for B)

    Returns:
        Reduced density matrix
    """
    if keep == 0:
        # Trace out subsystem B, keep A
        rho_a = np.zeros((dim_a, dim_a), dtype=complex)
        for i in range(dim_a):
            for j in range(dim_a):
                for k in range(dim_b):
                    rho_a[i, j] += state[i * dim_b + k, j * dim_b + k]
        return rho_a
    else:
        # Trace out subsystem A, keep B
        rho_b = np.zeros((dim_b, dim_b), dtype=complex)
        for k in range(dim_b):
            for l in range(dim_b):
                for i in range(dim_a):
                    rho_b[k, l] += state[i * dim_b + k, i * dim_b + l]
        return rho_b


def entanglement_of_formation(state: np.ndarray) -> float:
    """
    Calculate the entanglement of formation for a two-qubit state.

    For two qubits, EOF is related to concurrence C by:
    E(ρ) = h((1 + √(1-C²))/2)
    where h(x) = -x log₂(x) - (1-x)log₂(1-x) is the binary entropy.

    Args:
        state: 4x4 density matrix of two-qubit system

    Returns:
        Entanglement of formation
    """
    C = concurrence(state)

    if C == 0:
        return 0.0

    # Calculate binary entropy
    x = (1 + np.sqrt(1 - C**2)) / 2

    if x == 0 or x == 1:
        return 0.0

    h = -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    return float(h)


class EntanglementMeasure:
    """
    Class for computing various entanglement measures.
    """

    def __init__(self, state: np.ndarray):
        """
        Initialize with a quantum state.

        Args:
            state: Density matrix or state vector
        """
        # Convert state vector to density matrix if needed
        if state.ndim == 1:
            self.state = np.outer(state, state.conj())
        else:
            self.state = state

        self.dim = self.state.shape[0]

    def compute_all_measures(self) -> dict:
        """
        Compute all applicable entanglement measures.

        Returns:
            Dictionary of measure names and values
        """
        measures = {}

        # Try concurrence and EOF for two-qubit systems
        if self.dim == 4:
            try:
                measures['concurrence'] = concurrence(self.state)
                measures['entanglement_of_formation'] = entanglement_of_formation(self.state)
            except Exception as e:
                measures['concurrence'] = None
                measures['entanglement_of_formation'] = None

        # Negativity works for any bipartite system
        try:
            measures['negativity'] = negativity(self.state)
        except Exception:
            measures['negativity'] = None

        # Entropy of entanglement for pure states
        try:
            # Check if state is approximately pure
            purity = np.real(np.trace(self.state @ self.state))
            if purity > 0.99:
                measures['entropy_of_entanglement'] = entropy_of_entanglement(self.state)
            else:
                measures['entropy_of_entanglement'] = None
        except Exception:
            measures['entropy_of_entanglement'] = None

        return measures

    def is_entangled(self, threshold: float = 1e-6) -> bool:
        """
        Determine if the state is entangled based on negativity.

        Args:
            threshold: Threshold for considering a state entangled

        Returns:
            True if entangled, False if separable
        """
        try:
            neg = negativity(self.state)
            return neg > threshold
        except Exception:
            return False
