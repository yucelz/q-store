"""
Verification protocols for entangled states.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate fidelity between two quantum states.

    For pure states: F = |⟨ψ₁|ψ₂⟩|²
    For mixed states: F = (Tr√(√ρ₁ρ₂√ρ₁))²

    Args:
        state1: First state (density matrix or state vector)
        state2: Second state (density matrix or state vector)

    Returns:
        Fidelity value between 0 and 1
    """
    # Convert state vectors to density matrices
    if state1.ndim == 1:
        state1 = np.outer(state1, state1.conj())
    if state2.ndim == 1:
        state2 = np.outer(state2, state2.conj())

    # Compute fidelity using Uhlmann's theorem
    sqrt_state1 = _matrix_sqrt(state1)
    M = sqrt_state1 @ state2 @ sqrt_state1
    sqrt_M = _matrix_sqrt(M)

    fidelity = np.real(np.trace(sqrt_M)) ** 2

    return min(max(float(fidelity), 0.0), 1.0)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T


def verify_bell_state(state: np.ndarray, target_bell: str = 'phi_plus') -> Dict[str, float]:
    """
    Verify if a state is close to a target Bell state.

    Bell states:
    - |Φ⁺⟩ = (|00⟩ + |11⟩)/√2  (phi_plus)
    - |Φ⁻⟩ = (|00⟩ - |11⟩)/√2  (phi_minus)
    - |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2  (psi_plus)
    - |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2  (psi_minus)

    Args:
        state: Density matrix or state vector to verify
        target_bell: Which Bell state to compare against

    Returns:
        Dictionary with verification results
    """
    if state.shape[0] != 4:
        raise ValueError("Bell state verification requires a two-qubit state")

    # Define Bell states
    bell_states = {
        'phi_plus': np.array([1, 0, 0, 1]) / np.sqrt(2),
        'phi_minus': np.array([1, 0, 0, -1]) / np.sqrt(2),
        'psi_plus': np.array([0, 1, 1, 0]) / np.sqrt(2),
        'psi_minus': np.array([0, 1, -1, 0]) / np.sqrt(2)
    }

    if target_bell not in bell_states:
        raise ValueError(f"Unknown Bell state: {target_bell}")

    target = bell_states[target_bell]

    # Calculate fidelity
    fid = state_fidelity(state, target)

    # Calculate all Bell state overlaps
    overlaps = {}
    for name, bell_state in bell_states.items():
        overlaps[name] = state_fidelity(state, bell_state)

    return {
        'target_fidelity': fid,
        'is_bell_state': fid > 0.95,
        'bell_state_overlaps': overlaps,
        'closest_bell_state': max(overlaps, key=overlaps.get)
    }


def verify_ghz_state(state: np.ndarray, n_qubits: int = 3) -> Dict[str, float]:
    """
    Verify if a state is close to a GHZ state.

    GHZ state: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

    Args:
        state: Density matrix or state vector to verify
        n_qubits: Number of qubits

    Returns:
        Dictionary with verification results
    """
    dim = 2 ** n_qubits

    if state.shape[0] != dim:
        raise ValueError(f"Expected {dim}-dimensional state for {n_qubits} qubits")

    # Create GHZ state
    ghz = np.zeros(dim, dtype=complex)
    ghz[0] = 1.0 / np.sqrt(2)  # |00...0⟩
    ghz[-1] = 1.0 / np.sqrt(2)  # |11...1⟩

    # Calculate fidelity
    fid = state_fidelity(state, ghz)

    # Check parity
    parity = _check_ghz_parity(state, n_qubits)

    return {
        'fidelity': fid,
        'is_ghz_state': fid > 0.95,
        'parity_expectation': parity,
        'n_qubits': n_qubits
    }


def _check_ghz_parity(state: np.ndarray, n_qubits: int) -> float:
    """
    Check parity expectation value for GHZ state.

    For GHZ states, measuring all qubits in X basis gives ⟨X⊗X⊗...⊗X⟩ = 1
    """
    dim = 2 ** n_qubits

    # Create Pauli X operators
    X = np.array([[0, 1], [1, 0]])

    # Tensor product of n X operators
    X_all = X
    for _ in range(n_qubits - 1):
        X_all = np.kron(X_all, X)

    # Convert state vector to density matrix if needed
    if state.ndim == 1:
        state = np.outer(state, state.conj())

    # Calculate expectation value
    parity = np.real(np.trace(X_all @ state))

    return float(parity)


def verify_w_state(state: np.ndarray, n_qubits: int = 3) -> Dict[str, float]:
    """
    Verify if a state is close to a W state.

    W state: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n

    Args:
        state: Density matrix or state vector to verify
        n_qubits: Number of qubits

    Returns:
        Dictionary with verification results
    """
    dim = 2 ** n_qubits

    if state.shape[0] != dim:
        raise ValueError(f"Expected {dim}-dimensional state for {n_qubits} qubits")

    # Create W state
    w = np.zeros(dim, dtype=complex)
    for i in range(n_qubits):
        # Set computational basis state with single 1 in position i
        basis_index = 2 ** (n_qubits - 1 - i)
        w[basis_index] = 1.0 / np.sqrt(n_qubits)

    # Calculate fidelity
    fid = state_fidelity(state, w)

    # Check if state has Dicke structure (symmetric under permutation)
    symmetry = _check_permutation_symmetry(state, n_qubits)

    return {
        'fidelity': fid,
        'is_w_state': fid > 0.95,
        'permutation_symmetry': symmetry,
        'n_qubits': n_qubits
    }


def _check_permutation_symmetry(state: np.ndarray, n_qubits: int) -> float:
    """
    Check how symmetric a state is under qubit permutations.

    Returns a score from 0 (not symmetric) to 1 (fully symmetric).
    """
    # Convert state vector to density matrix if needed
    if state.ndim == 1:
        state = np.outer(state, state.conj())

    # For simplicity, just check a few permutations
    # Full check would require testing all n! permutations

    # Check that all single-excitation amplitudes are equal
    dim = 2 ** n_qubits
    single_excitation_indices = [2 ** (n_qubits - 1 - i) for i in range(n_qubits)]

    amplitudes = [np.abs(state[idx, idx]) for idx in single_excitation_indices]

    # Calculate variance normalized by mean
    if len(amplitudes) > 0 and np.mean(amplitudes) > 1e-10:
        symmetry = 1.0 - np.std(amplitudes) / np.mean(amplitudes)
        return max(0.0, min(1.0, float(symmetry)))

    return 0.0


class EntanglementVerifier:
    """
    Class for verifying entangled states.
    """

    def __init__(self, state: np.ndarray, n_qubits: Optional[int] = None):
        """
        Initialize with a quantum state.

        Args:
            state: Density matrix or state vector
            n_qubits: Number of qubits (inferred if not provided)
        """
        if state.ndim == 1:
            self.state = np.outer(state, state.conj())
        else:
            self.state = state

        self.dim = self.state.shape[0]

        if n_qubits is None:
            self.n_qubits = int(np.log2(self.dim))
        else:
            self.n_qubits = n_qubits

    def verify_all_standard_states(self) -> Dict[str, any]:
        """
        Test if state matches any standard entangled states.

        Returns:
            Dictionary with verification results for all standard states
        """
        results = {}

        # Bell states (two qubits)
        if self.n_qubits == 2:
            for bell_type in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
                try:
                    results[bell_type] = verify_bell_state(self.state, bell_type)
                except Exception as e:
                    results[bell_type] = {'error': str(e)}

        # GHZ state
        try:
            results['ghz'] = verify_ghz_state(self.state, self.n_qubits)
        except Exception as e:
            results['ghz'] = {'error': str(e)}

        # W state
        try:
            results['w'] = verify_w_state(self.state, self.n_qubits)
        except Exception as e:
            results['w'] = {'error': str(e)}

        return results

    def identify_state(self, threshold: float = 0.9) -> Tuple[str, float]:
        """
        Identify which standard entangled state this most closely matches.

        Args:
            threshold: Minimum fidelity to consider a match

        Returns:
            Tuple of (state_name, fidelity)
        """
        results = self.verify_all_standard_states()

        best_match = None
        best_fidelity = 0.0

        # Check Bell states
        if self.n_qubits == 2:
            for bell_type in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
                if bell_type in results and 'target_fidelity' in results[bell_type]:
                    fid = results[bell_type]['target_fidelity']
                    if fid > best_fidelity:
                        best_fidelity = fid
                        best_match = bell_type

        # Check GHZ
        if 'ghz' in results and 'fidelity' in results['ghz']:
            fid = results['ghz']['fidelity']
            if fid > best_fidelity:
                best_fidelity = fid
                best_match = 'ghz'

        # Check W
        if 'w' in results and 'fidelity' in results['w']:
            fid = results['w']['fidelity']
            if fid > best_fidelity:
                best_fidelity = fid
                best_match = 'w'

        if best_match and best_fidelity >= threshold:
            return best_match, best_fidelity
        else:
            return 'unknown', best_fidelity
