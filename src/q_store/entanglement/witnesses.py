"""
Entanglement witnesses and detection methods.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .measures import partial_transpose


def bell_inequality_test(measurements: Dict[str, float], inequality: str = 'CHSH') -> Tuple[float, bool]:
    """
    Test Bell inequality violation.

    Args:
        measurements: Dictionary of measurement outcomes
            For CHSH: {'AB', 'AB_prime', 'A_primeB', 'A_primeB_prime'}
        inequality: Type of Bell inequality ('CHSH', 'CH74')

    Returns:
        Tuple of (S value, is_violated)
        For CHSH: S > 2 indicates violation (entanglement)
    """
    if inequality == 'CHSH':
        # CHSH inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2
        E_ab = measurements.get('AB', 0.0)
        E_ab_prime = measurements.get('AB_prime', 0.0)
        E_a_prime_b = measurements.get('A_primeB', 0.0)
        E_a_prime_b_prime = measurements.get('A_primeB_prime', 0.0)

        S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)

        classical_bound = 2.0
        is_violated = S > classical_bound

        return S, is_violated

    else:
        raise ValueError(f"Unknown inequality type: {inequality}")


def witness_operator(state: np.ndarray, witness_type: str = 'PPT') -> float:
    """
    Apply an entanglement witness operator to a state.

    An entanglement witness W is an observable such that:
    - Tr(Wρ) ≥ 0 for all separable states
    - Tr(Wρ) < 0 for some entangled states

    Args:
        state: Density matrix to test
        witness_type: Type of witness ('PPT', 'Bell', 'decomposable')

    Returns:
        Witness value (negative indicates detected entanglement)
    """
    if witness_type == 'PPT':
        # Use PPT criterion as witness
        return ppt_criterion(state)

    elif witness_type == 'Bell':
        # Bell state witness for two qubits
        if state.shape[0] != 4:
            raise ValueError("Bell witness requires two-qubit state")

        # W = 1/2 * I - |Φ⁺⟩⟨Φ⁺|
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_plus_dm = np.outer(phi_plus, phi_plus.conj())

        W = 0.5 * np.eye(4) - phi_plus_dm

        witness_value = np.real(np.trace(W @ state))
        return float(witness_value)

    else:
        raise ValueError(f"Unknown witness type: {witness_type}")


def ppt_criterion(state: np.ndarray, subsystem: int = 0) -> float:
    """
    Apply the Peres-Horodecki PPT (Positive Partial Transpose) criterion.

    A state is separable if its partial transpose is positive semi-definite.
    If the partial transpose has negative eigenvalues, the state is entangled.

    Args:
        state: Density matrix to test
        subsystem: Which subsystem to transpose

    Returns:
        Minimum eigenvalue of partial transpose (negative indicates entanglement)
    """
    dim = state.shape[0]

    # Determine subsystem dimensions
    dim_a = int(np.sqrt(dim))
    dim_b = dim // dim_a

    if dim_a * dim_b != dim:
        raise ValueError("Cannot determine subsystem dimensions")

    # Compute partial transpose
    rho_pt = partial_transpose(state, dim_a, dim_b, subsystem)

    # Get minimum eigenvalue
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    min_eigenvalue = np.min(eigenvalues)

    return float(min_eigenvalue)


def ccnr_criterion(state: np.ndarray) -> bool:
    """
    Apply the CCNR (Computable Cross Norm or Realignment) criterion.

    This is a separability criterion based on realignment of the density matrix.

    Args:
        state: Density matrix to test

    Returns:
        True if criterion is satisfied (state is separable), False otherwise
    """
    dim = state.shape[0]

    # Determine subsystem dimensions
    dim_a = int(np.sqrt(dim))
    dim_b = dim // dim_a

    if dim_a * dim_b != dim:
        raise ValueError("Cannot determine subsystem dimensions")

    # Realignment (also called reshuffling)
    R = np.zeros((dim_a * dim_b, dim_a * dim_b), dtype=complex)

    for i in range(dim_a):
        for j in range(dim_b):
            for k in range(dim_a):
                for l in range(dim_b):
                    R[i * dim_b + j, k * dim_b + l] = state[i * dim_a + k, j * dim_a + l]

    # Compute trace norm
    singular_values = np.linalg.svd(R, compute_uv=False)
    trace_norm = np.sum(singular_values)

    # For separable states, trace norm ≤ 1
    return trace_norm <= 1.0


class EntanglementWitness:
    """
    Class for detecting entanglement using various witness methods.
    """

    def __init__(self, state: np.ndarray):
        """
        Initialize with a quantum state.

        Args:
            state: Density matrix to analyze
        """
        self.state = state
        self.dim = state.shape[0]

    def apply_all_tests(self) -> Dict[str, any]:
        """
        Apply all available entanglement detection tests.

        Returns:
            Dictionary with test results
        """
        results = {}

        # PPT criterion
        try:
            ppt_value = ppt_criterion(self.state)
            results['ppt_criterion'] = {
                'min_eigenvalue': ppt_value,
                'is_entangled': ppt_value < -1e-6
            }
        except Exception as e:
            results['ppt_criterion'] = {'error': str(e)}

        # CCNR criterion
        try:
            ccnr_satisfied = ccnr_criterion(self.state)
            results['ccnr_criterion'] = {
                'is_separable': ccnr_satisfied,
                'is_entangled': not ccnr_satisfied
            }
        except Exception as e:
            results['ccnr_criterion'] = {'error': str(e)}

        # Bell witness (for two qubits)
        if self.dim == 4:
            try:
                bell_value = witness_operator(self.state, 'Bell')
                results['bell_witness'] = {
                    'value': bell_value,
                    'is_entangled': bell_value < -1e-6
                }
            except Exception as e:
                results['bell_witness'] = {'error': str(e)}

        return results

    def is_entangled(self) -> bool:
        """
        Determine if state is entangled using PPT criterion.

        Returns:
            True if entanglement detected, False otherwise
        """
        try:
            ppt_value = ppt_criterion(self.state)
            return ppt_value < -1e-6
        except Exception:
            return False

    def separability_test(self) -> Tuple[bool, str]:
        """
        Test if state is separable using multiple criteria.

        Returns:
            Tuple of (is_separable, method_used)
        """
        # Try PPT first (necessary condition for separability)
        try:
            ppt_value = ppt_criterion(self.state)
            if ppt_value < -1e-6:
                return False, 'PPT criterion violated'
        except Exception:
            pass

        # Try CCNR
        try:
            if not ccnr_criterion(self.state):
                return False, 'CCNR criterion violated'
        except Exception:
            pass

        # If all tests passed, likely separable (but not guaranteed)
        return True, 'All criteria satisfied'
