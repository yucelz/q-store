"""
Quantum kernel for machine learning.

A quantum kernel computes similarity between data points by encoding them
into quantum states and measuring overlap.
"""

import numpy as np
from typing import Callable, Optional, List
from q_store.core import UnifiedCircuit
from q_store.embeddings import ZZFeatureMap


class QuantumKernel:
    """
    Quantum kernel for computing similarity between data points.

    The kernel k(x, x') measures similarity by:
    k(x, x') = |⟨ψ(x)|ψ(x')⟩|²

    where |ψ(x)⟩ is the quantum state after encoding x.
    """

    def __init__(
        self,
        feature_map: Callable[[np.ndarray], UnifiedCircuit],
        n_qubits: Optional[int] = None
    ):
        """
        Initialize quantum kernel.

        Args:
            feature_map: Function that encodes data into quantum circuit
            n_qubits: Number of qubits (inferred from feature map if None)
        """
        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self._kernel_cache = {}

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Evaluate kernel between two data points.

        Args:
            x1: First data point
            x2: Second data point

        Returns:
            Kernel value k(x1, x2)
        """
        # Create circuits for both points
        circuit1 = self.feature_map(x1)
        circuit2 = self.feature_map(x2)

        if self.n_qubits is None:
            self.n_qubits = circuit1.n_qubits

        # Compute overlap |⟨ψ(x1)|ψ(x2)⟩|²
        # This is a simplified implementation - in practice would use
        # statevector simulation or measurement-based estimation
        overlap = self._compute_overlap(circuit1, circuit2)

        return abs(overlap) ** 2

    def compute_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute kernel matrix for dataset(s).

        Args:
            X: First dataset (n_samples, n_features)
            Y: Second dataset (optional, uses X if None)

        Returns:
            Kernel matrix K where K[i,j] = k(X[i], Y[j])
        """
        if Y is None:
            Y = X

        K = np.zeros((len(X), len(Y)))

        for i, x1 in enumerate(X):
            for j, x2 in enumerate(Y):
                K[i, j] = self.evaluate(x1, x2)

        return K

    def _compute_overlap(self, circuit1: UnifiedCircuit, circuit2: UnifiedCircuit) -> complex:
        """
        Compute overlap between two quantum states.

        This is a simplified placeholder - real implementation would
        use statevector simulation.
        """
        # For now, return a dummy overlap based on circuit similarity
        # In production, this would simulate the circuits and compute
        # the inner product of their statevectors

        # Simple heuristic: overlap decreases with circuit differences
        gates1 = len(circuit1.gates)
        gates2 = len(circuit2.gates)

        # Placeholder calculation
        overlap = np.exp(-0.1 * abs(gates1 - gates2))
        return complex(overlap, 0)


def compute_kernel_matrix(
    X: np.ndarray,
    feature_map: Callable[[np.ndarray], UnifiedCircuit],
    Y: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute quantum kernel matrix.

    Args:
        X: First dataset
        feature_map: Quantum feature map
        Y: Second dataset (optional)

    Returns:
        Kernel matrix
    """
    kernel = QuantumKernel(feature_map)
    return kernel.compute_matrix(X, Y)


def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Compute alignment between two kernel matrices.

    Kernel alignment measures similarity between kernels:
    A(K1, K2) = ⟨K1, K2⟩_F / (||K1||_F ||K2||_F)

    Args:
        K1: First kernel matrix
        K2: Second kernel matrix

    Returns:
        Alignment score (0 to 1)
    """
    # Frobenius inner product
    inner = np.sum(K1 * K2)

    # Frobenius norms
    norm1 = np.sqrt(np.sum(K1 ** 2))
    norm2 = np.sqrt(np.sum(K2 ** 2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return inner / (norm1 * norm2)


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Compute kernel-target alignment.

    Measures how well kernel aligns with target labels:
    KTA = ⟨K, yy^T⟩_F / (||K||_F ||yy^T||_F)

    Args:
        K: Kernel matrix
        y: Target labels

    Returns:
        Alignment score
    """
    # Create ideal kernel from labels
    y = np.array(y).reshape(-1, 1)
    Y = y @ y.T

    return kernel_alignment(K, Y)


def compute_kernel_variance(K: np.ndarray) -> float:
    """
    Compute variance of kernel matrix.

    Args:
        K: Kernel matrix

    Returns:
        Variance
    """
    return np.var(K)


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    """
    Normalize kernel matrix to have unit diagonal.

    K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])

    Args:
        K: Kernel matrix

    Returns:
        Normalized kernel matrix
    """
    diag = np.diag(K)
    diag_sqrt = np.sqrt(np.outer(diag, diag))

    # Avoid division by zero
    diag_sqrt = np.where(diag_sqrt == 0, 1, diag_sqrt)

    return K / diag_sqrt


def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Center kernel matrix (remove mean).

    Args:
        K: Kernel matrix

    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n

    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    return K_centered


def create_default_kernel(n_features: int, n_qubits: Optional[int] = None) -> QuantumKernel:
    """
    Create default quantum kernel with ZZ feature map.

    Args:
        n_features: Number of input features
        n_qubits: Number of qubits (defaults to n_features)

    Returns:
        Quantum kernel
    """
    if n_qubits is None:
        n_qubits = n_features

    def feature_map(x):
        return ZZFeatureMap(n_qubits=n_qubits, data=x, reps=2)

    return QuantumKernel(feature_map, n_qubits)
