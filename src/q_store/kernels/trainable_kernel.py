"""
Trainable quantum kernel with parameter optimization.

Enables learning kernel parameters to improve performance on
specific tasks.
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple
from scipy.optimize import minimize
from q_store.core import UnifiedCircuit
from q_store.kernels.quantum_kernel import QuantumKernel, kernel_target_alignment


class TrainableQuantumKernel(QuantumKernel):
    """
    Quantum kernel with trainable parameters.

    Optimizes feature map parameters to maximize kernel-target alignment
    or minimize classification loss.
    """

    def __init__(
        self,
        parameterized_feature_map: Callable[[np.ndarray, np.ndarray], UnifiedCircuit],
        initial_parameters: np.ndarray,
        n_qubits: Optional[int] = None
    ):
        """
        Initialize trainable kernel.

        Args:
            parameterized_feature_map: Feature map that takes (data, params)
            initial_parameters: Initial parameter values
            n_qubits: Number of qubits
        """
        self.parameterized_feature_map = parameterized_feature_map
        self.parameters = initial_parameters.copy()
        self.n_qubits = n_qubits
        self._kernel_cache = {}

        # Create feature map with current parameters
        def feature_map(x):
            return self.parameterized_feature_map(x, self.parameters)

        super().__init__(feature_map, n_qubits)

    def update_parameters(self, new_parameters: np.ndarray):
        """Update kernel parameters."""
        self.parameters = new_parameters.copy()

        # Update feature map
        def feature_map(x):
            return self.parameterized_feature_map(x, self.parameters)

        self.feature_map = feature_map
        self._kernel_cache.clear()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        loss_fn: str = "alignment",
        max_iter: int = 100,
        method: str = "COBYLA"
    ) -> Dict[str, any]:
        """
        Train kernel parameters.

        Args:
            X_train: Training data
            y_train: Training labels
            loss_fn: Loss function ("alignment", "margin", "accuracy")
            max_iter: Maximum optimization iterations
            method: Optimization method

        Returns:
            Training results
        """
        def objective(params):
            self.update_parameters(params)
            K = self.compute_matrix(X_train)
            return -kernel_loss(K, y_train, loss_fn)

        result = minimize(
            objective,
            self.parameters,
            method=method,
            options={'maxiter': max_iter}
        )

        self.update_parameters(result.x)

        return {
            'final_parameters': result.x,
            'final_loss': -result.fun,
            'n_evaluations': result.nfev,
            'success': result.success
        }

    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        return self.parameters.copy()


def optimize_kernel_parameters(
    X: np.ndarray,
    y: np.ndarray,
    parameterized_feature_map: Callable[[np.ndarray, np.ndarray], UnifiedCircuit],
    initial_parameters: np.ndarray,
    loss_fn: str = "alignment",
    max_iter: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Optimize quantum kernel parameters.

    Args:
        X: Training data
        y: Training labels
        parameterized_feature_map: Parameterized feature map
        initial_parameters: Initial parameters
        loss_fn: Loss function
        max_iter: Maximum iterations

    Returns:
        Tuple of (optimized_parameters, final_loss)
    """
    kernel = TrainableQuantumKernel(
        parameterized_feature_map,
        initial_parameters
    )

    result = kernel.train(X, y, loss_fn, max_iter)

    return result['final_parameters'], result['final_loss']


def kernel_loss(K: np.ndarray, y: np.ndarray, loss_type: str = "alignment") -> float:
    """
    Compute kernel loss for training.

    Args:
        K: Kernel matrix
        y: Target labels
        loss_type: Type of loss ("alignment", "margin", "accuracy")

    Returns:
        Loss value
    """
    if loss_type == "alignment":
        return kernel_target_alignment(K, y)

    elif loss_type == "margin":
        # Compute margin-based loss
        # Higher kernel values for same-class pairs, lower for different-class
        y = np.array(y).reshape(-1, 1)
        same_class = (y == y.T).astype(float)

        # Want K high when same_class=1, low when same_class=0
        loss = np.sum(K * same_class) - np.sum(K * (1 - same_class))
        return loss / (len(y) ** 2)

    elif loss_type == "accuracy":
        # Pseudo-accuracy based on nearest neighbors in kernel space
        n = len(y)
        correct = 0

        for i in range(n):
            # Find nearest neighbor (excluding self)
            K_i = K[i].copy()
            K_i[i] = -np.inf
            nearest_idx = np.argmax(K_i)

            if y[i] == y[nearest_idx]:
                correct += 1

        return correct / n

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def gradient_free_optimization(
    X: np.ndarray,
    y: np.ndarray,
    kernel: TrainableQuantumKernel,
    n_iterations: int = 50,
    step_size: float = 0.1
) -> np.ndarray:
    """
    Gradient-free optimization using finite differences.

    Args:
        X: Training data
        y: Training labels
        kernel: Trainable kernel
        n_iterations: Number of iterations
        step_size: Step size for perturbations

    Returns:
        Optimized parameters
    """
    params = kernel.get_parameters()

    for iteration in range(n_iterations):
        # Compute current loss
        K = kernel.compute_matrix(X)
        current_loss = kernel_loss(K, y)

        # Try perturbations
        best_params = params.copy()
        best_loss = current_loss

        for i in range(len(params)):
            # Positive perturbation
            params_plus = params.copy()
            params_plus[i] += step_size
            kernel.update_parameters(params_plus)
            K_plus = kernel.compute_matrix(X)
            loss_plus = kernel_loss(K_plus, y)

            if loss_plus > best_loss:
                best_loss = loss_plus
                best_params = params_plus

            # Negative perturbation
            params_minus = params.copy()
            params_minus[i] -= step_size
            kernel.update_parameters(params_minus)
            K_minus = kernel.compute_matrix(X)
            loss_minus = kernel_loss(K_minus, y)

            if loss_minus > best_loss:
                best_loss = loss_minus
                best_params = params_minus

        params = best_params
        kernel.update_parameters(params)

    return params


def quantum_kernel_svm_score(
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> float:
    """
    Compute SVM classification score with quantum kernel.

    Args:
        K_train: Training kernel matrix
        K_test: Test kernel matrix
        y_train: Training labels
        y_test: Test labels

    Returns:
        Classification accuracy
    """
    # Simplified SVM scoring using nearest neighbors in kernel space
    n_test = len(y_test)
    correct = 0

    for i in range(n_test):
        # Find nearest neighbor in training set
        nearest_idx = np.argmax(K_test[i])
        predicted = y_train[nearest_idx]

        if predicted == y_test[i]:
            correct += 1

    return correct / n_test


def kernel_parameter_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    kernel: TrainableQuantumKernel,
    param_idx: int,
    param_range: np.ndarray
) -> np.ndarray:
    """
    Analyze kernel sensitivity to parameter changes.

    Args:
        X: Data
        y: Labels
        kernel: Trainable kernel
        param_idx: Parameter index to vary
        param_range: Range of parameter values

    Returns:
        Array of loss values
    """
    original_params = kernel.get_parameters()
    losses = np.zeros(len(param_range))

    for i, param_value in enumerate(param_range):
        params = original_params.copy()
        params[param_idx] = param_value

        kernel.update_parameters(params)
        K = kernel.compute_matrix(X)
        losses[i] = kernel_loss(K, y)

    # Restore original parameters
    kernel.update_parameters(original_params)

    return losses
