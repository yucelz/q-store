"""
Optimization loops for hybrid quantum-classical algorithms.

Provides tools for iterative parameter optimization in hybrid workflows.
"""

from typing import Callable, Optional, Dict, Any, List
import numpy as np


class ParameterUpdate:
    """
    Parameter update strategies for optimization.

    Implements various update rules for gradient-based and
    gradient-free optimization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        method: str = 'gradient_descent'
    ):
        """
        Initialize parameter updater.

        Args:
            learning_rate: Learning rate for updates
            method: Update method ('gradient_descent', 'adam', 'momentum')
        """
        self.learning_rate = learning_rate
        self.method = method

        # For momentum and Adam
        self.velocity = None
        self.m = None  # First moment (Adam)
        self.v = None  # Second moment (Adam)
        self.t = 0  # Time step

    def update(
        self,
        parameters: np.ndarray,
        gradients: Optional[np.ndarray] = None,
        loss_change: Optional[float] = None
    ) -> np.ndarray:
        """
        Update parameters based on gradients or loss change.

        Args:
            parameters: Current parameters
            gradients: Parameter gradients (if available)
            loss_change: Change in loss (for gradient-free methods)

        Returns:
            Updated parameters
        """
        if self.method == 'gradient_descent':
            if gradients is None:
                raise ValueError("Gradient descent requires gradients")
            return parameters - self.learning_rate * gradients

        elif self.method == 'momentum':
            if gradients is None:
                raise ValueError("Momentum requires gradients")

            if self.velocity is None:
                self.velocity = np.zeros_like(parameters)

            beta = 0.9
            self.velocity = beta * self.velocity + (1 - beta) * gradients
            return parameters - self.learning_rate * self.velocity

        elif self.method == 'adam':
            if gradients is None:
                raise ValueError("Adam requires gradients")

            if self.m is None:
                self.m = np.zeros_like(parameters)
                self.v = np.zeros_like(parameters)

            self.t += 1
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8

            # Update biased moments
            self.m = beta1 * self.m + (1 - beta1) * gradients
            self.v = beta2 * self.v + (1 - beta2) * (gradients ** 2)

            # Bias correction
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)

            return parameters - self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        else:
            raise ValueError(f"Unknown update method: {self.method}")

    def reset(self):
        """Reset optimizer state."""
        self.velocity = None
        self.m = None
        self.v = None
        self.t = 0


class ConvergenceChecker:
    """
    Check convergence criteria for optimization.

    Monitors loss, parameter changes, and gradient norms.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        patience: int = 10,
        check_gradient: bool = False
    ):
        """
        Initialize convergence checker.

        Args:
            tolerance: Convergence tolerance
            patience: Number of iterations without improvement
            check_gradient: Whether to check gradient norm
        """
        self.tolerance = tolerance
        self.patience = patience
        self.check_gradient = check_gradient

        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.loss_history = []

    def check(
        self,
        loss: float,
        parameters: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if optimization has converged.

        Args:
            loss: Current loss value
            parameters: Current parameters
            gradients: Current gradients

        Returns:
            True if converged, False otherwise
        """
        self.loss_history.append(loss)

        # Check loss improvement
        if loss < self.best_loss - self.tolerance:
            self.best_loss = loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check patience
        if self.no_improvement_count >= self.patience:
            return True

        # Check gradient norm if requested
        if self.check_gradient and gradients is not None:
            grad_norm = np.linalg.norm(gradients)
            if grad_norm < self.tolerance:
                return True

        # Check loss change
        if len(self.loss_history) > 1:
            loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
            if loss_change < self.tolerance:
                return True

        return False

    def reset(self):
        """Reset convergence state."""
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.loss_history = []


class OptimizationLoop:
    """
    Main optimization loop for hybrid quantum-classical algorithms.

    Coordinates parameter updates, loss evaluation, and convergence checking.
    """

    def __init__(
        self,
        loss_function: Callable,
        parameter_updater: ParameterUpdate,
        convergence_checker: Optional[ConvergenceChecker] = None,
        max_iterations: int = 100
    ):
        """
        Initialize optimization loop.

        Args:
            loss_function: Function to compute loss given parameters
            parameter_updater: Parameter update strategy
            convergence_checker: Convergence checker
            max_iterations: Maximum number of iterations
        """
        self.loss_function = loss_function
        self.parameter_updater = parameter_updater
        self.convergence_checker = convergence_checker or ConvergenceChecker()
        self.max_iterations = max_iterations

    def run(
        self,
        initial_parameters: np.ndarray,
        gradient_function: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run optimization loop.

        Args:
            initial_parameters: Starting parameters
            gradient_function: Function to compute gradients
            callback: Optional callback function called each iteration

        Returns:
            Dictionary with optimization results
        """
        parameters = initial_parameters.copy()
        loss_history = []

        for iteration in range(self.max_iterations):
            # Evaluate loss
            loss = self.loss_function(parameters)
            loss_history.append(loss)

            # Compute gradients if function provided
            gradients = None
            if gradient_function is not None:
                gradients = gradient_function(parameters)

            # Check convergence
            if self.convergence_checker.check(loss, parameters, gradients):
                break

            # Update parameters
            parameters = self.parameter_updater.update(
                parameters,
                gradients=gradients
            )

            # Call callback if provided
            if callback is not None:
                callback(iteration, parameters, loss)

        return {
            'parameters': parameters,
            'loss': loss_history[-1] if loss_history else None,
            'loss_history': loss_history,
            'iterations': len(loss_history),
            'converged': iteration < self.max_iterations - 1
        }


def run_optimization_loop(
    loss_function: Callable,
    initial_parameters: np.ndarray,
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    gradient_function: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run optimization loop with default settings.

    Args:
        loss_function: Function to minimize
        initial_parameters: Starting parameters
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        gradient_function: Optional gradient function

    Returns:
        Optimization results
    """
    updater = ParameterUpdate(
        learning_rate=learning_rate,
        method='adam' if gradient_function else 'gradient_descent'
    )

    checker = ConvergenceChecker(
        tolerance=tolerance,
        patience=10
    )

    loop = OptimizationLoop(
        loss_function=loss_function,
        parameter_updater=updater,
        convergence_checker=checker,
        max_iterations=max_iterations
    )

    return loop.run(
        initial_parameters=initial_parameters,
        gradient_function=gradient_function
    )
