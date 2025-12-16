"""
Quantum Gradient Computer
Computes gradients for quantum circuits using parameter shift rule
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..backends.quantum_backend_interface import ExecutionResult, QuantumBackend, QuantumCircuit

logger = logging.getLogger(__name__)


@dataclass
class GradientResult:
    """Result from gradient computation"""

    gradients: np.ndarray
    function_value: float
    n_circuit_executions: int
    computation_time_ms: float


class QuantumGradientComputer:
    """
    Computes gradients of quantum circuits using parameter shift rule

    For a parametrized quantum circuit with parameter θ_i:
    ∂⟨O⟩/∂θ_i = [⟨O⟩(θ_i + π/2) - ⟨O⟩(θ_i - π/2)] / 2

    This requires 2 circuit executions per parameter.
    """

    def __init__(self, backend: QuantumBackend, shift_amount: float = np.pi / 2):
        """
        Initialize gradient computer

        Args:
            backend: Quantum backend for circuit execution
            shift_amount: Amount to shift parameters (π/2 for standard gates)
        """
        self.backend = backend
        self.shift = shift_amount
        self._cache: Dict[str, float] = {}

    async def compute_gradients(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
        frozen_indices: Optional[List[int]] = None,
    ) -> GradientResult:
        """
        Compute all parameter gradients

        Args:
            circuit_builder: Function that builds circuit from parameters
            loss_function: Function that computes loss from ExecutionResult
            parameters: Current parameter values
            frozen_indices: Indices of frozen parameters (no gradient)

        Returns:
            GradientResult with gradients and metadata
        """

        start_time = time.time()

        gradients = np.zeros_like(parameters)
        frozen = set(frozen_indices or [])
        n_executions = 0

        # Compute gradient for each parameter
        tasks = []
        param_indices = []

        for i in range(len(parameters)):
            if i not in frozen:
                tasks.append(
                    self._compute_single_gradient(circuit_builder, loss_function, parameters, i)
                )
                param_indices.append(i)

        # Execute in parallel
        results = await asyncio.gather(*tasks)

        # Collect results
        for i, (grad, n_exec) in zip(param_indices, results):
            gradients[i] = grad
            n_executions += n_exec

        # Evaluate current loss
        current_circuit = circuit_builder(parameters)
        current_result = await self.backend.execute_circuit(current_circuit)
        current_loss = loss_function(current_result)
        n_executions += 1

        computation_time = (time.time() - start_time) * 1000

        logger.debug(
            f"Computed {len(param_indices)} gradients with "
            f"{n_executions} circuit executions in {computation_time:.2f}ms"
        )

        return GradientResult(
            gradients=gradients,
            function_value=current_loss,
            n_circuit_executions=n_executions,
            computation_time_ms=computation_time,
        )

    async def _compute_single_gradient(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
        param_idx: int,
    ) -> tuple:
        """
        Compute gradient for a single parameter using parameter shift

        Returns:
            Tuple of (gradient, number of executions)
        """
        # Forward shift
        params_plus = parameters.copy()
        params_plus[param_idx] += self.shift

        circuit_plus = circuit_builder(params_plus)
        result_plus = await self.backend.execute_circuit(circuit_plus)
        loss_plus = loss_function(result_plus)

        # Backward shift
        params_minus = parameters.copy()
        params_minus[param_idx] -= self.shift

        circuit_minus = circuit_builder(params_minus)
        result_minus = await self.backend.execute_circuit(circuit_minus)
        loss_minus = loss_function(result_minus)

        # Gradient via parameter shift
        gradient = (loss_plus - loss_minus) / 2

        return gradient, 2

    async def compute_gradient_stochastic(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
        batch_size: int = 4,
    ) -> GradientResult:
        """
        Compute gradients for a random subset of parameters
        Useful for large parameter spaces

        Args:
            circuit_builder: Circuit builder function
            loss_function: Loss function
            parameters: Current parameters
            batch_size: Number of random parameters to compute

        Returns:
            GradientResult with sparse gradients
        """

        start_time = time.time()

        gradients = np.zeros_like(parameters)

        # Select random parameter subset
        indices = np.random.choice(
            len(parameters), size=min(batch_size, len(parameters)), replace=False
        )

        n_executions = 0

        # Compute gradients for subset
        for idx in indices:
            grad, n_exec = await self._compute_single_gradient(
                circuit_builder, loss_function, parameters, idx
            )
            gradients[idx] = grad
            n_executions += n_exec

        # Evaluate current loss
        current_circuit = circuit_builder(parameters)
        current_result = await self.backend.execute_circuit(current_circuit)
        current_loss = loss_function(current_result)
        n_executions += 1

        computation_time = (time.time() - start_time) * 1000

        return GradientResult(
            gradients=gradients,
            function_value=current_loss,
            n_circuit_executions=n_executions,
            computation_time_ms=computation_time,
        )

    async def compute_hessian_diagonal(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
    ) -> np.ndarray:
        """
        Compute diagonal elements of Hessian matrix

        Uses finite differences on gradients:
        H_ii ≈ [∂²L/∂θ_i²] = [g(θ_i + ε) - g(θ_i - ε)] / (2ε)

        Args:
            circuit_builder: Circuit builder function
            loss_function: Loss function
            parameters: Current parameters

        Returns:
            Diagonal Hessian elements
        """
        hessian_diag = np.zeros_like(parameters)
        epsilon = 0.01

        for i in range(len(parameters)):
            # Gradient at θ_i + ε
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            grad_plus, _ = await self._compute_single_gradient(
                circuit_builder, loss_function, params_plus, i
            )

            # Gradient at θ_i - ε
            params_minus = parameters.copy()
            params_minus[i] -= epsilon
            grad_minus, _ = await self._compute_single_gradient(
                circuit_builder, loss_function, params_minus, i
            )

            # Second derivative
            hessian_diag[i] = (grad_plus - grad_minus) / (2 * epsilon)

        return hessian_diag

    def clear_cache(self):
        """Clear gradient cache"""
        self._cache.clear()


class FiniteDifferenceGradient:
    """
    Compute gradients using finite differences
    Fallback method when parameter shift is not applicable
    """

    def __init__(self, backend: QuantumBackend, epsilon: float = 1e-4):
        """
        Initialize finite difference gradient computer

        Args:
            backend: Quantum backend
            epsilon: Small perturbation for finite differences
        """
        self.backend = backend
        self.epsilon = epsilon

    async def compute_gradients(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
    ) -> GradientResult:
        """
        Compute gradients using finite differences

        ∂L/∂θ_i ≈ [L(θ_i + ε) - L(θ_i)] / ε
        """

        start_time = time.time()

        gradients = np.zeros_like(parameters)

        # Base loss
        base_circuit = circuit_builder(parameters)
        base_result = await self.backend.execute_circuit(base_circuit)
        base_loss = loss_function(base_result)

        n_executions = 1

        # Compute gradient for each parameter
        for i in range(len(parameters)):
            params_perturbed = parameters.copy()
            params_perturbed[i] += self.epsilon

            circuit = circuit_builder(params_perturbed)
            result = await self.backend.execute_circuit(circuit)
            loss = loss_function(result)

            gradients[i] = (loss - base_loss) / self.epsilon
            n_executions += 1

        computation_time = (time.time() - start_time) * 1000

        return GradientResult(
            gradients=gradients,
            function_value=base_loss,
            n_circuit_executions=n_executions,
            computation_time_ms=computation_time,
        )


class NaturalGradientComputer:
    """
    Compute quantum natural gradients
    Uses Quantum Fisher Information Matrix
    """

    def __init__(self, backend: QuantumBackend, regularization: float = 1e-8):
        """
        Initialize natural gradient computer

        Args:
            backend: Quantum backend
            regularization: Regularization for matrix inversion
        """
        self.backend = backend
        self.regularization = regularization
        self.gradient_computer = QuantumGradientComputer(backend)

    async def compute_natural_gradients(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
    ) -> GradientResult:
        """
        Compute natural gradients using Fisher information

        Natural gradient = F^(-1) * gradient
        where F is the Quantum Fisher Information Matrix

        Args:
            circuit_builder: Circuit builder function
            loss_function: Loss function
            parameters: Current parameters

        Returns:
            GradientResult with natural gradients
        """
        start_time = time.time()

        # Compute standard gradients
        grad_result = await self.gradient_computer.compute_gradients(
            circuit_builder, loss_function, parameters
        )

        # Compute Fisher information matrix (approximation)
        fisher = await self._compute_fisher_matrix(circuit_builder, parameters)

        # Add regularization for numerical stability
        fisher += self.regularization * np.eye(len(parameters))

        # Compute natural gradient: F^(-1) * g
        try:
            natural_gradients = np.linalg.solve(fisher, grad_result.gradients)
        except np.linalg.LinAlgError:
            logger.warning("Fisher matrix singular, using standard gradients")
            natural_gradients = grad_result.gradients

        computation_time = (time.time() - start_time) * 1000

        return GradientResult(
            gradients=natural_gradients,
            function_value=grad_result.function_value,
            n_circuit_executions=grad_result.n_circuit_executions,
            computation_time_ms=computation_time,
        )

    async def _compute_fisher_matrix(
        self, circuit_builder: Callable[[np.ndarray], QuantumCircuit], parameters: np.ndarray
    ) -> np.ndarray:
        """
        Compute quantum Fisher information matrix

        F_ij = Re[⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩]

        This is a simplified approximation using parameter shifts
        """
        n_params = len(parameters)
        fisher = np.zeros((n_params, n_params))

        # Simplified Fisher matrix using parameter shift overlaps
        for i in range(n_params):
            for j in range(i, n_params):
                # Compute overlap approximation
                overlap = await self._compute_parameter_overlap(circuit_builder, parameters, i, j)
                fisher[i, j] = overlap
                fisher[j, i] = overlap

        return fisher

    async def _compute_parameter_overlap(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        parameters: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """
        Compute overlap between parameter-shifted states

        Simplified metric tensor element
        """
        shift = np.pi / 2

        # Create circuits with parameter shifts
        params_ij = parameters.copy()
        params_ij[i] += shift
        params_ij[j] += shift

        circuit = circuit_builder(params_ij)
        result = await self.backend.execute_circuit(circuit)

        # Use measurement probabilities as overlap proxy
        # This is a simplified approximation
        prob_overlap = sum(result.probabilities.values())

        return prob_overlap
