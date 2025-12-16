"""
SPSA Gradient Estimator - v3.3
Simultaneous Perturbation Stochastic Approximation for quantum gradients

Key Innovation: Estimates ALL gradients with just 2 circuit evaluations
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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
    method: str = "spsa"
    metadata: Dict[str, Any] = None


class SPSAGradientEstimator:
    """
    Simultaneous Perturbation Stochastic Approximation

    SPSA estimates gradients for ALL parameters using only 2 circuit evaluations,
    compared to 2N evaluations required by parameter shift rule.

    Algorithm:
    ----------
    1. Generate random perturbation vector: δ ~ Rademacher({-1, +1}^n)
    2. Evaluate loss at perturbed points:
       L₊ = L(θ + c_k δ)
       L₋ = L(θ - c_k δ)
    3. Estimate gradient:
       ∇L ≈ [(L₊ - L₋) / (2c_k)] * δ

    where c_k is the perturbation magnitude (decreases with iteration k).

    Convergence Properties:
    ----------------------
    - Proven to converge to true gradient in expectation
    - Convergence rate: O(1/k^α) where α ∈ (0.5, 1]
    - Works well with noisy measurements (quantum case)
    - Robust to measurement errors

    References:
    ----------
    Spall, J.C. (1992). "Multivariate stochastic approximation using a
    simultaneous perturbation gradient approximation." IEEE TAC.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        c_decay: float = 0.101,
        a_decay: float = 0.602,
        c_initial: float = 0.1,
        a_initial: float = 0.01,
        min_perturbation: float = 1e-4,
    ):
        """
        Initialize SPSA gradient estimator

        Args:
            backend: Quantum backend for circuit execution
            c_decay: Decay exponent for c_k (perturbation magnitude)
            a_decay: Decay exponent for a_k (step size)
            c_initial: Initial perturbation magnitude
            a_initial: Initial step size
            min_perturbation: Minimum perturbation magnitude
        """
        self.backend = backend

        # SPSA gain sequences (tuned for quantum ML)
        self.c_decay = c_decay
        self.a_decay = a_decay
        self.c_initial = c_initial
        self.a_initial = a_initial
        self.min_perturbation = min_perturbation

        self.iteration = 0

        # Statistics
        self.total_circuits_executed = 0
        self.total_time_ms = 0.0

    def get_gain_parameters(self, iteration: int) -> tuple:
        """
        Compute gain parameters c_k and a_k

        Standard SPSA schedules:
        - c_k = c / (k + 1)^γ  where γ ≈ 0.101
        - a_k = a / (k + 1)^α  where α ≈ 0.602

        These values satisfy convergence conditions:
        - Σ a_k = ∞ (steps don't vanish too quickly)
        - Σ a_k² < ∞ (steps decrease sufficiently)

        Returns:
            (c_k, a_k): perturbation magnitude and step size
        """
        k = iteration + 1  # Avoid division by zero

        c_k = self.c_initial / (k**self.c_decay)
        a_k = self.a_initial / (k**self.a_decay)

        # Enforce minimum perturbation
        c_k = max(c_k, self.min_perturbation)

        return c_k, a_k

    async def estimate_gradient(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
        shots: int = 1000,
        frozen_indices: Optional[set] = None,
    ) -> GradientResult:
        """
        Estimate gradient using SPSA

        This is the core SPSA algorithm. It computes gradient estimates
        for ALL parameters using only 2 circuit evaluations.

        Args:
            circuit_builder: Function that builds circuit from parameters
            loss_function: Function that computes loss from ExecutionResult
            parameters: Current parameter values (θ)
            shots: Number of measurement shots per circuit
            frozen_indices: Parameter indices to freeze (gradient = 0)

        Returns:
            GradientResult with gradient estimate and metadata

        Example:
            >>> estimator = SPSAGradientEstimator(backend)
            >>>
            >>> def build_circuit(params):
            ...     circuit = create_variational_circuit(params)
            ...     return circuit
            >>>
            >>> def loss(result):
            ...     return compute_loss_from_measurements(result)
            >>>
            >>> result = await estimator.estimate_gradient(
            ...     build_circuit, loss, current_params, shots=1000
            ... )
            >>>
            >>> print(f"Gradient: {result.gradients}")
            >>> print(f"Only {result.n_circuit_executions} circuits used!")
        """
        start_time = time.time()

        # Get gain parameters for current iteration
        c_k, a_k = self.get_gain_parameters(self.iteration)

        # Generate random perturbation (Rademacher distribution)
        # Each element is ±1 with equal probability
        delta = np.random.choice([-1, 1], size=len(parameters))

        # Apply frozen parameters (set perturbation to 0)
        if frozen_indices:
            for idx in frozen_indices:
                delta[idx] = 0

        # Create perturbed parameter vectors
        params_plus = parameters + c_k * delta
        params_minus = parameters - c_k * delta

        # Build circuits for both perturbations
        circuit_plus = circuit_builder(params_plus)
        circuit_minus = circuit_builder(params_minus)

        # Execute both circuits in parallel
        logger.debug(f"SPSA iteration {self.iteration}: Executing 2 circuits with c_k={c_k:.6f}")

        results = await asyncio.gather(
            self.backend.execute_circuit(circuit_plus, shots=shots),
            self.backend.execute_circuit(circuit_minus, shots=shots),
        )

        result_plus, result_minus = results

        # Compute losses
        loss_plus = loss_function(result_plus)
        loss_minus = loss_function(result_minus)

        # SPSA gradient estimate
        # ∇L ≈ [(L(θ+cδ) - L(θ-cδ)) / (2c)] * δ
        gradient = ((loss_plus - loss_minus) / (2 * c_k)) * delta

        # Average loss (used as function value)
        avg_loss = (loss_plus + loss_minus) / 2.0

        # Update iteration counter
        self.iteration += 1
        self.total_circuits_executed += 2

        computation_time = (time.time() - start_time) * 1000
        self.total_time_ms += computation_time

        logger.info(
            f"SPSA gradient computed: loss={avg_loss:.4f}, "
            f"||∇||={np.linalg.norm(gradient):.4f}, "
            f"time={computation_time:.2f}ms, circuits=2"
        )

        return GradientResult(
            gradients=gradient,
            function_value=avg_loss,
            n_circuit_executions=2,  # 🔥 Only 2 circuits!
            computation_time_ms=computation_time,
            method="spsa",
            metadata={
                "iteration": self.iteration,
                "c_k": c_k,
                "a_k": a_k,
                "loss_plus": loss_plus,
                "loss_minus": loss_minus,
                "perturbation_norm": np.linalg.norm(delta),
                "total_circuits": self.total_circuits_executed,
            },
        )

    async def estimate_gradient_with_momentum(
        self,
        circuit_builder: Callable[[np.ndarray], QuantumCircuit],
        loss_function: Callable[[ExecutionResult], float],
        parameters: np.ndarray,
        momentum: float = 0.9,
        shots: int = 1000,
    ) -> GradientResult:
        """
        SPSA with momentum accumulation

        Improves convergence by accumulating gradient estimates:
        g_t = β * g_{t-1} + (1-β) * ∇L_t

        Args:
            momentum: Momentum coefficient (0-1)

        Returns:
            GradientResult with momentum-smoothed gradient
        """
        # Get SPSA gradient
        result = await self.estimate_gradient(circuit_builder, loss_function, parameters, shots)

        # Apply momentum if this isn't first iteration
        if hasattr(self, "_momentum_buffer"):
            smoothed_gradient = momentum * self._momentum_buffer + (1 - momentum) * result.gradients
            result.gradients = smoothed_gradient

        # Update momentum buffer
        self._momentum_buffer = result.gradients

        return result

    def reset(self):
        """Reset iteration counter and statistics"""
        self.iteration = 0
        self.total_circuits_executed = 0
        self.total_time_ms = 0.0
        if hasattr(self, "_momentum_buffer"):
            delattr(self, "_momentum_buffer")

    def get_statistics(self) -> Dict[str, Any]:
        """Get SPSA statistics"""
        return {
            "iteration": self.iteration,
            "total_circuits_executed": self.total_circuits_executed,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_gradient_ms": (
                self.total_time_ms / self.iteration if self.iteration > 0 else 0
            ),
            "current_c_k": self.get_gain_parameters(self.iteration)[0],
            "current_a_k": self.get_gain_parameters(self.iteration)[1],
        }


class SPSAOptimizerWithAdaptiveGains:
    """
    Enhanced SPSA with adaptive gain tuning

    Automatically adjusts c_k and a_k based on:
    - Gradient variance
    - Loss improvement rate
    - Parameter sensitivity
    """

    def __init__(self, backend: QuantumBackend, adapt_every: int = 10):
        self.backend = backend
        self.adapt_every = adapt_every

        # Base SPSA estimator
        self.spsa = SPSAGradientEstimator(backend)

        # Adaptation history
        self.gradient_history = []
        self.loss_history = []

    async def estimate_gradient(
        self, circuit_builder: Callable, loss_function: Callable, parameters: np.ndarray, **kwargs
    ) -> GradientResult:
        """Estimate gradient with adaptive gains"""

        # Adapt gains periodically
        if self.spsa.iteration > 0 and self.spsa.iteration % self.adapt_every == 0:
            self._adapt_gains()

        # Compute gradient
        result = await self.spsa.estimate_gradient(
            circuit_builder, loss_function, parameters, **kwargs
        )

        # Track history
        self.gradient_history.append(result.gradients)
        self.loss_history.append(result.function_value)

        return result

    def _adapt_gains(self):
        """
        Adapt gain parameters based on training progress

        Strategy:
        - If gradients are noisy: increase c_k (larger perturbations)
        - If loss is flat: increase a_k (larger steps)
        - If converging well: decrease both (refinement)
        """
        if len(self.gradient_history) < 2:
            return

        # Compute gradient variance
        recent_grads = self.gradient_history[-10:]
        grad_variance = np.var([np.linalg.norm(g) for g in recent_grads])

        # Compute loss improvement
        recent_losses = self.loss_history[-10:]
        loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]

        # Adapt c_initial based on gradient variance
        if grad_variance > 1.0:
            # High variance: increase perturbation
            self.spsa.c_initial *= 1.1
            logger.info("SPSA: Increased c_initial due to high gradient variance")
        elif grad_variance < 0.1:
            # Low variance: decrease perturbation
            self.spsa.c_initial *= 0.9
            logger.info("SPSA: Decreased c_initial due to low gradient variance")

        # Adapt a_initial based on loss improvement
        if abs(loss_improvement) < 0.01:
            # Slow improvement: increase step size
            self.spsa.a_initial *= 1.1
            logger.info("SPSA: Increased a_initial due to slow convergence")
        elif loss_improvement > 0.1:
            # Fast improvement: maintain current schedule
            pass

    def reset(self):
        """Reset optimizer state"""
        self.spsa.reset()
        self.gradient_history.clear()
        self.loss_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        stats = self.spsa.get_statistics()
        stats.update(
            {
                "gradient_history_length": len(self.gradient_history),
                "loss_history_length": len(self.loss_history),
            }
        )
        return stats
