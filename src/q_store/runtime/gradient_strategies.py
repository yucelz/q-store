"""
Gradient Strategy Abstraction - v4.1 Enhanced
Provides pluggable gradient estimation strategies for quantum circuits

Key Innovation: Extensible architecture supporting multiple gradient methods
- SPSA (v4.1 default)
- Parameter-shift (planned v4.2)
- Natural gradients (planned v4.2)
- Adaptive switching based on circuit properties

Design:
- Abstract base class for strategy pattern
- Async-first for non-blocking quantum execution
- Integrates with GradientNoiseTracker for adaptation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..ml.spsa_gradient_estimator import SPSAGradientEstimator as BaseSPSAEstimator
from ..backends.quantum_backend_interface import QuantumCircuit

logger = logging.getLogger(__name__)


@dataclass
class GradientEstimate:
    """
    Result from gradient estimation.

    Attributes
    ----------
    gradient : np.ndarray
        Estimated gradient vector
    variance : float
        Variance of gradient estimate
    n_circuit_executions : int
        Number of circuits executed
    computation_time_ms : float
        Time taken for computation
    strategy_name : str
        Name of strategy used
    metadata : dict, optional
        Additional strategy-specific data
    """
    gradient: np.ndarray
    variance: float
    n_circuit_executions: int
    computation_time_ms: float
    strategy_name: str
    metadata: Optional[Dict[str, Any]] = None


class GradientStrategy(ABC):
    """
    Abstract base class for quantum gradient estimation strategies.

    All gradient strategies must implement async gradient estimation
    for non-blocking quantum circuit execution.

    Examples
    --------
    >>> strategy = SPSAGradientEstimator(epsilon=0.1)
    >>> gradient = await strategy.estimate_gradient(
    ...     circuit=my_circuit,
    ...     params=theta,
    ...     loss_fn=my_loss
    ... )
    """

    @abstractmethod
    async def estimate_gradient(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        **kwargs
    ) -> GradientEstimate:
        """
        Estimate gradient of loss function with respect to parameters.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized quantum circuit
        params : np.ndarray
            Current parameter values
        loss_fn : callable
            Loss function to minimize
        **kwargs
            Additional strategy-specific arguments

        Returns
        -------
        GradientEstimate
            Gradient estimate with metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and tracking."""
        pass


class SPSAGradientEstimator(GradientStrategy):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimator.

    **v4.1 Default Strategy**

    Advantages:
    - Sample-efficient: 2 circuit evaluations per gradient
    - Works with any number of parameters
    - Noisy but unbiased estimate

    Disadvantages:
    - High variance for deep circuits
    - Slower convergence than exact methods

    Parameters
    ----------
    epsilon : float, default=0.1
        Perturbation magnitude
    samples_per_gradient : int, default=1
        Number of samples to average
    adaptive_epsilon : bool, default=True
        Decay epsilon over iterations
    backend : str, default='simulator'
        Quantum backend to use

    Examples
    --------
    >>> estimator = SPSAGradientEstimator(epsilon=0.1)
    >>> gradient = await estimator.estimate_gradient(circuit, params, loss_fn)
    >>> print(f"Gradient: {gradient.gradient}")
    >>> print(f"Variance: {gradient.variance}")
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        samples_per_gradient: int = 1,
        adaptive_epsilon: bool = True,
        backend: str = 'simulator'
    ):
        # Wrap existing SPSA implementation
        self._base_estimator = BaseSPSAEstimator(
            backend=backend,
            perturbation_magnitude=epsilon,
            adaptive_perturbation=adaptive_epsilon
        )

        self.epsilon = epsilon
        self.samples_per_gradient = samples_per_gradient
        self.adaptive_epsilon = adaptive_epsilon
        self.iteration = 0

        logger.info(
            f"Initialized SPSA gradient estimator: "
            f"epsilon={epsilon}, samples={samples_per_gradient}, "
            f"adaptive={adaptive_epsilon}"
        )

    async def estimate_gradient(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        **kwargs
    ) -> GradientEstimate:
        """
        Estimate gradient using SPSA method.

        Process:
        1. Generate random perturbation δ ~ Rademacher({-1, +1}^n)
        2. Evaluate loss at θ + ε·δ and θ - ε·δ
        3. Estimate gradient: ∇L ≈ [(L₊ - L₋) / (2ε)] · δ
        """
        import time
        start_time = time.time()

        n_params = len(params)
        gradient_sum = np.zeros(n_params)
        variance_sum = 0.0
        total_executions = 0

        # Adaptive epsilon
        if self.adaptive_epsilon:
            epsilon = self.epsilon / (1 + self.iteration / 100)
        else:
            epsilon = self.epsilon

        # Average over multiple samples
        gradients = []
        for _ in range(self.samples_per_gradient):
            # Random perturbation (±1 for each parameter)
            delta = np.random.choice([-1, 1], size=n_params)

            # Perturbed parameters
            params_plus = params + epsilon * delta
            params_minus = params - epsilon * delta

            # Evaluate loss (async if loss_fn supports it)
            if asyncio.iscoroutinefunction(loss_fn):
                loss_plus, loss_minus = await asyncio.gather(
                    loss_fn(params_plus),
                    loss_fn(params_minus)
                )
            else:
                loss_plus = loss_fn(params_plus)
                loss_minus = loss_fn(params_minus)

            # SPSA gradient estimate
            gradient = (loss_plus - loss_minus) / (2 * epsilon) * delta
            gradients.append(gradient)
            gradient_sum += gradient
            total_executions += 2  # Two circuit evaluations

        # Average gradient
        avg_gradient = gradient_sum / self.samples_per_gradient

        # Compute variance
        if len(gradients) > 1:
            variance = np.var(gradients, axis=0).mean()
        else:
            variance = 0.0

        self.iteration += 1

        elapsed_ms = (time.time() - start_time) * 1000

        return GradientEstimate(
            gradient=avg_gradient,
            variance=variance,
            n_circuit_executions=total_executions,
            computation_time_ms=elapsed_ms,
            strategy_name=self.name,
            metadata={
                'epsilon': epsilon,
                'iteration': self.iteration,
                'samples': self.samples_per_gradient
            }
        )

    @property
    def name(self) -> str:
        return "SPSA"


class AdaptiveGradientEstimator(GradientStrategy):
    """
    Adaptive gradient estimator that switches between strategies.

    **v4.1 Experimental**

    Switches based on:
    - Circuit depth (shallow → parameter-shift, deep → SPSA)
    - Gradient variance (high noise → increase SPSA samples)
    - Training phase (early → exploration, late → refinement)

    Note: Currently uses SPSA only. Parameter-shift will be added in v4.2.

    Parameters
    ----------
    spsa_estimator : SPSAGradientEstimator
        SPSA strategy instance
    variance_threshold : float, default=0.1
        Variance threshold for adaptation
    depth_threshold : int, default=10
        Circuit depth threshold

    Examples
    --------
    >>> spsa = SPSAGradientEstimator(epsilon=0.1)
    >>> adaptive = AdaptiveGradientEstimator(spsa, variance_threshold=0.1)
    >>> gradient = await adaptive.estimate_gradient(circuit, params, loss_fn)
    """

    def __init__(
        self,
        spsa_estimator: SPSAGradientEstimator,
        variance_threshold: float = 0.1,
        depth_threshold: int = 10
    ):
        self.spsa = spsa_estimator
        self.variance_threshold = variance_threshold
        self.depth_threshold = depth_threshold

        # Track variance history for adaptation
        self.variance_history = []

        logger.info(
            f"Initialized adaptive gradient estimator: "
            f"variance_threshold={variance_threshold}, "
            f"depth_threshold={depth_threshold}"
        )

    async def estimate_gradient(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        **kwargs
    ) -> GradientEstimate:
        """
        Adaptively choose and execute gradient estimation.

        Current logic (v4.1):
        - Always use SPSA
        - Adapt SPSA samples based on variance history

        Future (v4.2):
        - Use parameter-shift for shallow circuits (depth < threshold)
        - Use SPSA for deep circuits
        """
        # For now, always use SPSA (parameter-shift coming in v4.2)
        gradient = await self.spsa.estimate_gradient(circuit, params, loss_fn, **kwargs)

        # Track variance
        self.variance_history.append(gradient.variance)
        if len(self.variance_history) > 100:
            self.variance_history.pop(0)

        # Adapt SPSA samples based on variance
        if len(self.variance_history) >= 10:
            recent_variance = np.mean(self.variance_history[-10:])

            if recent_variance > self.variance_threshold:
                # High variance - increase samples
                old_samples = self.spsa.samples_per_gradient
                self.spsa.samples_per_gradient = min(5, old_samples + 1)

                if self.spsa.samples_per_gradient != old_samples:
                    logger.info(
                        f"Increased SPSA samples: {old_samples} → "
                        f"{self.spsa.samples_per_gradient} "
                        f"(variance={recent_variance:.4f})"
                    )

            elif recent_variance < self.variance_threshold / 2:
                # Low variance - can reduce samples
                old_samples = self.spsa.samples_per_gradient
                self.spsa.samples_per_gradient = max(1, old_samples - 1)

                if self.spsa.samples_per_gradient != old_samples:
                    logger.info(
                        f"Reduced SPSA samples: {old_samples} → "
                        f"{self.spsa.samples_per_gradient} "
                        f"(variance={recent_variance:.4f})"
                    )

        # Update metadata
        gradient.metadata = gradient.metadata or {}
        gradient.metadata['adaptive_strategy'] = True
        gradient.metadata['variance_history_size'] = len(self.variance_history)

        return gradient

    @property
    def name(self) -> str:
        return "Adaptive"

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        if not self.variance_history:
            return {}

        return {
            'variance_history_size': len(self.variance_history),
            'mean_variance': np.mean(self.variance_history),
            'std_variance': np.std(self.variance_history),
            'current_spsa_samples': self.spsa.samples_per_gradient,
        }


# Future strategies (placeholders for v4.2)

class ParameterShiftGradientEstimator(GradientStrategy):
    """
    Parameter-shift rule gradient estimator.

    **Planned for v4.2**

    Provides exact gradients for quantum circuits using the parameter-shift rule.
    Requires 2N circuit evaluations (N = number of parameters).

    Advantages:
    - Exact gradients (no noise from estimation)
    - Works well for shallow circuits

    Disadvantages:
    - Requires 2N circuits (expensive for many parameters)
    - Only works for gates with certain properties
    """

    def __init__(self):
        raise NotImplementedError(
            "Parameter-shift gradient estimator will be implemented in v4.2"
        )

    async def estimate_gradient(self, circuit, params, loss_fn, **kwargs):
        raise NotImplementedError("Not yet implemented")

    @property
    def name(self) -> str:
        return "ParameterShift"


class NaturalGradientEstimator(GradientStrategy):
    """
    Natural gradient estimator using Fisher information.

    **Research Phase**

    Uses the quantum Fisher information matrix to compute natural gradients,
    which can provide better convergence in certain optimization landscapes.

    Note: Computationally expensive on NISQ devices.
    """

    def __init__(self):
        raise NotImplementedError(
            "Natural gradient estimator is in research phase"
        )

    async def estimate_gradient(self, circuit, params, loss_fn, **kwargs):
        raise NotImplementedError("Not yet implemented")

    @property
    def name(self) -> str:
        return "NaturalGradient"
