"""
Adaptive Gradient Optimizer - v3.3
Automatically selects best gradient method based on training stage

Key Innovation: Optimal speed/accuracy tradeoff throughout training
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..backends.backend_manager import BackendManager
from ..backends.quantum_backend_interface import QuantumBackend
from .gradient_computer import QuantumGradientComputer
from .spsa_gradient_estimator import GradientResult, SPSAGradientEstimator

logger = logging.getLogger(__name__)


class AdaptiveGradientOptimizer:
    """
    Automatically selects best gradient method based on:
    - Training stage (early vs late)
    - Parameter sensitivity
    - Convergence rate
    - Available computational budget
    """

    def __init__(
        self, backend: QuantumBackend, initial_method: str = "spsa", enable_adaptation: bool = True
    ):
        """
        Initialize adaptive gradient optimizer

        Args:
            backend: Quantum backend
            initial_method: Starting gradient method
            enable_adaptation: Whether to auto-switch methods
        """
        self.backend = backend
        self.enable_adaptation = enable_adaptation

        # Available gradient methods
        self.methods = {
            "spsa": SPSAGradientEstimator(backend),
            "parameter_shift": QuantumGradientComputer(backend),
        }

        self.current_method = initial_method
        self.method_history = []

        # Adaptation state
        self.iteration = 0
        self.convergence_history = []
        self.gradient_variance_history = []

        logger.info(f"Initialized adaptive optimizer with method: {initial_method}")

    async def compute_gradients(
        self,
        circuit_builder: Callable,
        loss_function: Callable,
        parameters: np.ndarray,
        frozen_indices: Optional[set] = None,
        shots: int = 1000,
    ) -> GradientResult:
        """
        Compute gradients with adaptive method selection

        Args:
            circuit_builder: Function to build circuit from parameters
            loss_function: Function to compute loss from result
            parameters: Current parameter values
            frozen_indices: Indices of frozen parameters
            shots: Shots per circuit

        Returns:
            GradientResult with gradient estimate
        """
        # Select method adaptively
        if self.enable_adaptation:
            method = self._select_method()
        else:
            method = self.current_method

        # Get estimator
        estimator = self.methods[method]

        # Compute gradients
        if method == "spsa":
            result = await estimator.estimate_gradient(
                circuit_builder,
                loss_function,
                parameters,
                shots=shots,
                frozen_indices=frozen_indices,
            )
        else:  # parameter_shift
            result = await estimator.compute_gradients(
                circuit_builder, loss_function, parameters, frozen_indices=frozen_indices
            )

        # Track method usage
        self.method_history.append(method)
        self.current_method = method
        self.iteration += 1

        # Update adaptation state
        self._update_adaptation_state(result)

        logger.debug(
            f"Iteration {self.iteration}: Used {method}, "
            f"loss={result.function_value:.4f}, "
            f"||∇||={np.linalg.norm(result.gradients):.4f}"
        )

        return result

    def _select_method(self) -> str:
        """
        Select gradient method based on training progress

        Strategy:
        - Early training (< 10 iters): SPSA for speed
        - Every 10th iteration: Parameter shift for accuracy check
        - Slow convergence: Switch to parameter shift
        - Fast convergence: Continue with SPSA
        """
        # Early training: use fast SPSA
        if self.iteration < 10:
            return "spsa"

        # Periodic refinement with accurate gradients
        if self.iteration % 10 == 0:
            logger.info(f"Iteration {self.iteration}: Using parameter shift for accuracy check")
            return "parameter_shift"

        # Check convergence
        if self._is_converging_slowly():
            logger.info("Slow convergence detected, switching to parameter shift")
            return "parameter_shift"

        # Check if gradients are too noisy
        if self._gradients_too_noisy():
            logger.info("High gradient variance, using parameter shift")
            return "parameter_shift"

        # Default: continue with SPSA
        return "spsa"

    def _is_converging_slowly(self, window: int = 5) -> bool:
        """
        Detect slow convergence

        Args:
            window: Number of recent iterations to check

        Returns:
            True if convergence is slow
        """
        if len(self.convergence_history) < window + 1:
            return False

        recent = self.convergence_history[-window:]

        # Check if loss is improving
        if len(recent) < 2:
            return False

        improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)

        return improvement < 0.01  # Less than 1% improvement

    def _gradients_too_noisy(self, threshold: float = 2.0) -> bool:
        """
        Check if gradient estimates are too noisy

        Args:
            threshold: Variance threshold

        Returns:
            True if gradients are noisy
        """
        if len(self.gradient_variance_history) < 3:
            return False

        recent_variance = np.mean(self.gradient_variance_history[-3:])

        return recent_variance > threshold

    def _update_adaptation_state(self, result: GradientResult):
        """
        Update internal state for adaptation

        Args:
            result: Latest gradient result
        """
        # Track convergence (loss values)
        self.convergence_history.append(result.function_value)

        # Track gradient variance (as proxy for noise)
        grad_norm = np.linalg.norm(result.gradients)
        if len(self.convergence_history) > 1:
            # Estimate variance from gradient norm changes
            prev_norm = np.linalg.norm(
                self.gradient_variance_history[-1] if self.gradient_variance_history else 0
            )
            variance = abs(grad_norm - prev_norm)
            self.gradient_variance_history.append(variance)
        else:
            self.gradient_variance_history.append(grad_norm)

        # Keep history bounded
        max_history = 50
        if len(self.convergence_history) > max_history:
            self.convergence_history = self.convergence_history[-max_history:]
        if len(self.gradient_variance_history) > max_history:
            self.gradient_variance_history = self.gradient_variance_history[-max_history:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimizer statistics

        Returns:
            Dictionary with statistics
        """
        # Count method usage
        method_counts = {}
        for method in self.method_history:
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            "iteration": self.iteration,
            "current_method": self.current_method,
            "method_counts": method_counts,
            "convergence_rate": self._estimate_convergence_rate(),
            "gradient_variance": (
                np.mean(self.gradient_variance_history[-10:])
                if self.gradient_variance_history
                else 0
            ),
            "adaptation_enabled": self.enable_adaptation,
        }

    def _estimate_convergence_rate(self) -> float:
        """
        Estimate current convergence rate

        Returns:
            Convergence rate (loss improvement per iteration)
        """
        if len(self.convergence_history) < 2:
            return 0.0

        recent = self.convergence_history[-10:]
        if len(recent) < 2:
            return 0.0

        # Linear regression on recent loss values
        x = np.arange(len(recent))
        y = np.array(recent)

        # Slope indicates convergence rate
        slope = np.polyfit(x, y, 1)[0]

        return -slope  # Negative slope means improvement

    def reset(self):
        """Reset optimizer state"""
        self.iteration = 0
        self.method_history.clear()
        self.convergence_history.clear()
        self.gradient_variance_history.clear()

        # Reset individual estimators
        for estimator in self.methods.values():
            if hasattr(estimator, "reset"):
                estimator.reset()

        logger.info("Reset adaptive optimizer")

    def force_method(self, method: str):
        """
        Force a specific gradient method

        Args:
            method: Method to use ('spsa' or 'parameter_shift')
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")

        self.current_method = method
        self.enable_adaptation = False

        logger.info(f"Forced gradient method to: {method}")

    def enable_auto_adaptation(self):
        """Re-enable automatic method adaptation"""
        self.enable_adaptation = True
        logger.info("Enabled automatic method adaptation")


class GradientMethodScheduler:
    """
    Schedule gradient methods over training

    More deterministic than adaptive selection
    """

    def __init__(self, backend: QuantumBackend, schedule: Optional[List[tuple]] = None):
        """
        Initialize scheduler

        Args:
            backend: Quantum backend
            schedule: List of (iteration, method) tuples
        """
        self.backend = backend

        # Default schedule
        if schedule is None:
            schedule = [
                (0, "spsa"),  # Start with SPSA
                (50, "parameter_shift"),  # Switch to accurate gradients
                (100, "spsa"),  # Back to SPSA for fine-tuning
            ]

        self.schedule = sorted(schedule)
        self.iteration = 0

        # Gradient computers
        self.methods = {
            "spsa": SPSAGradientEstimator(backend),
            "parameter_shift": QuantumGradientComputer(backend),
        }

    async def compute_gradients(
        self, circuit_builder: Callable, loss_function: Callable, parameters: np.ndarray, **kwargs
    ) -> GradientResult:
        """Compute gradients using scheduled method"""

        # Find current method from schedule
        method = self.schedule[0][1]  # Default to first

        for iter_threshold, scheduled_method in self.schedule:
            if self.iteration >= iter_threshold:
                method = scheduled_method

        # Compute gradients
        estimator = self.methods[method]

        if method == "spsa":
            result = await estimator.estimate_gradient(
                circuit_builder, loss_function, parameters, **kwargs
            )
        else:
            result = await estimator.compute_gradients(
                circuit_builder, loss_function, parameters, **kwargs
            )

        self.iteration += 1

        return result

    def add_schedule_point(self, iteration: int, method: str):
        """
        Add point to schedule

        Args:
            iteration: Iteration number
            method: Method to use from this iteration
        """
        self.schedule.append((iteration, method))
        self.schedule = sorted(self.schedule)

        logger.info(f"Added schedule point: iteration {iteration} -> {method}")
