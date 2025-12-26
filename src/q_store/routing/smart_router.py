"""
Smart Backend Router for intelligent backend selection.

This module implements intelligent routing of quantum circuits to optimal backends
based on circuit properties, backend capabilities, performance history, and cost.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import defaultdict, deque
import numpy as np

from ..core import UnifiedCircuit
from ..backends import QuantumBackend, BackendCapabilities

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Strategy for backend selection."""
    SPEED = "speed"  # Optimize for fastest execution
    COST = "cost"  # Optimize for lowest cost
    PRECISION = "precision"  # Optimize for highest precision
    BALANCED = "balanced"  # Balance speed, cost, and precision
    ADAPTIVE = "adaptive"  # Adapt based on circuit properties


@dataclass
class BackendScore:
    """Score for a backend candidate."""
    backend_name: str
    total_score: float
    speed_score: float
    cost_score: float
    precision_score: float
    capability_score: float
    availability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"BackendScore({self.backend_name}, total={self.total_score:.3f}, "
                f"speed={self.speed_score:.2f}, cost={self.cost_score:.2f})")


@dataclass
class CircuitComplexity:
    """Analysis of circuit complexity."""
    n_qubits: int
    n_gates: int
    depth: int
    n_two_qubit_gates: int
    n_parameters: int
    estimated_runtime: float  # Estimated execution time in seconds

    @classmethod
    def analyze(cls, circuit: UnifiedCircuit) -> 'CircuitComplexity':
        """Analyze circuit complexity."""
        n_gates = len(circuit.gates)

        # Count two-qubit gates based on targets and controls
        n_two_qubit = 0
        for g in circuit.gates:
            qubit_count = len(g.targets)
            if g.controls:
                qubit_count += len(g.controls)
            if qubit_count == 2:
                n_two_qubit += 1

        n_params = len(circuit.parameters)

        # Estimate circuit depth (simplified)
        depth = circuit.depth if hasattr(circuit, 'depth') else n_gates // circuit.n_qubits

        # Estimate runtime based on circuit properties
        # Formula: base + qubits^2 * gates * depth_factor
        base_time = 0.001  # 1ms base
        qubit_factor = circuit.n_qubits ** 1.5
        gate_factor = n_gates * 0.0001
        estimated_runtime = base_time + qubit_factor * gate_factor

        return cls(
            n_qubits=circuit.n_qubits,
            n_gates=n_gates,
            depth=depth,
            n_two_qubit_gates=n_two_qubit,
            n_parameters=n_params,
            estimated_runtime=estimated_runtime
        )
class PerformanceTracker:
    """Track backend performance history."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.execution_times = defaultdict(lambda: deque(maxlen=history_size))
        self.success_rates = defaultdict(lambda: deque(maxlen=history_size))
        self.costs = defaultdict(lambda: deque(maxlen=history_size))

    def record_execution(
        self,
        backend_name: str,
        execution_time: float,
        success: bool,
        cost: float = 0.0
    ):
        """Record a backend execution."""
        self.execution_times[backend_name].append(execution_time)
        self.success_rates[backend_name].append(1.0 if success else 0.0)
        self.costs[backend_name].append(cost)

    def get_avg_execution_time(self, backend_name: str) -> float:
        """Get average execution time for backend."""
        times = self.execution_times.get(backend_name, [])
        return np.mean(times) if times else 1.0

    def get_success_rate(self, backend_name: str) -> float:
        """Get success rate for backend."""
        rates = self.success_rates.get(backend_name, [])
        return np.mean(rates) if rates else 1.0

    def get_avg_cost(self, backend_name: str) -> float:
        """Get average cost for backend."""
        costs_list = self.costs.get(backend_name, [])
        return np.mean(costs_list) if costs_list else 0.0


class SmartBackendRouter:
    """
    Intelligent backend selection router.

    Routes circuits to optimal backends based on:
    - Circuit complexity and requirements
    - Backend capabilities and constraints
    - Performance history and metrics
    - Cost optimization
    - Multi-objective scoring

    Args:
        strategy: Default routing strategy
        performance_weight: Weight for speed (0-1)
        cost_weight: Weight for cost optimization (0-1)
        precision_weight: Weight for precision (0-1)

    Example:
        >>> router = SmartBackendRouter(strategy=RoutingStrategy.BALANCED)
        >>> router.register_backend('qsim', qsim_backend)
        >>> router.register_backend('lightning', lightning_backend)
        >>> best = router.select_backend(circuit)
        >>> result = best.execute(circuit, shots=1000)
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        performance_weight: float = 0.4,
        cost_weight: float = 0.3,
        precision_weight: float = 0.3
    ):
        self.strategy = strategy
        self.performance_weight = performance_weight
        self.cost_weight = cost_weight
        self.precision_weight = precision_weight

        # Normalize weights
        total = performance_weight + cost_weight + precision_weight
        if total > 0:
            self.performance_weight /= total
            self.cost_weight /= total
            self.precision_weight /= total

        self.backends: Dict[str, QuantumBackend] = {}
        self.capabilities: Dict[str, BackendCapabilities] = {}
        self.performance_tracker = PerformanceTracker()
        self.cost_per_shot: Dict[str, float] = {}  # Cost per 1000 shots

        logger.info(f"SmartBackendRouter initialized with strategy={strategy}")

    def register_backend(
        self,
        name: str,
        backend: QuantumBackend,
        cost_per_shot: float = 0.0
    ):
        """
        Register a backend for routing.

        Args:
            name: Backend identifier
            backend: Backend instance
            cost_per_shot: Cost per 1000 shots (for cost optimization)
        """
        self.backends[name] = backend
        self.capabilities[name] = backend.get_capabilities()
        self.cost_per_shot[name] = cost_per_shot
        logger.info(f"Registered backend: {name} (cost={cost_per_shot}/1k shots)")

    def unregister_backend(self, name: str):
        """Remove a backend from routing."""
        if name in self.backends:
            del self.backends[name]
            del self.capabilities[name]
            if name in self.cost_per_shot:
                del self.cost_per_shot[name]
            logger.info(f"Unregistered backend: {name}")

    def _calculate_capability_score(
        self,
        backend_name: str,
        complexity: CircuitComplexity
    ) -> float:
        """Calculate capability match score (0-1)."""
        caps = self.capabilities[backend_name]

        # Check basic requirements
        if complexity.n_qubits > caps.max_qubits:
            return 0.0  # Cannot handle circuit

        # Score based on capability match
        score = 1.0

        # Penalize if using too few qubits (inefficient)
        utilization = complexity.n_qubits / caps.max_qubits
        if utilization < 0.3:
            score *= 0.8  # Small penalty for underutilization

        return score

    def _calculate_speed_score(
        self,
        backend_name: str,
        complexity: CircuitComplexity
    ) -> float:
        """Calculate speed score (0-1, higher is faster)."""
        # Get historical average execution time
        avg_time = self.performance_tracker.get_avg_execution_time(backend_name)

        # Estimate time for this circuit
        estimated_time = max(complexity.estimated_runtime, avg_time)

        # Convert to score (inverse relationship)
        # Fast backends (< 1s) get high scores, slow (> 10s) get low scores
        if estimated_time < 0.1:
            return 1.0
        elif estimated_time < 1.0:
            return 0.9
        elif estimated_time < 5.0:
            return 0.7
        elif estimated_time < 10.0:
            return 0.5
        else:
            return 0.3

    def _calculate_cost_score(
        self,
        backend_name: str,
        shots: int = 1000
    ) -> float:
        """Calculate cost score (0-1, higher is cheaper)."""
        cost = self.cost_per_shot.get(backend_name, 0.0)
        actual_cost = cost * (shots / 1000.0)

        # Convert to score (inverse relationship)
        if actual_cost == 0:
            return 1.0  # Free backends get perfect score
        elif actual_cost < 0.01:
            return 0.9
        elif actual_cost < 0.1:
            return 0.7
        elif actual_cost < 1.0:
            return 0.5
        else:
            return 0.3

    def _calculate_precision_score(
        self,
        backend_name: str
    ) -> float:
        """Calculate precision score (0-1)."""
        caps = self.capabilities[backend_name]

        # Simulators generally have high precision
        if caps.backend_type.value in ['simulator', 'noisy_simulator']:
            return 0.95 if caps.supports_state_vector else 0.85
        elif caps.backend_type.value == 'qpu':
            # Real hardware has lower precision due to noise
            return 0.6
        else:
            return 0.7

    def _calculate_availability_score(
        self,
        backend_name: str
    ) -> float:
        """Calculate availability score based on success rate (0-1)."""
        return self.performance_tracker.get_success_rate(backend_name)

    def score_backend(
        self,
        backend_name: str,
        circuit: UnifiedCircuit,
        shots: int = 1000
    ) -> BackendScore:
        """
        Score a backend for executing the given circuit.

        Args:
            backend_name: Name of backend to score
            circuit: Circuit to execute
            shots: Number of shots

        Returns:
            BackendScore with detailed scoring
        """
        complexity = CircuitComplexity.analyze(circuit)

        # Calculate individual scores
        capability_score = self._calculate_capability_score(backend_name, complexity)

        if capability_score == 0:
            # Backend cannot handle circuit
            return BackendScore(
                backend_name=backend_name,
                total_score=0.0,
                speed_score=0.0,
                cost_score=0.0,
                precision_score=0.0,
                capability_score=0.0,
                availability_score=0.0,
                metadata={'reason': 'insufficient_capability'}
            )

        speed_score = self._calculate_speed_score(backend_name, complexity)
        cost_score = self._calculate_cost_score(backend_name, shots)
        precision_score = self._calculate_precision_score(backend_name)
        availability_score = self._calculate_availability_score(backend_name)

        # Combine scores based on strategy
        if self.strategy == RoutingStrategy.SPEED:
            total_score = speed_score * 0.7 + availability_score * 0.3
        elif self.strategy == RoutingStrategy.COST:
            total_score = cost_score * 0.7 + availability_score * 0.3
        elif self.strategy == RoutingStrategy.PRECISION:
            total_score = precision_score * 0.7 + availability_score * 0.3
        elif self.strategy == RoutingStrategy.BALANCED:
            total_score = (
                speed_score * self.performance_weight +
                cost_score * self.cost_weight +
                precision_score * self.precision_weight
            ) * availability_score  # Modulated by availability
        else:  # ADAPTIVE
            # Adapt based on circuit complexity
            if complexity.n_qubits > 20:
                # Large circuits: prioritize speed and capability
                total_score = speed_score * 0.6 + availability_score * 0.4
            else:
                # Smaller circuits: balance all factors
                total_score = (
                    speed_score * 0.33 +
                    cost_score * 0.33 +
                    precision_score * 0.34
                ) * availability_score

        return BackendScore(
            backend_name=backend_name,
            total_score=total_score,
            speed_score=speed_score,
            cost_score=cost_score,
            precision_score=precision_score,
            capability_score=capability_score,
            availability_score=availability_score,
            metadata={
                'complexity': complexity,
                'strategy': self.strategy.value
            }
        )

    def select_backend(
        self,
        circuit: UnifiedCircuit,
        shots: int = 1000,
        strategy: Optional[RoutingStrategy] = None
    ) -> Tuple[QuantumBackend, BackendScore]:
        """
        Select the best backend for executing a circuit.

        Args:
            circuit: Circuit to execute
            shots: Number of measurement shots
            strategy: Override default strategy

        Returns:
            Tuple of (selected_backend, score)

        Raises:
            ValueError: If no suitable backend found
        """
        if not self.backends:
            raise ValueError("No backends registered")

        # Temporarily override strategy if provided
        original_strategy = self.strategy
        if strategy:
            self.strategy = strategy

        try:
            # Score all backends
            scores = []
            for name in self.backends:
                score = self.score_backend(name, circuit, shots)
                scores.append(score)

            # Sort by total score (descending)
            scores.sort(key=lambda s: s.total_score, reverse=True)

            # Select best backend
            if not scores or scores[0].total_score == 0:
                raise ValueError("No suitable backend found for circuit")

            best_score = scores[0]
            best_backend = self.backends[best_score.backend_name]

            logger.info(f"Selected backend: {best_score.backend_name} "
                       f"(score={best_score.total_score:.3f})")
            logger.debug(f"All scores: {scores[:3]}")  # Log top 3

            return best_backend, best_score

        finally:
            # Restore original strategy
            self.strategy = original_strategy

    def execute_with_fallback(
        self,
        circuit: UnifiedCircuit,
        shots: int = 1000,
        max_retries: int = 3
    ):
        """
        Execute circuit with automatic fallback on failure.

        Args:
            circuit: Circuit to execute
            shots: Number of shots
            max_retries: Maximum retry attempts with fallback

        Returns:
            Execution result

        Raises:
            RuntimeError: If all backends fail
        """
        # Get ranked list of backends
        scores = []
        for name in self.backends:
            score = self.score_backend(name, circuit, shots)
            if score.total_score > 0:
                scores.append(score)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        if not scores:
            raise ValueError("No suitable backends available")

        # Try backends in order
        last_error = None
        for i, score in enumerate(scores[:max_retries]):
            backend = self.backends[score.backend_name]
            logger.info(f"Attempting execution with {score.backend_name} "
                       f"(attempt {i+1}/{max_retries})")

            start_time = time.time()
            try:
                result = backend.execute(circuit, shots=shots)
                execution_time = time.time() - start_time

                # Record successful execution
                self.performance_tracker.record_execution(
                    score.backend_name,
                    execution_time,
                    success=True,
                    cost=self.cost_per_shot.get(score.backend_name, 0.0) * shots / 1000
                )

                logger.info(f"Execution successful with {score.backend_name} "
                           f"in {execution_time:.3f}s")
                return result

            except Exception as e:
                execution_time = time.time() - start_time
                last_error = e

                # Record failed execution
                self.performance_tracker.record_execution(
                    score.backend_name,
                    execution_time,
                    success=False,
                    cost=0.0
                )

                logger.warning(f"Execution failed with {score.backend_name}: {e}")
                continue

        # All backends failed
        raise RuntimeError(f"All backends failed. Last error: {last_error}")

    def get_backend_statistics(self) -> Dict[str, Dict]:
        """Get performance statistics for all backends."""
        stats = {}
        for name in self.backends:
            stats[name] = {
                'avg_execution_time': self.performance_tracker.get_avg_execution_time(name),
                'success_rate': self.performance_tracker.get_success_rate(name),
                'avg_cost': self.performance_tracker.get_avg_cost(name),
                'capabilities': self.capabilities[name]
            }
        return stats


def create_smart_router(
    backends: Optional[Dict[str, QuantumBackend]] = None,
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
) -> SmartBackendRouter:
    """
    Factory function to create a smart backend router.

    Args:
        backends: Optional dict of {name: backend} to register
        strategy: Routing strategy

    Returns:
        Configured SmartBackendRouter

    Example:
        >>> router = create_smart_router({
        ...     'qsim': qsim_backend,
        ...     'lightning': lightning_backend
        ... })
    """
    router = SmartBackendRouter(strategy=strategy)

    if backends:
        for name, backend in backends.items():
            router.register_backend(name, backend)

    return router
