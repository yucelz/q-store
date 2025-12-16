"""
Parallel SPSA Gradient Estimator - v3.3.1 CORRECTED
True batch gradient computation with parallel circuit execution

KEY FIX: Computes gradient of BATCH LOSS, not per-sample average
"""

import asyncio
import time
import numpy as np
import logging
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass

from ..backends.quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    ExecutionResult
)
from .circuit_batch_manager import CircuitBatchManager

logger = logging.getLogger(__name__)


@dataclass
class GradientResult:
    """Result from gradient computation"""
    gradients: np.ndarray
    function_value: float
    n_circuit_executions: int
    computation_time_ms: float
    method: str = 'spsa_parallel_batch'
    metadata: Dict[str, Any] = None


class ParallelSPSAEstimator:
    """
    SPSA with Parallel Batch Gradient Computation

    Critical Innovation:
    - Computes gradient of BATCH LOSS (not per-sample average)
    - Executes all circuits in parallel
    - Only 2 × batch_size circuit evaluations

    Performance:
    - Batch size 10 → 20 circuits total
    - With parallelization: ~10s per batch (vs 50s sequential)
    """

    def __init__(
        self,
        backend: QuantumBackend,
        batch_manager: Optional[CircuitBatchManager] = None,
        c_decay: float = 0.101,
        a_decay: float = 0.602,
        c_initial: float = 0.1,
        a_initial: float = 0.01
    ):
        """
        Initialize parallel SPSA estimator

        Args:
            backend: Quantum backend
            batch_manager: Circuit batch manager for parallel execution
            c_decay, a_decay, c_initial, a_initial: SPSA parameters
        """
        self.backend = backend
        self.batch_manager = batch_manager or CircuitBatchManager(backend)

        # SPSA parameters
        self.c_decay = c_decay
        self.a_decay = a_decay
        self.c_initial = c_initial
        self.a_initial = a_initial

        self.iteration = 0

        # Statistics
        self.total_circuits = 0
        self.total_time_ms = 0.0

    def get_gain_parameters(self, iteration: int) -> tuple:
        """Compute SPSA gain parameters"""
        k = iteration + 1
        c_k = self.c_initial / (k ** self.c_decay)
        a_k = self.a_initial / (k ** self.a_decay)
        return c_k, a_k

    async def estimate_batch_gradient(
        self,
        model,  # QuantumModel or QuantumLayer
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        loss_function: Callable,
        shots: int = 1000
    ) -> GradientResult:
        """
        Estimate gradient over ENTIRE BATCH

        This is the CORRECTED implementation that computes:
        ∇L_batch(θ) where L_batch(θ) = (1/B) Σ loss(f(x_i; θ), y_i)

        Algorithm:
        1. Generate random perturbation δ
        2. Build circuits for ALL samples at θ + c·δ
        3. Build circuits for ALL samples at θ - c·δ
        4. Execute ALL circuits in parallel
        5. Compute batch loss at each perturbation
        6. Estimate gradient: ∇L ≈ [L(θ+cδ) - L(θ-cδ)] / (2c) · δ

        Args:
            model: Quantum model/layer to train
            batch_x: Input batch [batch_size, features]
            batch_y: Target batch [batch_size, outputs]
            loss_function: Loss function(predictions, targets) → scalar
            shots: Measurement shots per circuit

        Returns:
            GradientResult with batch gradient estimate
        """
        start_time = time.time()
        batch_size = len(batch_x)

        # Store full model reference for output projection
        self._current_full_model = model

        # Get quantum layer for circuit building
        quantum_layer = model.quantum_layer if hasattr(model, 'quantum_layer') else model

        # Get current parameters
        params = quantum_layer.parameters.copy()

        # SPSA perturbation
        c_k, a_k = self.get_gain_parameters(self.iteration)
        delta = np.random.choice([-1, 1], size=len(params))

        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        # === Build ALL circuits (but don't execute yet) ===
        circuits_plus = []
        circuits_minus = []

        logger.debug(f"Building {batch_size * 2} circuits for batch gradient...")

        for x in batch_x:
            # Circuit at params_plus
            quantum_layer.parameters = params_plus
            circuit_plus = quantum_layer.build_circuit(x)
            circuits_plus.append(circuit_plus)

            # Circuit at params_minus
            quantum_layer.parameters = params_minus
            circuit_minus = quantum_layer.build_circuit(x)
            circuits_minus.append(circuit_minus)

        # Restore original parameters
        quantum_layer.parameters = params

        # === KEY: Parallel execution ===
        all_circuits = circuits_plus + circuits_minus

        logger.info(
            f"SPSA batch gradient: Submitting {len(all_circuits)} circuits "
            f"(batch_size={batch_size}, perturbations=2)"
        )

        # Execute all circuits in parallel (single API call to backend)
        results = await self.batch_manager.execute_batch(
            circuits=all_circuits,
            shots=shots,
            wait_for_results=True
        )

        # Split results
        results_plus = results[:batch_size]
        results_minus = results[batch_size:]

        # === Compute batch loss at each perturbation ===

        # Loss at params_plus
        loss_plus = 0.0
        for result, y in zip(results_plus, batch_y):
            # Process measurement result to get model output
            output = self._process_result(model, result)
            loss_plus += loss_function(output, y)
        loss_plus /= batch_size

        # Loss at params_minus
        loss_minus = 0.0
        for result, y in zip(results_minus, batch_y):
            output = self._process_result(model, result)
            loss_minus += loss_function(output, y)
        loss_minus /= batch_size

        # === SPSA gradient estimate ===
        gradient = ((loss_plus - loss_minus) / (2 * c_k)) * delta

        # Average loss for reporting
        avg_loss = (loss_plus + loss_minus) / 2.0

        # Update iteration
        self.iteration += 1
        self.total_circuits += len(all_circuits)

        computation_time = (time.time() - start_time) * 1000
        self.total_time_ms += computation_time

        logger.info(
            f"SPSA batch gradient: loss={avg_loss:.4f}, "
            f"||∇||={np.linalg.norm(gradient):.4f}, "
            f"time={computation_time:.2f}ms, "
            f"circuits={len(all_circuits)}, "
            f"throughput={len(all_circuits)/(computation_time/1000):.1f} circuits/s"
        )

        return GradientResult(
            gradients=gradient,
            function_value=avg_loss,
            n_circuit_executions=len(all_circuits),
            computation_time_ms=computation_time,
            method='spsa_parallel_batch',
            metadata={
                'iteration': self.iteration,
                'c_k': c_k,
                'a_k': a_k,
                'batch_size': batch_size,
                'loss_plus': loss_plus,
                'loss_minus': loss_minus,
                'perturbation_norm': np.linalg.norm(delta)
            }
        )

    def _process_result(self, model, result: ExecutionResult) -> np.ndarray:
        """Process measurement result to get model output"""
        # Use stored full model if available (for output projection)
        full_model = getattr(self, '_current_full_model', model)

        # Get quantum layer for processing
        quantum_layer = full_model.quantum_layer if hasattr(full_model, 'quantum_layer') else full_model

        # Process measurements
        if hasattr(quantum_layer, '_process_measurements'):
            output = quantum_layer._process_measurements(result)
        elif hasattr(result, 'counts'):
            # Fallback: extract probabilities from measurements
            counts = result.counts
            total = sum(counts.values())
            n_qubits = len(next(iter(counts.keys())))
            output = np.zeros(2 ** n_qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                output[idx] = count / total
        else:
            raise ValueError("Cannot process result without counts or _process_measurements method")

        # Project to output dimension if full model specifies it
        if hasattr(full_model, 'output_dim') and len(output) != full_model.output_dim:
            output = output[:full_model.output_dim]

        return output


class SubsampledSPSAEstimator(ParallelSPSAEstimator):
    """
    SPSA with Gradient Subsampling

    Further optimization: Compute gradient on SUBSET of batch

    Theory: E[∇L(θ; S)] = ∇L(θ) for random subset S

    Performance:
    - Batch size 10, subsample 5 → 10 circuits (vs 20)
    - Batch size 10, subsample 2 → 4 circuits (vs 20)
    - Faster convergence iterations, may need more epochs
    """

    def __init__(
        self,
        backend: QuantumBackend,
        batch_manager: Optional[CircuitBatchManager] = None,
        subsample_size: int = 5,
        **kwargs
    ):
        """
        Initialize subsampled SPSA

        Args:
            backend: Quantum backend
            batch_manager: Circuit batch manager
            subsample_size: Number of samples to use for gradient
            **kwargs: SPSA parameters
        """
        super().__init__(backend, batch_manager, **kwargs)
        self.subsample_size = subsample_size

    async def estimate_batch_gradient(
        self,
        model,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        loss_function: Callable,
        shots: int = 1000
    ) -> GradientResult:
        """
        Estimate batch gradient using random subsample

        Key Optimization: Only use k << batch_size samples

        Tradeoff:
        - Faster: 2k circuits vs 2·batch_size
        - Higher variance: May need more training iterations
        - Unbiased: E[gradient] = true gradient
        """
        start_time = time.time()
        batch_size = len(batch_x)

        # === Randomly subsample ===
        actual_subsample = min(self.subsample_size, batch_size)
        indices = np.random.choice(
            batch_size,
            size=actual_subsample,
            replace=False
        )

        subset_x = batch_x[indices]
        subset_y = batch_y[indices]

        logger.debug(
            f"Subsampled {actual_subsample}/{batch_size} samples "
            f"for gradient estimation"
        )

        # Get parameters and perturbation
        # Store full model reference for output projection
        self._current_full_model = model

        # Get quantum layer for circuit building
        quantum_layer = model.quantum_layer if hasattr(model, 'quantum_layer') else model

        params = quantum_layer.parameters.copy()
        c_k, a_k = self.get_gain_parameters(self.iteration)
        delta = np.random.choice([-1, 1], size=len(params))

        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        # Build circuits for SUBSET only
        circuits_plus = []
        circuits_minus = []

        for x in subset_x:
            quantum_layer.parameters = params_plus
            circuits_plus.append(quantum_layer.build_circuit(x))

            quantum_layer.parameters = params_minus
            circuits_minus.append(quantum_layer.build_circuit(x))

        quantum_layer.parameters = params

        # Execute in parallel
        all_circuits = circuits_plus + circuits_minus

        logger.info(
            f"SPSA subsampled gradient: {len(all_circuits)} circuits "
            f"(subsample={actual_subsample}/{batch_size})"
        )

        results = await self.batch_manager.execute_batch(
            all_circuits, shots=shots
        )

        results_plus = results[:actual_subsample]
        results_minus = results[actual_subsample:]

        # Compute losses on subset
        loss_plus = sum(
            loss_function(self._process_result(model, r), y)
            for r, y in zip(results_plus, subset_y)
        ) / actual_subsample

        loss_minus = sum(
            loss_function(self._process_result(model, r), y)
            for r, y in zip(results_minus, subset_y)
        ) / actual_subsample

        # SPSA gradient
        gradient = ((loss_plus - loss_minus) / (2 * c_k)) * delta
        avg_loss = (loss_plus + loss_minus) / 2.0

        self.iteration += 1
        self.total_circuits += len(all_circuits)

        computation_time = (time.time() - start_time) * 1000
        self.total_time_ms += computation_time

        logger.info(
            f"SPSA subsampled: loss={avg_loss:.4f}, "
            f"||∇||={np.linalg.norm(gradient):.4f}, "
            f"time={computation_time:.2f}ms, "
            f"circuits={len(all_circuits)}"
        )

        return GradientResult(
            gradients=gradient,
            function_value=avg_loss,
            n_circuit_executions=len(all_circuits),
            computation_time_ms=computation_time,
            method='spsa_subsampled',
            metadata={
                'iteration': self.iteration,
                'subsample_size': actual_subsample,
                'full_batch_size': batch_size,
                'subsample_ratio': actual_subsample / batch_size,
                'c_k': c_k,
                'a_k': a_k
            }
        )

    def adjust_subsample_size(
        self,
        gradient_variance: float,
        target_variance: float = 0.1
    ):
        """
        Dynamically adjust subsample size based on gradient variance

        If variance too high: increase subsample size
        If variance low: can decrease subsample size
        """
        if gradient_variance > target_variance:
            # Increase subsample
            new_size = min(self.subsample_size + 1, 10)
            if new_size != self.subsample_size:
                logger.info(
                    f"Increasing subsample size: "
                    f"{self.subsample_size} → {new_size} "
                    f"(variance={gradient_variance:.4f})"
                )
                self.subsample_size = new_size

        elif gradient_variance < target_variance / 2:
            # Decrease subsample
            new_size = max(self.subsample_size - 1, 2)
            if new_size != self.subsample_size:
                logger.info(
                    f"Decreasing subsample size: "
                    f"{self.subsample_size} → {new_size} "
                    f"(variance={gradient_variance:.4f})"
                )
                self.subsample_size = new_size


# For backward compatibility
SPSABatchGradientEstimator = ParallelSPSAEstimator
