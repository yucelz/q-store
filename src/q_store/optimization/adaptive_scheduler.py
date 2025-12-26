"""
Adaptive Batch Scheduler

Dynamically adjusts batch size based on:
- Queue depth (how many circuits pending?)
- Circuit complexity (gate count, qubit count)
- Historical latency (past execution times)
- Backend characteristics (simulator vs hardware)

Goal: Maximize throughput while minimizing latency.

Strategy:
- Small batches when queue is empty (low latency)
- Large batches when queue is full (high throughput)
- Adjust based on circuit complexity
- Learn from historical performance

Example:
    >>> scheduler = AdaptiveBatchScheduler(
    ...     min_batch_size=1,
    ...     max_batch_size=100,
    ...     target_latency_ms=100,
    ... )
    >>>
    >>> batch_size = scheduler.get_batch_size(
    ...     queue_depth=50,
    ...     circuit_complexity=100,
    ... )
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class AdaptiveBatchScheduler:
    """
    Adaptive batch size scheduler.

    Learns optimal batch size from execution history.

    Parameters
    ----------
    min_batch_size : int, default=1
        Minimum batch size
    max_batch_size : int, default=100
        Maximum batch size
    target_latency_ms : float, default=100.0
        Target latency per batch (milliseconds)
    learning_rate : float, default=0.1
        Learning rate for adaptation
    history_size : int, default=100
        Size of execution history

    Examples
    --------
    >>> scheduler = AdaptiveBatchScheduler()
    >>> batch_size = scheduler.get_batch_size(queue_depth=20)
    >>> scheduler.record_execution(batch_size=batch_size, latency_ms=50.0)
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        target_latency_ms: float = 100.0,
        learning_rate: float = 0.1,
        history_size: int = 100,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.learning_rate = learning_rate
        self.history_size = history_size

        # Current batch size
        self.current_batch_size = min_batch_size

        # Execution history
        self.history = deque(maxlen=history_size)

        # Statistics
        self.total_batches = 0
        self.total_circuits = 0
        self.total_time_ms = 0

    def get_batch_size(
        self,
        queue_depth: int,
        circuit_complexity: Optional[float] = None,
    ) -> int:
        """
        Get optimal batch size.

        Parameters
        ----------
        queue_depth : int
            Number of circuits in queue
        circuit_complexity : float, optional
            Circuit complexity score (gate count / qubits)

        Returns
        -------
        batch_size : int
            Recommended batch size
        """
        # Base batch size from queue depth
        if queue_depth == 0:
            # Empty queue: use minimum (low latency)
            base_size = self.min_batch_size
        elif queue_depth < 10:
            # Low queue: small batches
            base_size = min(queue_depth, self.min_batch_size * 2)
        elif queue_depth < 50:
            # Medium queue: medium batches
            base_size = min(queue_depth, self.max_batch_size // 2)
        else:
            # High queue: large batches (maximize throughput)
            base_size = min(queue_depth, self.max_batch_size)

        # Adjust for circuit complexity
        if circuit_complexity is not None:
            # Complex circuits → smaller batches
            complexity_factor = np.exp(-circuit_complexity / 100)
            base_size = int(base_size * complexity_factor)

        # Adjust based on historical performance
        if len(self.history) > 10:
            avg_latency = np.mean([h['latency_ms'] for h in self.history])

            if avg_latency > self.target_latency_ms * 1.5:
                # Too slow: reduce batch size
                base_size = int(base_size * 0.8)
            elif avg_latency < self.target_latency_ms * 0.5:
                # Too fast: increase batch size
                base_size = int(base_size * 1.2)

        # Clamp to bounds
        batch_size = max(self.min_batch_size, min(self.max_batch_size, base_size))

        # Smooth adaptation
        self.current_batch_size = int(
            (1 - self.learning_rate) * self.current_batch_size +
            self.learning_rate * batch_size
        )

        return max(self.min_batch_size, self.current_batch_size)

    def record_execution(
        self,
        batch_size: int,
        latency_ms: float,
        circuit_complexity: Optional[float] = None,
    ):
        """
        Record execution metrics.

        Parameters
        ----------
        batch_size : int
            Batch size used
        latency_ms : float
            Execution latency (milliseconds)
        circuit_complexity : float, optional
            Circuit complexity score
        """
        self.history.append({
            'batch_size': batch_size,
            'latency_ms': latency_ms,
            'circuit_complexity': circuit_complexity,
            'timestamp': time.time(),
            'throughput': batch_size / (latency_ms / 1000),  # circuits/sec
        })

        self.total_batches += 1
        self.total_circuits += batch_size
        self.total_time_ms += latency_ms

    def stats(self) -> Dict:
        """
        Get scheduler statistics.

        Returns
        -------
        stats : dict
            Scheduler statistics
        """
        if len(self.history) == 0:
            return {
                'total_batches': 0,
                'total_circuits': 0,
                'avg_throughput': 0,
                'current_batch_size': self.current_batch_size,
            }

        recent_history = list(self.history)[-20:]  # Last 20 batches

        return {
            'total_batches': self.total_batches,
            'total_circuits': self.total_circuits,
            'total_time_sec': self.total_time_ms / 1000,
            'avg_throughput': self.total_circuits / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0,
            'current_batch_size': self.current_batch_size,
            'recent_avg_latency_ms': np.mean([h['latency_ms'] for h in recent_history]),
            'recent_avg_throughput': np.mean([h['throughput'] for h in recent_history]),
            'history_size': len(self.history),
        }

    def reset(self):
        """Reset scheduler state."""
        self.current_batch_size = self.min_batch_size
        self.history.clear()
        self.total_batches = 0
        self.total_circuits = 0
        self.total_time_ms = 0


class CircuitComplexityEstimator:
    """
    Estimate circuit complexity.

    Complexity = f(n_qubits, n_gates, gate_types)

    Examples
    --------
    >>> estimator = CircuitComplexityEstimator()
    >>> complexity = estimator.estimate(circuit)
    """

    def __init__(self):
        # Gate costs (relative)
        self.gate_costs = {
            'X': 1,
            'Y': 1,
            'Z': 1,
            'H': 1,
            'RX': 2,
            'RY': 2,
            'RZ': 2,
            'CNOT': 10,
            'CZ': 10,
            'SWAP': 15,
            'TOFFOLI': 50,
        }

    def estimate(self, circuit) -> float:
        """
        Estimate circuit complexity.

        Parameters
        ----------
        circuit : Circuit
            Quantum circuit

        Returns
        -------
        complexity : float
            Complexity score (higher = more complex)
        """
        try:
            n_qubits = len(circuit.all_qubits())

            # Count gates
            gate_count = 0
            weighted_gate_count = 0

            for moment in circuit:
                for op in moment:
                    gate_count += 1
                    gate_name = str(op.gate).split('(')[0]
                    cost = self.gate_costs.get(gate_name, 5)
                    weighted_gate_count += cost

            # Complexity score
            complexity = weighted_gate_count * np.log2(n_qubits + 1)

            return complexity

        except Exception:
            # Fallback: assume moderate complexity
            return 50.0
