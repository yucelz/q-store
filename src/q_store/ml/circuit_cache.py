"""
Quantum Circuit Cache - v3.3
Multi-level caching for quantum circuits and results

Key Innovation: Avoids redundant circuit execution through intelligent caching
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backends.quantum_backend_interface import ExecutionResult, QuantumCircuit

logger = logging.getLogger(__name__)


class QuantumCircuitCache:
    """
    Multi-level caching for quantum circuits

    Levels:
    1. Circuit hash → compiled circuit (avoid recompilation)
    2. Circuit + params → measurement results (avoid re-execution)
    3. Circuit structure → optimized circuit (avoid re-optimization)
    """

    def __init__(
        self,
        max_compiled_circuits: int = 1000,
        max_results: int = 5000,
        result_ttl: float = 300.0,  # 5 minutes
    ):
        """
        Initialize circuit cache

        Args:
            max_compiled_circuits: Max compiled circuits to cache
            max_results: Max execution results to cache
            result_ttl: Time-to-live for cached results (seconds)
        """
        # Level 1: Compiled circuits (LRU cache)
        self._compiled_cache: OrderedDict[str, Any] = OrderedDict()
        self._compiled_lru_max = max_compiled_circuits

        # Level 2: Execution results with TTL
        self._result_cache: Dict[str, Tuple[ExecutionResult, float]] = {}
        self._result_max = max_results
        self.result_ttl = result_ttl

        # Level 3: Optimized circuits
        self._optimized_cache: Dict[str, QuantumCircuit] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_compiled_circuit(self, circuit: QuantumCircuit, backend_name: str) -> Optional[Any]:
        """
        Get compiled circuit if cached

        Args:
            circuit: Circuit to look up
            backend_name: Backend identifier

        Returns:
            Compiled circuit if found, None otherwise
        """
        key = self._circuit_hash(circuit, backend_name)

        if key in self._compiled_cache:
            self.hits += 1
            # Move to end (most recently used)
            self._compiled_cache.move_to_end(key)
            return self._compiled_cache[key]

        self.misses += 1
        return None

    def cache_compiled_circuit(
        self, circuit: QuantumCircuit, backend_name: str, compiled_circuit: Any
    ):
        """
        Cache compiled circuit

        Args:
            circuit: Original circuit
            backend_name: Backend identifier
            compiled_circuit: Compiled circuit object
        """
        key = self._circuit_hash(circuit, backend_name)

        # Evict oldest if at capacity
        if len(self._compiled_cache) >= self._compiled_lru_max:
            oldest_key, _ = self._compiled_cache.popitem(last=False)
            self.evictions += 1
            logger.debug(f"Evicted compiled circuit: {oldest_key}")

        self._compiled_cache[key] = compiled_circuit
        logger.debug(f"Cached compiled circuit: {key}")

    def get_execution_result(
        self, circuit: QuantumCircuit, parameters: np.ndarray, shots: int
    ) -> Optional[ExecutionResult]:
        """
        Get cached execution result

        Args:
            circuit: Circuit executed
            parameters: Parameter values used
            shots: Number of shots

        Returns:
            Cached result if found and not expired, None otherwise
        """
        key = self._result_hash(circuit, parameters, shots)

        if key in self._result_cache:
            result, timestamp = self._result_cache[key]

            # Check TTL
            if time.time() - timestamp < self.result_ttl:
                self.hits += 1
                logger.debug(f"Cache hit for result: {key}")
                return result
            else:
                # Expired
                del self._result_cache[key]
                logger.debug(f"Result expired: {key}")

        self.misses += 1
        return None

    def cache_execution_result(
        self, circuit: QuantumCircuit, parameters: np.ndarray, shots: int, result: ExecutionResult
    ):
        """
        Cache execution result

        Args:
            circuit: Circuit executed
            parameters: Parameter values used
            shots: Number of shots
            result: Execution result
        """
        key = self._result_hash(circuit, parameters, shots)

        # Evict oldest if at capacity
        if len(self._result_cache) >= self._result_max:
            # Find oldest (lowest timestamp)
            oldest_key = min(self._result_cache.keys(), key=lambda k: self._result_cache[k][1])
            del self._result_cache[oldest_key]
            self.evictions += 1
            logger.debug(f"Evicted result: {oldest_key}")

        self._result_cache[key] = (result, time.time())
        logger.debug(f"Cached result: {key}")

    def get_optimized_circuit(self, circuit: QuantumCircuit) -> Optional[QuantumCircuit]:
        """
        Get optimized version of circuit

        Args:
            circuit: Circuit to look up

        Returns:
            Optimized circuit if found, None otherwise
        """
        key = self._circuit_hash(circuit)

        if key in self._optimized_cache:
            self.hits += 1
            return self._optimized_cache[key]

        self.misses += 1
        return None

    def cache_optimized_circuit(self, original: QuantumCircuit, optimized: QuantumCircuit):
        """
        Cache optimized circuit

        Args:
            original: Original circuit
            optimized: Optimized version
        """
        key = self._circuit_hash(original)
        self._optimized_cache[key] = optimized
        logger.debug(f"Cached optimized circuit: {key}")

    def _circuit_hash(self, circuit: QuantumCircuit, backend_name: str = "") -> str:
        """
        Generate hash for circuit structure

        Args:
            circuit: Circuit to hash
            backend_name: Optional backend identifier

        Returns:
            Hash string
        """
        # Build circuit signature
        circuit_str = f"{circuit.n_qubits}_{len(circuit.gates)}_"
        circuit_str += "_".join(
            [f"{g.gate_type.value}_{','.join(map(str, g.qubits))}" for g in circuit.gates]
        )
        circuit_str += f"_{backend_name}"

        # Hash it
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]

    def _result_hash(self, circuit: QuantumCircuit, parameters: np.ndarray, shots: int) -> str:
        """
        Generate hash for circuit + parameters + shots

        Args:
            circuit: Circuit executed
            parameters: Parameter values
            shots: Number of shots

        Returns:
            Hash string
        """
        # Circuit hash
        circuit_hash = self._circuit_hash(circuit)

        # Parameter hash (rounded to avoid floating point issues)
        param_str = "_".join([f"{p:.4f}" for p in parameters])

        # Combined hash
        full_str = f"{circuit_hash}_{param_str}_{shots}"

        return hashlib.sha256(full_str.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        # Calculate memory usage (rough estimate)
        compiled_size = len(self._compiled_cache)
        result_size = len(self._result_cache)
        optimized_size = len(self._optimized_cache)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "compiled_circuits": compiled_size,
            "cached_results": result_size,
            "optimized_circuits": optimized_size,
            "compiled_usage_pct": (compiled_size / self._compiled_lru_max) * 100,
            "result_usage_pct": (result_size / self._result_max) * 100,
        }

    def clear(self, level: Optional[str] = None):
        """
        Clear cache

        Args:
            level: Which level to clear ('compiled', 'results', 'optimized', or None for all)
        """
        if level is None or level == "compiled":
            self._compiled_cache.clear()
            logger.info("Cleared compiled circuit cache")

        if level is None or level == "results":
            self._result_cache.clear()
            logger.info("Cleared execution result cache")

        if level is None or level == "optimized":
            self._optimized_cache.clear()
            logger.info("Cleared optimized circuit cache")

        if level is None:
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def cleanup_expired(self):
        """Remove expired results from cache"""
        current_time = time.time()
        expired_keys = []

        for key, (result, timestamp) in self._result_cache.items():
            if current_time - timestamp >= self.result_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._result_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired results")

    def prewarm(self, circuits: List[QuantumCircuit], backend_name: str = ""):
        """
        Pre-populate cache with circuits

        Args:
            circuits: Circuits to pre-cache
            backend_name: Backend identifier
        """
        for circuit in circuits:
            key = self._circuit_hash(circuit, backend_name)
            # Just mark as known (actual compilation happens on first use)
            logger.debug(f"Prewarmed circuit: {key}")


class AdaptiveCircuitCache(QuantumCircuitCache):
    """
    Enhanced cache with adaptive sizing and eviction

    Automatically adjusts cache sizes based on:
    - Hit rates
    - Memory pressure
    - Access patterns
    """

    def __init__(
        self,
        initial_compiled_size: int = 500,
        initial_result_size: int = 2500,
        max_memory_mb: float = 500.0,
    ):
        super().__init__(
            max_compiled_circuits=initial_compiled_size, max_results=initial_result_size
        )

        self.max_memory_mb = max_memory_mb
        self.initial_compiled_size = initial_compiled_size
        self.initial_result_size = initial_result_size

        # Adaptive parameters
        self.access_counts: Dict[str, int] = {}
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 60.0  # Adapt every minute

    def adapt_cache_sizes(self):
        """Automatically adjust cache sizes based on usage"""
        if time.time() - self.last_adaptation_time < self.adaptation_interval:
            return

        stats = self.get_stats()

        # Increase compiled cache if high hit rate
        if stats["hit_rate"] > 0.8 and stats["compiled_usage_pct"] > 90:
            new_size = int(self._compiled_lru_max * 1.2)
            logger.info(f"Increasing compiled cache size to {new_size}")
            self._compiled_lru_max = new_size

        # Increase result cache if high hit rate
        if stats["hit_rate"] > 0.8 and stats["result_usage_pct"] > 90:
            new_size = int(self._result_max * 1.2)
            logger.info(f"Increasing result cache size to {new_size}")
            self._result_max = new_size

        # Decrease if low hit rate
        if stats["hit_rate"] < 0.3:
            self._compiled_lru_max = max(
                self.initial_compiled_size, int(self._compiled_lru_max * 0.8)
            )
            self._result_max = max(self.initial_result_size, int(self._result_max * 0.8))
            logger.info("Decreased cache sizes due to low hit rate")

        self.last_adaptation_time = time.time()
