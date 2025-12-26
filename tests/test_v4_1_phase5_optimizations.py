"""
Comprehensive Tests for Q-Store v4.1 Phase 5: Optimizations

Tests all Phase 5 components:
- AdaptiveBatchScheduler with learning
- CircuitComplexityEstimator
- MultiLevelCache (L1/L2/L3)
- LRUCache implementation
- IonQNativeCompiler with gate decomposition
- Integrated optimization pipeline
"""

import pytest
import numpy as np
import time
import cirq
from unittest.mock import Mock, patch

from q_store.optimization import (
    AdaptiveBatchScheduler,
    CircuitComplexityEstimator,
    MultiLevelCache,
    LRUCache,
    IonQNativeCompiler,
)


# ============================================================================
# Test AdaptiveBatchScheduler
# ============================================================================

class TestAdaptiveBatchScheduler:
    """Test AdaptiveBatchScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100,
            target_latency_ms=100.0,
            learning_rate=0.1
        )

        assert scheduler.min_batch_size == 1
        assert scheduler.max_batch_size == 100
        assert scheduler.target_latency_ms == 100.0
        assert scheduler.learning_rate == 0.1

    def test_empty_queue_small_batch(self):
        """Test empty queue returns small batch size."""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100
        )

        # Empty queue should suggest small batch (low latency)
        batch_size = scheduler.get_batch_size(queue_depth=0)

        assert batch_size < 50  # Should be in lower half

    def test_full_queue_large_batch(self):
        """Test full queue returns large batch size."""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100
        )

        # Full queue should suggest large batch (high throughput)
        batch_size = scheduler.get_batch_size(queue_depth=200)

        assert batch_size > 50  # Should be in upper half

    def test_complexity_aware_scheduling(self):
        """Test scheduling adapts to circuit complexity."""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100
        )

        # Low complexity should allow larger batches
        batch_low = scheduler.get_batch_size(
            queue_depth=50,
            circuit_complexity=10
        )

        # High complexity should suggest smaller batches
        batch_high = scheduler.get_batch_size(
            queue_depth=50,
            circuit_complexity=1000
        )

        assert batch_low > batch_high

    def test_learning_from_history(self):
        """Test scheduler learns from execution history."""
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100,
            target_latency_ms=100.0
        )

        # Record execution history
        for _ in range(20):
            batch_size = 50
            # Simulate: batch_size=50 consistently achieves target latency
            latency_ms = 100.0 + np.random.randn() * 5
            scheduler.record_execution(batch_size, latency_ms)

        # Scheduler should learn that batch_size=50 is optimal
        optimal_batch = scheduler.get_batch_size(queue_depth=50)

        # Should be close to 50
        assert 40 <= optimal_batch <= 60

    def test_record_execution(self):
        """Test recording execution metrics."""
        scheduler = AdaptiveBatchScheduler()

        # Record multiple executions
        for i in range(10):
            scheduler.record_execution(
                batch_size=i + 1,
                latency_ms=50.0 + i * 10
            )

        stats = scheduler.stats()

        assert stats['total_batches'] == 10
        assert stats['total_circuits'] == sum(range(1, 11))

    def test_statistics(self):
        """Test statistics tracking."""
        scheduler = AdaptiveBatchScheduler()

        # Record some executions
        scheduler.record_execution(batch_size=10, latency_ms=100.0)
        scheduler.record_execution(batch_size=20, latency_ms=150.0)
        scheduler.record_execution(batch_size=15, latency_ms=120.0)

        stats = scheduler.stats()

        assert 'total_batches' in stats
        assert 'total_circuits' in stats
        assert 'avg_batch_size' in stats
        assert 'avg_latency' in stats
        assert 'avg_throughput' in stats

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        scheduler = AdaptiveBatchScheduler()

        # Record: 100 circuits in 1000ms = 100 circuits/sec
        scheduler.record_execution(batch_size=100, latency_ms=1000.0)

        stats = scheduler.stats()

        # Throughput should be ~100 circuits/sec
        assert 90 <= stats['avg_throughput'] <= 110


# ============================================================================
# Test CircuitComplexityEstimator
# ============================================================================

class TestCircuitComplexityEstimator:
    """Test CircuitComplexityEstimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = CircuitComplexityEstimator()

        assert estimator is not None

    def test_simple_circuit_complexity(self):
        """Test complexity estimation for simple circuit."""
        estimator = CircuitComplexityEstimator()

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])

        complexity = estimator.estimate(circuit)

        # Should be positive and reasonable
        assert complexity > 0
        assert complexity < 100

    def test_complex_circuit_higher_score(self):
        """Test complex circuits get higher scores."""
        estimator = CircuitComplexityEstimator()

        qubits = cirq.LineQubit.range(4)

        # Simple circuit
        simple_circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])

        # Complex circuit
        complex_circuit = cirq.Circuit([
            cirq.H.on_each(*qubits),
            *[cirq.CNOT(qubits[i], qubits[(i+1)%4]) for i in range(4)],
            *[cirq.rx(0.5).on(q) for q in qubits],
            *[cirq.ry(0.5).on(q) for q in qubits],
            *[cirq.rz(0.5).on(q) for q in qubits],
        ])

        simple_complexity = estimator.estimate(simple_circuit)
        complex_complexity = estimator.estimate(complex_circuit)

        assert complex_complexity > simple_complexity

    def test_qubit_count_impact(self):
        """Test more qubits increases complexity."""
        estimator = CircuitComplexityEstimator()

        # 2-qubit circuit
        qubits_2 = cirq.LineQubit.range(2)
        circuit_2 = cirq.Circuit([cirq.H.on_each(*qubits_2)])

        # 8-qubit circuit
        qubits_8 = cirq.LineQubit.range(8)
        circuit_8 = cirq.Circuit([cirq.H.on_each(*qubits_8)])

        complexity_2 = estimator.estimate(circuit_2)
        complexity_8 = estimator.estimate(circuit_8)

        assert complexity_8 > complexity_2

    def test_gate_type_weighting(self):
        """Test different gate types have different weights."""
        estimator = CircuitComplexityEstimator()

        qubits = cirq.LineQubit.range(2)

        # Single-qubit gates
        circuit_single = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.X(qubits[1])
        ])

        # Two-qubit gates
        circuit_two = cirq.Circuit([
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[0])
        ])

        complexity_single = estimator.estimate(circuit_single)
        complexity_two = estimator.estimate(circuit_two)

        # Two-qubit gates should be weighted higher
        assert complexity_two > complexity_single


# ============================================================================
# Test LRUCache
# ============================================================================

class TestLRUCache:
    """Test LRUCache implementation."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(max_size=10)

        assert cache.max_size == 10
        assert len(cache) == 0

    def test_put_and_get(self):
        """Test basic put/get operations."""
        cache = LRUCache(max_size=5)

        # Put items
        for i in range(3):
            cache.put(f"key_{i}", f"value_{i}")

        assert len(cache) == 3

        # Get items
        for i in range(3):
            value = cache.get(f"key_{i}")
            assert value == f"value_{i}"

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=3)

        # Fill cache
        cache.put("key_1", "value_1")
        cache.put("key_2", "value_2")
        cache.put("key_3", "value_3")

        # Access key_1 to make it recently used
        cache.get("key_1")

        # Add new item, should evict key_2 (least recently used)
        cache.put("key_4", "value_4")

        assert cache.get("key_1") is not None
        assert cache.get("key_2") is None  # Evicted
        assert cache.get("key_3") is not None
        assert cache.get("key_4") is not None

    def test_hit_and_miss_tracking(self):
        """Test hit/miss statistics."""
        cache = LRUCache(max_size=5)

        cache.put("key_1", "value_1")

        # Hit
        cache.get("key_1")

        # Miss
        cache.get("key_2")

        stats = cache.stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5

    def test_eviction_count(self):
        """Test eviction counting."""
        cache = LRUCache(max_size=3)

        # Fill and overflow cache
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")

        stats = cache.stats()

        # Should have evicted 7 items (10 - 3)
        assert stats['evictions'] == 7

    def test_utilization(self):
        """Test utilization calculation."""
        cache = LRUCache(max_size=10)

        # Fill half
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        stats = cache.stats()

        assert stats['utilization'] == 0.5


# ============================================================================
# Test MultiLevelCache
# ============================================================================

class TestMultiLevelCache:
    """Test MultiLevelCache system."""

    def test_initialization(self):
        """Test multi-level cache initialization."""
        cache = MultiLevelCache(
            l1_size=10,
            l2_size=100,
            l3_size=1000
        )

        assert cache.l1_size == 10
        assert cache.l2_size == 100
        assert cache.l3_size == 1000

    def test_l1_cache_params(self):
        """Test L1 cache for hot parameters."""
        cache = MultiLevelCache(l1_size=10, l2_size=100, l3_size=1000)

        params = np.random.randn(10)
        key = cache.hash_params(params)

        # Put in L1
        cache.put_l1(key, {'compiled': True})

        # Get from L1
        value = cache.get_l1(key)

        assert value is not None
        assert value['compiled'] == True

    def test_l2_cache_circuits(self):
        """Test L2 cache for compiled circuits."""
        cache = MultiLevelCache(l1_size=10, l2_size=100, l3_size=1000)

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])
        key = cache.hash_circuit(circuit)

        # Put in L2
        cache.put_l2(key, {'compiled_circuit': circuit})

        # Get from L2
        value = cache.get_l2(key)

        assert value is not None

    def test_l3_cache_results(self):
        """Test L3 cache for measurement results."""
        cache = MultiLevelCache(l1_size=10, l2_size=100, l3_size=1000)

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])
        key = cache.hash_result(circuit, shots=1000)

        # Put in L3
        result = {'measurements': np.random.randn(10)}
        cache.put_l3(key, result)

        # Get from L3
        cached_result = cache.get_l3(key)

        assert cached_result is not None

    def test_cache_hierarchy(self):
        """Test cache hierarchy operates correctly."""
        cache = MultiLevelCache(
            l1_size=5,
            l2_size=10,
            l3_size=20
        )

        # Fill all cache levels
        for i in range(30):
            cache.put_l1(f"l1_key_{i}", f"l1_value_{i}")
            cache.put_l2(f"l2_key_{i}", f"l2_value_{i}")
            cache.put_l3(f"l3_key_{i}", f"l3_value_{i}")

        # Check sizes
        stats = cache.stats()

        assert stats['l1']['size'] <= 5
        assert stats['l2']['size'] <= 10
        assert stats['l3']['size'] <= 20

    def test_overall_hit_rate(self):
        """Test overall hit rate calculation."""
        cache = MultiLevelCache(l1_size=5, l2_size=10, l3_size=20)

        # Add and access items
        for i in range(10):
            cache.put_l3(f"key_{i}", f"value_{i}")

        # Access some items (hits)
        for i in range(5):
            cache.get_l3(f"key_{i}")

        # Access non-existent items (misses)
        for i in range(10, 15):
            cache.get_l3(f"key_{i}")

        stats = cache.stats()

        # Should have some hit rate
        assert 'overall_hit_rate' in stats
        assert 0 <= stats['overall_hit_rate'] <= 1

    def test_hash_functions(self):
        """Test hash functions for different data types."""
        cache = MultiLevelCache()

        # Hash parameters
        params = np.random.randn(10)
        params_hash = cache.hash_params(params)
        assert isinstance(params_hash, str)
        assert len(params_hash) > 0

        # Hash circuit
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])
        circuit_hash = cache.hash_circuit(circuit)
        assert isinstance(circuit_hash, str)
        assert len(circuit_hash) > 0

        # Hash result
        result_hash = cache.hash_result(circuit, shots=1000)
        assert isinstance(result_hash, str)
        assert len(result_hash) > 0


# ============================================================================
# Test IonQNativeCompiler
# ============================================================================

class TestIonQNativeCompiler:
    """Test IonQNativeCompiler."""

    def test_initialization(self):
        """Test compiler initialization."""
        compiler = IonQNativeCompiler(optimization_level=2)

        assert compiler.optimization_level == 2

    def test_basic_gate_decomposition(self):
        """Test basic gate decomposition."""
        compiler = IonQNativeCompiler()

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.X(qubits[1])
        ])

        native_circuit = compiler.compile(circuit)

        # Should produce a valid circuit
        assert native_circuit is not None
        assert len(list(native_circuit.all_operations())) > 0

    def test_rotation_gate_decomposition(self):
        """Test rotation gate decomposition."""
        compiler = IonQNativeCompiler()

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.rx(0.5).on(qubits[0]),
            cirq.ry(0.7).on(qubits[1]),
            cirq.rz(0.3).on(qubits[0])
        ])

        native_circuit = compiler.compile(circuit)

        assert native_circuit is not None

    def test_cnot_compilation(self):
        """Test CNOT gate compilation."""
        compiler = IonQNativeCompiler()

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.CNOT(qubits[0], qubits[1])
        ])

        native_circuit = compiler.compile(circuit)

        # CNOT is native to IonQ, should be preserved or optimized
        assert native_circuit is not None

    def test_optimization_levels(self):
        """Test different optimization levels."""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.H(qubits[0]),  # Should cancel out
            cirq.X(qubits[1])
        ])

        # Level 0: No optimization
        compiler_0 = IonQNativeCompiler(optimization_level=0)
        circuit_0 = compiler_0.compile(circuit)

        # Level 2: With optimization
        compiler_2 = IonQNativeCompiler(optimization_level=2)
        circuit_2 = compiler_2.compile(circuit)

        gates_0 = len(list(circuit_0.all_operations()))
        gates_2 = len(list(circuit_2.all_operations()))

        # Optimization should reduce gate count
        assert gates_2 <= gates_0

    def test_gate_count_reduction(self):
        """Test compilation reduces gate count."""
        compiler = IonQNativeCompiler(optimization_level=2)

        qubits = cirq.LineQubit.range(4)
        circuit = cirq.Circuit([
            cirq.H.on_each(*qubits),
            *[cirq.CNOT(qubits[i], qubits[(i+1)%4]) for i in range(4)],
            cirq.X.on_each(*qubits),
        ])

        gates_before = len(list(circuit.all_operations()))

        native_circuit = compiler.compile(circuit)
        gates_after = len(list(native_circuit.all_operations()))

        print(f"Gates: {gates_before} -> {gates_after}")

        # May reduce or stay same depending on optimization
        assert gates_after <= gates_before * 1.5  # Allow some expansion

    def test_compilation_statistics(self):
        """Test compilation statistics tracking."""
        compiler = IonQNativeCompiler()

        qubits = cirq.LineQubit.range(2)

        # Compile multiple circuits
        for i in range(5):
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.rx(i * 0.1).on(qubits[1])
            ])
            compiler.compile(circuit)

        stats = compiler.stats()

        assert stats['circuits_compiled'] == 5
        assert 'avg_gates_before' in stats
        assert 'avg_gates_after' in stats
        assert 'estimated_speedup' in stats

    def test_speedup_estimation(self):
        """Test speedup estimation."""
        compiler = IonQNativeCompiler(optimization_level=2)

        qubits = cirq.LineQubit.range(4)
        circuit = cirq.Circuit([
            cirq.H.on_each(*qubits),
            *[cirq.CNOT(qubits[i], qubits[(i+1)%4]) for i in range(4)],
        ])

        compiler.compile(circuit)

        stats = compiler.stats()

        # Should estimate some speedup
        assert stats['estimated_speedup'] >= 1.0
        assert stats['estimated_speedup'] <= 2.0  # Typically 1.3x

    def test_identity_removal(self):
        """Test identity gate removal."""
        compiler = IonQNativeCompiler(optimization_level=2)

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.X(qubits[0]),
            cirq.X(qubits[0]),  # X^2 = I (identity)
            cirq.Y(qubits[1])
        ])

        native_circuit = compiler.compile(circuit)
        gates_after = len(list(native_circuit.all_operations()))

        # Should remove redundant X gates
        assert gates_after < 3


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 optimizations."""

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Initialize all components
        scheduler = AdaptiveBatchScheduler(
            min_batch_size=1,
            max_batch_size=100
        )

        cache = MultiLevelCache(
            l1_size=10,
            l2_size=100,
            l3_size=1000
        )

        compiler = IonQNativeCompiler(optimization_level=2)

        complexity_estimator = CircuitComplexityEstimator()

        # Simulate workload
        qubits = cirq.LineQubit.range(4)

        for i in range(20):
            # Create circuit
            circuit = cirq.Circuit([
                cirq.H.on_each(*qubits),
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.rx(i * 0.1).on(qubits[2])
            ])

            # Estimate complexity
            complexity = complexity_estimator.estimate(circuit)

            # Get adaptive batch size
            queue_depth = 20 - i
            batch_size = scheduler.get_batch_size(
                queue_depth=queue_depth,
                circuit_complexity=complexity
            )

            # Check cache (L3)
            circuit_key = cache.hash_result(circuit, shots=1000)
            cached_result = cache.get_l3(circuit_key)

            if cached_result is None:
                # Compile circuit
                native_circuit = compiler.compile(circuit)

                # Simulate execution
                result = {'measurements': np.random.randn(10)}

                # Cache result
                cache.put_l3(circuit_key, result)

            # Record execution
            scheduler.record_execution(
                batch_size=batch_size,
                latency_ms=50.0
            )

        # Check final statistics
        sched_stats = scheduler.stats()
        cache_stats = cache.stats()
        comp_stats = compiler.stats()

        print("\nOptimization Pipeline Statistics:")
        print(f"  Scheduler - Avg batch size: {sched_stats['avg_batch_size']:.1f}")
        print(f"  Scheduler - Avg throughput: {sched_stats['avg_throughput']:.1f} circuits/sec")
        print(f"  Cache - Overall hit rate: {cache_stats['overall_hit_rate']:.2%}")
        print(f"  Compiler - Circuits compiled: {comp_stats['circuits_compiled']}")
        print(f"  Compiler - Estimated speedup: {comp_stats['estimated_speedup']:.2f}x")

    def test_performance_improvement(self):
        """Test optimizations provide performance improvement."""
        cache = MultiLevelCache(l1_size=10, l2_size=100, l3_size=1000)

        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])

        # First execution (cache miss)
        key = cache.hash_result(circuit, shots=1000)

        start = time.time()
        result = cache.get_l3(key)
        if result is None:
            # Simulate execution
            time.sleep(0.01)  # 10ms
            result = {'measurements': np.random.randn(10)}
            cache.put_l3(key, result)
        time_miss = time.time() - start

        # Second execution (cache hit)
        start = time.time()
        result = cache.get_l3(key)
        time_hit = time.time() - start

        print(f"\nCache performance:")
        print(f"  Miss: {time_miss*1000:.2f}ms")
        print(f"  Hit: {time_hit*1000:.4f}ms")
        print(f"  Speedup: {time_miss/time_hit:.1f}x")

        # Cache hit should be much faster
        assert time_hit < time_miss


# ============================================================================
# Performance Tests
# ============================================================================

class TestPhase5Performance:
    """Performance tests for Phase 5 optimizations."""

    def test_scheduler_overhead(self):
        """Test scheduler has minimal overhead."""
        scheduler = AdaptiveBatchScheduler()

        start = time.time()

        for _ in range(1000):
            batch_size = scheduler.get_batch_size(queue_depth=50)

        duration = time.time() - start

        print(f"Scheduler overhead: {duration*1000:.2f}ms for 1000 calls")

        # Should be very fast
        assert duration < 0.1  # <100ms for 1000 calls

    def test_cache_lookup_speed(self):
        """Test cache lookup is fast."""
        cache = MultiLevelCache(l1_size=100, l2_size=1000, l3_size=10000)

        # Fill cache
        for i in range(100):
            cache.put_l3(f"key_{i}", f"value_{i}")

        # Benchmark lookups
        start = time.time()

        for _ in range(10000):
            cache.get_l3("key_50")

        duration = time.time() - start

        print(f"Cache lookup: {duration*1000000/10000:.2f}μs per lookup")

        # Should be very fast (< 1μs per lookup)
        assert duration < 0.01

    def test_compiler_throughput(self):
        """Test compiler throughput."""
        compiler = IonQNativeCompiler(optimization_level=2)

        qubits = cirq.LineQubit.range(4)

        circuits = []
        for i in range(10):
            circuit = cirq.Circuit([
                cirq.H.on_each(*qubits),
                cirq.rx(i * 0.1).on(qubits[0])
            ])
            circuits.append(circuit)

        start = time.time()

        for circuit in circuits:
            compiler.compile(circuit)

        duration = time.time() - start

        print(f"Compiler throughput: {len(circuits)/duration:.1f} circuits/sec")

        # Should compile reasonable number of circuits per second
        assert len(circuits) / duration > 5  # >5 circuits/sec


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
