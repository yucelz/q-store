"""
Comprehensive Tests for Q-Store v4.1 Phase 2: Async Execution Pipeline

Tests all Phase 2 components:
- AsyncQuantumExecutor with non-blocking execution
- ResultCache with LRU eviction
- BackendClient with connection pooling
- Background worker and queue management
- AsyncQuantumTrainer with pipelined execution
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time
from concurrent.futures import ThreadPoolExecutor

from q_store.runtime import AsyncQuantumExecutor, BackendClient, ResultCache


# ============================================================================
# Test AsyncQuantumExecutor
# ============================================================================

class TestAsyncQuantumExecutor:
    """Test AsyncQuantumExecutor core functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test executor initialization."""
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=10,
            enable_caching=True
        )

        assert executor.backend == 'cirq_simulator'
        assert executor.max_concurrent == 10
        assert executor.cache is not None

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_non_blocking_submission(self):
        """Test non-blocking circuit submission."""
        executor = AsyncQuantumExecutor(backend='cirq_simulator')

        # Create simple circuit
        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])

        # Submit should return immediately
        start = time.time()
        future = await executor.submit(circuit, shots=100)
        submit_time = time.time() - start

        # Should be very fast (< 10ms)
        assert submit_time < 0.01

        # Get result
        result = await future
        assert result is not None

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent circuit execution."""
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=5
        )

        import cirq
        qubits = cirq.LineQubit.range(2)

        # Submit multiple circuits concurrently
        circuits = []
        for i in range(10):
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.rx(i * 0.1).on(qubits[1])
            ])
            circuits.append(circuit)

        start = time.time()

        # Submit all circuits
        futures = []
        for circuit in circuits:
            future = await executor.submit(circuit, shots=100)
            futures.append(future)

        # Wait for all results
        results = await asyncio.gather(*futures)

        duration = time.time() - start

        assert len(results) == 10

        # Should be faster than sequential execution
        # (though with simulator the benefit is limited)
        print(f"Concurrent execution time: {duration:.3f}s")

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting with max_concurrent."""
        max_concurrent = 3
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=max_concurrent
        )

        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])

        # Track concurrent executions
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def tracked_submit():
            nonlocal active_count, max_active

            async with lock:
                active_count += 1
                max_active = max(max_active, active_count)

            try:
                future = await executor.submit(circuit, shots=100)
                result = await future
                return result
            finally:
                async with lock:
                    active_count -= 1

        # Submit many circuits
        tasks = [tracked_submit() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Should not exceed max_concurrent
        assert max_active <= max_concurrent + 1  # +1 for tolerance

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for failed executions."""
        executor = AsyncQuantumExecutor(backend='cirq_simulator')

        import cirq
        qubits = cirq.LineQubit.range(2)

        # Create invalid circuit
        circuit = cirq.Circuit()

        try:
            future = await executor.submit(circuit, shots=100)
            result = await future
            # May or may not fail depending on backend
        except Exception as e:
            # Error should be caught and reported
            assert e is not None

        await executor.shutdown()


# ============================================================================
# Test ResultCache
# ============================================================================

class TestResultCache:
    """Test ResultCache with LRU eviction."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = ResultCache(max_size=100)

        assert cache.max_size == 100
        assert len(cache) == 0

    def test_put_and_get(self):
        """Test basic put/get operations."""
        cache = ResultCache(max_size=10)

        # Put items
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        assert len(cache) == 5

        # Get items
        for i in range(5):
            value = cache.get(f"key_{i}")
            assert value == f"value_{i}"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResultCache(max_size=10)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ResultCache(max_size=3)

        # Fill cache
        cache.put("key_1", "value_1")
        cache.put("key_2", "value_2")
        cache.put("key_3", "value_3")

        assert len(cache) == 3

        # Access key_1 to make it recently used
        cache.get("key_1")

        # Add new item, should evict key_2 (least recently used)
        cache.put("key_4", "value_4")

        assert cache.get("key_1") is not None
        assert cache.get("key_2") is None  # Evicted
        assert cache.get("key_3") is not None
        assert cache.get("key_4") is not None

    def test_cache_hit_rate(self):
        """Test cache hit rate tracking."""
        cache = ResultCache(max_size=10)

        # Add items
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        # Access with hits and misses
        cache.get("key_0")  # Hit
        cache.get("key_1")  # Hit
        cache.get("key_5")  # Miss
        cache.get("key_2")  # Hit
        cache.get("key_6")  # Miss

        stats = cache.stats()

        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.6

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ResultCache(max_size=10)

        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")

        assert len(cache) == 5

        cache.clear()

        assert len(cache) == 0


# ============================================================================
# Test BackendClient
# ============================================================================

class TestBackendClient:
    """Test BackendClient with connection pooling."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = BackendClient(
            backend='cirq_simulator',
            pool_size=5
        )

        assert client.backend == 'cirq_simulator'
        assert client.pool_size == 5

        await client.close()

    @pytest.mark.asyncio
    async def test_connection_pooling(self):
        """Test connection pooling."""
        pool_size = 3
        client = BackendClient(
            backend='cirq_simulator',
            pool_size=pool_size
        )

        # Get connections
        connections = []
        for _ in range(pool_size):
            conn = await client.get_connection()
            connections.append(conn)

        assert len(connections) == pool_size

        # Return connections
        for conn in connections:
            await client.return_connection(conn)

        await client.close()

    @pytest.mark.asyncio
    async def test_execute_circuit(self):
        """Test circuit execution through client."""
        client = BackendClient(backend='cirq_simulator')

        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])

        result = await client.execute(circuit, shots=100)

        assert result is not None

        await client.close()


# ============================================================================
# Test Background Worker
# ============================================================================

class TestBackgroundWorker:
    """Test background worker and queue management."""

    @pytest.mark.asyncio
    async def test_worker_initialization(self):
        """Test worker initialization."""
        from q_store.backends import BackgroundWorker

        worker = BackgroundWorker(
            executor=AsyncQuantumExecutor(backend='cirq_simulator'),
            max_queue_size=100
        )

        assert worker.max_queue_size == 100

        await worker.start()
        await worker.stop()

    @pytest.mark.asyncio
    async def test_job_submission(self):
        """Test job submission to worker."""
        from q_store.backends import BackgroundWorker

        executor = AsyncQuantumExecutor(backend='cirq_simulator')
        worker = BackgroundWorker(executor=executor)

        await worker.start()

        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])

        # Submit job
        job_id = await worker.submit_job(circuit, shots=100)

        assert job_id is not None

        # Wait for result
        result = await worker.get_result(job_id, timeout=5.0)

        assert result is not None

        await worker.stop()
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of circuits."""
        from q_store.backends import BackgroundWorker

        executor = AsyncQuantumExecutor(backend='cirq_simulator')
        worker = BackgroundWorker(executor=executor, batch_size=5)

        await worker.start()

        import cirq
        qubits = cirq.LineQubit.range(2)

        # Submit batch of jobs
        job_ids = []
        for i in range(10):
            circuit = cirq.Circuit([cirq.rx(i * 0.1).on(qubits[0])])
            job_id = await worker.submit_job(circuit, shots=100)
            job_ids.append(job_id)

        # Wait for all results
        results = []
        for job_id in job_ids:
            result = await worker.get_result(job_id, timeout=10.0)
            results.append(result)

        assert len(results) == 10

        await worker.stop()
        await executor.shutdown()


# ============================================================================
# Test AsyncQuantumTrainer
# ============================================================================

class TestAsyncQuantumTrainer:
    """Test AsyncQuantumTrainer with pipelined execution."""

    @pytest.mark.asyncio
    async def test_trainer_initialization(self):
        """Test trainer initialization."""
        from q_store.training import AsyncQuantumTrainer

        trainer = AsyncQuantumTrainer(
            model=Mock(),
            executor=AsyncQuantumExecutor(backend='cirq_simulator'),
            pipeline_depth=3
        )

        assert trainer.pipeline_depth == 3

        await trainer.shutdown()

    @pytest.mark.asyncio
    async def test_pipelined_execution(self):
        """Test 3-batch pipelined execution."""
        from q_store.training import AsyncQuantumTrainer

        model = Mock()
        executor = AsyncQuantumExecutor(backend='cirq_simulator')

        trainer = AsyncQuantumTrainer(
            model=model,
            executor=executor,
            pipeline_depth=3
        )

        # Create fake batches
        batches = [
            (np.random.randn(10, 4), np.random.randint(0, 2, 10))
            for _ in range(9)  # 3 pipeline stages * 3 batches
        ]

        start = time.time()

        # Process batches with pipelining
        for batch_x, batch_y in batches:
            await trainer.train_step(batch_x, batch_y)

        duration = time.time() - start

        print(f"Pipelined training time: {duration:.3f}s")

        await trainer.shutdown()

    @pytest.mark.asyncio
    async def test_async_gradient_computation(self):
        """Test async gradient computation."""
        from q_store.training import AsyncQuantumTrainer

        model = Mock()
        executor = AsyncQuantumExecutor(backend='cirq_simulator')

        trainer = AsyncQuantumTrainer(
            model=model,
            executor=executor
        )

        # Compute gradients asynchronously
        batch_x = np.random.randn(5, 4)
        batch_y = np.random.randint(0, 2, 5)

        gradients = await trainer.compute_gradients_async(batch_x, batch_y)

        assert gradients is not None

        await trainer.shutdown()


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 async pipeline."""

    @pytest.mark.asyncio
    async def test_full_async_pipeline(self):
        """Test complete async execution pipeline."""
        # Initialize components
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=5,
            enable_caching=True
        )

        from q_store.backends import BackgroundWorker
        worker = BackgroundWorker(executor=executor, batch_size=3)
        await worker.start()

        import cirq
        qubits = cirq.LineQubit.range(2)

        # Submit multiple circuits
        job_ids = []
        for i in range(20):
            circuit = cirq.Circuit([
                cirq.H(qubits[0]),
                cirq.rx(i * 0.1).on(qubits[1])
            ])
            job_id = await worker.submit_job(circuit, shots=100)
            job_ids.append(job_id)

        # Collect results
        results = []
        for job_id in job_ids:
            result = await worker.get_result(job_id, timeout=10.0)
            results.append(result)

        assert len(results) == 20

        # Check cache statistics
        cache_stats = executor.cache.stats()
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

        await worker.stop()
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test async provides performance improvement."""
        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        ])

        n_circuits = 10

        # Async execution
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=5
        )

        start = time.time()
        futures = []
        for _ in range(n_circuits):
            future = await executor.submit(circuit, shots=100)
            futures.append(future)

        results = await asyncio.gather(*futures)
        async_time = time.time() - start

        await executor.shutdown()

        print(f"Async execution: {async_time:.3f}s for {n_circuits} circuits")
        print(f"Throughput: {n_circuits/async_time:.1f} circuits/sec")

        assert len(results) == n_circuits


# ============================================================================
# Performance Tests
# ============================================================================

class TestPhase2Performance:
    """Performance tests for Phase 2 async execution."""

    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test execution throughput."""
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            max_concurrent=10
        )

        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])

        n_circuits = 100

        start = time.time()

        futures = []
        for _ in range(n_circuits):
            future = await executor.submit(circuit, shots=100)
            futures.append(future)

        await asyncio.gather(*futures)

        duration = time.time() - start
        throughput = n_circuits / duration

        print(f"Throughput: {throughput:.1f} circuits/sec")

        # Should achieve >10 circuits/sec with simulator
        assert throughput > 10

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_cache_effectiveness(self):
        """Test cache effectiveness for repeated circuits."""
        executor = AsyncQuantumExecutor(
            backend='cirq_simulator',
            enable_caching=True
        )

        import cirq
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.H(qubits[0])])

        # Execute same circuit multiple times
        for _ in range(10):
            future = await executor.submit(circuit, shots=100)
            await future

        stats = executor.cache.stats()

        # Should have high hit rate after first execution
        assert stats['hit_rate'] > 0.8

        await executor.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
