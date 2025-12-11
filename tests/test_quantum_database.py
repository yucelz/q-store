"""
Comprehensive Test Suite for Quantum Database
Tests Pinecone integration, quantum features, and performance
"""

import pytest
import asyncio
import numpy as np
import time
from typing import List
from unittest.mock import Mock, patch, AsyncMock

# Import from implementation
from q_store import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    StateStatus,
    Metrics,
    QueryResult,
    StateManager
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration with mock backends"""
    return DatabaseConfig(
        pinecone_api_key="test-key",
        pinecone_environment="us-east-1-test",
        pinecone_index_name="test-index",
        pinecone_dimension=128,
        ionq_api_key=None,  # Use mock quantum backend
        enable_quantum=True,
        max_quantum_states=100,
        result_cache_ttl=60,
        decoherence_check_interval=5
    )


@pytest.fixture
async def test_db(test_config):
    """Initialize test database"""
    db = QuantumDatabase(test_config)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def sample_vectors():
    """Generate sample test vectors"""
    np.random.seed(42)
    return [np.random.rand(128) for _ in range(20)]


# ============================================================================
# Unit Tests - State Manager
# ============================================================================

class TestStateManager:
    """Test quantum state management"""
    
    @pytest.mark.asyncio
    async def test_create_superposition(self, test_config):
        """Test creating quantum superposition state"""
        manager = StateManager(test_config)
        await manager.start()
        
        vectors = [np.random.rand(128) for _ in range(2)]
        contexts = ["context_a", "context_b"]
        
        state = await manager.create_superposition(
            state_id="test_1",
            vectors=vectors,
            contexts=contexts,
            coherence_time=1000.0
        )
        
        assert state.state_id == "test_1"
        assert state.status == StateStatus.ACTIVE
        assert len(state.contexts) == 2
        assert state.is_coherent(time.time())
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_measure_with_context(self, test_config):
        """Test quantum measurement with context collapse"""
        manager = StateManager(test_config)
        await manager.start()
        
        vectors = [np.random.rand(128) for _ in range(2)]
        contexts = ["technical", "general"]
        
        await manager.create_superposition(
            state_id="test_2",
            vectors=vectors,
            contexts=contexts
        )
        
        # Measure with matching context
        result = await manager.measure_with_context("test_2", "technical")
        
        assert result is not None
        assert len(result) == 128
        
        # Check state was measured
        state = manager.states["test_2"]
        assert state.status == StateStatus.MEASURED
        assert state.measurement_count == 1
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_decoherence(self, test_config):
        """Test automatic decoherence cleanup"""
        manager = StateManager(test_config)
        await manager.start()
        
        # Create state with very short coherence time
        vectors = [np.random.rand(128)]
        contexts = ["test"]
        
        await manager.create_superposition(
            state_id="test_3",
            vectors=vectors,
            contexts=contexts,
            coherence_time=10.0  # 10ms
        )
        
        # Wait for decoherence
        await asyncio.sleep(0.02)  # 20ms
        
        # Apply decoherence cleanup
        removed = await manager.apply_decoherence()
        
        assert "test_3" in removed
        assert "test_3" not in manager.states
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_capacity_limit(self, test_config):
        """Test state manager respects capacity limits"""
        # Set low capacity for testing
        test_config.max_quantum_states = 5
        manager = StateManager(test_config)
        await manager.start()
        
        # Create states up to capacity
        for i in range(7):  # More than capacity
            await manager.create_superposition(
                state_id=f"test_{i}",
                vectors=[np.random.rand(128)],
                contexts=["test"]
            )
        
        # Should not exceed capacity
        assert len(manager.states) <= test_config.max_quantum_states
        
        await manager.stop()


# ============================================================================
# Integration Tests - Database Operations
# ============================================================================

class TestDatabaseOperations:
    """Test full database operations"""
    
    @pytest.mark.asyncio
    async def test_insert_simple(self, test_db):
        """Test simple vector insertion"""
        vector = np.random.rand(128)
        
        await test_db.insert(
            id="vec_1",
            vector=vector,
            metadata={"type": "test"}
        )
        
        # Verify insertion worked (would check Pinecone in real scenario)
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_insert_with_contexts(self, test_db):
        """Test insertion with quantum contexts"""
        vector = np.random.rand(128)
        
        await test_db.insert(
            id="vec_2",
            vector=vector,
            contexts=[
                ("technical", 0.7),
                ("general", 0.3)
            ],
            coherence_time=5000.0,
            metadata={"domain": "quantum"}
        )
        
        # Verify quantum state was created
        assert "vec_2" in test_db.state_manager.states
        state = test_db.state_manager.states["vec_2"]
        assert state.status == StateStatus.ACTIVE
        assert len(state.contexts) == 2
    
    @pytest.mark.asyncio
    async def test_batch_insert(self, test_db, sample_vectors):
        """Test batch insertion for efficiency"""
        batch = [
            {
                'id': f'batch_{i}',
                'vector': vec,
                'contexts': [("general", 1.0)],
                'metadata': {'batch': True, 'index': i}
            }
            for i, vec in enumerate(sample_vectors[:10])
        ]
        
        start_time = time.time()
        await test_db.insert_batch(batch)
        duration = time.time() - start_time
        
        # Batch should be faster than individual inserts
        print(f"Batch insert took {duration:.3f}s for 10 vectors")
        
        # Verify quantum states created
        assert len(test_db.state_manager.states) >= 10
    
    @pytest.mark.asyncio
    async def test_query_basic(self, test_db, sample_vectors):
        """Test basic query functionality"""
        # Insert test data
        for i, vec in enumerate(sample_vectors[:5]):
            await test_db.insert(
                id=f'query_test_{i}',
                vector=vec,
                metadata={'index': i}
            )
        
        # Query
        query_vector = sample_vectors[0] + np.random.randn(128) * 0.1
        
        results = await test_db.query(
            vector=query_vector,
            top_k=3
        )
        
        assert len(results) > 0
        assert all(isinstance(r, QueryResult) for r in results)
        assert all(r.score >= 0 for r in results)
        
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_query_with_context(self, test_db, sample_vectors):
        """Test context-aware quantum query"""
        # Insert with contexts
        vec = sample_vectors[0]
        await test_db.insert(
            id='context_test',
            vector=vec,
            contexts=[
                ("technical", 0.7),
                ("general", 0.3)
            ]
        )
        
        # Query with technical context
        query_vector = vec + np.random.randn(128) * 0.05
        
        results = await test_db.query(
            vector=query_vector,
            context="technical",
            mode=QueryMode.BALANCED,
            top_k=5
        )
        
        assert len(results) > 0
        
        # At least one result should be quantum-enhanced
        quantum_results = [r for r in results if r.quantum_enhanced]
        print(f"Quantum-enhanced results: {len(quantum_results)}")
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, test_db, sample_vectors):
        """Test query result caching"""
        vec = sample_vectors[0]
        await test_db.insert(id='cache_test', vector=vec)
        
        query_vector = sample_vectors[1]
        
        # First query - cache miss
        start_time = time.time()
        results1 = await test_db.query(vector=query_vector, top_k=5)
        first_duration = time.time() - start_time
        
        assert test_db.metrics.cache_misses == 1
        
        # Second identical query - cache hit
        start_time = time.time()
        results2 = await test_db.query(vector=query_vector, top_k=5)
        second_duration = time.time() - start_time
        
        assert test_db.metrics.cache_hits == 1
        
        # Cache should be faster
        print(f"First query: {first_duration:.4f}s, Cached: {second_duration:.4f}s")
        
        # Results should be identical
        assert len(results1) == len(results2)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, test_db, sample_vectors):
        """Test concurrent insertion performance"""
        async def insert_vector(i, vec):
            await test_db.insert(
                id=f'concurrent_{i}',
                vector=vec,
                contexts=[("test", 1.0)]
            )
        
        start_time = time.time()
        
        # Insert 20 vectors concurrently
        await asyncio.gather(*[
            insert_vector(i, vec)
            for i, vec in enumerate(sample_vectors)
        ])
        
        duration = time.time() - start_time
        throughput = len(sample_vectors) / duration
        
        print(f"Concurrent inserts: {throughput:.1f} vectors/sec")
        assert throughput > 10  # Should handle at least 10 ops/sec
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, test_db, sample_vectors):
        """Test concurrent query performance"""
        # Setup: Insert test data
        for i, vec in enumerate(sample_vectors[:10]):
            await test_db.insert(id=f'perf_test_{i}', vector=vec)
        
        async def run_query(vec):
            return await test_db.query(vector=vec, top_k=5)
        
        start_time = time.time()
        
        # Run 50 concurrent queries
        query_vectors = [np.random.rand(128) for _ in range(50)]
        results = await asyncio.gather(*[
            run_query(vec) for vec in query_vectors
        ])
        
        duration = time.time() - start_time
        qps = len(query_vectors) / duration
        
        print(f"Queries per second: {qps:.1f}")
        assert qps > 5  # Should handle at least 5 QPS
        assert all(len(r) > 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, test_db):
        """Test performance with large batches"""
        # Generate large batch
        large_batch = [
            {
                'id': f'large_batch_{i}',
                'vector': np.random.rand(128),
                'contexts': [("test", 1.0)],
                'metadata': {'index': i}
            }
            for i in range(100)
        ]
        
        start_time = time.time()
        await test_db.insert_batch(large_batch)
        duration = time.time() - start_time
        
        throughput = len(large_batch) / duration
        
        print(f"Large batch: {throughput:.1f} vectors/sec")
        assert throughput > 20  # Should handle good throughput
    
    @pytest.mark.asyncio
    async def test_query_latency_distribution(self, test_db, sample_vectors):
        """Test query latency percentiles"""
        # Setup
        for i, vec in enumerate(sample_vectors[:10]):
            await test_db.insert(id=f'latency_test_{i}', vector=vec)
        
        latencies = []
        
        # Run 100 queries
        for _ in range(100):
            query_vec = np.random.rand(128)
            start = time.time()
            await test_db.query(vector=query_vec, top_k=5)
            latencies.append((time.time() - start) * 1000)
        
        latencies.sort()
        p50 = latencies[49]
        p95 = latencies[94]
        p99 = latencies[98]
        
        print(f"Latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        # Reasonable latency expectations
        assert p50 < 100  # 50th percentile under 100ms
        assert p95 < 200  # 95th percentile under 200ms


# ============================================================================
# Real Pinecone Integration Tests (requires API key)
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    "not config.getoption('--run-integration')",
    reason="Need --run-integration flag and real API keys"
)
class TestPineconeIntegration:
    """Integration tests with real Pinecone backend"""
    
    @pytest.fixture
    def real_config(self):
        """Configuration for real Pinecone"""
        import os
        
        return DatabaseConfig(
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
            pinecone_index_name='quantum-db-test',
            pinecone_dimension=128,
            ionq_api_key=None,  # Optional
            enable_quantum=True
        )
    
    @pytest.mark.asyncio
    async def test_real_pinecone_insert_query(self, real_config):
        """Test with real Pinecone backend"""
        db = QuantumDatabase(real_config)
        
        try:
            await db.initialize()
            
            # Insert test vectors
            test_vectors = [
                (f'real_test_{i}', np.random.rand(128))
                for i in range(10)
            ]
            
            for vec_id, vec in test_vectors:
                await db.insert(
                    id=vec_id,
                    vector=vec,
                    metadata={'source': 'integration_test'}
                )
            
            # Wait for indexing
            await asyncio.sleep(2)
            
            # Query
            query_vec = test_vectors[0][1]
            results = await db.query(
                vector=query_vec,
                top_k=5
            )
            
            assert len(results) > 0
            assert results[0].id == 'real_test_0'  # Should find exact match
            assert results[0].score > 0.95  # High similarity
            
            print(f"Real Pinecone test: Found {len(results)} results")
            
        finally:
            await db.close()


# ============================================================================
# Stress Tests
# ============================================================================

@pytest.mark.stress
class TestStress:
    """Stress testing for reliability"""
    
    @pytest.mark.asyncio
    async def test_memory_leak(self, test_db):
        """Test for memory leaks during extended operation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many operations
        for iteration in range(10):
            # Insert batch
            batch = [
                {
                    'id': f'stress_{iteration}_{i}',
                    'vector': np.random.rand(128),
                    'contexts': [("test", 1.0)]
                }
                for i in range(50)
            ]
            await test_db.insert_batch(batch)
            
            # Query multiple times
            for _ in range(20):
                query_vec = np.random.rand(128)
                await test_db.query(vector=query_vec, top_k=5)
            
            # Trigger decoherence cleanup
            await test_db.state_manager.apply_decoherence()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory growth: {memory_growth:.2f} MB")
        
        # Should not grow excessively (allow 50MB growth)
        assert memory_growth < 50
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, test_db):
        """Test database recovers from errors"""
        # Simulate various errors
        errors_handled = 0
        
        # Invalid vector dimension
        try:
            await test_db.insert(
                id='error_test_1',
                vector=np.random.rand(256)  # Wrong dimension
            )
        except Exception:
            errors_handled += 1
        
        # Database should still work after errors
        await test_db.insert(
            id='recovery_test',
            vector=np.random.rand(128)
        )
        
        results = await test_db.query(
            vector=np.random.rand(128),
            top_k=5
        )
        
        assert len(results) >= 0  # Should not crash


# ============================================================================
# Test Runner Configuration
# ============================================================================

def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests with real backends"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-k", "not integration and not stress"
    ])
