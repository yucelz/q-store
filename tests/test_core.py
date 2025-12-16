"""
Comprehensive Test Suite for Core Components
Tests StateManager, EntanglementRegistry, and TunnelingEngine
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock

from q_store.core.state_manager import (
    StateManager,
    QuantumState,
    StateStatus
)
from q_store.core.entanglement_registry import EntanglementRegistry
from q_store.core.tunneling_engine import TunnelingEngine
from q_store.core.quantum_database import DatabaseConfig


@pytest.fixture
def test_config():
    """Test configuration"""
    return DatabaseConfig(
        pinecone_api_key="test-key",
        pinecone_environment="test-env",
        pinecone_index_name="test-index",
        pinecone_dimension=128,
        enable_quantum=True,
        max_quantum_states=100
    )


class TestQuantumState:
    """Test QuantumState data structure"""

    def test_quantum_state_creation(self):
        """Test creating quantum state"""
        vectors = [np.random.rand(128) for _ in range(2)]
        contexts = ["context_a", "context_b"]

        state = QuantumState(
            state_id="test_state",
            vectors=vectors,
            contexts=contexts,
            coherence_time=1000.0,
            creation_time=time.time()
        )

        assert state.state_id == "test_state"
        assert state.status == StateStatus.ACTIVE
        assert len(state.vectors) == 2
        assert len(state.contexts) == 2
        assert state.measurement_count == 0

    def test_quantum_state_is_coherent(self):
        """Test coherence checking"""
        state = QuantumState(
            state_id="test",
            vectors=[np.random.rand(128)],
            contexts=["test"],
            coherence_time=100.0,
            creation_time=time.time()
        )

        # Should be coherent immediately
        assert state.is_coherent(time.time())

        # Should not be coherent after waiting
        future_time = time.time() + 200.0
        assert not state.is_coherent(future_time)

    def test_quantum_state_measure(self):
        """Test state measurement"""
        vectors = [np.random.rand(128), np.random.rand(128)]
        contexts = ["ctx1", "ctx2"]

        state = QuantumState(
            state_id="test",
            vectors=vectors,
            contexts=contexts,
            coherence_time=1000.0,
            creation_time=time.time()
        )

        # Measure with first context
        result = state.measure("ctx1")

        assert result is not None
        assert len(result) == 128
        assert state.status == StateStatus.MEASURED
        assert state.measurement_count == 1
        assert state.measured_context == "ctx1"

    def test_quantum_state_measure_wrong_context(self):
        """Test measurement with non-existent context"""
        state = QuantumState(
            state_id="test",
            vectors=[np.random.rand(128)],
            contexts=["valid_context"],
            coherence_time=1000.0,
            creation_time=time.time()
        )

        result = state.measure("invalid_context")

        assert result is None


class TestStateManager:
    """Test StateManager functionality"""

    @pytest.mark.asyncio
    async def test_state_manager_creation(self, test_config):
        """Test state manager initialization"""
        manager = StateManager(test_config)

        assert manager.config == test_config
        assert manager.states == {}

    @pytest.mark.asyncio
    async def test_start_stop(self, test_config):
        """Test starting and stopping state manager"""
        manager = StateManager(test_config)

        await manager.start()
        assert manager._decoherence_task is not None

        await manager.stop()
        assert manager._decoherence_task is None or manager._decoherence_task.cancelled()

    @pytest.mark.asyncio
    async def test_create_superposition(self, test_config):
        """Test creating superposition state"""
        manager = StateManager(test_config)
        await manager.start()

        vectors = [np.random.rand(128) for _ in range(3)]
        contexts = ["tech", "general", "science"]

        state = await manager.create_superposition(
            state_id="super_1",
            vectors=vectors,
            contexts=contexts,
            coherence_time=2000.0
        )

        assert state.state_id == "super_1"
        assert len(state.vectors) == 3
        assert len(state.contexts) == 3
        assert state.status == StateStatus.ACTIVE
        assert "super_1" in manager.states

        await manager.stop()

    @pytest.mark.asyncio
    async def test_measure_with_context(self, test_config):
        """Test measuring state with context"""
        manager = StateManager(test_config)
        await manager.start()

        vectors = [np.random.rand(128), np.random.rand(128)]
        contexts = ["context_a", "context_b"]

        await manager.create_superposition(
            state_id="measure_test",
            vectors=vectors,
            contexts=contexts
        )

        result = await manager.measure_with_context("measure_test", "context_a")

        assert result is not None
        assert len(result) == 128

        state = manager.states["measure_test"]
        assert state.status == StateStatus.MEASURED
        assert state.measured_context == "context_a"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_state(self, test_config):
        """Test getting state by ID"""
        manager = StateManager(test_config)
        await manager.start()

        await manager.create_superposition(
            state_id="get_test",
            vectors=[np.random.rand(128)],
            contexts=["test"]
        )

        state = manager.get_state("get_test")

        assert state is not None
        assert state.state_id == "get_test"

        # Non-existent state
        none_state = manager.get_state("nonexistent")
        assert none_state is None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_apply_decoherence(self, test_config):
        """Test decoherence cleanup"""
        manager = StateManager(test_config)
        await manager.start()

        # Create state with very short coherence
        await manager.create_superposition(
            state_id="short_lived",
            vectors=[np.random.rand(128)],
            contexts=["test"],
            coherence_time=10.0  # 10ms
        )

        # Wait for decoherence
        await asyncio.sleep(0.02)  # 20ms

        removed = await manager.apply_decoherence()

        assert "short_lived" in removed
        assert "short_lived" not in manager.states

        await manager.stop()

    @pytest.mark.asyncio
    async def test_capacity_limit(self, test_config):
        """Test state capacity limits"""
        test_config.max_quantum_states = 5
        manager = StateManager(test_config)
        await manager.start()

        # Try to create more states than capacity
        for i in range(10):
            await manager.create_superposition(
                state_id=f"state_{i}",
                vectors=[np.random.rand(128)],
                contexts=["test"]
            )

        # Should not exceed capacity
        assert len(manager.states) <= test_config.max_quantum_states

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_metrics(self, test_config):
        """Test getting state metrics"""
        manager = StateManager(test_config)
        await manager.start()

        # Create various states
        await manager.create_superposition(
            state_id="active_1",
            vectors=[np.random.rand(128)],
            contexts=["test"]
        )

        await manager.create_superposition(
            state_id="to_measure",
            vectors=[np.random.rand(128)],
            contexts=["test"]
        )
        await manager.measure_with_context("to_measure", "test")

        metrics = manager.get_metrics()

        assert 'total_states' in metrics
        assert 'active_states' in metrics
        assert 'measured_states' in metrics
        assert metrics['total_states'] >= 2

        await manager.stop()


class TestEntanglementRegistry:
    """Test EntanglementRegistry functionality"""

    def test_entanglement_registry_creation(self):
        """Test creating entanglement registry"""
        registry = EntanglementRegistry()

        assert registry is not None
        assert hasattr(registry, 'entanglements')

    def test_register_entanglement(self):
        """Test registering entangled states"""
        registry = EntanglementRegistry()

        registry.register_entanglement("state_a", "state_b", strength=0.8)

        assert registry.is_entangled("state_a", "state_b")

    def test_get_entangled_states(self):
        """Test getting entangled partners"""
        registry = EntanglementRegistry()

        registry.register_entanglement("state_1", "state_2", strength=0.9)
        registry.register_entanglement("state_1", "state_3", strength=0.7)

        partners = registry.get_entangled_states("state_1")

        assert "state_2" in partners
        assert "state_3" in partners
        assert len(partners) == 2

    def test_get_entanglement_strength(self):
        """Test getting entanglement strength"""
        registry = EntanglementRegistry()

        registry.register_entanglement("state_x", "state_y", strength=0.85)

        strength = registry.get_entanglement_strength("state_x", "state_y")

        assert strength == 0.85

    def test_remove_entanglement(self):
        """Test removing entanglement"""
        registry = EntanglementRegistry()

        registry.register_entanglement("state_a", "state_b", strength=0.9)
        assert registry.is_entangled("state_a", "state_b")

        registry.remove_entanglement("state_a", "state_b")

        assert not registry.is_entangled("state_a", "state_b")

    def test_cleanup_state(self):
        """Test cleaning up all entanglements for a state"""
        registry = EntanglementRegistry()

        registry.register_entanglement("state_1", "state_2", strength=0.8)
        registry.register_entanglement("state_1", "state_3", strength=0.7)

        registry.cleanup_state("state_1")

        assert not registry.is_entangled("state_1", "state_2")
        assert not registry.is_entangled("state_1", "state_3")


class TestTunnelingEngine:
    """Test TunnelingEngine functionality"""

    @pytest.mark.asyncio
    async def test_tunneling_engine_creation(self, test_config):
        """Test creating tunneling engine"""
        engine = TunnelingEngine(test_config)

        assert engine is not None
        assert engine.config == test_config

    @pytest.mark.asyncio
    async def test_tunnel_query(self, test_config):
        """Test quantum tunneling query enhancement"""
        engine = TunnelingEngine(test_config)

        query_vector = np.random.rand(128)

        # Tunnel the query
        tunneled = await engine.tunnel_query(query_vector, barrier_height=0.5)

        assert tunneled is not None
        assert len(tunneled) == len(query_vector)
        # Tunneling should modify the vector
        assert not np.allclose(tunneled, query_vector)

    @pytest.mark.asyncio
    async def test_compute_tunneling_probability(self, test_config):
        """Test computing tunneling probability"""
        engine = TunnelingEngine(test_config)

        vector1 = np.random.rand(128)
        vector2 = np.random.rand(128)

        prob = await engine.compute_tunneling_probability(
            vector1, vector2, barrier_height=0.5
        )

        assert 0 <= prob <= 1

    @pytest.mark.asyncio
    async def test_apply_tunneling_effect(self, test_config):
        """Test applying tunneling effect to vector"""
        engine = TunnelingEngine(test_config)

        vector = np.random.rand(128)
        target = np.random.rand(128)

        result = await engine.apply_tunneling_effect(
            vector, target, strength=0.3
        )

        assert result is not None
        assert len(result) == len(vector)
        # Result should be influenced by target
        assert not np.allclose(result, vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
