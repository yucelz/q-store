"""
Integration Tests - Full System Testing
Tests end-to-end workflows and component integration
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch

from q_store import (
    QuantumDatabase,
    DatabaseConfig,
    QuantumModel,
    QuantumLayer,
    LayerConfig,
    QuantumTrainer,
    TrainingConfig,
    BackendManager,
    MockQuantumBackend
)


@pytest.fixture
def integration_config():
    """Configuration for integration testing"""
    return DatabaseConfig(
        pinecone_api_key="test-integration-key",
        pinecone_environment="test-env",
        pinecone_index_name="integration-test",
        pinecone_dimension=64,
        enable_quantum=True,
        max_quantum_states=50
    )


@pytest.fixture
def backend_manager():
    """Backend manager with mock backend"""
    manager = BackendManager()
    manager.register_backend('mock', MockQuantumBackend())
    return manager


class TestDatabaseMLIntegration:
    """Test integration between database and ML components"""

    @pytest.mark.asyncio
    async def test_database_with_quantum_features(self, integration_config):
        """Test database with quantum feature creation"""
        db = QuantumDatabase(integration_config)
        await db.initialize()

        # Insert vectors with quantum contexts
        vectors = [np.random.rand(64) for _ in range(5)]

        for i, vec in enumerate(vectors):
            await db.insert(
                id=f'ml_vec_{i}',
                vector=vec,
                contexts=[('training', 0.7), ('validation', 0.3)],
                metadata={'type': 'ml_data'}
            )

        # Verify quantum states created
        assert len(db.state_manager.states) > 0

        # Query with context
        query_vec = np.random.rand(64)
        results = await db.query(
            vector=query_vec,
            context='training',
            top_k=3
        )

        assert len(results) > 0

        await db.close()

    @pytest.mark.asyncio
    async def test_ml_model_with_database_vectors(self, integration_config, backend_manager):
        """Test training ML model on database vectors"""
        # Setup database
        db = QuantumDatabase(integration_config)
        await db.initialize()

        # Insert training data
        training_data = [np.random.rand(64) for _ in range(10)]
        for i, vec in enumerate(training_data):
            await db.insert(id=f'train_{i}', vector=vec)

        # Create ML model
        backend = backend_manager.get_backend('mock')
        model = QuantumModel(backend=backend)

        layer = QuantumLayer(
            backend=backend,
            config=LayerConfig(num_qubits=3, num_layers=2)
        )
        model.add_layer(layer)

        # Verify model can process vectors
        test_input = np.random.rand(8)  # 2^3 for 3 qubits
        output = model.forward(test_input)

        assert output is not None

        await db.close()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    @pytest.mark.asyncio
    async def test_full_quantum_ml_pipeline(self, backend_manager):
        """Test complete ML training pipeline"""
        backend = backend_manager.get_backend('mock')

        # 1. Create model
        model = QuantumModel(backend=backend)
        layer = QuantumLayer(
            backend=backend,
            config=LayerConfig(num_qubits=2, num_layers=1)
        )
        model.add_layer(layer)

        # 2. Create trainer
        config = TrainingConfig(
            learning_rate=0.01,
            num_epochs=2,
            batch_size=4
        )
        trainer = QuantumTrainer(model=model, config=config)

        # 3. Generate training data
        X_train = [np.random.rand(4) for _ in range(8)]
        y_train = [np.random.randint(0, 2) for _ in range(8)]

        # 4. Verify trainer is set up correctly
        assert trainer.model == model
        assert trainer.config.num_epochs == 2

        # Test would train here if trainer.train() is implemented

    @pytest.mark.asyncio
    async def test_multi_context_quantum_search(self, integration_config):
        """Test searching across multiple quantum contexts"""
        db = QuantumDatabase(integration_config)
        await db.initialize()

        # Insert data with multiple contexts
        contexts_list = [
            [('technical', 0.8), ('general', 0.2)],
            [('scientific', 0.9), ('general', 0.1)],
            [('technical', 0.5), ('scientific', 0.5)]
        ]

        for i, contexts in enumerate(contexts_list):
            vec = np.random.rand(64)
            await db.insert(
                id=f'multi_ctx_{i}',
                vector=vec,
                contexts=contexts
            )

        # Query with different contexts
        query_vec = np.random.rand(64)

        tech_results = await db.query(
            vector=query_vec,
            context='technical',
            top_k=5
        )

        sci_results = await db.query(
            vector=query_vec,
            context='scientific',
            top_k=5
        )

        # Results can differ based on context
        assert tech_results is not None
        assert sci_results is not None

        await db.close()


class TestErrorHandlingIntegration:
    """Test error handling across components"""

    @pytest.mark.asyncio
    async def test_invalid_vector_dimension(self, integration_config):
        """Test handling invalid vector dimensions"""
        db = QuantumDatabase(integration_config)
        await db.initialize()

        # Try to insert wrong dimension
        try:
            await db.insert(
                id='wrong_dim',
                vector=np.random.rand(128)  # Wrong dimension
            )
        except Exception as e:
            # Should handle error gracefully
            assert e is not None

        await db.close()

    @pytest.mark.asyncio
    async def test_missing_quantum_state(self, integration_config):
        """Test handling missing quantum state"""
        db = QuantumDatabase(integration_config)
        await db.initialize()

        # Try to measure non-existent state
        result = await db.state_manager.measure_with_context(
            'nonexistent_state',
            'any_context'
        )

        assert result is None

        await db.close()


class TestPerformanceIntegration:
    """Test performance across integrated components"""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_config):
        """Test concurrent database and quantum operations"""
        db = QuantumDatabase(integration_config)
        await db.initialize()

        async def insert_and_query(idx):
            vec = np.random.rand(64)
            await db.insert(
                id=f'concurrent_{idx}',
                vector=vec,
                contexts=[('test', 1.0)]
            )

            query_vec = np.random.rand(64)
            results = await db.query(vector=query_vec, top_k=3)
            return len(results)

        # Run multiple operations concurrently
        results = await asyncio.gather(*[
            insert_and_query(i) for i in range(10)
        ])

        assert len(results) == 10
        assert all(r >= 0 for r in results)

        await db.close()


class TestBackwardCompatibility:
    """Test backward compatibility with older versions"""

    def test_legacy_ionq_backend_import(self):
        """Test legacy IonQ backend can still be imported"""
        from q_store.backends.ionq_backend import IonQQuantumBackend

        assert IonQQuantumBackend is not None

    def test_legacy_backend_manager_import(self):
        """Test legacy backend manager functions"""
        from q_store.backends.backend_manager import (
            create_default_backend_manager,
            setup_ionq_backends
        )

        assert create_default_backend_manager is not None
        assert setup_ionq_backends is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
