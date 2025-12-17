"""
Simple Test Coverage Summary
Tests core imports and basic functionality
"""

import pytest
import numpy as np


class TestImports:
    """Test all major imports work"""

    def test_import_backends(self):
        """Test backend imports"""
        from q_store.backends import (
            BackendManager,
            MockQuantumBackend,
            create_default_backend_manager
        )
        assert BackendManager is not None
        assert MockQuantumBackend is not None

    def test_import_core(self):
        """Test core imports"""
        from q_store.core import (
            QuantumDatabase,
            DatabaseConfig,
            StateManager,
            EntanglementRegistry,
            TunnelingEngine
        )
        assert all([QuantumDatabase, DatabaseConfig, StateManager,
                    EntanglementRegistry, TunnelingEngine])

    def test_import_ml(self):
        """Test ML imports"""
        from q_store.ml import (
            QuantumLayer,
            LayerConfig,
            QuantumTrainer,
            TrainingConfig,
            QuantumModel
        )
        assert all([QuantumLayer, LayerConfig, QuantumTrainer,
                    TrainingConfig, QuantumModel])

    def test_import_constants(self):
        """Test constants import"""
        from q_store import constants
        assert constants is not None

    def test_import_exceptions(self):
        """Test exceptions import"""
        from q_store import exceptions
        assert exceptions is not None


class TestBasicFunctionality:
    """Test basic functionality without external dependencies"""

    def test_backend_manager(self):
        """Test backend manager creation"""
        from q_store.backends import BackendManager, MockQuantumBackend

        manager = BackendManager()
        backend = MockQuantumBackend()

        assert manager is not None
        assert backend is not None

    def test_database_config(self):
        """Test database configuration"""
        from q_store.core import DatabaseConfig

        config = DatabaseConfig(
            pinecone_api_key="test",
            pinecone_environment="test",
            pinecone_index_name="test",
            pinecone_dimension=128
        )

        assert config.pinecone_api_key == "test"
        assert config.pinecone_dimension == 128

    def test_layer_config(self):
        """Test layer configuration"""
        from q_store.ml import LayerConfig

        config = LayerConfig(
            n_qubits=4,
            depth=2,
            entanglement='linear'
        )

        assert config.n_qubits == 4
        assert config.depth == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
