"""
Comprehensive Test Suite for ML Components
Tests quantum layers, trainers, gradient computation, and optimization
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from q_store.ml.quantum_layer import QuantumLayer, LayerConfig, QuantumConvolutionalLayer, QuantumPoolingLayer
from q_store.ml.quantum_layer_v2 import HardwareEfficientQuantumLayer, HardwareEfficientLayerConfig
from q_store.ml.gradient_computer import (
    QuantumGradientComputer,
    FiniteDifferenceGradient,
    NaturalGradientComputer,
    GradientResult
)
from q_store.ml.data_encoder import QuantumDataEncoder, QuantumFeatureMap
from q_store.ml.quantum_trainer import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    TrainingMetrics
)
from q_store.ml.spsa_gradient_estimator import SPSAGradientEstimator
from q_store.ml.adaptive_optimizer import AdaptiveGradientOptimizer
from q_store.ml.circuit_cache import QuantumCircuitCache
from q_store.ml.circuit_batch_manager import CircuitBatchManager
from q_store.ml.performance_tracker import PerformanceTracker
from q_store.backends.backend_manager import MockQuantumBackend


@pytest.fixture
def mock_backend():
    """Mock quantum backend for testing"""
    return MockQuantumBackend()


@pytest.fixture
def layer_config():
    """Standard layer configuration"""
    return LayerConfig(
        num_qubits=4,
        num_layers=2,
        entanglement_pattern='linear'
    )


class TestQuantumLayer:
    """Test basic quantum layer"""

    def test_quantum_layer_creation(self, mock_backend, layer_config):
        """Test creating quantum layer"""
        layer = QuantumLayer(
            backend=mock_backend,
            config=layer_config
        )

        assert layer.config.num_qubits == 4
        assert layer.config.num_layers == 2
        assert layer.num_parameters > 0

    def test_quantum_layer_forward(self, mock_backend, layer_config):
        """Test forward pass through layer"""
        layer = QuantumLayer(backend=mock_backend, config=layer_config)

        input_data = np.random.rand(16)  # 2^4 = 16 for 4 qubits

        output = layer.forward(input_data)

        assert output is not None
        assert len(output) > 0

    def test_quantum_layer_get_parameters(self, mock_backend, layer_config):
        """Test getting layer parameters"""
        layer = QuantumLayer(backend=mock_backend, config=layer_config)

        params = layer.get_parameters()

        assert params is not None
        assert len(params) == layer.num_parameters

    def test_quantum_layer_set_parameters(self, mock_backend, layer_config):
        """Test setting layer parameters"""
        layer = QuantumLayer(backend=mock_backend, config=layer_config)

        new_params = np.random.rand(layer.num_parameters)
        layer.set_parameters(new_params)

        retrieved_params = layer.get_parameters()
        assert np.allclose(retrieved_params, new_params)


class TestQuantumLayerV2:
    """Test advanced quantum layers"""

    def test_hardware_efficient_layer_creation(self, mock_backend):
        """Test creating hardware efficient quantum layer"""
        config = HardwareEfficientLayerConfig(
            num_qubits=6,
            num_layers=2
        )
        layer = HardwareEfficientQuantumLayer(
            backend=mock_backend,
            config=config
        )

        assert layer.config.num_qubits == 6
        assert layer.config.num_layers == 2

    def test_hardware_efficient_layer_forward(self, mock_backend):
        """Test hardware efficient layer forward pass"""
        config = HardwareEfficientLayerConfig(
            num_qubits=4,
            num_layers=1
        )
        layer = HardwareEfficientQuantumLayer(
            backend=mock_backend,
            config=config
        )

        input_data = np.random.rand(16)
        output = layer.forward(input_data)

        assert output is not None


class TestGradientComputation:
    """Test gradient computation methods"""

    def test_finite_difference_gradient(self, mock_backend, layer_config):
        """Test finite difference gradient computation"""
        layer = QuantumLayer(backend=mock_backend, config=layer_config)
        grad_computer = FiniteDifferenceGradient(epsilon=0.01)

        def loss_fn(params):
            layer.set_parameters(params)
            output = layer.forward(np.random.rand(16))
            return np.sum(output ** 2)

        params = layer.get_parameters()
        gradient = grad_computer.compute_gradient(loss_fn, params)

        assert gradient is not None
        assert len(gradient) == len(params)

    def test_quantum_gradient_computer(self, mock_backend, layer_config):
        """Test quantum gradient computer"""
        layer = QuantumLayer(backend=mock_backend, config=layer_config)
        grad_computer = QuantumGradientComputer(backend=mock_backend)

        params = layer.get_parameters()

        # Mock circuit and observable
        from q_store.backends.quantum_backend_interface import CircuitBuilder
        builder = CircuitBuilder(num_qubits=4)
        builder.h(0)
        circuit = builder.build()

        observable = np.eye(2)

        result = grad_computer.compute_gradient(circuit, params, observable)

        assert isinstance(result, GradientResult)
        assert result.gradients is not None

    def test_natural_gradient_computer(self, mock_backend):
        """Test natural gradient computation"""
        grad_computer = NaturalGradientComputer(backend=mock_backend)

        from q_store.backends.quantum_backend_interface import CircuitBuilder
        builder = CircuitBuilder(num_qubits=2)
        builder.h(0)
        circuit = builder.build()

        params = np.array([0.5, 0.3])
        observable = np.eye(2)

        result = grad_computer.compute_gradient(circuit, params, observable)

        assert isinstance(result, GradientResult)


class TestSPSAGradientEstimator:
    """Test SPSA gradient estimation"""

    def test_spsa_estimator_creation(self):
        """Test creating SPSA estimator"""
        estimator = SPSAGradientEstimator(
            perturbation_size=0.1,
            num_samples=2
        )

        assert estimator.perturbation_size == 0.1
        assert estimator.num_samples == 2

    def test_spsa_estimate_gradient(self):
        """Test SPSA gradient estimation"""
        estimator = SPSAGradientEstimator(perturbation_size=0.1)

        def loss_fn(params):
            return np.sum(params ** 2)

        params = np.array([1.0, 2.0, 3.0])
        gradient = estimator.estimate_gradient(loss_fn, params)

        assert gradient is not None
        assert len(gradient) == len(params)


class TestAdaptiveOptimizer:
    """Test adaptive optimizer"""

    def test_adaptive_optimizer_creation(self):
        """Test creating adaptive optimizer"""
        optimizer = AdaptiveGradientOptimizer(
            initial_learning_rate=0.01
        )

        assert optimizer.learning_rate == 0.01

    def test_optimizer_step(self):
        """Test optimization step"""
        optimizer = AdaptiveGradientOptimizer(initial_learning_rate=0.1)

        params = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.3])

        updated = optimizer.step(params, gradients)

        assert updated is not None
        assert len(updated) == len(params)
        # Parameters should change
        assert not np.allclose(updated, params)


class TestCircuitCache:
    """Test circuit caching"""

    def test_circuit_cache_creation(self):
        """Test creating circuit cache"""
        cache = QuantumCircuitCache(max_size=100)

        assert cache.max_size == 100

    def test_cache_put_get(self):
        """Test caching and retrieving circuits"""
        cache = QuantumCircuitCache(max_size=10)

        from q_store.backends.quantum_backend_interface import CircuitBuilder
        builder = CircuitBuilder(num_qubits=2)
        builder.h(0)
        circuit = builder.build()

        key = "test_circuit"
        cache.put(key, circuit)

        retrieved = cache.get(key)
        assert retrieved is not None

    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = QuantumCircuitCache(max_size=10)

        from q_store.backends.quantum_backend_interface import CircuitBuilder
        builder = CircuitBuilder(num_qubits=2)
        builder.h(0)
        circuit = builder.build()

        cache.put("circuit1", circuit)

        # Cache hit
        cache.get("circuit1")

        # Cache miss
        cache.get("nonexistent")

        stats = cache.get_statistics()
        assert 'hits' in stats or 'total_gets' in stats


class TestCircuitBatchManager:
    """Test circuit batch manager"""

    def test_batch_manager_creation(self):
        """Test creating batch manager"""
        manager = CircuitBatchManager(batch_size=10)

        assert manager.batch_size == 10

    def test_add_to_batch(self):
        """Test adding circuits to batch"""
        manager = CircuitBatchManager(batch_size=5)

        from q_store.backends.quantum_backend_interface import CircuitBuilder

        for i in range(3):
            builder = CircuitBuilder(num_qubits=2)
            builder.h(0)
            manager.add_circuit(builder.build())

        assert manager.current_batch_size == 3

    def test_batch_execution(self, mock_backend):
        """Test executing circuit batch"""
        manager = CircuitBatchManager(batch_size=5)

        from q_store.backends.quantum_backend_interface import CircuitBuilder

        circuits = []
        for i in range(3):
            builder = CircuitBuilder(num_qubits=2)
            builder.h(0)
            circuit = builder.build()
            circuits.append(circuit)
            manager.add_circuit(circuit)

        results = manager.execute_batch(mock_backend, shots=100)

        assert results is not None
        assert len(results) == 3


class TestPerformanceTracker:
    """Test performance tracking"""

    def test_performance_tracker_creation(self):
        """Test creating performance tracker"""
        tracker = PerformanceTracker()

        assert tracker is not None

    def test_track_metric(self):
        """Test tracking a metric"""
        tracker = PerformanceTracker()

        tracker.track('loss', 0.5)
        tracker.track('loss', 0.4)
        tracker.track('loss', 0.3)

        assert len(tracker.metrics['loss']) == 3

    def test_get_metric_history(self):
        """Test getting metric history"""
        tracker = PerformanceTracker()

        values = [0.5, 0.4, 0.3, 0.2]
        for val in values:
            tracker.track('accuracy', val)

        history = tracker.get_history('accuracy')

        assert len(history) == len(values)
        assert history == values

    def test_get_latest_metric(self):
        """Test getting latest metric value"""
        tracker = PerformanceTracker()

        tracker.track('loss', 0.5)
        tracker.track('loss', 0.3)

        latest = tracker.get_latest('loss')

        assert latest == 0.3

    def test_compute_statistics(self):
        """Test computing metric statistics"""
        tracker = PerformanceTracker()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            tracker.track('metric', val)

        stats = tracker.get_statistics('metric')

        assert 'mean' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0


class TestQuantumDataEncoder:
    """Test quantum data encoder"""

    def test_encoder_creation(self, mock_backend):
        """Test creating data encoder"""
        encoder = QuantumDataEncoder(backend=mock_backend, num_qubits=4)

        assert encoder.num_qubits == 4

    def test_amplitude_encoding(self, mock_backend):
        """Test amplitude encoding"""
        encoder = QuantumDataEncoder(backend=mock_backend, num_qubits=2)

        data = np.array([1.0, 0.0, 0.0, 0.0])
        circuit = encoder.amplitude_encode(data)

        assert circuit is not None
        assert circuit.num_qubits == 2

    def test_angle_encoding(self, mock_backend):
        """Test angle encoding"""
        encoder = QuantumDataEncoder(backend=mock_backend, num_qubits=3)

        data = np.array([0.1, 0.2, 0.3])
        circuit = encoder.angle_encode(data)

        assert circuit is not None
        assert circuit.num_qubits == 3


class TestQuantumFeatureMap:
    """Test quantum feature map"""

    def test_feature_map_creation(self, mock_backend):
        """Test creating feature map"""
        feature_map = QuantumFeatureMap(
            backend=mock_backend,
            num_qubits=4,
            feature_dimension=8
        )

        assert feature_map.num_qubits == 4
        assert feature_map.feature_dimension == 8

    def test_feature_map_transform(self, mock_backend):
        """Test transforming features"""
        feature_map = QuantumFeatureMap(
            backend=mock_backend,
            num_qubits=2,
            feature_dimension=4
        )

        features = np.random.rand(4)
        circuit = feature_map.transform(features)

        assert circuit is not None


class TestQuantumModel:
    """Test quantum model"""

    def test_quantum_model_creation(self, mock_backend):
        """Test creating quantum model"""
        model = QuantumModel(backend=mock_backend)

        assert model is not None
        assert len(model.layers) == 0

    def test_add_layer(self, mock_backend, layer_config):
        """Test adding layer to model"""
        model = QuantumModel(backend=mock_backend)
        layer = QuantumLayer(backend=mock_backend, config=layer_config)

        model.add_layer(layer)

        assert len(model.layers) == 1

    def test_model_forward(self, mock_backend, layer_config):
        """Test model forward pass"""
        model = QuantumModel(backend=mock_backend)
        layer = QuantumLayer(backend=mock_backend, config=layer_config)
        model.add_layer(layer)

        input_data = np.random.rand(16)
        output = model.forward(input_data)

        assert output is not None

    def test_get_all_parameters(self, mock_backend, layer_config):
        """Test getting all model parameters"""
        model = QuantumModel(backend=mock_backend)

        layer1 = QuantumLayer(backend=mock_backend, config=layer_config)
        layer2 = QuantumLayer(backend=mock_backend, config=layer_config)

        model.add_layer(layer1)
        model.add_layer(layer2)

        params = model.get_parameters()

        assert params is not None
        assert len(params) == layer1.num_parameters + layer2.num_parameters


class TestQuantumTrainer:
    """Test quantum trainer"""

    def test_trainer_creation(self, mock_backend):
        """Test creating trainer"""
        model = QuantumModel(backend=mock_backend)
        config = TrainingConfig(
            learning_rate=0.01,
            num_epochs=10,
            batch_size=32
        )

        trainer = QuantumTrainer(model=model, config=config)

        assert trainer.model == model
        assert trainer.config.learning_rate == 0.01

    def test_training_config(self):
        """Test training configuration"""
        config = TrainingConfig(
            learning_rate=0.05,
            num_epochs=20,
            batch_size=64
        )

        assert config.learning_rate == 0.05
        assert config.num_epochs == 20
        assert config.batch_size == 64

    def test_training_metrics(self):
        """Test training metrics"""
        metrics = TrainingMetrics()

        metrics.add_loss(0.5)
        metrics.add_loss(0.4)

        assert len(metrics.losses) == 2
        assert metrics.get_average_loss() < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
