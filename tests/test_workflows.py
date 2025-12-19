"""
Tests for hybrid quantum-classical workflows.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.workflows import (
    QuantumClassicalHybrid,
    QuantumPreprocessor,
    WorkflowQuantumLayer,
    ClassicalPostprocessor,
    create_hybrid_model,
    DataPipeline,
    QuantumDataEncoder,
    BatchProcessor,
    ResultAggregator,
    create_data_pipeline,
    OptimizationLoop,
    ParameterUpdate,
    ConvergenceChecker,
    run_optimization_loop
)


def simple_quantum_function(x):
    """Simple quantum function for testing."""
    circuit = UnifiedCircuit(n_qubits=2)
    circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': x[0]})
    circuit.add_gate(GateType.RY, targets=[1], parameters={'angle': x[1]})
    return circuit


def simple_preprocessing(x):
    """Simple preprocessing function."""
    return x * 2.0


def simple_postprocessing(x):
    """Simple postprocessing function."""
    return x + 1.0


class TestQuantumClassicalHybrid:
    """Test hybrid quantum-classical models."""

    def test_hybrid_creation(self):
        """Test creating hybrid model."""
        model = QuantumClassicalHybrid(
            quantum_component=simple_quantum_function
        )
        assert model.quantum_component == simple_quantum_function

    def test_hybrid_forward_quantum_only(self):
        """Test forward pass with quantum component only."""
        model = QuantumClassicalHybrid(
            quantum_component=simple_quantum_function
        )
        x = np.array([0.5, 0.3])
        result = model.forward(x)
        assert isinstance(result, UnifiedCircuit)

    def test_hybrid_with_preprocessing(self):
        """Test hybrid model with preprocessing."""
        model = QuantumClassicalHybrid(
            quantum_component=simple_quantum_function,
            classical_preprocessor=simple_preprocessing
        )
        x = np.array([0.5, 0.3])
        result = model.forward(x)
        assert isinstance(result, UnifiedCircuit)

    def test_hybrid_with_postprocessing(self):
        """Test hybrid model with postprocessing."""
        def quantum_func(x):
            return np.array([1.0, 2.0])

        model = QuantumClassicalHybrid(
            quantum_component=quantum_func,
            classical_postprocessor=simple_postprocessing
        )
        x = np.array([0.5, 0.3])
        result = model.forward(x)
        assert np.allclose(result, np.array([2.0, 3.0]))

    def test_hybrid_call(self):
        """Test calling hybrid model directly."""
        model = QuantumClassicalHybrid(
            quantum_component=simple_quantum_function
        )
        x = np.array([0.5, 0.3])
        result = model(x)
        assert isinstance(result, UnifiedCircuit)


class TestQuantumPreprocessor:
    """Test quantum preprocessing."""

    def test_preprocessor_creation(self):
        """Test creating quantum preprocessor."""
        preprocessor = QuantumPreprocessor(
            encoding_circuit=simple_quantum_function,
            n_qubits=2
        )
        assert preprocessor.n_qubits == 2

    def test_preprocessor_encode(self):
        """Test encoding data."""
        preprocessor = QuantumPreprocessor(
            encoding_circuit=simple_quantum_function,
            n_qubits=2
        )
        data = np.array([0.5, 0.3])
        circuit = preprocessor.encode(data)
        assert isinstance(circuit, UnifiedCircuit)

    def test_preprocessor_call(self):
        """Test calling preprocessor directly."""
        preprocessor = QuantumPreprocessor(
            encoding_circuit=simple_quantum_function,
            n_qubits=2
        )
        data = np.array([0.5, 0.3])
        circuit = preprocessor(data)
        assert isinstance(circuit, UnifiedCircuit)


class TestQuantumLayer:
    """Test quantum layer."""

    def test_layer_creation(self):
        """Test creating quantum layer."""
        def circuit_builder(x, params):
            circuit = UnifiedCircuit(n_qubits=2)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit

        layer = WorkflowQuantumLayer(
            circuit_builder=circuit_builder,
            n_qubits=2,
            n_parameters=3
        )
        assert layer.n_qubits == 2
        assert layer.n_parameters == 3
        assert len(layer.parameters) == 3

    def test_layer_forward(self):
        """Test forward pass through layer."""
        def circuit_builder(x, params):
            circuit = UnifiedCircuit(n_qubits=2)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit

        layer = WorkflowQuantumLayer(
            circuit_builder=circuit_builder,
            n_qubits=2,
            n_parameters=3
        )
        x = np.array([0.5, 0.3])
        circuit = layer.forward(x)
        assert isinstance(circuit, UnifiedCircuit)

    def test_layer_update_parameters(self):
        """Test updating layer parameters."""
        def circuit_builder(x, params):
            return UnifiedCircuit(n_qubits=2)

        layer = WorkflowQuantumLayer(
            circuit_builder=circuit_builder,
            n_qubits=2,
            n_parameters=3
        )
        new_params = np.array([1.0, 2.0, 3.0])
        layer.update_parameters(new_params)
        assert np.allclose(layer.get_parameters(), new_params)

    def test_layer_call(self):
        """Test calling layer directly."""
        def circuit_builder(x, params):
            return UnifiedCircuit(n_qubits=2)

        layer = WorkflowQuantumLayer(
            circuit_builder=circuit_builder,
            n_qubits=2,
            n_parameters=3
        )
        x = np.array([0.5, 0.3])
        circuit = layer(x)
        assert isinstance(circuit, UnifiedCircuit)


class TestClassicalPostprocessor:
    """Test classical postprocessing."""

    def test_postprocessor_creation(self):
        """Test creating postprocessor."""
        postprocessor = ClassicalPostprocessor(
            processing_function=simple_postprocessing
        )
        assert postprocessor.processing_function == simple_postprocessing

    def test_postprocessor_process(self):
        """Test processing data."""
        postprocessor = ClassicalPostprocessor(
            processing_function=lambda x: x * 2
        )
        data = np.array([1.0, 2.0])
        result = postprocessor.process(data)
        assert np.allclose(result, np.array([2.0, 4.0]))

    def test_postprocessor_call(self):
        """Test calling postprocessor directly."""
        postprocessor = ClassicalPostprocessor(
            processing_function=lambda x: x * 2
        )
        data = np.array([1.0, 2.0])
        result = postprocessor(data)
        assert np.allclose(result, np.array([2.0, 4.0]))


class TestHybridModelFactory:
    """Test hybrid model factory."""

    def test_create_hybrid_model(self):
        """Test creating hybrid model with factory."""
        model = create_hybrid_model(
            quantum_circuit=simple_quantum_function
        )
        assert isinstance(model, QuantumClassicalHybrid)

    def test_create_hybrid_with_processing(self):
        """Test creating hybrid model with processing."""
        model = create_hybrid_model(
            quantum_circuit=simple_quantum_function,
            preprocessing=simple_preprocessing,
            postprocessing=simple_postprocessing
        )
        assert model.classical_preprocessor is not None
        assert model.classical_postprocessor is not None


class TestQuantumDataEncoder:
    """Test quantum data encoder."""

    def test_encoder_creation(self):
        """Test creating data encoder."""
        encoder = QuantumDataEncoder(
            encoding_function=simple_quantum_function,
            normalization='minmax'
        )
        assert encoder.normalization == 'minmax'

    def test_encoder_fit_minmax(self):
        """Test fitting minmax normalization."""
        encoder = QuantumDataEncoder(
            encoding_function=simple_quantum_function,
            normalization='minmax'
        )
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        encoder.fit(data)
        assert encoder._min is not None
        assert encoder._max is not None

    def test_encoder_normalize_minmax(self):
        """Test minmax normalization."""
        encoder = QuantumDataEncoder(
            encoding_function=simple_quantum_function,
            normalization='minmax'
        )
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        encoder.fit(data)
        normalized = encoder.normalize(np.array([2.0, 3.0]))
        assert 0 <= normalized[0] <= 1
        assert 0 <= normalized[1] <= 1

    def test_encoder_fit_standard(self):
        """Test fitting standard normalization."""
        encoder = QuantumDataEncoder(
            encoding_function=simple_quantum_function,
            normalization='standard'
        )
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        encoder.fit(data)
        assert encoder._mean is not None
        assert encoder._std is not None

    def test_encoder_encode(self):
        """Test encoding data."""
        encoder = QuantumDataEncoder(
            encoding_function=simple_quantum_function
        )
        data = np.array([0.5, 0.3])
        circuit = encoder.encode(data)
        assert isinstance(circuit, UnifiedCircuit)


class TestBatchProcessor:
    """Test batch processor."""

    def test_processor_creation(self):
        """Test creating batch processor."""
        processor = BatchProcessor(
            quantum_function=lambda x: x * 2,
            batch_size=32
        )
        assert processor.batch_size == 32

    def test_process_batch(self):
        """Test processing a batch."""
        processor = BatchProcessor(
            quantum_function=lambda x: x * 2,
            batch_size=2
        )
        batch = np.array([[1.0], [2.0]])
        results = processor.process_batch(batch)
        assert len(results) == 2

    def test_process_dataset(self):
        """Test processing full dataset."""
        processor = BatchProcessor(
            quantum_function=lambda x: x * 2,
            batch_size=2
        )
        dataset = np.array([[1.0], [2.0], [3.0], [4.0]])
        results = processor.process_dataset(dataset)
        assert len(results) == 4

    def test_processor_call(self):
        """Test calling processor directly."""
        processor = BatchProcessor(
            quantum_function=lambda x: x * 2,
            batch_size=2
        )
        dataset = np.array([[1.0], [2.0]])
        results = processor(dataset)
        assert len(results) == 2


class TestResultAggregator:
    """Test result aggregator."""

    def test_aggregator_creation(self):
        """Test creating aggregator."""
        aggregator = ResultAggregator(aggregation_method='mean')
        assert aggregator.aggregation_method == 'mean'

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        aggregator = ResultAggregator(aggregation_method='mean')
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        aggregated = aggregator.aggregate(results)
        assert np.allclose(aggregated, np.array([2.0, 3.0]))

    def test_aggregate_sum(self):
        """Test sum aggregation."""
        aggregator = ResultAggregator(aggregation_method='sum')
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        aggregated = aggregator.aggregate(results)
        assert np.allclose(aggregated, np.array([4.0, 6.0]))

    def test_aggregate_max(self):
        """Test max aggregation."""
        aggregator = ResultAggregator(aggregation_method='max')
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        aggregated = aggregator.aggregate(results)
        assert np.allclose(aggregated, np.array([3.0, 4.0]))

    def test_aggregator_call(self):
        """Test calling aggregator directly."""
        aggregator = ResultAggregator(aggregation_method='mean')
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        aggregated = aggregator(results)
        assert np.allclose(aggregated, np.array([2.0, 3.0]))


class TestDataPipeline:
    """Test data pipeline."""

    def test_pipeline_creation(self):
        """Test creating data pipeline."""
        pipeline = DataPipeline()
        assert pipeline.encoder is None
        assert pipeline.processor is None
        assert pipeline.aggregator is None

    def test_pipeline_with_components(self):
        """Test pipeline with all components."""
        encoder = QuantumDataEncoder(simple_quantum_function)
        processor = BatchProcessor(lambda x: x, batch_size=2)
        aggregator = ResultAggregator('mean')

        pipeline = DataPipeline(encoder, processor, aggregator)
        assert pipeline.encoder is not None
        assert pipeline.processor is not None
        assert pipeline.aggregator is not None

    def test_create_data_pipeline(self):
        """Test creating pipeline with factory."""
        pipeline = create_data_pipeline(
            processing_function=lambda x: np.array([1.0, 2.0]),
            aggregation_method='mean',
            batch_size=2
        )
        assert isinstance(pipeline, DataPipeline)


class TestParameterUpdate:
    """Test parameter update strategies."""

    def test_updater_creation(self):
        """Test creating parameter updater."""
        updater = ParameterUpdate(learning_rate=0.01, method='gradient_descent')
        assert updater.learning_rate == 0.01
        assert updater.method == 'gradient_descent'

    def test_gradient_descent_update(self):
        """Test gradient descent update."""
        updater = ParameterUpdate(learning_rate=0.1, method='gradient_descent')
        params = np.array([1.0, 2.0])
        gradients = np.array([0.5, 1.0])

        new_params = updater.update(params, gradients=gradients)
        expected = params - 0.1 * gradients
        assert np.allclose(new_params, expected)

    def test_momentum_update(self):
        """Test momentum update."""
        updater = ParameterUpdate(learning_rate=0.1, method='momentum')
        params = np.array([1.0, 2.0])
        gradients = np.array([0.5, 1.0])

        new_params = updater.update(params, gradients=gradients)
        assert len(new_params) == 2

    def test_adam_update(self):
        """Test Adam update."""
        updater = ParameterUpdate(learning_rate=0.1, method='adam')
        params = np.array([1.0, 2.0])
        gradients = np.array([0.5, 1.0])

        new_params = updater.update(params, gradients=gradients)
        assert len(new_params) == 2

    def test_updater_reset(self):
        """Test resetting updater state."""
        updater = ParameterUpdate(learning_rate=0.1, method='adam')
        params = np.array([1.0, 2.0])
        gradients = np.array([0.5, 1.0])

        updater.update(params, gradients=gradients)
        updater.reset()
        assert updater.m is None
        assert updater.v is None


class TestConvergenceChecker:
    """Test convergence checker."""

    def test_checker_creation(self):
        """Test creating convergence checker."""
        checker = ConvergenceChecker(tolerance=1e-6, patience=10)
        assert checker.tolerance == 1e-6
        assert checker.patience == 10

    def test_check_improvement(self):
        """Test checking for improvement."""
        checker = ConvergenceChecker(tolerance=1e-3, patience=2)

        assert not checker.check(loss=1.0)
        assert not checker.check(loss=0.5)  # Improvement
        assert not checker.check(loss=0.45)  # Small improvement

    def test_check_patience(self):
        """Test patience-based convergence."""
        checker = ConvergenceChecker(tolerance=1e-3, patience=2)

        checker.check(loss=1.0)
        checker.check(loss=1.0)  # No improvement
        converged = checker.check(loss=1.0)  # Patience exceeded
        assert converged

    def test_checker_reset(self):
        """Test resetting checker."""
        checker = ConvergenceChecker(tolerance=1e-3, patience=2)
        checker.check(loss=1.0)
        checker.reset()
        assert checker.best_loss == float('inf')
        assert len(checker.loss_history) == 0


class TestOptimizationLoop:
    """Test optimization loop."""

    def test_loop_creation(self):
        """Test creating optimization loop."""
        def loss_fn(params):
            return np.sum(params ** 2)

        updater = ParameterUpdate(learning_rate=0.1)
        loop = OptimizationLoop(
            loss_function=loss_fn,
            parameter_updater=updater,
            max_iterations=10
        )
        assert loop.max_iterations == 10

    def test_loop_run(self):
        """Test running optimization loop."""
        def loss_fn(params):
            return np.sum(params ** 2)

        def grad_fn(params):
            return 2 * params

        updater = ParameterUpdate(learning_rate=0.1, method='gradient_descent')
        checker = ConvergenceChecker(tolerance=1e-3, patience=5)
        loop = OptimizationLoop(
            loss_function=loss_fn,
            parameter_updater=updater,
            convergence_checker=checker,
            max_iterations=50
        )

        initial_params = np.array([1.0, 2.0])
        result = loop.run(initial_params, gradient_function=grad_fn)

        assert 'parameters' in result
        assert 'loss' in result
        assert 'loss_history' in result
        assert 'iterations' in result
        assert 'converged' in result

    def test_run_optimization_loop(self):
        """Test convenience function for optimization."""
        def loss_fn(params):
            return np.sum(params ** 2)

        def grad_fn(params):
            return 2 * params

        initial_params = np.array([1.0, 2.0])
        result = run_optimization_loop(
            loss_function=loss_fn,
            initial_parameters=initial_params,
            learning_rate=0.1,
            max_iterations=50,
            gradient_function=grad_fn
        )

        assert 'parameters' in result
        assert result['loss'] < 1.0  # Should decrease


class TestIntegration:
    """Integration tests for workflows."""

    def test_full_hybrid_pipeline(self):
        """Test complete hybrid workflow."""
        # Create hybrid model
        def quantum_func(x):
            return np.array([np.sum(x)])

        model = create_hybrid_model(
            quantum_circuit=quantum_func,
            preprocessing=lambda x: x / 2,
            postprocessing=lambda x: x * 2
        )

        # Test model
        x = np.array([1.0, 2.0])
        result = model(x)
        assert isinstance(result, np.ndarray)

    def test_data_pipeline_integration(self):
        """Test data pipeline integration."""
        def process_fn(x):
            return np.sum(x)

        pipeline = create_data_pipeline(
            processing_function=process_fn,
            aggregation_method='mean',
            batch_size=2
        )

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pipeline.run(data)
        assert isinstance(result, (np.ndarray, float, int))

    def test_optimization_integration(self):
        """Test optimization integration."""
        def loss_fn(params):
            return np.sum((params - np.array([1.0, 2.0])) ** 2)

        def grad_fn(params):
            return 2 * (params - np.array([1.0, 2.0]))

        result = run_optimization_loop(
            loss_function=loss_fn,
            initial_parameters=np.array([0.0, 0.0]),
            learning_rate=0.1,
            max_iterations=100,
            gradient_function=grad_fn
        )

        # Should converge close to [1.0, 2.0]
        assert np.allclose(result['parameters'], np.array([1.0, 2.0]), atol=0.3)
