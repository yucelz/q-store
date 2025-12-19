"""
Tests for quantum kernel methods.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.embeddings import ZZFeatureMap
from q_store.kernels import (
    QuantumKernel,
    compute_kernel_matrix,
    kernel_alignment,
    kernel_target_alignment,
    FidelityQuantumKernel,
    state_fidelity,
    compute_fidelity_kernel,
    ProjectedQuantumKernel,
    measurement_kernel,
    compute_projected_kernel,
    TrainableQuantumKernel,
    optimize_kernel_parameters,
    kernel_loss
)


def simple_feature_map(x):
    """Simple feature map for testing."""
    n_qubits = len(x)
    circuit = UnifiedCircuit(n_qubits=n_qubits)
    for i in range(n_qubits):
        circuit.add_gate(GateType.RY, targets=[i], parameters={'angle': x[i]})
    return circuit


class TestQuantumKernel:
    """Test basic quantum kernel."""

    def test_kernel_creation(self):
        """Test creating quantum kernel."""
        kernel = QuantumKernel(simple_feature_map, n_qubits=2)
        assert kernel.n_qubits == 2

    def test_kernel_evaluation(self):
        """Test evaluating kernel between two points."""
        kernel = QuantumKernel(simple_feature_map)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.6, 0.4])

        k_value = kernel.evaluate(x1, x2)
        assert isinstance(k_value, (float, np.floating))
        assert 0 <= k_value <= 1

    def test_kernel_matrix(self):
        """Test computing kernel matrix."""
        kernel = QuantumKernel(simple_feature_map)
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        K = kernel.compute_matrix(X)
        assert K.shape == (3, 3)
        # Kernel should be symmetric
        assert np.allclose(K, K.T, atol=1e-6)

    def test_kernel_diagonal(self):
        """Test that kernel diagonal is positive."""
        kernel = QuantumKernel(simple_feature_map)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])

        K = kernel.compute_matrix(X)
        diag = np.diag(K)
        assert np.all(diag > 0)

    def test_compute_kernel_matrix_function(self):
        """Test standalone kernel matrix function."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = compute_kernel_matrix(X, simple_feature_map)

        assert K.shape == (2, 2)

    def test_rectangular_kernel_matrix(self):
        """Test kernel matrix for different X and Y."""
        kernel = QuantumKernel(simple_feature_map)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        Y = np.array([[0.5, 0.6]])

        K = kernel.compute_matrix(X, Y)
        assert K.shape == (2, 1)


class TestKernelMetrics:
    """Test kernel alignment and metrics."""

    def test_kernel_alignment(self):
        """Test kernel alignment computation."""
        K1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        K2 = np.array([[1.0, 0.6], [0.6, 1.0]])

        alignment = kernel_alignment(K1, K2)
        assert 0 <= alignment <= 1

    def test_identical_kernels_alignment(self):
        """Test that identical kernels have alignment 1."""
        K = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])

        alignment = kernel_alignment(K, K)
        assert np.isclose(alignment, 1.0, atol=1e-6)

    def test_kernel_target_alignment(self):
        """Test kernel-target alignment."""
        K = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        y = np.array([1, 1, -1])

        kta = kernel_target_alignment(K, y)
        assert isinstance(kta, float)

    def test_zero_kernel_alignment(self):
        """Test alignment with zero kernel."""
        K1 = np.zeros((3, 3))
        K2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        alignment = kernel_alignment(K1, K2)
        assert alignment == 0.0


class TestFidelityKernel:
    """Test fidelity-based quantum kernel."""

    def test_fidelity_kernel_creation(self):
        """Test creating fidelity kernel."""
        kernel = FidelityQuantumKernel(simple_feature_map, n_qubits=2)
        assert kernel.n_qubits == 2

    def test_fidelity_evaluation(self):
        """Test fidelity kernel evaluation."""
        kernel = FidelityQuantumKernel(simple_feature_map)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.6, 0.4])

        fidelity = kernel.evaluate(x1, x2)
        assert 0 <= fidelity <= 1

    def test_state_fidelity(self):
        """Test state fidelity function."""
        state1 = np.array([1.0, 0.0])
        state2 = np.array([1.0, 0.0])

        fidelity = state_fidelity(state1, state2)
        assert np.isclose(fidelity, 1.0)

    def test_orthogonal_states_fidelity(self):
        """Test fidelity of orthogonal states."""
        state1 = np.array([1.0, 0.0])
        state2 = np.array([0.0, 1.0])

        fidelity = state_fidelity(state1, state2)
        assert np.isclose(fidelity, 0.0)

    def test_compute_fidelity_kernel(self):
        """Test computing fidelity kernel matrix."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = compute_fidelity_kernel(X, simple_feature_map)

        assert K.shape == (2, 2)

    def test_measurement_based_fidelity(self):
        """Test measurement-based fidelity estimation."""
        kernel = FidelityQuantumKernel(
            simple_feature_map,
            use_measurement=True
        )
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.6, 0.4])

        fidelity = kernel.evaluate(x1, x2)
        assert 0 <= fidelity <= 1


class TestProjectedKernel:
    """Test projected quantum kernel."""

    def test_projected_kernel_creation(self):
        """Test creating projected kernel."""
        kernel = ProjectedQuantumKernel(
            simple_feature_map,
            n_qubits=2,
            measurement_basis="computational"
        )
        assert kernel.measurement_basis == "computational"

    def test_projected_kernel_evaluation(self):
        """Test projected kernel evaluation."""
        kernel = ProjectedQuantumKernel(simple_feature_map)
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.6, 0.4])

        k_value = kernel.evaluate(x1, x2)
        assert isinstance(k_value, (float, np.floating))

    def test_measurement_basis_x(self):
        """Test projected kernel with X basis."""
        kernel = ProjectedQuantumKernel(
            simple_feature_map,
            measurement_basis="X"
        )
        x1 = np.array([0.5, 0.3])
        x2 = np.array([0.6, 0.4])

        k_value = kernel.evaluate(x1, x2)
        assert 0 <= k_value <= 1

    def test_measurement_kernel_function(self):
        """Test measurement kernel function."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = measurement_kernel(X, simple_feature_map)

        assert K.shape == (2, 2)

    def test_compute_projected_kernel(self):
        """Test projected kernel with multiple observables."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        observables = ["computational", "X"]

        K = compute_projected_kernel(X, simple_feature_map, observables)
        assert K.shape == (2, 2)


class TestTrainableKernel:
    """Test trainable quantum kernel."""

    def test_trainable_kernel_creation(self):
        """Test creating trainable kernel."""
        def param_feature_map(x, params):
            circuit = UnifiedCircuit(n_qubits=len(x))
            for i in range(len(x)):
                angle = x[i] * params[i]
                circuit.add_gate(GateType.RY, targets=[i], parameters={'angle': angle})
            return circuit

        initial_params = np.array([1.0, 1.0])
        kernel = TrainableQuantumKernel(param_feature_map, initial_params)

        assert len(kernel.get_parameters()) == 2

    def test_parameter_update(self):
        """Test updating kernel parameters."""
        def param_feature_map(x, params):
            circuit = UnifiedCircuit(n_qubits=len(x))
            for i in range(len(x)):
                circuit.add_gate(GateType.RY, targets=[i], parameters={'angle': params[i]})
            return circuit

        kernel = TrainableQuantumKernel(param_feature_map, np.array([1.0, 1.0]))

        new_params = np.array([2.0, 3.0])
        kernel.update_parameters(new_params)

        assert np.allclose(kernel.get_parameters(), new_params)

    def test_kernel_training(self):
        """Test training kernel parameters."""
        def param_feature_map(x, params):
            circuit = UnifiedCircuit(n_qubits=len(x))
            for i in range(len(x)):
                circuit.add_gate(GateType.RY, targets=[i], parameters={'angle': x[i] * params[i]})
            return circuit

        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        y = np.array([1, 1, -1])

        kernel = TrainableQuantumKernel(param_feature_map, np.array([1.0, 1.0]))
        result = kernel.train(X, y, max_iter=10)

        assert 'final_parameters' in result
        assert 'final_loss' in result

    def test_kernel_loss_alignment(self):
        """Test kernel loss with alignment."""
        K = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        y = np.array([1, 1, -1])

        loss = kernel_loss(K, y, loss_type="alignment")
        assert isinstance(loss, float)

    def test_kernel_loss_margin(self):
        """Test kernel loss with margin."""
        K = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        y = np.array([1, 1, -1])

        loss = kernel_loss(K, y, loss_type="margin")
        assert isinstance(loss, float)

    def test_optimize_kernel_parameters(self):
        """Test parameter optimization function."""
        def param_feature_map(x, params):
            circuit = UnifiedCircuit(n_qubits=len(x))
            for i in range(len(x)):
                circuit.add_gate(GateType.RY, targets=[i], parameters={'angle': x[i] * params[i]})
            return circuit

        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([1, -1])

        params, loss = optimize_kernel_parameters(
            X, y, param_feature_map, np.array([1.0, 1.0]), max_iter=5
        )

        assert len(params) == 2
        assert isinstance(loss, float)


class TestKernelIntegration:
    """Integration tests for quantum kernels."""

    def test_kernel_with_zz_feature_map(self):
        """Test kernel with ZZ feature map."""
        def zz_map(x):
            # ZZFeatureMap.encode(data) returns circuit
            feature_map = ZZFeatureMap(n_features=len(x), reps=1)
            return feature_map.encode(x)
        
        kernel = QuantumKernel(zz_map)
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        K = kernel.compute_matrix(X)
        assert K.shape == (2, 2)
    
    def test_multiple_kernel_types(self):
        """Test creating different kernel types."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])

        quantum_k = QuantumKernel(simple_feature_map)
        fidelity_k = FidelityQuantumKernel(simple_feature_map)
        projected_k = ProjectedQuantumKernel(simple_feature_map)

        K1 = quantum_k.compute_matrix(X)
        K2 = fidelity_k.compute_matrix(X)
        K3 = projected_k.compute_matrix(X)

        assert K1.shape == K2.shape == K3.shape == (2, 2)

    def test_kernel_classification_pipeline(self):
        """Test simple classification pipeline with quantum kernel."""
        X_train = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
        y_train = np.array([1, 1, -1, -1])

        kernel = QuantumKernel(simple_feature_map)
        K = kernel.compute_matrix(X_train)

        # Check kernel quality with target alignment
        kta = kernel_target_alignment(K, y_train)
        assert isinstance(kta, float)
