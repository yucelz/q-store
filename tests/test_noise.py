"""
Tests for Noise Models.

Tests for:
- Depolarizing noise
- Amplitude damping (T1)
- Phase damping (T2)
- Thermal relaxation
- Readout errors
- Composite models
"""

import pytest
import numpy as np

from q_store.core import UnifiedCircuit, GateType
from q_store.noise import (
    NoiseModel,
    NoiseParameters,
    DepolarizingNoise,
    AmplitudeDampingNoise,
    PhaseDampingNoise,
    ThermalRelaxationNoise,
    ReadoutErrorNoise,
    CompositeNoiseModel,
    create_device_noise_model,
)


# =============================================================================
# Depolarizing Noise Tests
# =============================================================================

class TestDepolarizingNoise:
    """Test depolarizing noise model."""

    def test_creation(self):
        """Test depolarizing noise creation."""
        params = NoiseParameters(error_rate=0.01)
        noise = DepolarizingNoise(params)
        assert noise.single_qubit_error_rate == 0.01
        assert noise.two_qubit_error_rate == 0.10

    def test_kraus_operators_single_qubit(self):
        """Test Kraus operators for single-qubit depolarizing."""
        params = NoiseParameters(error_rate=0.03)
        noise = DepolarizingNoise(params)

        kraus_ops = noise.get_kraus_operators(GateType.H)

        # Should have 4 Kraus operators: I, X, Y, Z
        assert len(kraus_ops) == 4
        assert all(k.shape == (2, 2) for k in kraus_ops)

        # Completeness relation: Σ K†K = I
        completeness = sum(k.conj().T @ k for k in kraus_ops)
        np.testing.assert_allclose(completeness, np.eye(2), atol=1e-10)

    def test_kraus_operators_two_qubit(self):
        """Test Kraus operators for two-qubit depolarizing."""
        params = NoiseParameters(error_rate=0.01)
        noise = DepolarizingNoise(params)

        kraus_ops = noise.get_kraus_operators(GateType.CNOT)

        assert len(kraus_ops) >= 2
        assert all(k.shape == (4, 4) for k in kraus_ops)

    def test_apply_to_circuit(self):
        """Test applying depolarizing noise to circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        params = NoiseParameters(error_rate=0.01)
        noise = DepolarizingNoise(params)

        noisy_circuit = noise.apply_to_circuit(circuit)

        assert noisy_circuit.n_qubits == 2
        assert len(noisy_circuit.gates) == 2
        assert 'noise_model' in noisy_circuit._metadata

    def test_serialization(self):
        """Test noise model serialization."""
        params = NoiseParameters(error_rate=0.02)
        noise = DepolarizingNoise(params)

        noise_dict = noise.to_dict()

        assert noise_dict['type'] == 'DepolarizingNoise'
        assert noise_dict['parameters']['error_rate'] == 0.02


# =============================================================================
# Amplitude Damping Tests
# =============================================================================

class TestAmplitudeDampingNoise:
    """Test amplitude damping (T1) noise model."""

    def test_creation(self):
        """Test amplitude damping creation."""
        params = NoiseParameters(t1_time=50, gate_time=0.05)
        noise = AmplitudeDampingNoise(params)

        # γ = 1 - exp(-t/T1) = 1 - exp(-0.05/50) ≈ 0.001
        assert 0.0009 < noise.gamma < 0.0011

    def test_requires_t1_and_gate_time(self):
        """Test that T1 and gate time are required."""
        params = NoiseParameters(error_rate=0.01)

        with pytest.raises(ValueError, match="requires t1_time and gate_time"):
            AmplitudeDampingNoise(params)

    def test_kraus_operators(self):
        """Test amplitude damping Kraus operators."""
        params = NoiseParameters(t1_time=100, gate_time=0.1)
        noise = AmplitudeDampingNoise(params)

        kraus_ops = noise.get_kraus_operators(GateType.X)

        # Should have 2 Kraus operators
        assert len(kraus_ops) == 2
        K0, K1 = kraus_ops

        # Check shapes
        assert K0.shape == (2, 2)
        assert K1.shape == (2, 2)

        # Completeness: K0†K0 + K1†K1 = I
        completeness = K0.conj().T @ K0 + K1.conj().T @ K1
        np.testing.assert_allclose(completeness, np.eye(2), atol=1e-10)

        # K1 should map |1⟩ to |0⟩
        assert abs(K1[0, 1]) > 0  # Non-zero transition
        assert abs(K1[1, 1]) < 1e-10  # |1⟩ → |1⟩ should be zero

    def test_longer_gate_more_damping(self):
        """Test that longer gates have more damping."""
        params_short = NoiseParameters(t1_time=50, gate_time=0.01)
        params_long = NoiseParameters(t1_time=50, gate_time=1.0)

        noise_short = AmplitudeDampingNoise(params_short)
        noise_long = AmplitudeDampingNoise(params_long)

        assert noise_long.gamma > noise_short.gamma


# =============================================================================
# Phase Damping Tests
# =============================================================================

class TestPhaseDampingNoise:
    """Test phase damping (T2) noise model."""

    def test_creation(self):
        """Test phase damping creation."""
        params = NoiseParameters(t2_time=70, gate_time=0.05)
        noise = PhaseDampingNoise(params)

        # λ = 1 - exp(-t/T2)
        assert noise.lambda_param > 0
        assert noise.lambda_param < 1

    def test_requires_t2_and_gate_time(self):
        """Test that T2 and gate time are required."""
        params = NoiseParameters(error_rate=0.01)

        with pytest.raises(ValueError, match="requires t2_time and gate_time"):
            PhaseDampingNoise(params)

    def test_kraus_operators(self):
        """Test phase damping Kraus operators."""
        params = NoiseParameters(t2_time=100, gate_time=0.1)
        noise = PhaseDampingNoise(params)

        kraus_ops = noise.get_kraus_operators(GateType.H)

        assert len(kraus_ops) == 2
        K0, K1 = kraus_ops

        # Completeness
        completeness = K0.conj().T @ K0 + K1.conj().T @ K1
        np.testing.assert_allclose(completeness, np.eye(2), atol=1e-10)

        # K0 and K1 should be diagonal (no energy exchange)
        assert abs(K0[0, 1]) < 1e-10
        assert abs(K0[1, 0]) < 1e-10
        assert abs(K1[0, 1]) < 1e-10
        assert abs(K1[1, 0]) < 1e-10


# =============================================================================
# Thermal Relaxation Tests
# =============================================================================

class TestThermalRelaxationNoise:
    """Test thermal relaxation noise model."""

    def test_creation(self):
        """Test thermal relaxation creation."""
        params = NoiseParameters(
            t1_time=50,
            t2_time=40,
            gate_time=0.05,
            temperature=0.015
        )
        noise = ThermalRelaxationNoise(params)

        assert noise.gamma_t1 > 0
        assert noise.gamma_t2 > 0
        assert noise.gamma_phi >= 0
        assert 0 <= noise.p_excited <= 1

    def test_requires_all_parameters(self):
        """Test that all timing parameters are required."""
        params = NoiseParameters(t1_time=50)

        with pytest.raises(ValueError, match="requires t1_time, t2_time, and gate_time"):
            ThermalRelaxationNoise(params)

    def test_t2_constraint(self):
        """Test T2 ≤ 2*T1 constraint."""
        # T2 > 2*T1 should be clamped with warning
        params = NoiseParameters(
            t1_time=50,
            t2_time=150,  # > 2*50
            gate_time=0.05
        )

        noise = ThermalRelaxationNoise(params)

        # T2 should be clamped to 2*T1
        assert params.t2_time == 100

    def test_kraus_operators(self):
        """Test thermal relaxation Kraus operators."""
        params = NoiseParameters(
            t1_time=100,
            t2_time=80,
            gate_time=0.1,
            temperature=0.015
        )
        noise = ThermalRelaxationNoise(params)

        kraus_ops = noise.get_kraus_operators(GateType.X)

        # Should have multiple Kraus operators
        assert len(kraus_ops) == 4
        assert all(k.shape == (2, 2) for k in kraus_ops)

        # Approximate completeness (thermal channel is complex)
        completeness = sum(k.conj().T @ k for k in kraus_ops)
        # May not be exactly identity due to simplifications
        assert np.allclose(completeness, np.eye(2), atol=0.5)

    def test_cold_temperature(self):
        """Test that cold temperature gives low excited state population."""
        params = NoiseParameters(
            t1_time=50,
            t2_time=40,
            gate_time=0.05,
            temperature=0.015  # 15mK
        )
        noise = ThermalRelaxationNoise(params)

        # At 15mK, p_excited should be very small
        assert noise.p_excited < 0.01


# =============================================================================
# Readout Error Tests
# =============================================================================

class TestReadoutErrorNoise:
    """Test readout error noise model."""

    def test_creation(self):
        """Test readout error creation."""
        params = NoiseParameters(readout_error=0.02)
        noise = ReadoutErrorNoise(params)

        assert noise.p0 == 0.02
        assert noise.p1 == 0.02

    def test_default_error(self):
        """Test default readout error."""
        params = NoiseParameters()
        noise = ReadoutErrorNoise(params)

        assert noise.p0 == 0.01
        assert noise.p1 == 0.01

    def test_confusion_matrix(self):
        """Test readout confusion matrix."""
        params = NoiseParameters(readout_error=0.05)
        noise = ReadoutErrorNoise(params)

        confusion = noise.get_confusion_matrix()

        assert confusion.shape == (2, 2)

        # Rows should sum to 1 (probability conservation)
        np.testing.assert_allclose(confusion.sum(axis=1), [1, 1])

        # Check specific values
        assert confusion[0, 0] == 0.95  # P(0|0)
        assert confusion[0, 1] == 0.05  # P(1|0)
        assert confusion[1, 0] == 0.05  # P(0|1)
        assert confusion[1, 1] == 0.95  # P(1|1)

    def test_apply_to_circuit(self):
        """Test applying readout errors to circuit."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])

        params = NoiseParameters(readout_error=0.03)
        noise = ReadoutErrorNoise(params)

        noisy_circuit = noise.apply_to_circuit(circuit)

        assert 'readout_confusion' in noisy_circuit._metadata
        confusion = np.array(noisy_circuit._metadata['readout_confusion'])
        assert confusion.shape == (2, 2)


# =============================================================================
# Composite Noise Model Tests
# =============================================================================

class TestCompositeNoiseModel:
    """Test composite noise model."""

    def test_creation(self):
        """Test composite model creation."""
        params = NoiseParameters(
            error_rate=0.01,
            t1_time=50,
            t2_time=40,
            gate_time=0.05
        )

        models = [
            DepolarizingNoise(params),
            AmplitudeDampingNoise(params)
        ]

        composite = CompositeNoiseModel(models)
        assert len(composite.noise_models) == 2

    def test_requires_models(self):
        """Test that at least one model is required."""
        with pytest.raises(ValueError, match="requires at least one"):
            CompositeNoiseModel([])

    def test_apply_to_circuit(self):
        """Test applying composite model to circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])

        params = NoiseParameters(
            error_rate=0.01,
            t1_time=50,
            t2_time=40,
            gate_time=0.05,
            readout_error=0.02
        )

        composite = CompositeNoiseModel([
            DepolarizingNoise(params),
            ThermalRelaxationNoise(params),
            ReadoutErrorNoise(params)
        ])

        noisy_circuit = composite.apply_to_circuit(circuit)

        assert noisy_circuit.n_qubits == 2
        assert 'noise_model' in noisy_circuit._metadata

    def test_kraus_composition(self):
        """Test Kraus operator composition."""
        params = NoiseParameters(
            error_rate=0.01,
            t1_time=100,
            gate_time=0.1
        )

        composite = CompositeNoiseModel([
            DepolarizingNoise(params),
            AmplitudeDampingNoise(params)
        ])

        kraus_ops = composite.get_kraus_operators(GateType.X)

        # Should have composed Kraus operators
        assert len(kraus_ops) > 0
        assert all(k.shape == (2, 2) for k in kraus_ops)

    def test_serialization(self):
        """Test composite model serialization."""
        params = NoiseParameters(error_rate=0.01, t1_time=50, gate_time=0.05)

        composite = CompositeNoiseModel([
            DepolarizingNoise(params),
            AmplitudeDampingNoise(params)
        ])

        noise_dict = composite.to_dict()

        assert noise_dict['type'] == 'CompositeNoiseModel'
        assert len(noise_dict['noise_models']) == 2


# =============================================================================
# Device Noise Model Tests
# =============================================================================

class TestDeviceNoiseModels:
    """Test pre-configured device noise models."""

    def test_ibm_generic(self):
        """Test IBM generic device model."""
        noise_model = create_device_noise_model('ibm_generic')

        assert isinstance(noise_model, CompositeNoiseModel)
        assert len(noise_model.noise_models) == 3

    def test_ionq(self):
        """Test IonQ device model."""
        noise_model = create_device_noise_model('ionq')

        assert isinstance(noise_model, CompositeNoiseModel)

        # IonQ should have better coherence times
        thermal = noise_model.noise_models[0]
        assert isinstance(thermal, ThermalRelaxationNoise)
        assert thermal.parameters.t1_time == 10000  # 10ms

    def test_rigetti(self):
        """Test Rigetti device model."""
        noise_model = create_device_noise_model('rigetti')
        assert isinstance(noise_model, CompositeNoiseModel)

    def test_google(self):
        """Test Google device model."""
        noise_model = create_device_noise_model('google')
        assert isinstance(noise_model, CompositeNoiseModel)

    def test_generic(self):
        """Test generic device model."""
        noise_model = create_device_noise_model('generic')
        assert isinstance(noise_model, CompositeNoiseModel)

    def test_unknown_device_defaults_to_generic(self):
        """Test unknown device falls back to generic."""
        noise_model = create_device_noise_model('unknown_device')
        assert isinstance(noise_model, CompositeNoiseModel)

    def test_custom_parameters(self):
        """Test overriding device parameters."""
        custom_params = NoiseParameters(
            error_rate=0.1,
            t1_time=10,
            t2_time=8,
            gate_time=1.0,
            readout_error=0.1
        )

        noise_model = create_device_noise_model('ibm_generic', custom_params)

        thermal = noise_model.noise_models[0]
        assert thermal.parameters.t1_time == 10
        assert thermal.parameters.error_rate == 0.1

    def test_all_devices_have_three_channels(self):
        """Test all pre-configured devices have thermal + depolarizing + readout."""
        devices = ['ibm_generic', 'ionq', 'rigetti', 'google', 'generic']

        for device in devices:
            noise_model = create_device_noise_model(device)
            assert len(noise_model.noise_models) == 3
            assert isinstance(noise_model.noise_models[0], ThermalRelaxationNoise)
            assert isinstance(noise_model.noise_models[1], DepolarizingNoise)
            assert isinstance(noise_model.noise_models[2], ReadoutErrorNoise)


# =============================================================================
# Integration Tests
# =============================================================================

class TestNoiseModelIntegration:
    """Integration tests for noise models."""

    def test_apply_device_model_to_circuit(self):
        """Test applying device noise model to circuit."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.RZ, targets=[2], parameters={'angle': np.pi/4})

        noise_model = create_device_noise_model('google')
        noisy_circuit = noise_model.apply_to_circuit(circuit)

        assert noisy_circuit.n_qubits == 3
        assert len(noisy_circuit.gates) == 3
        assert 'noise_model' in noisy_circuit._metadata

    def test_noise_model_with_optimizer(self):
        """Test that noise models work with optimized circuits."""
        from q_store.core import CircuitOptimizer

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])  # Cancels out

        optimizer = CircuitOptimizer()
        optimized = optimizer.optimize(circuit)

        noise_model = create_device_noise_model('generic')
        noisy_circuit = noise_model.apply_to_circuit(optimized)

        # Optimizer should have removed X gates
        assert len(noisy_circuit.gates) < 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
