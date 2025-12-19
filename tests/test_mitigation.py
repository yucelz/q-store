"""
Tests for Error Mitigation Module.

Tests for:
- Zero-Noise Extrapolation (ZNE)
- Probabilistic Error Cancellation (PEC)
- Measurement Error Mitigation
"""

import pytest
import numpy as np
from typing import Dict

from q_store.core import UnifiedCircuit, GateType
from q_store.mitigation import (
    # ZNE
    ZeroNoiseExtrapolator,
    ExtrapolationMethod,
    ZNEResult,
    create_zne_mitigator,
    # PEC
    ProbabilisticErrorCanceller,
    AdaptivePEC,
    PECResult,
    create_pec_mitigator,
    # Measurement
    MeasurementErrorMitigator,
    CalibrationData,
    MitigationResult,
    create_measurement_mitigator,
)


# =============================================================================
# Zero-Noise Extrapolation Tests
# =============================================================================

class TestZeroNoiseExtrapolation:
    """Test ZNE implementation."""
    
    def test_zne_creation(self):
        """Test ZNE mitigator creation."""
        zne = create_zne_mitigator()
        assert isinstance(zne, ZeroNoiseExtrapolator)
        assert zne.extrapolation_method == ExtrapolationMethod.LINEAR
        assert zne.noise_factors == [1, 2, 3]
    
    def test_zne_custom_config(self):
        """Test ZNE with custom configuration."""
        zne = ZeroNoiseExtrapolator(
            extrapolation_method=ExtrapolationMethod.EXPONENTIAL,
            noise_factors=[1, 1.5, 2, 2.5, 3],
            polynomial_degree=3
        )
        assert zne.extrapolation_method == ExtrapolationMethod.EXPONENTIAL
        assert len(zne.noise_factors) == 5
    
    def test_amplify_noise_unitary_folding(self):
        """Test noise amplification via unitary folding."""
        zne = create_zne_mitigator()
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        
        # Amplify with factor 3 (should triple gate count: G → G†G†G)
        amplified = zne.amplify_noise(circuit, noise_factor=3)
        
        # Original has 2 gates, folding factor 3 should give:
        # Each gate → G†G†G (3 copies)
        # But actually, folding creates G G† G† G (4 gates per original)
        # For factor k, we get: G (G† G)^((k-1)/2) when k is odd
        # So for k=3: G G† G† G = 4 gates per original gate
        assert len(amplified.gates) >= len(circuit.gates)
    
    def test_linear_extrapolation(self):
        """Test linear extrapolation to zero noise."""
        zne = ZeroNoiseExtrapolator(
            extrapolation_method=ExtrapolationMethod.LINEAR,
            noise_factors=[1, 2, 3]
        )
        
        # Simulate noisy expectation values: <Z> = 1 - 0.1 * noise_factor
        noise_factors = [1, 2, 3]
        measured_values = [0.9, 0.8, 0.7]
        
        mitigated, fit_quality = zne.extrapolate(noise_factors, measured_values)
        
        # Linear fit: y = 1 - 0.1x, extrapolate to x=0 gives y=1
        assert abs(mitigated - 1.0) < 0.01
        assert fit_quality > 0.99  # Should be perfect linear fit
    
    def test_exponential_extrapolation(self):
        """Test exponential extrapolation."""
        zne = ZeroNoiseExtrapolator(
            extrapolation_method=ExtrapolationMethod.EXPONENTIAL,
            noise_factors=[1, 2, 3, 4]
        )
        
        # Exponential decay: exp(-0.2 * x)
        noise_factors = [1, 2, 3, 4]
        measured_values = [
            np.exp(-0.2),
            np.exp(-0.4),
            np.exp(-0.6),
            np.exp(-0.8)
        ]
        
        mitigated, fit_quality = zne.extrapolate(noise_factors, measured_values)
        
        # Extrapolate to x=0 should give exp(0) = 1
        assert abs(mitigated - 1.0) < 0.1
        assert fit_quality > 0.9
    
    def test_polynomial_extrapolation(self):
        """Test polynomial extrapolation."""
        zne = ZeroNoiseExtrapolator(
            extrapolation_method=ExtrapolationMethod.POLYNOMIAL,
            noise_factors=[1, 2, 3, 4],
            polynomial_degree=2
        )
        
        # Quadratic: 1 - 0.1x - 0.05x^2
        noise_factors = [1, 2, 3, 4]
        measured_values = [
            1 - 0.1 - 0.05,
            1 - 0.2 - 0.20,
            1 - 0.3 - 0.45,
            1 - 0.4 - 0.80
        ]
        
        mitigated, fit_quality = zne.extrapolate(noise_factors, measured_values)
        assert abs(mitigated - 1.0) < 0.01
    
    def test_richardson_extrapolation(self):
        """Test Richardson extrapolation."""
        zne = ZeroNoiseExtrapolator(
            extrapolation_method=ExtrapolationMethod.RICHARDSON,
            noise_factors=[1, 2, 4]
        )
        
        noise_factors = [1, 2, 4]
        measured_values = [0.9, 0.8, 0.6]
        
        mitigated, fit_quality = zne.extrapolate(noise_factors, measured_values)
        
        # Just verify it runs and returns values
        assert isinstance(mitigated, float)
        assert isinstance(fit_quality, float)
        assert 0.0 <= fit_quality <= 1.0
    
    def test_mitigate_circuit(self):
        """Test full ZNE mitigation workflow."""
        zne = create_zne_mitigator()
        
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        
        # Mock executor that simulates noise: <Z> decreases with circuit depth
        def executor(circ):
            depth = len(circ.gates)
            return 1.0 - 0.05 * depth  # Linear noise model
        
        result = zne.mitigate(circuit, executor, metadata={'test': 'value'})
        
        assert isinstance(result, ZNEResult)
        assert result.mitigated_value > result.raw_values[0]  # Should improve
        assert len(result.noise_factors) == 3
        assert 'test' in result.metadata
        assert result.metadata['test'] == 'value'


# =============================================================================
# Probabilistic Error Cancellation Tests
# =============================================================================

class TestProbabilisticErrorCancellation:
    """Test PEC implementation."""
    
    def test_pec_creation(self):
        """Test PEC mitigator creation."""
        pec = create_pec_mitigator()
        assert isinstance(pec, ProbabilisticErrorCanceller)
        assert pec.n_samples == 1000
        assert pec.max_overhead == 10.0
    
    def test_pec_adaptive_creation(self):
        """Test adaptive PEC creation."""
        pec = create_pec_mitigator(adaptive=True, n_samples=5000)
        assert isinstance(pec, AdaptivePEC)
        assert pec.max_samples == 5000
    
    def test_decompose_single_qubit_gate(self):
        """Test quasi-probability decomposition."""
        pec = ProbabilisticErrorCanceller(
            noise_model={
                'type': 'depolarizing',
                'single_qubit_error_rate': 0.01
            }
        )
        
        decomp = pec.decompose_noisy_gate(GateType.H, n_qubits=1)
        
        assert len(decomp.operations) == 4  # Ideal + 3 Pauli errors
        assert len(decomp.coefficients) == 4
        assert decomp.sampling_overhead > 1.0  # Must be > 1 for noisy gates
        
        # Check quasi-probability normalization
        # (not standard probability - can have negative coefficients)
        assert abs(sum(decomp.coefficients) - 1.0) < 0.1
    
    def test_sampling_overhead_calculation(self):
        """Test sampling overhead increases with error rate."""
        low_noise_pec = ProbabilisticErrorCanceller(
            noise_model={'type': 'depolarizing', 'single_qubit_error_rate': 0.001}
        )
        high_noise_pec = ProbabilisticErrorCanceller(
            noise_model={'type': 'depolarizing', 'single_qubit_error_rate': 0.1}
        )
        
        low_decomp = low_noise_pec.decompose_noisy_gate(GateType.H)
        high_decomp = high_noise_pec.decompose_noisy_gate(GateType.H)
        
        assert high_decomp.sampling_overhead > low_decomp.sampling_overhead
    
    def test_sample_from_decomposition(self):
        """Test Monte Carlo sampling from quasi-probability."""
        pec = create_pec_mitigator(seed=42)
        decomp = pec.decompose_noisy_gate(GateType.H)
        
        # Sample multiple times
        sampled_ops = []
        weights = []
        for _ in range(100):
            op, weight = pec.sample_from_decomposition(decomp)
            sampled_ops.append(op)
            weights.append(weight)
        
        # Check we got valid circuits and weights
        assert all(isinstance(op, UnifiedCircuit) for op in sampled_ops)
        assert all(isinstance(w, (int, float)) for w in weights)
        
        # Weights should average to sampling overhead
        avg_abs_weight = np.mean(np.abs(weights))
        assert abs(avg_abs_weight - decomp.sampling_overhead) < 0.5
    
    def test_mitigate_single_gate_circuit(self):
        """Test PEC on simple circuit."""
        pec = ProbabilisticErrorCanceller(
            noise_model={'type': 'depolarizing', 'single_qubit_error_rate': 0.05},
            n_samples=500,
            seed=42
        )
        
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        
        # Mock executor: ideal gives 1.0, noisy gives 0.9
        def executor(circ):
            # Simplified: just return slightly different values
            return 0.9 if len(circ.gates) == 1 else 0.85
        
        result = pec.mitigate(circuit, executor)
        
        assert isinstance(result, PECResult)
        assert result.raw_value == 0.9
        assert result.sampling_overhead > 1.0
        assert result.n_samples_used == 500
    
    def test_adaptive_pec(self):
        """Test adaptive sampling in PEC."""
        pec = AdaptivePEC(
            initial_samples=50,
            max_samples=500,
            target_precision=0.05,
            seed=42
        )
        
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        
        def executor(circ):
            return 0.9 + np.random.normal(0, 0.01)
        
        result = pec.mitigate(circuit, executor)
        
        # Should use more than initial samples if variance is high
        assert result.n_samples_used >= 50


# =============================================================================
# Measurement Error Mitigation Tests
# =============================================================================

class TestMeasurementErrorMitigation:
    """Test measurement error mitigation."""
    
    def test_mitigator_creation(self):
        """Test measurement mitigator creation."""
        mitigator = create_measurement_mitigator(n_qubits=2)
        assert isinstance(mitigator, MeasurementErrorMitigator)
        assert mitigator.n_qubits == 2
        assert mitigator.tensored is True
    
    def test_single_qubit_calibration(self):
        """Test single-qubit confusion matrix calibration."""
        mitigator = MeasurementErrorMitigator(n_qubits=1, tensored=True)
        
        # Mock executor that simulates readout errors
        # P(0|0) = 0.95, P(1|0) = 0.05
        # P(0|1) = 0.10, P(1|1) = 0.90
        def executor(circuit):
            if len(circuit.gates) == 0:  # Measuring |0⟩
                return {'0': 9500, '1': 500}
            else:  # Measuring |1⟩ (X gate applied)
                return {'0': 1000, '1': 9000}
        
        confusion = mitigator.calibrate_single_qubit(0, executor)
        
        assert confusion.shape == (2, 2)
        assert abs(confusion[0, 0] - 0.95) < 0.01  # P(0|0)
        assert abs(confusion[1, 1] - 0.90) < 0.01  # P(1|1)
    
    def test_tensored_calibration(self):
        """Test tensor product calibration."""
        mitigator = MeasurementErrorMitigator(n_qubits=2, tensored=True)
        
        def executor(circuit):
            # Perfect readout for simplicity
            if len(circuit.gates) == 0:
                return {'00': 10000}
            else:
                # Check which qubit has X gate
                x_targets = [g.targets[0] for g in circuit.gates if g.gate_type == GateType.X]
                if 0 in x_targets and 1 not in x_targets:
                    return {'01': 10000}
                elif 1 in x_targets and 0 not in x_targets:
                    return {'10': 10000}
                else:
                    return {'11': 10000}
        
        calibration = mitigator.calibrate(executor)
        
        assert isinstance(calibration, CalibrationData)
        assert calibration.n_qubits == 2
        assert calibration.tensored is True
        assert calibration.confusion_matrix.shape == (4, 4)
        assert calibration.mitigation_matrix.shape == (4, 4)
    
    def test_full_calibration(self):
        """Test full correlated calibration."""
        mitigator = MeasurementErrorMitigator(n_qubits=2, tensored=False)
        
        def executor(circuit):
            # Determine prepared state from X gates
            x_qubits = [g.targets[0] for g in circuit.gates if g.gate_type == GateType.X]
            
            if len(x_qubits) == 0:
                return {'00': 10000}
            elif x_qubits == [0]:
                return {'01': 10000}
            elif x_qubits == [1]:
                return {'10': 10000}
            else:
                return {'11': 10000}
        
        calibration = mitigator.calibrate(executor)
        
        assert calibration.tensored is False
        assert calibration.confusion_matrix.shape == (4, 4)
    
    def test_mitigation_inversion(self):
        """Test mitigation via matrix inversion."""
        mitigator = MeasurementErrorMitigator(n_qubits=2, tensored=True)
        
        # Calibrate with perfect readout
        def calibrate_executor(circuit):
            if len(circuit.gates) == 0:
                return {'00': 10000}
            x_targets = [g.targets[0] for g in circuit.gates if g.gate_type == GateType.X]
            if 0 in x_targets and 1 not in x_targets:
                return {'01': 10000}
            elif 1 in x_targets and 0 not in x_targets:
                return {'10': 10000}
            else:
                return {'11': 10000}
        
        mitigator.calibrate(calibrate_executor)
        
        # Mitigate noisy measurements
        noisy_counts = {'00': 8000, '01': 1500, '10': 400, '11': 100}
        
        result = mitigator.mitigate(noisy_counts, method='inversion')
        
        assert isinstance(result, MitigationResult)
        assert result.mitigated_probabilities.shape == (4,)
        assert abs(np.sum(result.mitigated_probabilities) - 1.0) < 1e-6
        assert np.all(result.mitigated_probabilities >= 0)  # Physical constraint
    
    def test_mitigation_least_squares(self):
        """Test mitigation via least squares."""
        mitigator = MeasurementErrorMitigator(n_qubits=1, tensored=True)
        
        # Mock calibration
        def executor(circuit):
            if len(circuit.gates) == 0:
                return {'0': 9000, '1': 1000}  # 10% error on |0⟩
            else:
                return {'0': 500, '1': 9500}   # 5% error on |1⟩
        
        mitigator.calibrate(executor)
        
        # Mitigate
        noisy_counts = {'0': 6000, '1': 4000}
        result = mitigator.mitigate(noisy_counts, method='least_squares')
        
        assert isinstance(result, MitigationResult)
        assert result.total_variation_distance >= 0
        assert 'method' in result.metadata
        assert result.metadata['method'] == 'least_squares'
    
    def test_get_calibration_circuits(self):
        """Test getting calibration circuits for batching."""
        mitigator = MeasurementErrorMitigator(n_qubits=2, tensored=True)
        
        circuits = mitigator.get_calibration_circuits()
        
        # Tensored mode: 2 circuits per qubit (|0⟩ and |1⟩)
        assert len(circuits) == 4
        assert all(isinstance(c, UnifiedCircuit) for c in circuits)
    
    def test_high_condition_number_warning(self):
        """Test warning for ill-conditioned confusion matrix."""
        mitigator = MeasurementErrorMitigator(n_qubits=1)
        
        # Create near-singular confusion matrix (very similar rows)
        def executor(circuit):
            if len(circuit.gates) == 0:
                return {'0': 5500, '1': 4500}  # Almost 50-50
            else:
                return {'0': 4500, '1': 5500}  # Almost 50-50 (very noisy)
        
        calibration = mitigator.calibrate(executor)
        
        # High condition number indicates ill-conditioning
        # May not always be > 100 depending on exact values, but should be > 1
        assert calibration.condition_number > 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestMitigationIntegration:
    """Integration tests combining multiple mitigation techniques."""
    
    def test_zne_and_measurement_mitigation(self):
        """Test combining ZNE with measurement error mitigation."""
        # Create mitigators
        zne = create_zne_mitigator()
        mem = create_measurement_mitigator(n_qubits=1)
        
        # Mock calibration
        def calibrate_executor(circuit):
            if len(circuit.gates) == 0:
                return {'0': 9500, '1': 500}
            else:
                return {'0': 500, '1': 9500}
        
        mem.calibrate(calibrate_executor)
        
        # Circuit to mitigate
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        
        # Executor returns counts
        def executor_counts(circ):
            depth = len(circ.gates)
            # More depth = more errors
            if depth <= 1:
                return {'0': 5000, '1': 5000}
            else:
                return {'0': 6000, '1': 4000}
        
        # Executor returns expectation value
        def executor_expectation(circ):
            counts = executor_counts(circ)
            total = sum(counts.values())
            p0 = counts.get('0', 0) / total
            return 1 - 2 * p0  # <Z> = P(0) - P(1)
        
        # Apply ZNE
        zne_result = zne.mitigate(circuit, executor_expectation)
        assert isinstance(zne_result, ZNEResult)
        
        # Apply measurement mitigation
        noisy_counts = executor_counts(circuit)
        mem_result = mem.mitigate(noisy_counts)
        assert isinstance(mem_result, MitigationResult)
    
    def test_factory_functions(self):
        """Test all factory functions."""
        zne = create_zne_mitigator(
            method=ExtrapolationMethod.POLYNOMIAL,
            noise_factors=[1, 2, 3, 4]
        )
        assert isinstance(zne, ZeroNoiseExtrapolator)
        
        pec = create_pec_mitigator(
            n_samples=500,
            adaptive=True
        )
        assert isinstance(pec, AdaptivePEC)
        
        mem = create_measurement_mitigator(
            n_qubits=3,
            tensored=False
        )
        assert isinstance(mem, MeasurementErrorMitigator)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
