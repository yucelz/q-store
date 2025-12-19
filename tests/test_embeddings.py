"""
Tests for Quantum Embeddings.

Tests for:
- Feature maps (ZZ, Pauli, IQP)
- Amplitude encoding
- Angle encoding
- Basis encoding
"""

import pytest
import numpy as np

from q_store.core import UnifiedCircuit, GateType
from q_store.embeddings import (
    ZZFeatureMap,
    PauliFeatureMap,
    IQPFeatureMap,
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
)
from q_store.embeddings.amplitude_encoding import amplitude_encode
from q_store.embeddings.angle_encoding import angle_encode
from q_store.embeddings.basis_encoding import basis_encode, basis_encode_integer


# =============================================================================
# Feature Map Tests
# =============================================================================

class TestZZFeatureMap:
    """Test ZZ Feature Map."""

    def test_creation(self):
        """Test creating ZZ feature map."""
        fmap = ZZFeatureMap(n_features=3, reps=2)
        assert fmap.n_features == 3
        assert fmap.reps == 2

    def test_encode_basic(self):
        """Test basic encoding."""
        fmap = ZZFeatureMap(n_features=2, reps=1)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        assert circuit.n_qubits == 2
        assert len(circuit.gates) > 0
        # Should have H gates for initialization
        assert any(g.gate_type == GateType.H for g in circuit.gates)

    def test_encode_wrong_dimension(self):
        """Test encoding with wrong data dimension."""
        fmap = ZZFeatureMap(n_features=3, reps=1)
        data = np.array([0.5, 1.0])  # Wrong size

        with pytest.raises(ValueError):
            fmap.encode(data)

    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        data = np.array([0.5, 1.0, 1.5])

        for pattern in ['full', 'linear', 'circular']:
            fmap = ZZFeatureMap(n_features=3, reps=1, entanglement=pattern)
            circuit = fmap.encode(data)

            assert circuit.n_qubits == 3
            # Should have CNOT gates for entanglement
            assert any(g.gate_type == GateType.CNOT for g in circuit.gates)

    def test_multiple_reps(self):
        """Test with multiple repetitions."""
        fmap = ZZFeatureMap(n_features=2, reps=3)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        # More reps should create deeper circuit
        assert len(circuit.gates) > 10

    def test_data_map_function(self):
        """Test custom data mapping function."""
        fmap = ZZFeatureMap(n_features=2, reps=1, data_map_func=lambda x: 2*x)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        assert isinstance(circuit, UnifiedCircuit)


class TestPauliFeatureMap:
    """Test Pauli Feature Map."""

    def test_creation(self):
        """Test creating Pauli feature map."""
        fmap = PauliFeatureMap(n_features=3, paulis=['Z', 'ZZ'])
        assert fmap.n_features == 3
        assert fmap.paulis == ['Z', 'ZZ']

    def test_encode_z_paulis(self):
        """Test encoding with Z Paulis."""
        fmap = PauliFeatureMap(n_features=2, paulis=['Z'], reps=1)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        assert circuit.n_qubits == 2
        # Should have RZ gates
        assert any(g.gate_type == GateType.RZ for g in circuit.gates)

    def test_encode_zz_paulis(self):
        """Test encoding with ZZ interactions."""
        fmap = PauliFeatureMap(n_features=2, paulis=['ZZ'], reps=1)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        # Should have entangling gates
        assert any(g.gate_type == GateType.CNOT for g in circuit.gates)

    def test_encode_multiple_paulis(self):
        """Test with multiple Pauli types."""
        fmap = PauliFeatureMap(n_features=3, paulis=['X', 'Y', 'Z', 'ZZ'], reps=1)
        data = np.array([0.5, 1.0, 1.5])

        circuit = fmap.encode(data)

        assert circuit.n_qubits == 3
        # Should have various rotation types
        gate_types = {g.gate_type for g in circuit.gates}
        assert len(gate_types) > 1


class TestIQPFeatureMap:
    """Test IQP Feature Map."""

    def test_creation(self):
        """Test creating IQP feature map."""
        fmap = IQPFeatureMap(n_features=3, reps=2)
        assert fmap.n_features == 3
        assert fmap.reps == 2

    def test_encode_basic(self):
        """Test basic IQP encoding."""
        fmap = IQPFeatureMap(n_features=2, reps=1)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        assert circuit.n_qubits == 2
        # IQP uses Hadamards and diagonal gates
        assert any(g.gate_type == GateType.H for g in circuit.gates)
        assert any(g.gate_type == GateType.RZ for g in circuit.gates)

    def test_diagonal_structure(self):
        """Test that IQP is diagonal (uses only Z-basis gates)."""
        fmap = IQPFeatureMap(n_features=2, reps=1)
        data = np.array([0.5, 1.0])

        circuit = fmap.encode(data)

        # IQP should only use H, RZ, CNOT gates
        allowed_gates = {GateType.H, GateType.RZ, GateType.CNOT}
        for gate in circuit.gates:
            assert gate.gate_type in allowed_gates

    def test_multiple_reps(self):
        """Test IQP with multiple repetitions."""
        fmap = IQPFeatureMap(n_features=3, reps=3)
        data = np.array([0.5, 1.0, 1.5])

        circuit = fmap.encode(data)

        # More reps = deeper circuit
        assert len(circuit.gates) > 15


# =============================================================================
# Amplitude Encoding Tests
# =============================================================================

class TestAmplitudeEncoding:
    """Test amplitude encoding."""

    def test_creation(self):
        """Test creating amplitude encoder."""
        encoder = AmplitudeEncoding(normalize=True)
        assert encoder.normalize is True

    def test_encode_power_of_two(self):
        """Test encoding with power-of-2 data."""
        encoder = AmplitudeEncoding(normalize=True)
        data = np.array([1.0, 0.0, 0.0, 0.0])

        circuit = encoder.encode(data)

        assert circuit.n_qubits == 2  # log2(4) = 2

    def test_encode_non_power_of_two(self):
        """Test encoding with non-power-of-2 data (should pad)."""
        encoder = AmplitudeEncoding(normalize=True)
        data = np.array([1.0, 2.0, 3.0])  # 3 elements

        circuit = encoder.encode(data)

        assert circuit.n_qubits == 2  # Padded to 4 elements

    def test_normalization(self):
        """Test automatic normalization."""
        encoder = AmplitudeEncoding(normalize=True)
        data = np.array([1.0, 2.0, 3.0, 4.0])

        circuit = encoder.encode(data)

        # Should succeed (normalized internally)
        assert isinstance(circuit, UnifiedCircuit)

    def test_no_normalization(self):
        """Test without normalization."""
        encoder = AmplitudeEncoding(normalize=False)
        # Provide already normalized data
        data = np.array([1.0, 0.0, 0.0, 0.0])

        circuit = encoder.encode(data)

        assert circuit.n_qubits == 2

    def test_convenience_function(self):
        """Test amplitude_encode convenience function."""
        data = np.array([1.0, 0.0, 0.0, 0.0])
        circuit = amplitude_encode(data)

        assert isinstance(circuit, UnifiedCircuit)
        assert circuit.n_qubits == 2


# =============================================================================
# Angle Encoding Tests
# =============================================================================

class TestAngleEncoding:
    """Test angle encoding."""

    def test_creation(self):
        """Test creating angle encoder."""
        encoder = AngleEncoding(rotation='Y')
        assert encoder.rotation == 'Y'

    def test_invalid_rotation(self):
        """Test invalid rotation type."""
        with pytest.raises(ValueError):
            AngleEncoding(rotation='A')

    def test_encode_y_rotation(self):
        """Test encoding with Y rotations."""
        encoder = AngleEncoding(rotation='Y')
        data = np.array([0.5, 1.0, 1.5])

        circuit = encoder.encode(data)

        assert circuit.n_qubits == 3
        # Should have RY gates
        assert all(g.gate_type == GateType.RY for g in circuit.gates)

    def test_encode_x_rotation(self):
        """Test encoding with X rotations."""
        encoder = AngleEncoding(rotation='X')
        data = np.array([0.5, 1.0])

        circuit = encoder.encode(data)

        # Should have RX gates
        assert all(g.gate_type == GateType.RX for g in circuit.gates)

    def test_encode_z_rotation(self):
        """Test encoding with Z rotations."""
        encoder = AngleEncoding(rotation='Z')
        data = np.array([0.5, 1.0])

        circuit = encoder.encode(data)

        # Should have RZ gates
        assert all(g.gate_type == GateType.RZ for g in circuit.gates)

    def test_convenience_function(self):
        """Test angle_encode convenience function."""
        data = np.array([0.5, 1.0])
        circuit = angle_encode(data, rotation='Y')

        assert isinstance(circuit, UnifiedCircuit)
        assert circuit.n_qubits == 2


# =============================================================================
# Basis Encoding Tests
# =============================================================================

class TestBasisEncoding:
    """Test basis encoding."""

    def test_creation(self):
        """Test creating basis encoder."""
        encoder = BasisEncoding()
        assert encoder is not None

    def test_encode_binary(self):
        """Test encoding binary data."""
        encoder = BasisEncoding()
        data = np.array([1, 0, 1, 1])

        circuit = encoder.encode(data)

        assert circuit.n_qubits == 4
        # Should have X gates where data is 1
        x_gates = [g for g in circuit.gates if g.gate_type == GateType.X]
        assert len(x_gates) == 3  # Three 1s in data

    def test_encode_all_zeros(self):
        """Test encoding all zeros."""
        encoder = BasisEncoding()
        data = np.array([0, 0, 0])

        circuit = encoder.encode(data)

        # No gates needed for |000⟩
        assert len(circuit.gates) == 0

    def test_encode_all_ones(self):
        """Test encoding all ones."""
        encoder = BasisEncoding()
        data = np.array([1, 1, 1])

        circuit = encoder.encode(data)

        # Should have X on all qubits
        assert len(circuit.gates) == 3
        assert all(g.gate_type == GateType.X for g in circuit.gates)

    def test_encode_integer(self):
        """Test encoding integer as binary."""
        encoder = BasisEncoding()
        value = 5  # Binary: 101

        circuit = encoder.encode_integer(value, n_qubits=3)

        assert circuit.n_qubits == 3
        # 5 = 101, so X on qubits 0 and 2
        x_gates = [g for g in circuit.gates if g.gate_type == GateType.X]
        assert len(x_gates) == 2

    def test_encode_integer_too_large(self):
        """Test encoding integer too large for qubits."""
        encoder = BasisEncoding()

        with pytest.raises(ValueError):
            encoder.encode_integer(16, n_qubits=3)  # 16 needs 5 bits, not 3

    def test_convenience_functions(self):
        """Test convenience functions."""
        data = np.array([1, 0, 1])
        circuit1 = basis_encode(data)
        assert isinstance(circuit1, UnifiedCircuit)

        circuit2 = basis_encode_integer(5, n_qubits=3)
        assert isinstance(circuit2, UnifiedCircuit)


# =============================================================================
# Integration Tests
# =============================================================================

class TestEmbeddingIntegration:
    """Integration tests for embeddings."""

    def test_different_encodings_same_data(self):
        """Test encoding same data with different methods."""
        data = np.array([0.5, 1.0])

        # Angle encoding
        angle_circuit = angle_encode(data)
        assert angle_circuit.n_qubits == 2

        # ZZ feature map
        zz_map = ZZFeatureMap(n_features=2, reps=1)
        zz_circuit = zz_map.encode(data)
        assert zz_circuit.n_qubits == 2

        # IQP feature map
        iqp_map = IQPFeatureMap(n_features=2, reps=1)
        iqp_circuit = iqp_map.encode(data)
        assert iqp_circuit.n_qubits == 2

    def test_embedding_for_ml_pipeline(self):
        """Test embeddings in ML-style pipeline."""
        # Feature map for quantum kernel
        data = np.random.rand(4)

        zz_map = ZZFeatureMap(n_features=4, reps=2, entanglement='linear')
        circuit = zz_map.encode(data)

        # Should create valid circuit
        assert circuit.n_qubits == 4
        assert len(circuit.gates) > 0

    def test_amplitude_vs_basis_encoding(self):
        """Compare amplitude and basis encoding."""
        # Amplitude encoding: efficient but requires normalization
        data_amp = np.array([1.0, 0.0, 0.0, 0.0])
        amp_circuit = amplitude_encode(data_amp)
        assert amp_circuit.n_qubits == 2  # log2(4)

        # Basis encoding: simple but uses more qubits
        data_basis = np.array([1, 0, 0, 0])
        basis_circuit = basis_encode(data_basis)
        assert basis_circuit.n_qubits == 4  # linear in data size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
