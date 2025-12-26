"""
Tests for advanced verification module.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.verification import (
    check_unitary_equivalence,
    check_state_equivalence,
    check_circuit_equivalence,
    circuits_equal_up_to_phase,
    EquivalenceChecker,
    is_unitary,
    is_reversible,
    check_commutativity,
    verify_gate_decomposition,
    PropertyVerifier,
    verify_circuit_identity,
    check_algebraic_property,
    symbolic_circuit_analysis,
    FormalVerifier
)


class TestUnitaryEquivalence:
    """Test unitary equivalence checking."""

    def test_identical_unitaries(self):
        """Test equivalence of identical unitaries."""
        U = np.array([[0, 1], [1, 0]])  # Pauli X

        is_equiv, distance = check_unitary_equivalence(U, U)

        assert is_equiv == True
        assert distance < 1e-10

    def test_global_phase_equivalence(self):
        """Test equivalence with global phase."""
        U1 = np.array([[0, 1], [1, 0]])
        U2 = 1j * U1  # Global phase e^(iπ/2)

        is_equiv, distance = check_unitary_equivalence(U1, U2)

        assert is_equiv == True
        assert distance < 1e-10

    def test_different_unitaries(self):
        """Test non-equivalent unitaries."""
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])

        is_equiv, distance = check_unitary_equivalence(X, Z)

        assert is_equiv == False
        assert distance > 0.1


class TestStateEquivalence:
    """Test state equivalence checking."""

    def test_identical_states(self):
        """Test equivalence of identical states."""
        state = np.array([1, 0])

        is_equiv, distance = check_state_equivalence(state, state)

        assert is_equiv == True
        assert distance < 1e-10

    def test_global_phase_state(self):
        """Test states with global phase."""
        state1 = np.array([1, 0])
        state2 = 1j * state1

        is_equiv, distance = check_state_equivalence(state1, state2)

        assert is_equiv == True
        assert distance < 1e-10

    def test_orthogonal_states(self):
        """Test orthogonal states."""
        state1 = np.array([1, 0])
        state2 = np.array([0, 1])

        is_equiv, distance = check_state_equivalence(state1, state2)

        assert is_equiv == False
        assert distance > 0.9


class TestCircuitEquivalence:
    """Test circuit equivalence checking."""

    def test_identical_circuits(self):
        """Test equivalence of identical circuits."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])

        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        circuit2.add_gate(GateType.CNOT, [0, 1])

        is_equiv, details = check_circuit_equivalence(circuit1, circuit2)

        assert is_equiv == True
        assert details['unitary_distance'] < 1e-10

    def test_different_implementations(self):
        """Test equivalent circuits with different implementations."""
        # X gate
        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.X, [0])

        # HZH = X
        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.H, [0])
        circuit2.add_gate(GateType.Z, [0])
        circuit2.add_gate(GateType.H, [0])

        is_equiv, details = check_circuit_equivalence(circuit1, circuit2)

        assert is_equiv == True

    def test_different_circuits(self):
        """Test non-equivalent circuits."""
        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.X, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.Z, [0])

        is_equiv, details = check_circuit_equivalence(circuit1, circuit2)

        assert is_equiv == False

    def test_circuits_equal_up_to_phase(self):
        """Test convenience function."""
        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.H, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.H, [0])

        assert circuits_equal_up_to_phase(circuit1, circuit2) == True


class TestEquivalenceChecker:
    """Test EquivalenceChecker class."""

    def test_checker_creation(self):
        """Test creating equivalence checker."""
        checker = EquivalenceChecker(tolerance=1e-10)
        assert checker.tolerance == 1e-10

    def test_check_unitary(self):
        """Test unitary checking."""
        checker = EquivalenceChecker()

        X = np.array([[0, 1], [1, 0]])
        result = checker.check_unitary(X, X)

        assert result['equivalent'] == True
        assert 'global_phase' in result

    def test_check_circuit(self):
        """Test circuit checking."""
        checker = EquivalenceChecker()

        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.H, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.H, [0])

        result = checker.check_circuit(circuit1, circuit2)

        assert result['equivalent'] == True
        assert 'basis_state_tests' in result

    def test_compare_multiple_circuits(self):
        """Test comparing multiple circuits."""
        checker = EquivalenceChecker()

        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.X, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.X, [0])

        circuit3 = UnifiedCircuit(1)
        circuit3.add_gate(GateType.Z, [0])

        result = checker.compare_multiple_circuits([circuit1, circuit2, circuit3])

        assert result['n_circuits'] == 3
        assert result['equivalence_matrix'][0, 1] == True
        assert result['equivalence_matrix'][0, 2] == False
        assert result['all_equivalent'] == False


class TestUnitarity:
    """Test unitarity checking."""

    def test_unitary_matrix(self):
        """Test unitary matrix."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        is_unitary_flag, deviation = is_unitary(H)

        assert is_unitary_flag == True
        assert deviation < 1e-10

    def test_non_unitary_matrix(self):
        """Test non-unitary matrix."""
        M = np.array([[1, 0], [0, 2]])

        is_unitary_flag, deviation = is_unitary(M)

        assert is_unitary_flag == False
        assert deviation > 0.1


class TestReversibility:
    """Test reversibility checking."""

    def test_reversible_circuit(self):
        """Test reversible circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        assert is_reversible(circuit) == True

    def test_single_gate_reversibility(self):
        """Test single gate reversibility."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.X, [0])

        assert is_reversible(circuit) == True


class TestCommutativity:
    """Test commutativity checking."""

    def test_commuting_gates(self):
        """Test gates that commute."""
        # X on qubit 0 and X on qubit 1 commute
        do_commute, norm = check_commutativity(
            GateType.X, GateType.X,
            [0], [1],
            n_qubits=2
        )

        assert do_commute == True
        assert norm < 1e-10

    def test_non_commuting_gates(self):
        """Test gates that don't commute."""
        # X and Z on same qubit don't commute
        do_commute, norm = check_commutativity(
            GateType.X, GateType.Z,
            [0], [0],
            n_qubits=1
        )

        assert do_commute == False
        assert norm > 0.1


class TestGateDecomposition:
    """Test gate decomposition verification."""

    def test_correct_decomposition(self):
        """Test correct gate decomposition."""
        # X = HZH
        target = np.array([[0, 1], [1, 0]])

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.Z, [0])
        circuit.add_gate(GateType.H, [0])

        is_correct, details = verify_gate_decomposition(target, circuit)

        assert is_correct == True
        assert details['n_gates'] == 3

    def test_incorrect_decomposition(self):
        """Test incorrect decomposition."""
        target = np.array([[0, 1], [1, 0]])  # X gate

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.Z, [0])  # Wrong gate

        is_correct, details = verify_gate_decomposition(target, circuit)

        assert is_correct == False


class TestPropertyVerifier:
    """Test PropertyVerifier class."""

    def test_verifier_creation(self):
        """Test creating property verifier."""
        verifier = PropertyVerifier(tolerance=1e-10)
        assert verifier.tolerance == 1e-10

    def test_verify_unitarity(self):
        """Test unitarity verification."""
        verifier = PropertyVerifier()

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])

        result = verifier.verify_unitarity(circuit)

        assert result['is_unitary'] == True
        assert result['deviation'] < 1e-10

    def test_verify_reversibility(self):
        """Test reversibility verification."""
        verifier = PropertyVerifier()

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.X, [0])

        result = verifier.verify_reversibility(circuit)

        assert result['is_reversible'] == True

    def test_analyze_commutativity(self):
        """Test commutativity analysis."""
        verifier = PropertyVerifier()

        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.X, [0])
        circuit.add_gate(GateType.X, [1])
        circuit.add_gate(GateType.Z, [0])

        result = verifier.analyze_commutativity(circuit)

        assert result['n_gates'] == 3
        assert 'commute_matrix' in result
        assert result['total_pairs'] == 3

    def test_verify_identity(self):
        """Test identity verification."""
        verifier = PropertyVerifier()

        # Empty circuit is identity
        circuit = UnifiedCircuit(1)

        result = verifier.verify_identity(circuit)

        assert result['is_identity'] == True

    def test_check_all_properties(self):
        """Test checking all properties."""
        verifier = PropertyVerifier()

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])

        result = verifier.check_all_properties(circuit)

        assert 'unitarity' in result
        assert 'reversibility' in result
        assert 'commutativity' in result
        assert 'identity' in result
        assert 'summary' in result


class TestCircuitIdentity:
    """Test circuit identity verification."""

    def test_verify_inverse(self):
        """Test inverse verification."""
        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.X, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.X, [0])  # X is self-inverse

        result = verify_circuit_identity(circuit1, circuit2, "inverse")

        assert result['verified'] == True
        assert result['property'] == 'inverse'

    def test_verify_equivalence(self):
        """Test equivalence verification."""
        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.H, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.H, [0])

        result = verify_circuit_identity(circuit1, circuit2, "equivalent")

        assert result['verified'] == True

    def test_verify_commute(self):
        """Test commutation verification."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.X, [0])

        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.X, [1])

        result = verify_circuit_identity(circuit1, circuit2, "commute")

        assert result['verified'] == True


class TestAlgebraicProperty:
    """Test algebraic property checking."""

    def test_hermitian_check(self):
        """Test checking hermitian property."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.X, [0])

        def is_hermitian(U):
            return np.allclose(U, U.conj().T)

        result = check_algebraic_property(circuit, is_hermitian, "hermitian")

        assert result['verified'] == True

    def test_custom_property(self):
        """Test custom property check."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])

        def has_unit_modulus_determinant(U):
            return np.abs(np.abs(np.linalg.det(U)) - 1.0) < 1e-10

        result = check_algebraic_property(circuit, has_unit_modulus_determinant, "unit_det")

        assert result['verified'] == True


class TestSymbolicAnalysis:
    """Test symbolic circuit analysis."""

    def test_basic_analysis(self):
        """Test basic symbolic analysis."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.X, [1])

        result = symbolic_circuit_analysis(circuit)

        assert result['n_qubits'] == 2
        assert result['n_gates'] == 3
        assert 'gate_counts' in result

    def test_pattern_detection(self):
        """Test pattern detection."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.X, [0])
        circuit.add_gate(GateType.X, [0])  # Repeated gate

        result = symbolic_circuit_analysis(circuit)

        assert len(result['patterns']) > 0

    def test_inverse_pair_detection(self):
        """Test inverse pair detection."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [0])  # H is self-inverse

        result = symbolic_circuit_analysis(circuit)

        # Should detect inverse pair pattern
        assert 'patterns' in result


class TestFormalVerifier:
    """Test FormalVerifier class."""

    def test_verifier_creation(self):
        """Test creating formal verifier."""
        verifier = FormalVerifier(tolerance=1e-10)
        assert verifier.tolerance == 1e-10

    def test_verify_relationship(self):
        """Test relationship verification."""
        verifier = FormalVerifier()

        circuit1 = UnifiedCircuit(1)
        circuit1.add_gate(GateType.H, [0])

        circuit2 = UnifiedCircuit(1)
        circuit2.add_gate(GateType.H, [0])

        result = verifier.verify_relationship(circuit1, circuit2, "equivalent")

        assert result['verified'] == True

    def test_verify_transformation(self):
        """Test transformation verification."""
        verifier = FormalVerifier()

        original = UnifiedCircuit(1)
        original.add_gate(GateType.H, [0])

        transformed = UnifiedCircuit(1)
        transformed.add_gate(GateType.H, [0])

        result = verifier.verify_transformation(original, transformed, "test_transform")

        assert result['transformation'] == 'test_transform'
        assert result['preserves_functionality'] == True

    def test_verify_optimization(self):
        """Test optimization verification."""
        verifier = FormalVerifier()

        original = UnifiedCircuit(1)
        original.add_gate(GateType.X, [0])
        original.add_gate(GateType.X, [0])  # Redundant

        optimized = UnifiedCircuit(1)  # Empty (optimized away)

        result = verifier.verify_optimization(original, optimized)

        assert 'improvements' in result
        assert result['improvements']['gate_count_reduction'] == 2

    def test_verify_decomposition(self):
        """Test decomposition verification."""
        verifier = FormalVerifier()

        target = np.array([[0, 1], [1, 0]])  # X gate

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.Z, [0])
        circuit.add_gate(GateType.H, [0])

        result = verifier.verify_decomposition(target, circuit)

        assert result['decomposition_correct'] == True

    def test_comprehensive_verification(self):
        """Test comprehensive verification."""
        verifier = FormalVerifier()

        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])

        result = verifier.comprehensive_verification(circuit)

        assert 'properties' in result
        assert 'symbolic_analysis' in result
        assert 'is_identity' in result


class TestIntegration:
    """Integration tests for verification module."""

    def test_full_equivalence_pipeline(self):
        """Test complete equivalence checking pipeline."""
        # Create two equivalent circuits
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])

        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        circuit2.add_gate(GateType.CNOT, [0, 1])

        # Check equivalence
        checker = EquivalenceChecker()
        result = checker.check_circuit(circuit1, circuit2)

        assert result['equivalent'] == True

        # Verify properties
        verifier = PropertyVerifier()
        props = verifier.check_all_properties(circuit1)

        assert props['summary']['is_unitary'] == True

    def test_optimization_verification_workflow(self):
        """Test verifying circuit optimization."""
        # Original with redundant gates
        original = UnifiedCircuit(1)
        original.add_gate(GateType.X, [0])
        original.add_gate(GateType.Y, [0])
        original.add_gate(GateType.Z, [0])

        # Optimized version
        optimized = UnifiedCircuit(1)
        optimized.add_gate(GateType.X, [0])
        optimized.add_gate(GateType.Y, [0])
        optimized.add_gate(GateType.Z, [0])

        # Verify optimization
        verifier = FormalVerifier()
        result = verifier.verify_optimization(original, optimized)

        assert result['preserves_functionality'] == True

    def test_decomposition_verification_workflow(self):
        """Test verifying gate decomposition."""
        # Target: CNOT gate
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        # Decomposition
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.CNOT, [0, 1])

        # Verify
        verifier = FormalVerifier()
        result = verifier.verify_decomposition(CNOT, circuit)

        assert result['decomposition_correct'] == True
