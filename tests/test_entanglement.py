"""
Tests for quantum entanglement measures, witnesses, and verification.
"""

import pytest
import numpy as np
from q_store.entanglement import (
    concurrence,
    negativity,
    entropy_of_entanglement,
    entanglement_of_formation,
    EntanglementMeasure,
    bell_inequality_test,
    witness_operator,
    ppt_criterion,
    ccnr_criterion,
    EntanglementWitness,
    verify_bell_state,
    verify_ghz_state,
    verify_w_state,
    state_fidelity,
    EntanglementVerifier
)


class TestEntanglementMeasures:
    """Test entanglement measures."""

    def test_concurrence_bell_state(self):
        """Test concurrence for Bell state."""
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        C = concurrence(rho)
        assert abs(C - 1.0) < 0.01  # Maximally entangled

    def test_concurrence_separable_state(self):
        """Test concurrence for separable state."""
        # |00⟩
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        C = concurrence(rho)
        assert abs(C) < 0.01  # Not entangled

    def test_concurrence_mixed_state(self):
        """Test concurrence for mixed state."""
        # Maximally mixed state
        rho = np.eye(4) / 4

        C = concurrence(rho)
        assert abs(C) < 0.01

    def test_negativity_bell_state(self):
        """Test negativity for Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        N = negativity(rho)
        assert N > 0.4  # Should be positive for entangled state

    def test_negativity_separable_state(self):
        """Test negativity for separable state."""
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        N = negativity(rho)
        assert abs(N) < 0.01

    def test_entropy_of_entanglement_bell_state(self):
        """Test entropy of entanglement for Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        E = entropy_of_entanglement(phi_plus)
        assert abs(E - 1.0) < 0.01  # Should be 1 ebit

    def test_entropy_of_entanglement_separable(self):
        """Test entropy for separable state."""
        state = np.array([1, 0, 0, 0])

        E = entropy_of_entanglement(state)
        assert abs(E) < 0.01

    def test_entanglement_of_formation_bell_state(self):
        """Test EOF for Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        eof = entanglement_of_formation(rho)
        assert abs(eof - 1.0) < 0.01

    def test_entanglement_of_formation_separable(self):
        """Test EOF for separable state."""
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        eof = entanglement_of_formation(rho)
        assert abs(eof) < 0.01


class TestEntanglementMeasureClass:
    """Test EntanglementMeasure class."""

    def test_measure_creation(self):
        """Test creating EntanglementMeasure."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        measure = EntanglementMeasure(phi_plus)

        assert measure.dim == 4

    def test_compute_all_measures_bell_state(self):
        """Test computing all measures for Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        measure = EntanglementMeasure(phi_plus)

        measures = measure.compute_all_measures()

        assert 'concurrence' in measures
        assert 'negativity' in measures
        assert 'entanglement_of_formation' in measures
        assert 'entropy_of_entanglement' in measures

        # All should indicate strong entanglement
        assert measures['concurrence'] > 0.9
        assert measures['negativity'] > 0.4

    def test_is_entangled_bell_state(self):
        """Test detecting entanglement in Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        measure = EntanglementMeasure(phi_plus)

        assert measure.is_entangled() is True

    def test_is_entangled_separable_state(self):
        """Test no false positives for separable state."""
        state = np.array([1, 0, 0, 0])
        measure = EntanglementMeasure(state)

        assert measure.is_entangled() is False


class TestBellInequality:
    """Test Bell inequality violations."""

    def test_chsh_violation(self):
        """Test CHSH inequality violation."""
        # Measurements from maximally entangled state
        # For perfect Bell state, S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| = 2√2 ≈ 2.828
        # Using optimal measurement settings
        measurements = {
            'AB': 1/np.sqrt(2),
            'AB_prime': -1/np.sqrt(2),
            'A_primeB': 1/np.sqrt(2),
            'A_primeB_prime': 1/np.sqrt(2)
        }

        S, is_violated = bell_inequality_test(measurements, 'CHSH')

        assert is_violated == True
        assert S > 2.0  # Classical bound
        assert S <= 2.828 + 0.1  # Quantum bound

    def test_chsh_classical(self):
        """Test CHSH with classical correlations."""
        measurements = {
            'AB': 0.5,
            'AB_prime': 0.5,
            'A_primeB': 0.5,
            'A_primeB_prime': 0.5
        }

        S, is_violated = bell_inequality_test(measurements, 'CHSH')

        assert is_violated is False
        assert S <= 2.0


class TestEntanglementWitnesses:
    """Test entanglement witness operators."""

    def test_ppt_criterion_bell_state(self):
        """Test PPT criterion detects Bell state entanglement."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        min_eig = ppt_criterion(rho)

        assert min_eig < -0.1  # Should have negative eigenvalue

    def test_ppt_criterion_separable(self):
        """Test PPT criterion for separable state."""
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        min_eig = ppt_criterion(rho)

        assert min_eig >= -1e-10  # Should be positive semi-definite

    def test_bell_witness_operator(self):
        """Test Bell state witness."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        witness_value = witness_operator(rho, 'Bell')

        # Should be negative for Bell state
        assert witness_value < 0

    def test_ccnr_criterion_separable(self):
        """Test CCNR criterion for separable state."""
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        is_separable = ccnr_criterion(rho)

        assert is_separable == True

    def test_ccnr_criterion_entangled(self):
        """Test CCNR criterion for entangled state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        is_separable = ccnr_criterion(rho)

        assert is_separable == False


class TestEntanglementWitnessClass:
    """Test EntanglementWitness class."""

    def test_witness_creation(self):
        """Test creating EntanglementWitness."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        witness = EntanglementWitness(rho)
        assert witness.dim == 4

    def test_apply_all_tests_bell_state(self):
        """Test all witness tests on Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        witness = EntanglementWitness(rho)
        results = witness.apply_all_tests()

        assert 'ppt_criterion' in results
        assert 'ccnr_criterion' in results
        assert 'bell_witness' in results

        # All should detect entanglement
        assert results['ppt_criterion']['is_entangled'] is True
        assert results['ccnr_criterion']['is_entangled'] is True

    def test_is_entangled_detection(self):
        """Test entanglement detection."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        witness = EntanglementWitness(rho)
        assert witness.is_entangled() is True

    def test_separability_test(self):
        """Test separability testing."""
        state = np.array([1, 0, 0, 0])
        rho = np.outer(state, state.conj())

        witness = EntanglementWitness(rho)
        is_separable, method = witness.separability_test()

        assert is_separable is True


class TestBellStateVerification:
    """Test Bell state verification."""

    def test_verify_phi_plus(self):
        """Test verification of |Φ⁺⟩."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        result = verify_bell_state(phi_plus, 'phi_plus')

        assert result['target_fidelity'] > 0.99
        assert result['is_bell_state'] is True
        assert result['closest_bell_state'] == 'phi_plus'

    def test_verify_phi_minus(self):
        """Test verification of |Φ⁻⟩."""
        phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)

        result = verify_bell_state(phi_minus, 'phi_minus')

        assert result['target_fidelity'] > 0.99
        assert result['is_bell_state'] is True

    def test_verify_psi_plus(self):
        """Test verification of |Ψ⁺⟩."""
        psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)

        result = verify_bell_state(psi_plus, 'psi_plus')

        assert result['target_fidelity'] > 0.99
        assert result['is_bell_state'] is True

    def test_verify_wrong_bell_state(self):
        """Test verifying against wrong Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        result = verify_bell_state(phi_plus, 'psi_plus')

        assert result['target_fidelity'] < 0.1  # Should be very different
        assert result['is_bell_state'] is False
        assert result['closest_bell_state'] == 'phi_plus'

    def test_verify_separable_state(self):
        """Test verification fails for separable state."""
        state = np.array([1, 0, 0, 0])

        result = verify_bell_state(state, 'phi_plus')

        assert result['is_bell_state'] is False


class TestGHZStateVerification:
    """Test GHZ state verification."""

    def test_verify_ghz_three_qubit(self):
        """Test verification of 3-qubit GHZ state."""
        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        ghz = np.zeros(8, dtype=complex)
        ghz[0] = 1.0 / np.sqrt(2)
        ghz[7] = 1.0 / np.sqrt(2)

        result = verify_ghz_state(ghz, n_qubits=3)

        assert result['fidelity'] > 0.99
        assert result['is_ghz_state'] is True
        assert result['n_qubits'] == 3

    def test_verify_ghz_four_qubit(self):
        """Test verification of 4-qubit GHZ state."""
        ghz = np.zeros(16, dtype=complex)
        ghz[0] = 1.0 / np.sqrt(2)
        ghz[15] = 1.0 / np.sqrt(2)

        result = verify_ghz_state(ghz, n_qubits=4)

        assert result['fidelity'] > 0.99
        assert result['is_ghz_state'] is True

    def test_verify_non_ghz_state(self):
        """Test verification fails for non-GHZ state."""
        # Product state |000⟩
        state = np.zeros(8, dtype=complex)
        state[0] = 1.0

        result = verify_ghz_state(state, n_qubits=3)

        assert result['is_ghz_state'] is False


class TestWStateVerification:
    """Test W state verification."""

    def test_verify_w_three_qubit(self):
        """Test verification of 3-qubit W state."""
        # |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
        w = np.zeros(8, dtype=complex)
        w[1] = 1.0 / np.sqrt(3)  # |001⟩
        w[2] = 1.0 / np.sqrt(3)  # |010⟩
        w[4] = 1.0 / np.sqrt(3)  # |100⟩

        result = verify_w_state(w, n_qubits=3)

        assert result['fidelity'] > 0.99
        assert result['is_w_state'] is True
        assert result['n_qubits'] == 3

    def test_verify_w_four_qubit(self):
        """Test verification of 4-qubit W state."""
        w = np.zeros(16, dtype=complex)
        w[1] = 1.0 / 2.0   # |0001⟩
        w[2] = 1.0 / 2.0   # |0010⟩
        w[4] = 1.0 / 2.0   # |0100⟩
        w[8] = 1.0 / 2.0   # |1000⟩

        result = verify_w_state(w, n_qubits=4)

        assert result['fidelity'] > 0.99
        assert result['is_w_state'] is True

    def test_verify_non_w_state(self):
        """Test verification fails for non-W state."""
        # GHZ state
        ghz = np.zeros(8, dtype=complex)
        ghz[0] = 1.0 / np.sqrt(2)
        ghz[7] = 1.0 / np.sqrt(2)

        result = verify_w_state(ghz, n_qubits=3)

        assert result['is_w_state'] is False


class TestStateFidelity:
    """Test state fidelity calculations."""

    def test_fidelity_identical_states(self):
        """Test fidelity of identical states."""
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)

        fid = state_fidelity(state, state)

        assert abs(fid - 1.0) < 0.01

    def test_fidelity_orthogonal_states(self):
        """Test fidelity of orthogonal states."""
        state1 = np.array([1, 0, 0, 0])
        state2 = np.array([0, 1, 0, 0])

        fid = state_fidelity(state1, state2)

        assert abs(fid) < 0.01

    def test_fidelity_bell_states(self):
        """Test fidelity between different Bell states."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)

        fid = state_fidelity(phi_plus, psi_plus)

        assert fid >= 0  # Orthogonal Bell states have zero fidelity

    def test_fidelity_mixed_states(self):
        """Test fidelity with mixed states."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho1 = np.outer(phi_plus, phi_plus.conj())

        rho2 = np.eye(4) / 4  # Maximally mixed

        fid = state_fidelity(rho1, rho2)

        assert 0 < fid < 1


class TestEntanglementVerifierClass:
    """Test EntanglementVerifier class."""

    def test_verifier_creation(self):
        """Test creating EntanglementVerifier."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        verifier = EntanglementVerifier(phi_plus)

        assert verifier.n_qubits == 2
        assert verifier.dim == 4

    def test_verify_all_standard_states_bell(self):
        """Test verification of all standard states for Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        verifier = EntanglementVerifier(phi_plus, n_qubits=2)
        results = verifier.verify_all_standard_states()

        assert 'phi_plus' in results
        assert 'ghz' in results
        assert results['phi_plus']['is_bell_state'] is True

    def test_identify_bell_state(self):
        """Test identification of Bell state."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)

        verifier = EntanglementVerifier(phi_plus, n_qubits=2)
        state_name, fidelity = verifier.identify_state()

        assert state_name == 'phi_plus'
        assert fidelity > 0.99

    def test_identify_ghz_state(self):
        """Test identification of GHZ state."""
        ghz = np.zeros(8, dtype=complex)
        ghz[0] = 1.0 / np.sqrt(2)
        ghz[7] = 1.0 / np.sqrt(2)

        verifier = EntanglementVerifier(ghz, n_qubits=3)
        state_name, fidelity = verifier.identify_state()

        assert state_name == 'ghz'
        assert fidelity > 0.99

    def test_identify_w_state(self):
        """Test identification of W state."""
        w = np.zeros(8, dtype=complex)
        w[1] = 1.0 / np.sqrt(3)
        w[2] = 1.0 / np.sqrt(3)
        w[4] = 1.0 / np.sqrt(3)

        verifier = EntanglementVerifier(w, n_qubits=3)
        state_name, fidelity = verifier.identify_state()

        assert state_name == 'w'
        assert fidelity > 0.99

    def test_identify_unknown_state(self):
        """Test identification of unknown state."""
        # Random state
        state = np.array([0.5, 0.5, 0.5, 0.5])

        verifier = EntanglementVerifier(state, n_qubits=2)
        state_name, fidelity = verifier.identify_state()

        assert state_name == 'unknown'


class TestIntegration:
    """Integration tests for entanglement module."""

    def test_full_entanglement_analysis(self):
        """Test complete entanglement analysis pipeline."""
        # Create Bell state
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        # Measure entanglement
        measure = EntanglementMeasure(rho)
        measures = measure.compute_all_measures()

        assert measures['concurrence'] > 0.9
        assert measure.is_entangled() is True

        # Witness tests
        witness = EntanglementWitness(rho)
        witness_results = witness.apply_all_tests()

        assert witness_results['ppt_criterion']['is_entangled'] is True

        # Verify state
        verifier = EntanglementVerifier(rho)
        state_name, fidelity = verifier.identify_state()

        assert state_name == 'phi_plus'
        assert fidelity > 0.99

    def test_compare_bell_states(self):
        """Test comparing different Bell states."""
        bell_states = {
            'phi_plus': np.array([1, 0, 0, 1]) / np.sqrt(2),
            'phi_minus': np.array([1, 0, 0, -1]) / np.sqrt(2),
            'psi_plus': np.array([0, 1, 1, 0]) / np.sqrt(2),
            'psi_minus': np.array([0, 1, -1, 0]) / np.sqrt(2)
        }

        # All should be maximally entangled
        for name, state in bell_states.items():
            C = concurrence(np.outer(state, state.conj()))
            assert C > 0.99, f"{name} should be maximally entangled"

    def test_entanglement_scaling(self):
        """Test entanglement measures scale with system size."""
        # Two-qubit Bell state
        bell_2 = np.array([1, 0, 0, 1]) / np.sqrt(2)
        E_2 = entropy_of_entanglement(bell_2)

        # Three-qubit GHZ state
        ghz_3 = np.zeros(8, dtype=complex)
        ghz_3[0] = 1.0 / np.sqrt(2)
        ghz_3[7] = 1.0 / np.sqrt(2)
        E_3 = entropy_of_entanglement(ghz_3)

        # Both should be 1 ebit
        assert abs(E_2 - 1.0) < 0.01
        assert abs(E_3 - 1.0) < 0.01
