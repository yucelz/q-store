"""
Tests for quantum tomography.
"""

import pytest
import numpy as np
from q_store.tomography import (
    StateTomoGraphy,
    reconstruct_state,
    generate_measurement_bases,
    linear_inversion,
    maximum_likelihood_estimation,
    ProcessTomography,
    reconstruct_process,
    generate_input_states,
    pauli_transfer_matrix,
    chi_matrix_reconstruction
)


class TestStateTomoGraphy:
    """Test quantum state tomography."""

    def test_tomography_creation(self):
        """Test creating state tomography."""
        tomo = StateTomoGraphy(n_qubits=1)
        assert tomo.n_qubits == 1
        assert len(tomo.measurement_bases) > 0

    def test_add_measurement(self):
        """Test adding measurements."""
        tomo = StateTomoGraphy(n_qubits=1)
        probs = np.array([0.8, 0.2])
        tomo.add_measurement('Z', probs)
        assert 'Z' in tomo.measurements
        assert np.allclose(tomo.measurements['Z'], probs)

    def test_reconstruct_linear(self):
        """Test linear inversion reconstruction."""
        tomo = StateTomoGraphy(n_qubits=1)

        # Measurements for |0> state
        tomo.add_measurement('Z', np.array([1.0, 0.0]))
        tomo.add_measurement('X', np.array([0.5, 0.5]))
        tomo.add_measurement('Y', np.array([0.5, 0.5]))

        rho = tomo.reconstruct(method='linear')
        assert rho.shape == (2, 2)
        assert np.abs(np.trace(rho) - 1.0) < 0.1

    def test_reconstruct_mle(self):
        """Test maximum likelihood reconstruction."""
        tomo = StateTomoGraphy(n_qubits=1)

        tomo.add_measurement('Z', np.array([1.0, 0.0]))
        tomo.add_measurement('X', np.array([0.5, 0.5]))
        tomo.add_measurement('Y', np.array([0.5, 0.5]))

        rho = tomo.reconstruct(method='mle')
        assert rho.shape == (2, 2)

    def test_fidelity_calculation(self):
        """Test fidelity calculation."""
        tomo = StateTomoGraphy(n_qubits=1)

        # Perfect measurements for |0>
        tomo.add_measurement('Z', np.array([1.0, 0.0]))
        tomo.add_measurement('X', np.array([0.5, 0.5]))
        tomo.add_measurement('Y', np.array([0.5, 0.5]))

        # Target state |0>
        target = np.array([[1, 0], [0, 0]])
        fidelity = tomo.fidelity(target)

        assert 0 <= fidelity <= 1
        assert fidelity > 0.5  # Should be reasonably high

    def test_two_qubit_tomography(self):
        """Test two-qubit tomography."""
        tomo = StateTomoGraphy(n_qubits=2)
        assert tomo.n_qubits == 2

        # Add some measurements
        tomo.add_measurement('ZZ', np.array([0.25, 0.25, 0.25, 0.25]))
        rho = tomo.reconstruct(method='linear')
        assert rho.shape == (4, 4)


class TestStateTomographyFunctions:
    """Test standalone state tomography functions."""

    def test_generate_measurement_bases_single_qubit(self):
        """Test generating measurement bases for single qubit."""
        bases = generate_measurement_bases(n_qubits=1)
        assert 'Z' in bases
        assert 'X' in bases
        assert 'Y' in bases

    def test_generate_measurement_bases_two_qubit(self):
        """Test generating measurement bases for two qubits."""
        bases = generate_measurement_bases(n_qubits=2)
        assert len(bases) == 9
        assert 'ZZ' in bases
        assert 'XX' in bases

    def test_linear_inversion_function(self):
        """Test linear inversion function."""
        measurements = {
            'Z': np.array([1.0, 0.0]),
            'X': np.array([0.5, 0.5]),
            'Y': np.array([0.5, 0.5])
        }
        rho = linear_inversion(measurements, n_qubits=1)
        assert rho.shape == (2, 2)

    def test_mle_function(self):
        """Test MLE function."""
        measurements = {
            'Z': np.array([1.0, 0.0]),
            'X': np.array([0.5, 0.5]),
            'Y': np.array([0.5, 0.5])
        }
        rho = maximum_likelihood_estimation(measurements, n_qubits=1)
        assert rho.shape == (2, 2)

    def test_reconstruct_state_function(self):
        """Test reconstruct_state convenience function."""
        measurements = {
            'Z': np.array([0.8, 0.2]),
            'X': np.array([0.6, 0.4]),
            'Y': np.array([0.5, 0.5])
        }
        rho = reconstruct_state(measurements, n_qubits=1, method='linear')
        assert rho.shape == (2, 2)

    def test_reconstruct_mixed_state(self):
        """Test reconstructing mixed state."""
        measurements = {
            'Z': np.array([0.5, 0.5]),
            'X': np.array([0.5, 0.5]),
            'Y': np.array([0.5, 0.5])
        }
        rho = reconstruct_state(measurements, n_qubits=1)
        assert rho.shape == (2, 2)
        # Mixed state should have similar diagonal elements
        assert abs(rho[0, 0] - rho[1, 1]) < 0.5


class TestProcessTomography:
    """Test quantum process tomography."""

    def test_process_tomography_creation(self):
        """Test creating process tomography."""
        tomo = ProcessTomography(n_qubits=1)
        assert tomo.n_qubits == 1
        assert len(tomo.input_states) > 0

    def test_add_input_output_pair(self):
        """Test adding input-output pairs."""
        tomo = ProcessTomography(n_qubits=1)

        input_state = np.array([[1, 0], [0, 0]])
        output_state = np.array([[1, 0], [0, 0]])

        tomo.add_input_output(input_state, output_state)
        assert len(tomo.input_output_pairs) == 1

    def test_reconstruct_chi_matrix(self):
        """Test chi matrix reconstruction."""
        tomo = ProcessTomography(n_qubits=1)

        # Add identity process
        input_state = np.array([[1, 0], [0, 0]])
        output_state = np.array([[1, 0], [0, 0]])
        tomo.add_input_output(input_state, output_state)

        chi = tomo.reconstruct_chi_matrix()
        assert chi.shape == (4, 4)

    def test_reconstruct_ptm(self):
        """Test Pauli transfer matrix reconstruction."""
        tomo = ProcessTomography(n_qubits=1)

        # Add some data
        input_state = np.array([[1, 0], [0, 0]])
        output_state = np.array([[1, 0], [0, 0]])
        tomo.add_input_output(input_state, output_state)

        ptm = tomo.reconstruct_pauli_transfer_matrix()
        assert ptm.shape == (4, 4)

    def test_process_fidelity(self):
        """Test process fidelity calculation."""
        tomo = ProcessTomography(n_qubits=1)

        # Add identity process
        for state in tomo.input_states:
            tomo.add_input_output(state, state)

        target = np.eye(4) / 4
        fidelity = tomo.process_fidelity(target)
        assert 0 <= fidelity <= 1

    def test_average_gate_fidelity(self):
        """Test average gate fidelity."""
        tomo = ProcessTomography(n_qubits=1)

        # Perfect identity channel
        for state in tomo.input_states:
            tomo.add_input_output(state, state)

        avg_fid = tomo.average_gate_fidelity()
        assert 0 <= avg_fid <= 1

    def test_two_qubit_process_tomography(self):
        """Test two-qubit process tomography."""
        tomo = ProcessTomography(n_qubits=2)
        assert tomo.n_qubits == 2
        assert tomo.dim == 4


class TestProcessTomographyFunctions:
    """Test standalone process tomography functions."""

    def test_generate_input_states_single_qubit(self):
        """Test generating input states for single qubit."""
        states = generate_input_states(n_qubits=1)
        assert len(states) >= 4
        for state in states:
            assert state.shape == (2, 2)

    def test_generate_input_states_two_qubit(self):
        """Test generating input states for two qubits."""
        states = generate_input_states(n_qubits=2)
        assert len(states) >= 1
        for state in states:
            assert state.shape == (4, 4)

    def test_pauli_transfer_matrix_function(self):
        """Test PTM function."""
        input_state = np.array([[1, 0], [0, 0]])
        output_state = np.array([[1, 0], [0, 0]])

        pairs = [(input_state, output_state)]
        ptm = pauli_transfer_matrix(pairs, n_qubits=1)
        assert ptm.shape == (4, 4)

    def test_chi_matrix_reconstruction_function(self):
        """Test chi matrix reconstruction function."""
        input_state = np.array([[1, 0], [0, 0]])
        output_state = np.array([[1, 0], [0, 0]])

        pairs = [(input_state, output_state)]
        chi = chi_matrix_reconstruction(pairs, n_qubits=1)
        assert chi.shape == (4, 4)

    def test_reconstruct_process_function(self):
        """Test reconstruct_process convenience function."""
        # Identity process
        def identity_process(state):
            return state

        result = reconstruct_process(identity_process, n_qubits=1)

        assert 'chi_matrix' in result
        assert 'ptm' in result
        assert 'average_fidelity' in result
        assert result['chi_matrix'].shape == (4, 4)
        assert result['ptm'].shape == (4, 4)


class TestTomographyProperties:
    """Test tomography mathematical properties."""

    def test_density_matrix_trace(self):
        """Test that reconstructed states have trace 1."""
        measurements = {
            'Z': np.array([0.7, 0.3]),
            'X': np.array([0.6, 0.4]),
            'Y': np.array([0.5, 0.5])
        }
        rho = reconstruct_state(measurements, n_qubits=1)
        trace = np.trace(rho)
        assert np.abs(trace - 1.0) < 0.2

    def test_density_matrix_hermitian(self):
        """Test that reconstructed states are Hermitian."""
        measurements = {
            'Z': np.array([0.8, 0.2]),
            'X': np.array([0.6, 0.4]),
            'Y': np.array([0.5, 0.5])
        }
        rho = reconstruct_state(measurements, n_qubits=1)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_chi_matrix_dimensions(self):
        """Test chi matrix has correct dimensions."""
        tomo = ProcessTomography(n_qubits=1)
        chi = tomo.reconstruct_chi_matrix()
        expected_dim = 4 ** tomo.n_qubits
        assert chi.shape == (expected_dim, expected_dim)

    def test_ptm_dimensions(self):
        """Test PTM has correct dimensions."""
        tomo = ProcessTomography(n_qubits=1)
        ptm = tomo.reconstruct_pauli_transfer_matrix()
        expected_dim = 4 ** tomo.n_qubits
        assert ptm.shape == (expected_dim, expected_dim)


class TestIntegration:
    """Integration tests for tomography."""

    def test_full_state_tomography_pipeline(self):
        """Test complete state tomography workflow."""
        # Generate measurement bases
        bases = generate_measurement_bases(n_qubits=1)

        # Simulate measurements for |0> state
        measurements = {
            'Z': np.array([1.0, 0.0]),
            'X': np.array([0.5, 0.5]),
            'Y': np.array([0.5, 0.5])
        }

        # Reconstruct
        rho = reconstruct_state(measurements, n_qubits=1, method='linear')

        # Verify
        assert rho.shape == (2, 2)
        assert np.abs(np.trace(rho) - 1.0) < 0.2

    def test_full_process_tomography_pipeline(self):
        """Test complete process tomography workflow."""
        # Generate input states
        input_states = generate_input_states(n_qubits=1)

        # Identity process
        def identity(state):
            return state

        # Reconstruct
        result = reconstruct_process(identity, n_qubits=1)

        # Verify
        assert result['chi_matrix'].shape == (4, 4)
        assert result['ptm'].shape == (4, 4)
        assert 0 <= result['average_fidelity'] <= 1

    def test_tomography_with_noise(self):
        """Test tomography with noisy measurements."""
        # Noisy measurements
        measurements = {
            'Z': np.array([0.85, 0.15]),  # Should be [1, 0]
            'X': np.array([0.55, 0.45]),  # Should be [0.5, 0.5]
            'Y': np.array([0.52, 0.48])   # Should be [0.5, 0.5]
        }

        rho = reconstruct_state(measurements, n_qubits=1)

        # Should still reconstruct something reasonable
        assert rho.shape == (2, 2)
        assert np.real(rho[0, 0]) > 0.5  # Mostly in |0>

    def test_compare_reconstruction_methods(self):
        """Test different reconstruction methods."""
        measurements = {
            'Z': np.array([0.8, 0.2]),
            'X': np.array([0.6, 0.4]),
            'Y': np.array([0.5, 0.5])
        }

        rho_linear = reconstruct_state(measurements, n_qubits=1, method='linear')
        rho_mle = reconstruct_state(measurements, n_qubits=1, method='mle')

        assert rho_linear.shape == rho_mle.shape
        assert rho_linear.shape == (2, 2)

    def test_tomography_multiple_qubits(self):
        """Test tomography scales to multiple qubits."""
        # Two-qubit tomography
        tomo = StateTomoGraphy(n_qubits=2)
        assert tomo.n_qubits == 2

        # Add measurement
        tomo.add_measurement('ZZ', np.array([0.25, 0.25, 0.25, 0.25]))

        rho = tomo.reconstruct(method='linear')
        assert rho.shape == (4, 4)
