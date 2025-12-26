"""
Tests for quantum algorithm templates.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.templates import (
    qft, inverse_qft, qft_rotations,
    grover_circuit, grover_diffusion, create_oracle_from_bitstring, grover_search,
    phase_estimation, iterative_phase_estimation, create_phase_estimation_unitary,
    amplitude_amplification, fixed_point_amplification
)


class TestQFT:
    """Test Quantum Fourier Transform."""

    def test_qft_creates_circuit(self):
        """Test that QFT creates a valid circuit."""
        circuit = qft(n_qubits=3)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_qft_has_hadamards(self):
        """Test that QFT contains Hadamard gates."""
        circuit = qft(n_qubits=3)
        hadamard_gates = [g for g in circuit.gates if g.gate_type == GateType.H]
        assert len(hadamard_gates) == 3

    def test_qft_has_swaps(self):
        """Test that QFT contains SWAP gates."""
        circuit = qft(n_qubits=4)
        swap_gates = [g for g in circuit.gates if g.gate_type == GateType.SWAP]
        assert len(swap_gates) == 2  # n_qubits // 2

    def test_inverse_qft_creates_circuit(self):
        """Test inverse QFT creation."""
        circuit = inverse_qft(n_qubits=3)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_qft_approximate(self):
        """Test approximate QFT with reduced depth."""
        full_circuit = qft(n_qubits=5)
        approx_circuit = qft(n_qubits=5, approximation_degree=2)
        assert len(approx_circuit.gates) < len(full_circuit.gates)

    def test_qft_rotations_no_swaps(self):
        """Test QFT rotations without final swaps."""
        circuit = qft_rotations(n_qubits=3)
        swap_gates = [g for g in circuit.gates if g.gate_type == GateType.SWAP]
        assert len(swap_gates) == 0

    def test_qft_single_qubit(self):
        """Test QFT on single qubit (should just be Hadamard)."""
        circuit = qft(n_qubits=1)
        assert circuit.n_qubits == 1
        # Single qubit QFT is just H
        h_gates = [g for g in circuit.gates if g.gate_type == GateType.H]
        assert len(h_gates) == 1

    def test_qft_two_qubits(self):
        """Test QFT on two qubits."""
        circuit = qft(n_qubits=2)
        assert circuit.n_qubits == 2
        assert len(circuit.gates) > 2  # H gates + controlled rotations + swap


class TestGrover:
    """Test Grover's algorithm."""

    def test_grover_circuit_creates(self):
        """Test Grover circuit creation."""
        def dummy_oracle(circuit):
            circuit.add_gate(GateType.Z, targets=[0])

        circuit = grover_circuit(n_qubits=3, oracle=dummy_oracle, n_iterations=1)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_grover_has_hadamards(self):
        """Test that Grover circuit initializes with Hadamards."""
        def dummy_oracle(circuit):
            pass

        circuit = grover_circuit(n_qubits=3, oracle=dummy_oracle, n_iterations=1)
        hadamards = [g for g in circuit.gates if g.gate_type == GateType.H]
        assert len(hadamards) >= 3  # Initial superposition

    def test_grover_default_iterations(self):
        """Test default iteration count."""
        def dummy_oracle(circuit):
            pass

        # For 3 qubits, optimal iterations ≈ π/4 * √8 ≈ 2.22 → 2
        circuit = grover_circuit(n_qubits=3, oracle=dummy_oracle)
        # Circuit should have multiple iterations
        assert len(circuit.gates) > 3

    def test_grover_diffusion_operator(self):
        """Test diffusion operator creation."""
        circuit = grover_diffusion(n_qubits=3)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_create_oracle_from_bitstring(self):
        """Test oracle creation from bitstring."""
        oracle = create_oracle_from_bitstring("101")
        circuit = UnifiedCircuit(n_qubits=3)
        oracle(circuit)
        assert len(circuit.gates) > 0

    def test_grover_search_single_target(self):
        """Test Grover search for single target."""
        circuit = grover_search(n_qubits=3, marked_states=["101"], n_iterations=1)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_grover_search_multiple_targets(self):
        """Test Grover search for multiple targets."""
        circuit = grover_search(
            n_qubits=3,
            marked_states=["101", "110"],
            n_iterations=1
        )
        assert circuit.n_qubits == 3

    def test_grover_small_system(self):
        """Test Grover on 2-qubit system."""
        circuit = grover_search(n_qubits=2, marked_states=["11"], n_iterations=1)
        assert circuit.n_qubits == 2


class TestPhaseEstimation:
    """Test Quantum Phase Estimation."""

    def test_phase_estimation_creates_circuit(self):
        """Test QPE circuit creation."""
        def dummy_unitary(circuit, power):
            # Simple unitary for testing
            circuit.add_gate(GateType.RZ, targets=[2], parameters={'angle': np.pi / 4})

        circuit = phase_estimation(
            n_counting_qubits=3,
            n_target_qubits=1,
            unitary=dummy_unitary
        )
        assert circuit.n_qubits == 4
        assert len(circuit.gates) > 0

    def test_phase_estimation_has_hadamards(self):
        """Test that QPE initializes counting qubits with Hadamards."""
        def dummy_unitary(circuit, power):
            pass

        circuit = phase_estimation(
            n_counting_qubits=3,
            n_target_qubits=1,
            unitary=dummy_unitary
        )
        hadamards = [g for g in circuit.gates if g.gate_type == GateType.H]
        assert len(hadamards) >= 3

    def test_phase_estimation_with_state_prep(self):
        """Test QPE with state preparation."""
        def state_prep(circuit):
            circuit.add_gate(GateType.X, targets=[3])

        def dummy_unitary(circuit, power):
            pass

        circuit = phase_estimation(
            n_counting_qubits=3,
            n_target_qubits=1,
            unitary=dummy_unitary,
            state_preparation=state_prep
        )
        # Should have X gate from state prep
        x_gates = [g for g in circuit.gates if g.gate_type == GateType.X]
        assert len(x_gates) >= 1

    def test_iterative_phase_estimation(self):
        """Test iterative phase estimation (1 counting qubit)."""
        def dummy_unitary(circuit, power):
            circuit.add_gate(GateType.RZ, targets=[1], parameters={'angle': np.pi / 8})

        circuit = iterative_phase_estimation(
            n_target_qubits=1,
            unitary=dummy_unitary,
            precision_bits=8
        )
        assert circuit.n_qubits == 2  # 1 counting + 1 target

    def test_create_phase_estimation_unitary(self):
        """Test helper for creating simple unitaries."""
        unitary = create_phase_estimation_unitary(
            gate_type=GateType.RZ,
            target_qubit=1,
            angle=np.pi / 4
        )
        circuit = UnifiedCircuit(n_qubits=2)
        unitary(circuit, power=2)
        # Should apply gate twice
        rz_gates = [g for g in circuit.gates if g.gate_type == GateType.RZ]
        assert len(rz_gates) == 2

    def test_phase_estimation_multiple_target_qubits(self):
        """Test QPE with multiple target qubits."""
        def dummy_unitary(circuit, power):
            pass

        circuit = phase_estimation(
            n_counting_qubits=4,
            n_target_qubits=3,
            unitary=dummy_unitary
        )
        assert circuit.n_qubits == 7


class TestAmplitudeAmplification:
    """Test Amplitude Amplification."""

    def test_amplitude_amplification_creates_circuit(self):
        """Test amplitude amplification circuit creation."""
        def state_prep(circuit):
            for i in range(circuit.n_qubits):
                circuit.add_gate(GateType.H, targets=[i])

        def oracle(circuit):
            circuit.add_gate(GateType.Z, targets=[0])

        circuit = amplitude_amplification(
            n_qubits=3,
            state_preparation=state_prep,
            oracle=oracle,
            n_iterations=1
        )
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_amplitude_amplification_has_state_prep(self):
        """Test that amplitude amplification applies state preparation."""
        def state_prep(circuit):
            circuit.add_gate(GateType.X, targets=[0])

        def oracle(circuit):
            pass

        circuit = amplitude_amplification(
            n_qubits=2,
            state_preparation=state_prep,
            oracle=oracle,
            n_iterations=1
        )
        x_gates = [g for g in circuit.gates if g.gate_type == GateType.X]
        assert len(x_gates) > 0

    def test_amplitude_amplification_default_iterations(self):
        """Test default iteration calculation."""
        def state_prep(circuit):
            for i in range(circuit.n_qubits):
                circuit.add_gate(GateType.H, targets=[i])

        def oracle(circuit):
            pass

        circuit = amplitude_amplification(
            n_qubits=3,
            state_preparation=state_prep,
            oracle=oracle
        )
        # Should have calculated iterations automatically
        assert len(circuit.gates) > 3

    def test_fixed_point_amplification(self):
        """Test fixed-point amplitude amplification."""
        def state_prep(circuit):
            for i in range(circuit.n_qubits):
                circuit.add_gate(GateType.H, targets=[i])

        def oracle(circuit):
            circuit.add_gate(GateType.Z, targets=[0])

        circuit = fixed_point_amplification(
            n_qubits=3,
            state_preparation=state_prep,
            oracle=oracle
        )
        assert circuit.n_qubits == 3
        assert len(circuit.gates) > 0

    def test_amplitude_amplification_with_custom_oracle(self):
        """Test amplitude amplification with custom oracle."""
        def state_prep(circuit):
            circuit.add_gate(GateType.H, targets=[0])
            circuit.add_gate(GateType.H, targets=[1])

        def custom_oracle(circuit):
            # Mark |11⟩
            circuit.add_gate(GateType.CZ, targets=[0, 1])

        circuit = amplitude_amplification(
            n_qubits=2,
            state_preparation=state_prep,
            oracle=custom_oracle,
            n_iterations=1
        )
        assert circuit.n_qubits == 2


class TestTemplateIntegration:
    """Integration tests for templates."""

    def test_qft_composition(self):
        """Test composing QFT with other operations."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])

        qft_circuit = qft(n_qubits=3)
        for gate in qft_circuit.gates:
            circuit.add_gate(gate.gate_type, targets=gate.targets, parameters=gate.parameters)

        assert len(circuit.gates) > 1

    def test_grover_with_qft(self):
        """Test using Grover with QFT."""
        # Create Grover circuit
        grover = grover_search(n_qubits=3, marked_states=["101"], n_iterations=1)

        # Add QFT after search
        qft_circuit = qft(n_qubits=3)
        for gate in qft_circuit.gates:
            grover.add_gate(gate.gate_type, targets=gate.targets, parameters=gate.parameters)

        assert len(grover.gates) > 10

    def test_template_gate_counts(self):
        """Test that templates produce reasonable gate counts."""
        qft_circuit = qft(n_qubits=4)
        grover = grover_search(n_qubits=3, marked_states=["101"], n_iterations=1)

        # QFT should have O(n^2) gates
        assert len(qft_circuit.gates) < 50

        # Grover should have reasonable gate count
        assert len(grover.gates) < 100

    def test_multiple_templates(self):
        """Test creating multiple templates."""
        circuits = [
            qft(n_qubits=3),
            inverse_qft(n_qubits=3),
            grover_diffusion(n_qubits=3),
            qft_rotations(n_qubits=3)
        ]
        for circuit in circuits:
            assert circuit.n_qubits == 3
            assert len(circuit.gates) > 0
