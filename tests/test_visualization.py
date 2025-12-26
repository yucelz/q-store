"""
Tests for visualization module.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.visualization import (
    CircuitVisualizer,
    visualize_circuit,
    VisualizationConfig,
    StateVisualizer,
    visualize_state,
    BlochSphere,
    BlochVector,
    generate_ascii_circuit,
    circuit_to_text
)


class TestCircuitVisualizer:
    """Test circuit visualization."""

    def test_visualizer_creation(self):
        """Test creating a visualizer."""
        visualizer = CircuitVisualizer()
        assert visualizer is not None
        assert visualizer.config is not None

    def test_visualize_simple_circuit(self):
        """Test visualizing a simple circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        visualizer = CircuitVisualizer()
        ascii_art = visualizer.visualize(circuit)

        assert isinstance(ascii_art, str)
        assert "q0:" in ascii_art
        assert "q1:" in ascii_art
        assert "H" in ascii_art

    def test_visualization_config(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(width=100, show_measurements=False)
        visualizer = CircuitVisualizer(config)

        assert visualizer.config.width == 100
        assert visualizer.config.show_measurements == False

    def test_single_qubit_gates(self):
        """Test visualizing single-qubit gates."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.X, [1])
        circuit.add_gate(GateType.Z, [2])

        visualizer = CircuitVisualizer()
        ascii_art = visualizer.visualize(circuit)

        assert "H" in ascii_art
        assert "X" in ascii_art
        assert "Z" in ascii_art

    def test_two_qubit_gates(self):
        """Test visualizing two-qubit gates."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.CNOT, [0, 1])

        visualizer = CircuitVisualizer()
        ascii_art = visualizer.visualize(circuit)

        assert "●" in ascii_art or "⊕" in ascii_art

    def test_circuit_diagram(self):
        """Test getting circuit diagram data."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        visualizer = CircuitVisualizer()
        diagram = visualizer.get_circuit_diagram(circuit)

        assert diagram['n_qubits'] == 2
        assert diagram['n_gates'] == 2
        assert diagram['depth'] == 2
        assert 'layers' in diagram
        assert 'ascii' in diagram

    def test_circuit_layers(self):
        """Test computing circuit layers."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])  # Parallel with first
        circuit.add_gate(GateType.CNOT, [0, 1])  # New layer

        visualizer = CircuitVisualizer()
        diagram = visualizer.get_circuit_diagram(circuit)
        layers = diagram['layers']

        assert len(layers) >= 1
        assert isinstance(layers, list)

    def test_compare_circuits(self):
        """Test comparing two circuit visualizations."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])

        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])

        visualizer = CircuitVisualizer()
        comparison = visualizer.compare_circuits(circuit1, circuit2)

        assert 'circuit1' in comparison
        assert 'circuit2' in comparison
        assert 'differences' in comparison
        assert comparison['differences']['gate_diff'] == -1

    def test_latex_export(self):
        """Test exporting to LaTeX."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])

        visualizer = CircuitVisualizer()
        latex = visualizer.export_latex(circuit)

        assert "quantikz" in latex
        assert "gate" in latex

    def test_convenience_function(self):
        """Test visualize_circuit convenience function."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])

        ascii_art = visualize_circuit(circuit)

        assert isinstance(ascii_art, str)
        assert "q0:" in ascii_art


class TestBlochSphere:
    """Test Bloch sphere representation."""

    def test_bloch_vector_from_zero_state(self):
        """Test Bloch vector for |0⟩ state."""
        state = np.array([1, 0])
        vector = BlochVector.from_state(state)

        assert np.isclose(vector.z, 1.0)
        assert np.isclose(vector.x, 0.0)
        assert np.isclose(vector.y, 0.0)

    def test_bloch_vector_from_one_state(self):
        """Test Bloch vector for |1⟩ state."""
        state = np.array([0, 1])
        vector = BlochVector.from_state(state)

        assert np.isclose(vector.z, -1.0)

    def test_bloch_vector_from_plus_state(self):
        """Test Bloch vector for |+⟩ state."""
        state = np.array([1, 1]) / np.sqrt(2)
        vector = BlochVector.from_state(state)

        assert np.isclose(vector.x, 1.0)
        assert np.isclose(vector.z, 0.0)

    def test_bloch_vector_angles(self):
        """Test converting to spherical angles."""
        state = np.array([1, 0])
        vector = BlochVector.from_state(state)
        theta, phi = vector.to_angles()

        assert np.isclose(theta, 0.0)  # North pole

    def test_bloch_sphere_creation(self):
        """Test creating a Bloch sphere."""
        sphere = BlochSphere()
        assert sphere is not None
        assert len(sphere.states) == 0

    def test_add_state_to_sphere(self):
        """Test adding states to Bloch sphere."""
        sphere = BlochSphere()
        state = np.array([1, 0])

        sphere.add_state(state, "zero")

        assert len(sphere.states) == 1
        assert sphere.states[0][0] == "zero"

    def test_clear_sphere(self):
        """Test clearing Bloch sphere."""
        sphere = BlochSphere()
        sphere.add_state(np.array([1, 0]), "test")
        sphere.clear()

        assert len(sphere.states) == 0

    def test_ascii_representation(self):
        """Test ASCII representation of Bloch sphere."""
        sphere = BlochSphere()
        sphere.add_state(np.array([1, 0]), "|0⟩")

        ascii_art = sphere.get_ascii_representation()

        assert isinstance(ascii_art, str)
        assert "|z⟩" in ascii_art
        assert "States:" in ascii_art

    def test_state_data(self):
        """Test getting state data."""
        sphere = BlochSphere()
        state = np.array([1, 1]) / np.sqrt(2)
        sphere.add_state(state, "|+⟩")

        data = sphere.get_state_data()

        assert len(data) == 1
        assert 'x' in data[0]
        assert 'y' in data[0]
        assert 'z' in data[0]
        assert 'theta' in data[0]
        assert 'phi' in data[0]


class TestStateVisualizer:
    """Test state visualization."""

    def test_visualizer_creation(self):
        """Test creating a state visualizer."""
        visualizer = StateVisualizer()
        assert visualizer is not None

    def test_visualize_simple_statevector(self):
        """Test visualizing a simple state vector."""
        state = np.array([1, 0])

        visualizer = StateVisualizer()
        text = visualizer.visualize_statevector(state)

        assert isinstance(text, str)
        assert "|0⟩" in text or "|00⟩" in text

    def test_visualize_superposition(self):
        """Test visualizing superposition state."""
        state = np.array([1, 1]) / np.sqrt(2)

        visualizer = StateVisualizer()
        text = visualizer.visualize_statevector(state)

        assert "prob:" in text.lower()

    def test_visualize_density_matrix(self):
        """Test visualizing density matrix."""
        # Pure state density matrix
        state = np.array([1, 0])
        rho = np.outer(state, state.conj())

        visualizer = StateVisualizer()
        text = visualizer.visualize_density_matrix(rho)

        assert "Density matrix" in text
        assert "Trace:" in text
        assert "Purity:" in text

    def test_visualize_probabilities(self):
        """Test visualizing measurement probabilities."""
        state = np.array([1, 1, 0, 0]) / np.sqrt(2)

        visualizer = StateVisualizer()
        text = visualizer.visualize_probabilities(state)

        assert "probabilities" in text.lower()
        assert "█" in text  # Bar chart symbol

    def test_probability_top_k(self):
        """Test showing only top k probabilities."""
        state = np.array([1, 1, 1, 1]) / 2

        visualizer = StateVisualizer()
        text = visualizer.visualize_probabilities(state, top_k=2)

        # Should only show 2 states
        lines = [l for l in text.split('\n') if '|' in l and '⟩' in l]
        assert len(lines) == 2

    def test_create_bloch_sphere(self):
        """Test creating Bloch sphere from state."""
        state = np.array([1, 0])

        visualizer = StateVisualizer()
        sphere = visualizer.create_bloch_sphere(state)

        assert isinstance(sphere, BlochSphere)
        assert len(sphere.states) == 1

    def test_bloch_sphere_invalid_state(self):
        """Test error for non-single-qubit state."""
        state = np.array([1, 0, 0, 0])  # 2-qubit state

        visualizer = StateVisualizer()

        with pytest.raises(ValueError):
            visualizer.create_bloch_sphere(state)

    def test_compare_identical_states(self):
        """Test comparing identical states."""
        state = np.array([1, 0])

        visualizer = StateVisualizer()
        comparison = visualizer.compare_states(state, state)

        assert np.isclose(comparison['fidelity'], 1.0)
        assert np.isclose(comparison['overlap'], 1.0)
        assert comparison['are_orthogonal'] == False

    def test_compare_orthogonal_states(self):
        """Test comparing orthogonal states."""
        state1 = np.array([1, 0])
        state2 = np.array([0, 1])

        visualizer = StateVisualizer()
        comparison = visualizer.compare_states(state1, state2)

        assert np.isclose(comparison['fidelity'], 0.0)
        assert comparison['are_orthogonal'] == True

    def test_convenience_function_statevector(self):
        """Test visualize_state convenience function."""
        state = np.array([1, 0])

        text = visualize_state(state, mode='statevector')

        assert isinstance(text, str)
        assert "State vector" in text

    def test_convenience_function_probabilities(self):
        """Test visualize_state with probabilities mode."""
        state = np.array([1, 1]) / np.sqrt(2)

        text = visualize_state(state, mode='probabilities')

        assert "probabilities" in text.lower()


class TestVisualizationUtils:
    """Test visualization utilities."""

    def test_generate_ascii_circuit(self):
        """Test generating ASCII circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        ascii_art = generate_ascii_circuit(circuit)

        assert isinstance(ascii_art, str)
        assert "Circuit:" in ascii_art
        assert "qubits" in ascii_art

    def test_circuit_to_text_simple(self):
        """Test simple circuit to text."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])

        text = circuit_to_text(circuit, detailed=False)

        assert "UnifiedCircuit" in text
        assert "Gates:" in text

    def test_circuit_to_text_detailed(self):
        """Test detailed circuit to text."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.CNOT, [0, 1])

        text = circuit_to_text(circuit, detailed=True)

        assert "Gate counts:" in text
        assert "Gate sequence:" in text


class TestIntegration:
    """Test integration between visualization components."""

    def test_circuit_and_state_visualization(self):
        """Test visualizing both circuit and resulting state."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])

        # Visualize circuit
        circuit_viz = visualize_circuit(circuit)
        assert "H" in circuit_viz

        # Visualize a state (simulated result)
        state = np.array([1, 1, 0, 0]) / np.sqrt(2)
        state_viz = visualize_state(state, mode='probabilities')
        assert "probabilities" in state_viz.lower()

    def test_complete_workflow(self):
        """Test complete visualization workflow."""
        # Create circuit
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])

        # Get all visualizations
        visualizer = CircuitVisualizer()
        ascii_art = visualizer.visualize(circuit)
        diagram = visualizer.get_circuit_diagram(circuit)
        latex = visualizer.export_latex(circuit)

        assert all([ascii_art, diagram, latex])

        # State visualization
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        state_visualizer = StateVisualizer()
        state_text = state_visualizer.visualize_statevector(bell_state)
        prob_text = state_visualizer.visualize_probabilities(bell_state)

        assert all([state_text, prob_text])
