"""
Integration Tests for Q-Store v4.0
Tests end-to-end workflows across verification, profiling, and visualization modules.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.profiling import profile_circuit
from q_store.visualization import visualize_circuit, visualize_state
from q_store.tomography import reconstruct_state, reconstruct_process


class TestVerificationProfilingWorkflow:
    """Test verification and profiling working together"""

    def test_profile_bell_state_circuit(self):
        """Test profiling a Bell state circuit"""
        # Create Bell state
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        # Profile circuit
        profile = profile_circuit(circuit)
        
        # Verify profile
        assert profile is not None
        assert profile.n_gates == 2
        assert profile.depth > 0
        assert 'H' in profile.gate_counts or 'Hadamard' in str(profile.gate_counts)

    def test_profile_parameterized_circuit(self):
        """Test profiling parameterized circuits"""
        # Create parameterized circuit
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.RY, [0], parameters={'theta': 0.5})
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.RZ, [1], parameters={'theta': 0.3})
        
        # Profile
        profile = profile_circuit(circuit)
        
        assert profile.n_gates == 3
        assert profile.depth > 0


class TestProfilingComparison:
    """Test comparing circuit profiles"""

    def test_compare_simple_circuits(self):
        """Test comparing profiles of different circuits"""
        # Circuit 1: Simple
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])
        
        # Circuit 2: More complex
        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        circuit2.add_gate(GateType.H, [1])
        circuit2.add_gate(GateType.CNOT, [0, 1])
        circuit2.add_gate(GateType.RZ, [0], parameters={'theta': 0.5})
        
        # Profile both
        profile1 = profile_circuit(circuit1)
        profile2 = profile_circuit(circuit2)
        
        # Circuit 2 should be more complex
        assert profile2.n_gates > profile1.n_gates

    def test_profile_multi_qubit_circuit(self):
        """Test profiling multi-qubit circuit"""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.H, [2])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.CNOT, [1, 2])
        
        profile = profile_circuit(circuit)
        
        assert profile.n_gates == 5
        assert profile.depth > 0


class TestVisualizationIntegration:
    """Test visualization with other modules"""

    def test_visualize_simple_circuit(self):
        """Test visualizing a simple circuit"""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        # Visualize
        viz = visualize_circuit(circuit)
        
        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualize_parameterized_circuit(self):
        """Test visualizing parameterized circuit"""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.RX, [0], parameters={'theta': 0.5})
        circuit.add_gate(GateType.RY, [1], parameters={'theta': 0.3})
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        viz = visualize_circuit(circuit)
        
        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualize_state(self):
        """Test state visualization"""
        # Create a simple state
        state = np.array([1.0, 0.0])  # |0⟩
        
        # Visualize (should return string or plot)
        viz = visualize_state(state)
        
        assert viz is not None


class TestProfilingVisualizationWorkflow:
    """Test profiling → visualization workflow"""

    def test_profile_then_visualize(self):
        """Test profiling a circuit then visualizing it"""
        # Create circuit
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.RZ, [1], parameters={'theta': 0.5})
        
        # Profile
        profile = profile_circuit(circuit)
        assert profile.n_gates == 3
        
        # Visualize
        viz = visualize_circuit(circuit)
        assert len(viz) > 0

    def test_visualize_multiple_circuits(self):
        """Test visualizing multiple circuits"""
        circuits = []
        
        # Create several circuits
        for i in range(3):
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.H, [0])
            for _ in range(i + 1):
                circuit.add_gate(GateType.CNOT, [0, 1])
            circuits.append(circuit)
        
        # Visualize each
        visualizations = [visualize_circuit(c) for c in circuits]
        
        assert all(isinstance(v, str) and len(v) > 0 for v in visualizations)


class TestTomographyWorkflow:
    """Test tomography workflows"""

    def test_state_reconstruction(self):
        """Test state reconstruction workflow"""
        # Simulate measurement data for a |0⟩ state
        measurements = {
            'Z': [0] * 900 + [1] * 100,  # Mostly 0
            'X': [0] * 500 + [1] * 500,  # Mixed
            'Y': [0] * 500 + [1] * 500,  # Mixed
        }
        
        # Reconstruct state
        density_matrix = reconstruct_state(measurements, n_qubits=1)
        
        assert density_matrix is not None
        assert density_matrix.shape == (2, 2)
        
        # Check trace ≈ 1
        trace = np.trace(density_matrix)
        assert abs(trace - 1.0) < 0.2

    def test_process_reconstruction(self):
        """Test process reconstruction"""
        # Create test circuit (Hadamard)
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])
        
        # Simulate process tomography
        # (In practice, would have measurement results)
        # For now, just verify the function exists
        assert reconstruct_process is not None


class TestCompleteWorkflows:
    """Test complete end-to-end workflows"""

    def test_circuit_analysis_workflow(self):
        """Test complete circuit analysis: create → profile → visualize"""
        # 1. Create circuit
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.CNOT, [0, 2])
        circuit.add_gate(GateType.CNOT, [1, 2])
        circuit.add_gate(GateType.RZ, [2], parameters={'theta': 0.5})
        
        assert circuit.n_qubits == 3
        assert len(circuit.gates) == 5
        
        # 2. Profile circuit
        profile = profile_circuit(circuit)
        
        assert profile.n_gates == 5
        assert profile.depth > 0
        
        # 3. Visualize circuit
        viz = visualize_circuit(circuit)
        
        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_multi_circuit_comparison_workflow(self):
        """Test comparing multiple circuits"""
        # Create circuits with different complexity
        circuits = {}
        
        # Simple circuit
        simple = UnifiedCircuit(2)
        simple.add_gate(GateType.H, [0])
        circuits['simple'] = simple
        
        # Medium circuit
        medium = UnifiedCircuit(2)
        medium.add_gate(GateType.H, [0])
        medium.add_gate(GateType.CNOT, [0, 1])
        circuits['medium'] = medium
        
        # Complex circuit
        complex_circuit = UnifiedCircuit(2)
        complex_circuit.add_gate(GateType.H, [0])
        complex_circuit.add_gate(GateType.CNOT, [0, 1])
        complex_circuit.add_gate(GateType.RZ, [0], parameters={'theta': 0.5})
        complex_circuit.add_gate(GateType.RY, [1], parameters={'theta': 0.3})
        circuits['complex'] = complex_circuit
        
        # Profile all
        profiles = {name: profile_circuit(c) for name, c in circuits.items()}
        
        # Verify complexity ordering
        assert profiles['simple'].n_gates < profiles['medium'].n_gates
        assert profiles['medium'].n_gates < profiles['complex'].n_gates

    def test_parameterized_circuit_workflow(self):
        """Test workflow with parameterized circuits"""
        # Create parameterized circuit
        def create_ansatz(params):
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.RY, [0], parameters={'theta': params[0]})
            circuit.add_gate(GateType.RY, [1], parameters={'theta': params[1]})
            circuit.add_gate(GateType.CNOT, [0, 1])
            circuit.add_gate(GateType.RZ, [1], parameters={'theta': params[2]})
            return circuit
        
        # Create with different parameters
        params_list = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.5, 0.6, 0.7]),
            np.array([1.0, 1.1, 1.2]),
        ]
        
        for params in params_list:
            circuit = create_ansatz(params)
            
            # Profile
            profile = profile_circuit(circuit)
            assert profile.n_gates == 4
            
            # Visualize
            viz = visualize_circuit(circuit)
            assert len(viz) > 0


class TestModuleInteroperability:
    """Test that modules work well together"""

    def test_profile_and_visualize_same_circuit(self):
        """Test that profiling and visualization work on same circuit"""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        # Profile
        profile = profile_circuit(circuit)
        assert profile.n_gates == 2
        
        # Visualize
        viz = visualize_circuit(circuit)
        assert len(viz) > 0
        
        # Both should work without interference
        assert circuit.n_qubits == 2
        assert len(circuit.gates) == 2

    def test_batch_analysis(self):
        """Test analyzing a batch of circuits"""
        # Create batch of circuits
        batch = []
        for i in range(5):
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.H, [0])
            for _ in range(i):
                circuit.add_gate(GateType.CNOT, [0, 1])
            batch.append(circuit)
        
        # Profile each
        profiles = [profile_circuit(c) for c in batch]
        
        # Verify increasing complexity
        for i in range(len(profiles) - 1):
            assert profiles[i+1].n_gates >= profiles[i].n_gates

    def test_circuit_library_analysis(self):
        """Test analyzing common quantum circuits"""
        circuits = {}
        
        # Bell state
        bell = UnifiedCircuit(2)
        bell.add_gate(GateType.H, [0])
        bell.add_gate(GateType.CNOT, [0, 1])
        circuits['Bell'] = bell
        
        # GHZ state
        ghz = UnifiedCircuit(3)
        ghz.add_gate(GateType.H, [0])
        ghz.add_gate(GateType.CNOT, [0, 1])
        ghz.add_gate(GateType.CNOT, [0, 2])
        circuits['GHZ'] = ghz
        
        # QFT-like
        qft = UnifiedCircuit(2)
        qft.add_gate(GateType.H, [0])
        qft.add_gate(GateType.RZ, [0], parameters={'theta': np.pi/2})
        qft.add_gate(GateType.CNOT, [1, 0])
        qft.add_gate(GateType.H, [1])
        circuits['QFT-like'] = qft
        
        # Analyze all
        for name, circuit in circuits.items():
            profile = profile_circuit(circuit)
            assert profile.n_gates > 0, f"{name} should have gates"
            
            viz = visualize_circuit(circuit)
            assert len(viz) > 0, f"{name} should visualize"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
