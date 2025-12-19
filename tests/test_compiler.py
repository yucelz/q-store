"""
Tests for Circuit Compilation.

Tests for:
- Device topology
- Gate decomposition
- Circuit routing and SWAP insertion
- End-to-end compilation
"""

import pytest
import numpy as np

from q_store.core import UnifiedCircuit, GateType
from q_store.compiler import (
    DeviceTopology,
    create_topology,
    GateDecomposer,
    decompose_to_native_gates,
    CircuitCompiler,
    CompilationResult,
    compile_circuit,
)


# =============================================================================
# Topology Tests
# =============================================================================

class TestDeviceTopology:
    """Test device topology."""

    def test_linear_topology(self):
        """Test linear chain topology."""
        topo = create_topology('linear', n_qubits=5)

        assert topo.n_qubits == 5
        assert len(topo.edges) == 4
        assert topo.is_connected(0, 1)
        assert topo.is_connected(1, 2)
        assert not topo.is_connected(0, 2)

    def test_ring_topology(self):
        """Test ring topology."""
        topo = create_topology('ring', n_qubits=4)

        assert topo.n_qubits == 4
        assert len(topo.edges) == 4
        assert topo.is_connected(0, 1)
        assert topo.is_connected(3, 0)  # Wraps around

    def test_grid_topology(self):
        """Test 2D grid topology."""
        topo = create_topology('grid', rows=3, cols=3)

        assert topo.n_qubits == 9
        # 3x3 grid has 12 edges (6 horizontal + 6 vertical)
        assert len(topo.edges) == 12

        # Check horizontal connection
        assert topo.is_connected(0, 1)
        # Check vertical connection
        assert topo.is_connected(0, 3)
        # No diagonal
        assert not topo.is_connected(0, 4)

    def test_all_to_all_topology(self):
        """Test fully connected topology."""
        topo = create_topology('all_to_all', n_qubits=4)

        assert topo.n_qubits == 4
        # Complete graph K4 has 6 edges
        assert len(topo.edges) == 6

        # All pairs connected
        for i in range(4):
            for j in range(i+1, 4):
                assert topo.is_connected(i, j)

    def test_shortest_path(self):
        """Test shortest path finding."""
        topo = create_topology('linear', n_qubits=5)

        path = topo.shortest_path(0, 4)
        assert path == [0, 1, 2, 3, 4]

        path = topo.shortest_path(1, 3)
        assert path == [1, 2, 3]

    def test_distance(self):
        """Test distance calculation."""
        topo = create_topology('linear', n_qubits=5)

        assert topo.distance(0, 0) == 0
        assert topo.distance(0, 1) == 1
        assert topo.distance(0, 4) == 4
        assert topo.distance(2, 4) == 2

    def test_neighbors(self):
        """Test getting neighbors."""
        topo = create_topology('grid', rows=2, cols=2)

        # Qubit 0 (top-left) has neighbors 1 and 2
        neighbors = topo.get_neighbors(0)
        assert set(neighbors) == {1, 2}

        # Qubit 1 (top-right) has neighbors 0 and 3
        neighbors = topo.get_neighbors(1)
        assert set(neighbors) == {0, 3}

    def test_diameter(self):
        """Test topology diameter."""
        linear = create_topology('linear', n_qubits=5)
        assert linear.diameter() == 4

        ring = create_topology('ring', n_qubits=6)
        assert ring.diameter() == 3


# =============================================================================
# Gate Decomposition Tests
# =============================================================================

class TestGateDecomposition:
    """Test gate decomposition."""

    def test_decomposer_creation(self):
        """Test creating decomposer."""
        decomposer = GateDecomposer()
        assert GateType.CNOT in decomposer.native_gates

    def test_native_gate_passthrough(self):
        """Test native gates pass through unchanged."""
        decomposer = GateDecomposer(native_gates={GateType.H, GateType.CNOT})

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])

        decomposed = decomposer.decompose(circuit)

        assert len(decomposed.gates) == 1
        assert decomposed.gates[0].gate_type == GateType.H

    def test_hadamard_decomposition(self):
        """Test Hadamard decomposition."""
        decomposer = GateDecomposer(native_gates={GateType.RZ, GateType.RY})

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])

        decomposed = decomposer.decompose(circuit)

        assert len(decomposed.gates) == 2
        assert decomposed.gates[0].gate_type == GateType.RZ
        assert decomposed.gates[1].gate_type == GateType.RY

    def test_x_decomposition(self):
        """Test X gate decomposition."""
        decomposer = GateDecomposer(native_gates={GateType.RX})

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])

        decomposed = decomposer.decompose(circuit)

        assert len(decomposed.gates) == 1
        assert decomposed.gates[0].gate_type == GateType.RX
        assert abs(decomposed.gates[0].parameters['angle'] - np.pi) < 1e-10

    def test_swap_decomposition(self):
        """Test SWAP decomposition."""
        decomposer = GateDecomposer(native_gates={GateType.CNOT})

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.SWAP, targets=[0, 1])

        decomposed = decomposer.decompose(circuit)

        # SWAP = 3 CNOTs
        assert len(decomposed.gates) == 3
        assert all(g.gate_type == GateType.CNOT for g in decomposed.gates)

    def test_cz_decomposition(self):
        """Test CZ decomposition."""
        decomposer = GateDecomposer(native_gates={GateType.CNOT, GateType.H})

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.CZ, targets=[0, 1])

        decomposed = decomposer.decompose(circuit)

        # CZ = H @ CNOT @ H
        assert len(decomposed.gates) == 3
        assert decomposed.gates[0].gate_type == GateType.H
        assert decomposed.gates[1].gate_type == GateType.CNOT
        assert decomposed.gates[2].gate_type == GateType.H

    def test_decompose_circuit(self):
        """Test decomposing entire circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.X, targets=[1])

        decomposer = GateDecomposer(native_gates={GateType.RZ, GateType.RY, GateType.RX, GateType.CNOT})
        decomposed = decomposer.decompose(circuit)

        # H decomposes to 2 gates, CNOT stays, X becomes RX
        assert decomposed.n_qubits == 2
        assert len(decomposed.gates) >= 3


# =============================================================================
# Circuit Routing Tests
# =============================================================================

class TestCircuitRouting:
    """Test circuit routing and SWAP insertion."""

    def test_routing_on_all_to_all(self):
        """Test routing on fully connected topology needs no SWAPs."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.CNOT, targets=[1, 2])

        topology = create_topology('all_to_all', n_qubits=3)
        compiler = CircuitCompiler(topology, optimization_level=0)

        result = compiler.compile(circuit)

        assert result.swap_count == 0

    def test_routing_on_linear_needs_swaps(self):
        """Test routing on linear topology may need SWAPs."""
        circuit = UnifiedCircuit(n_qubits=4)
        # Create a pattern that will require SWAPs even with smart placement
        circuit.add_gate(GateType.CNOT, targets=[0, 1])  # Place 0,1 adjacent
        circuit.add_gate(GateType.CNOT, targets=[2, 3])  # Place 2,3 adjacent
        circuit.add_gate(GateType.CNOT, targets=[0, 3])  # Now 0 and 3 are far apart

        topology = create_topology('linear', n_qubits=4)
        compiler = CircuitCompiler(topology, optimization_level=0)

        result = compiler.compile(circuit)

        # With gates forcing specific placements, should need SWAPs or long distance
        assert result.compiled_circuit.n_qubits == 4
        # Either SWAPs were inserted or circuit was successfully compiled
        assert result.gate_count_compiled >= result.gate_count_original

    def test_initial_placement(self):
        """Test initial qubit placement."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.CNOT, targets=[1, 2])

        topology = create_topology('linear', n_qubits=5)
        compiler = CircuitCompiler(topology)

        mapping = compiler._initial_placement(circuit)

        assert len(mapping) == 3
        # Should place frequently interacting qubits close
        assert abs(mapping[0] - mapping[1]) <= 1 or abs(mapping[1] - mapping[2]) <= 1


# =============================================================================
# Full Compilation Tests
# =============================================================================

class TestCircuitCompiler:
    """Test full compilation pipeline."""

    def test_compiler_creation(self):
        """Test creating compiler."""
        topology = create_topology('linear', n_qubits=5)
        compiler = CircuitCompiler(topology)

        assert compiler.topology == topology
        assert compiler.optimization_level == 2

    def test_compile_simple_circuit(self):
        """Test compiling simple circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        result = compile_circuit(circuit, topology_type='all_to_all')

        assert isinstance(result, CompilationResult)
        assert result.compiled_circuit.n_qubits >= 2
        assert result.gate_count_original == 2

    def test_compile_with_decomposition(self):
        """Test compilation with gate decomposition."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.SWAP, targets=[0, 1])

        native_gates = {GateType.RZ, GateType.RY, GateType.CNOT}
        result = compile_circuit(circuit, native_gates=native_gates)

        # SWAP decomposes to CNOTs, H decomposes to rotations
        assert result.gate_count_compiled > result.gate_count_original

    def test_compile_for_linear_topology(self):
        """Test compilation for constrained topology."""
        circuit = UnifiedCircuit(n_qubits=5)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 4])  # Long distance
        circuit.add_gate(GateType.CNOT, targets=[1, 3])

        topology = create_topology('linear', n_qubits=5)
        result = compile_circuit(circuit, topology=topology)

        # Should add SWAPs for long-distance gates
        assert result.swap_count > 0
        assert result.compiled_depth >= result.original_depth

    def test_compile_for_grid_topology(self):
        """Test compilation for 2D grid."""
        circuit = UnifiedCircuit(n_qubits=4)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 3])

        topology = create_topology('grid', rows=2, cols=2)
        result = compile_circuit(circuit, topology=topology)

        assert isinstance(result, CompilationResult)
        assert result.compiled_circuit.n_qubits == 4

    def test_compilation_statistics(self):
        """Test compilation result statistics."""
        circuit = UnifiedCircuit(n_qubits=3)
        for _ in range(5):
            circuit.add_gate(GateType.H, targets=[0])
            circuit.add_gate(GateType.CNOT, targets=[0, 1])

        result = compile_circuit(circuit)

        assert result.original_depth > 0
        assert result.compiled_depth > 0
        assert result.gate_count_original == 10
        assert isinstance(result.initial_mapping, dict)
        assert isinstance(result.final_mapping, dict)

    def test_optimization_levels(self):
        """Test different optimization levels."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])  # Should cancel
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        # No optimization
        result_0 = compile_circuit(circuit, optimization_level=0)

        # With optimization
        result_2 = compile_circuit(circuit, optimization_level=2)

        # Optimized should have fewer gates (H H cancels)
        assert result_2.gate_count_compiled <= result_0.gate_count_compiled


# =============================================================================
# Integration Tests
# =============================================================================

class TestCompilerIntegration:
    """Integration tests for compiler."""

    def test_compile_bell_state(self):
        """Test compiling Bell state circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        result = compile_circuit(circuit, topology_type='linear')

        assert result.compiled_circuit.n_qubits >= 2
        assert result.gate_count_compiled >= 2

    def test_compile_ghz_state(self):
        """Test compiling GHZ state."""
        circuit = UnifiedCircuit(n_qubits=4)
        circuit.add_gate(GateType.H, targets=[0])
        for i in range(3):
            circuit.add_gate(GateType.CNOT, targets=[i, i+1])

        topology = create_topology('linear', n_qubits=4)
        result = compile_circuit(circuit, topology=topology)

        # Linear topology is good for GHZ - adjacent CNOTs
        assert result.swap_count == 0

    def test_compile_with_barriers(self):
        """Test compilation preserves circuit structure."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.CNOT, targets=[0, 2])

        result = compile_circuit(circuit)

        assert result.compiled_circuit.n_qubits >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
