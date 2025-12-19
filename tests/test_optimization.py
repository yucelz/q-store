"""
Tests for advanced circuit optimization.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.optimization import (
    CommutationAnalyzer,
    can_commute,
    commute_gates,
    reorder_commuting_gates,
    GateFuser,
    fuse_single_qubit_gates,
    fuse_rotation_gates,
    identify_fusion_opportunities,
    ParallelizationAnalyzer,
    find_parallel_layers,
    compute_circuit_depth,
    optimize_for_parallelism,
    TemplateOptimizer,
    match_and_replace_templates,
    create_optimization_template,
    standard_templates
)


class TestCommutation:
    """Test gate commutation analysis."""
    
    def test_disjoint_qubits_commute(self):
        """Test that gates on different qubits commute."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[1])
        
        analyzer = CommutationAnalyzer(circuit)
        assert analyzer.check_commutation(0, 1) == True
    
    def test_same_qubit_not_commute(self):
        """Test that non-commuting gates on same qubit don't commute."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        
        analyzer = CommutationAnalyzer(circuit)
        assert analyzer.check_commutation(0, 1) == False
    
    def test_pauli_commutation(self):
        """Test Pauli gate commutation rules."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.Z, targets=[0])
        circuit.add_gate(GateType.Z, targets=[0])
        
        assert can_commute(circuit.gates[0], circuit.gates[1]) == True
    
    def test_cnot_z_commutation(self):
        """Test CNOT commutes with Z on control."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.Z, targets=[0])
        
        assert can_commute(circuit.gates[0], circuit.gates[1]) == True
    
    def test_commutation_analyzer(self):
        """Test full commutation analysis."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.X, targets=[0])
        
        analyzer = CommutationAnalyzer(circuit)
        relations = analyzer.analyze()
        
        assert len(relations) == 3  # 3 pairwise comparisons
        # Gates 0 and 1 should commute (different qubits)
        assert relations[0].commute == True
    
    def test_commute_gates_function(self):
        """Test swapping commuting gates."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[1])
        
        new_circuit = commute_gates(circuit, 0, 1)
        assert new_circuit.gates[0].gate_type == GateType.X
        assert new_circuit.gates[1].gate_type == GateType.H
    
    def test_commute_gates_fails_noncommuting(self):
        """Test that commuting non-commuting gates raises error."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        
        with pytest.raises(ValueError):
            commute_gates(circuit, 0, 1)
    
    def test_reorder_for_depth(self):
        """Test reordering for depth optimization."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.H, targets=[2])
        
        optimized = reorder_commuting_gates(circuit, optimize_for="depth")
        assert optimized.n_qubits == 3
        assert len(optimized.gates) == 3


class TestGateFusion:
    """Test gate fusion optimization."""
    
    def test_fusion_opportunities(self):
        """Test finding fusion opportunities."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.RX, targets=[0], parameters={'angle': 0.5})
        circuit.add_gate(GateType.RX, targets=[0], parameters={'angle': 0.3})
        
        opportunities = identify_fusion_opportunities(circuit)
        assert len(opportunities) > 0
        assert opportunities[0].savings >= 1
    
    def test_rotation_fusion(self):
        """Test fusing rotation gates."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RZ, targets=[0], parameters={'angle': np.pi / 4})
        circuit.add_gate(GateType.RZ, targets=[0], parameters={'angle': np.pi / 4})
        
        fused = fuse_rotation_gates(circuit)
        assert len(fused.gates) == 1
        assert abs(fused.gates[0].parameters['angle'] - np.pi / 2) < 1e-10
    
    def test_inverse_cancellation(self):
        """Test canceling inverse gates."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        
        fuser = GateFuser(circuit)
        opportunities = fuser.find_fusion_opportunities()
        
        # Should find cancellation opportunity
        cancel_opp = [o for o in opportunities if "Cancel" in o.description]
        assert len(cancel_opp) > 0
    
    def test_single_qubit_sequence(self):
        """Test identifying single-qubit gate sequences."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.Z, targets=[0])
        
        fuser = GateFuser(circuit)
        opportunities = fuser.find_fusion_opportunities()
        
        assert len(opportunities) > 0
    
    def test_fuse_single_qubit_gates(self):
        """Test fusing single-qubit gates."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        
        fused = fuse_single_qubit_gates(circuit)
        assert fused.n_qubits == 1


class TestParallelization:
    """Test parallelization analysis."""
    
    def test_parallel_layers(self):
        """Test finding parallel layers."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.H, targets=[2])
        
        layers = find_parallel_layers(circuit)
        assert len(layers) == 1  # All gates can run in parallel
        assert len(layers[0].gates) == 3
    
    def test_sequential_gates(self):
        """Test sequential gates on same qubit."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.Z, targets=[0])
        
        layers = find_parallel_layers(circuit)
        assert len(layers) == 3  # Must be sequential
    
    def test_circuit_depth(self):
        """Test computing circuit depth."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        
        depth = compute_circuit_depth(circuit)
        assert depth == 2  # H layer + CNOT layer
    
    def test_parallelization_analyzer(self):
        """Test parallelization analyzer."""
        circuit = UnifiedCircuit(n_qubits=3)
        for i in range(3):
            circuit.add_gate(GateType.H, targets=[i])
        
        analyzer = ParallelizationAnalyzer(circuit)
        stats = analyzer.get_parallelism_stats()
        
        assert stats['total_gates'] == 3
        assert stats['depth'] == 1
        assert stats['max_parallelism'] == 3
    
    def test_optimize_for_parallelism(self):
        """Test optimizing for parallelism."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_gate(GateType.Z, targets=[2])
        
        optimized = optimize_for_parallelism(circuit)
        assert optimized.n_qubits == 3
        assert len(optimized.gates) == 4
    
    def test_visualize_layers(self):
        """Test layer visualization."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        
        analyzer = ParallelizationAnalyzer(circuit)
        vis = analyzer.visualize_layers()
        
        assert "Circuit Depth" in vis
        assert "Layer 0" in vis


class TestTemplateMatching:
    """Test template-based optimization."""
    
    def test_standard_templates(self):
        """Test loading standard templates."""
        templates = standard_templates()
        assert len(templates) > 0
        assert any(t.name == "hadamard_cancel" for t in templates)
    
    def test_hadamard_cancellation(self):
        """Test canceling adjacent Hadamard gates."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])
        
        optimized = match_and_replace_templates(circuit)
        assert len(optimized.gates) == 0  # Both gates canceled
    
    def test_pauli_cancellation(self):
        """Test canceling adjacent Pauli gates."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        
        optimized = match_and_replace_templates(circuit)
        assert len(optimized.gates) == 0
    
    def test_s_to_z_template(self):
        """Test S^2 = Z template."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.S, targets=[0])
        circuit.add_gate(GateType.S, targets=[0])
        
        optimized = match_and_replace_templates(circuit)
        assert len(optimized.gates) == 1
        assert optimized.gates[0].gate_type == GateType.Z
    
    def test_hxh_to_z_template(self):
        """Test H X H = Z template."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])
        
        optimized = match_and_replace_templates(circuit)
        assert len(optimized.gates) == 1
        assert optimized.gates[0].gate_type == GateType.Z
    
    def test_template_optimizer(self):
        """Test template optimizer class."""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])
        
        optimizer = TemplateOptimizer(circuit)
        matches = optimizer.find_matches()
        
        assert len(matches) > 0
    
    def test_create_custom_template(self):
        """Test creating custom template."""
        template = create_optimization_template(
            name="custom",
            pattern_gates=[GateType.X, GateType.Y],
            replacement_gates=[GateType.Z],
            description="Custom XY->Z"
        )
        
        assert template.name == "custom"
        assert len(template.pattern) == 2
        assert len(template.replacement) == 1


class TestIntegratedOptimization:
    """Test combining multiple optimizations."""
    
    def test_commutation_and_fusion(self):
        """Test combining commutation and fusion."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.RX, targets=[0], parameters={'angle': 0.5})
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.RX, targets=[0], parameters={'angle': 0.3})
        
        # First reorder
        reordered = reorder_commuting_gates(circuit, optimize_for="depth")
        # Then fuse
        fused = fuse_rotation_gates(reordered)
        
        assert len(fused.gates) <= len(circuit.gates)
    
    def test_template_and_parallelization(self):
        """Test templates then parallelization."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[1])
        
        # Apply templates
        optimized = match_and_replace_templates(circuit)
        # Analyze parallelism
        depth = compute_circuit_depth(optimized)
        
        assert depth == 1  # After canceling H H, just X remains
    
    def test_full_optimization_pipeline(self):
        """Test full optimization pipeline."""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.RZ, targets=[1], parameters={'angle': 0.5})
        circuit.add_gate(GateType.RZ, targets=[1], parameters={'angle': 0.3})
        circuit.add_gate(GateType.H, targets=[2])
        
        # Apply optimizations
        step1 = match_and_replace_templates(circuit)  # Cancel X X
        step2 = fuse_rotation_gates(step1)  # Fuse RZ
        step3 = optimize_for_parallelism(step2)  # Reorder for parallelism
        
        assert len(step3.gates) < len(circuit.gates)
