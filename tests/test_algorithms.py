"""
Tests for Variational Algorithms (VQE and QAOA).

Tests for:
- VQE with different ansatzes
- QAOA for optimization problems
- Convergence and optimization
"""

import pytest
import numpy as np

from q_store.core import UnifiedCircuit, GateType
from q_store.algorithms import VQE, VQEResult, QAOA, QAOAResult
from q_store.algorithms.vqe import create_hardware_efficient_ansatz, create_uccsd_ansatz
from q_store.algorithms.qaoa import create_maxcut_hamiltonian, create_partition_hamiltonian


# =============================================================================
# VQE Tests
# =============================================================================

class TestVQE:
    """Test Variational Quantum Eigensolver."""
    
    def test_vqe_creation(self):
        """Test creating VQE instance."""
        # Simple 1-qubit Hamiltonian: H = Z
        hamiltonian = [(1.0, 'Z')]
        
        def simple_ansatz(params):
            circuit = UnifiedCircuit(n_qubits=1)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit
        
        vqe = VQE(hamiltonian, simple_ansatz)
        assert vqe.hamiltonian == hamiltonian
        assert vqe.optimizer_method == 'COBYLA'
    
    def test_vqe_simple_hamiltonian(self):
        """Test VQE on simple Hamiltonian."""
        # H = Z (eigenvalues ±1)
        hamiltonian = [(1.0, 'Z')]
        
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=1)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit
        
        vqe = VQE(hamiltonian, ansatz, max_iterations=50)
        result = vqe.run(initial_parameters=np.array([0.5]))
        
        assert isinstance(result, VQEResult)
        # Ground state energy should be close to -1
        assert abs(result.eigenvalue - (-1.0)) < 0.1
        assert result.n_iterations > 0
    
    def test_vqe_two_qubit_hamiltonian(self):
        """Test VQE on two-qubit system."""
        # H = Z0 + Z1 (ground state |11⟩ with E = -2)
        hamiltonian = [(1.0, 'ZI'), (1.0, 'IZ')]
        
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=2)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            circuit.add_gate(GateType.RY, targets=[1], parameters={'angle': params[1]})
            return circuit
        
        vqe = VQE(hamiltonian, ansatz, max_iterations=100)
        result = vqe.run(initial_parameters=np.array([1.0, 1.0]))
        
        # Should find ground state with E ≈ -2
        assert result.eigenvalue < 0
        assert result.eigenvalue > -2.5
    
    def test_vqe_hardware_efficient_ansatz(self):
        """Test VQE with hardware-efficient ansatz."""
        n_qubits = 2
        n_layers = 1
        
        # Simple Hamiltonian
        hamiltonian = [(1.0, 'ZI'), (-0.5, 'IZ')]
        
        ansatz_factory = create_hardware_efficient_ansatz(n_qubits, n_layers)
        
        # 2 qubits * 2 rotations * 1 layer = 4 parameters
        vqe = VQE(hamiltonian, ansatz_factory, max_iterations=50)
        result = vqe.run(initial_parameters=np.random.rand(4))
        
        assert isinstance(result, VQEResult)
        assert result.optimal_circuit.n_qubits == n_qubits
        assert len(result.iteration_history) > 0
    
    def test_vqe_uccsd_ansatz(self):
        """Test VQE with UCCSD ansatz."""
        n_qubits = 4
        n_electrons = 2
        
        # Simple Hamiltonian
        hamiltonian = [(1.0, 'ZIII'), (-0.5, 'IZII')]
        
        ansatz_factory = create_uccsd_ansatz(n_qubits, n_electrons)
        
        # n_electrons * (n_qubits - n_electrons) = 2 * 2 = 4 parameters
        vqe = VQE(hamiltonian, ansatz_factory, max_iterations=30)
        result = vqe.run(initial_parameters=np.random.rand(4))
        
        assert isinstance(result, VQEResult)
        assert result.optimal_circuit.n_qubits == n_qubits
        # Should initialize first n_electrons qubits to |1⟩
        assert any(g.gate_type == GateType.X for g in result.optimal_circuit.gates)
    
    def test_vqe_convergence(self):
        """Test VQE convergence tracking."""
        hamiltonian = [(1.0, 'Z')]
        
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=1)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit
        
        vqe = VQE(hamiltonian, ansatz, max_iterations=50, tol=1e-4)
        result = vqe.run(initial_parameters=np.array([1.0]))
        
        # Check convergence history
        assert len(result.iteration_history) > 0
        energies = [e for _, e in result.iteration_history]
        # Energy should generally decrease
        assert energies[-1] <= energies[0] + 0.1
    
    def test_vqe_different_optimizers(self):
        """Test VQE with different optimizers."""
        hamiltonian = [(1.0, 'Z')]
        
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=1)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit
        
        for method in ['COBYLA', 'Powell']:
            vqe = VQE(hamiltonian, ansatz, optimizer_method=method, max_iterations=30)
            result = vqe.run(initial_parameters=np.array([1.0]))
            
            assert isinstance(result, VQEResult)
            # Should find reasonable ground state
            assert result.eigenvalue < 0


# =============================================================================
# QAOA Tests
# =============================================================================

class TestQAOA:
    """Test Quantum Approximate Optimization Algorithm."""
    
    def test_qaoa_creation(self):
        """Test creating QAOA instance."""
        # Simple cost Hamiltonian
        hamiltonian = [(-1.0, 'ZZ')]
        
        qaoa = QAOA(hamiltonian, n_layers=1)
        assert qaoa.cost_hamiltonian == hamiltonian
        assert qaoa.n_layers == 1
        assert qaoa.n_qubits == 2
    
    def test_qaoa_circuit_creation(self):
        """Test QAOA circuit structure."""
        hamiltonian = [(-1.0, 'ZZ')]
        qaoa = QAOA(hamiltonian, n_layers=1)
        
        params = np.array([0.5, 1.0])  # [beta, gamma]
        circuit = qaoa._create_circuit(params)
        
        assert circuit.n_qubits == 2
        # Should have H gates for initial state
        assert any(g.gate_type == GateType.H for g in circuit.gates)
        # Should have rotation gates
        assert any(g.gate_type in (GateType.RX, GateType.RY, GateType.RZ) for g in circuit.gates)
    
    def test_qaoa_simple_problem(self):
        """Test QAOA on simple optimization problem."""
        # Two qubits, want them antiparallel: H = -Z0*Z1
        hamiltonian = [(-1.0, 'ZZ')]
        
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=20)
        result = qaoa.run(n_shots=100)
        
        assert isinstance(result, QAOAResult)
        # Cost should be negative (optimizing -ZZ)
        assert result.optimal_cost < 0.5
        # Solution length should be correct
        assert len(result.optimal_solution) == 2
    
    def test_qaoa_maxcut(self):
        """Test QAOA on MaxCut problem."""
        # Triangle graph: 0-1, 1-2, 0-2
        edges = [(0, 1), (1, 2), (0, 2)]
        hamiltonian = create_maxcut_hamiltonian(edges)
        
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=30)
        result = qaoa.run(n_shots=200)
        
        assert isinstance(result, QAOAResult)
        assert result.n_iterations > 0
        assert len(result.optimal_solution) == 3
        # MaxCut value should be reasonable (max is 2 for triangle)
        assert result.optimal_cost < 0  # Negative because we minimize -cuts
    
    def test_qaoa_multiple_layers(self):
        """Test QAOA with multiple layers."""
        hamiltonian = [(-1.0, 'ZZ')]
        
        qaoa1 = QAOA(hamiltonian, n_layers=1, max_iterations=20)
        result1 = qaoa1.run(n_shots=100)
        
        qaoa2 = QAOA(hamiltonian, n_layers=2, max_iterations=20)
        result2 = qaoa2.run(n_shots=100)
        
        # More layers should give deeper circuit
        assert len(result2.optimal_circuit.gates) >= len(result1.optimal_circuit.gates)
        # Both should find good solutions
        assert result1.optimal_cost < 0.5
        assert result2.optimal_cost < 0.5
    
    def test_qaoa_partition_problem(self):
        """Test QAOA on number partition problem."""
        values = [3.0, 1.0, 1.0, 2.0]
        hamiltonian = create_partition_hamiltonian(values)
        
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=20)
        result = qaoa.run(n_shots=100)
        
        assert isinstance(result, QAOAResult)
        assert len(result.optimal_solution) == len(values)
    
    def test_qaoa_convergence(self):
        """Test QAOA optimization convergence."""
        hamiltonian = [(-1.0, 'ZZ')]
        
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=30, tol=1e-3)
        result = qaoa.run(n_shots=100)
        
        # Check iteration history
        assert len(result.iteration_history) > 0
        costs = [c for _, c in result.iteration_history]
        # Cost should improve (decrease for minimization)
        assert costs[-1] <= costs[0] + 0.5
    
    def test_qaoa_solution_probability(self):
        """Test QAOA solution probability extraction."""
        hamiltonian = [(-1.0, 'ZZ')]
        
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=20)
        result = qaoa.run(n_shots=500)
        
        # Probability should be between 0 and 1
        assert 0 <= result.solution_probability <= 1
        # With enough shots and good optimization, should be reasonably high
        assert result.solution_probability > 0.1


# =============================================================================
# Hamiltonian Construction Tests
# =============================================================================

class TestHamiltonianConstruction:
    """Test Hamiltonian construction utilities."""
    
    def test_maxcut_hamiltonian(self):
        """Test MaxCut Hamiltonian construction."""
        edges = [(0, 1), (1, 2)]
        hamiltonian = create_maxcut_hamiltonian(edges)
        
        assert len(hamiltonian) == 2
        for coeff, pauli in hamiltonian:
            assert coeff == -0.5
            assert 'Z' in pauli
    
    def test_partition_hamiltonian(self):
        """Test partition Hamiltonian construction."""
        values = [1.0, 2.0, 3.0]
        hamiltonian = create_partition_hamiltonian(values)
        
        assert len(hamiltonian) > 0
        # Should have quadratic terms
        for coeff, pauli in hamiltonian:
            assert 'Z' in pauli
            # Count number of Z's (should be 2 for quadratic terms)
            assert pauli.count('Z') == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestVariationalIntegration:
    """Integration tests for variational algorithms."""
    
    def test_vqe_qaoa_compatibility(self):
        """Test VQE and QAOA work with same Hamiltonian format."""
        hamiltonian = [(1.0, 'ZI'), (-0.5, 'IZ')]
        
        # VQE
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=2)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            circuit.add_gate(GateType.RY, targets=[1], parameters={'angle': params[1]})
            return circuit
        
        vqe = VQE(hamiltonian, ansatz, max_iterations=20)
        vqe_result = vqe.run(initial_parameters=np.array([1.0, 1.0]))
        
        # QAOA
        qaoa = QAOA(hamiltonian, n_layers=1, max_iterations=20)
        qaoa_result = qaoa.run(n_shots=100)
        
        # Both should produce valid results
        assert isinstance(vqe_result, VQEResult)
        assert isinstance(qaoa_result, QAOAResult)
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        # Define problem
        hamiltonian = [(1.0, 'Z')]
        
        def ansatz(params):
            circuit = UnifiedCircuit(n_qubits=1)
            circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': params[0]})
            return circuit
        
        # Run VQE
        vqe = VQE(hamiltonian, ansatz, max_iterations=30)
        result = vqe.run(initial_parameters=np.array([1.0]))
        
        # Verify optimization worked
        assert result.success or result.n_iterations > 10
        assert result.eigenvalue < 0.5
        assert result.optimal_circuit is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
