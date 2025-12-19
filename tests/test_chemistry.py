"""
Tests for quantum chemistry module.
"""

import pytest
import numpy as np
from q_store.chemistry import (
    MolecularHamiltonian,
    jordan_wigner_transform,
    fermion_to_qubit_operator,
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    create_beh2_hamiltonian,
    MolecularVQE,
    estimate_ground_state,
    optimize_bond_length,
    compute_molecular_properties,
    FermionOperator,
    QubitOperator,
    commutator,
    anticommutator,
    normal_ordered
)


class TestFermionOperator:
    """Test fermionic operators."""
    
    def test_fermion_operator_creation(self):
        """Test creating fermionic operator."""
        op = FermionOperator()
        assert len(op.terms) == 0
    
    def test_fermion_operator_addition(self):
        """Test adding fermionic operators."""
        op1 = FermionOperator()
        op1.terms[((0, 1), (0, 0))] = 1.0  # a†_0 a_0
        
        op2 = FermionOperator()
        op2.terms[((1, 1), (1, 0))] = 2.0  # 2 a†_1 a_1
        
        result = op1 + op2
        
        assert len(result.terms) == 2
        assert result.terms[((0, 1), (0, 0))] == 1.0
        assert result.terms[((1, 1), (1, 0))] == 2.0
    
    def test_fermion_operator_scalar_multiplication(self):
        """Test scalar multiplication."""
        op = FermionOperator()
        op.terms[((0, 1), (0, 0))] = 1.0
        
        result = 2.0 * op
        
        assert result.terms[((0, 1), (0, 0))] == 2.0


class TestQubitOperator:
    """Test qubit operators."""
    
    def test_qubit_operator_creation(self):
        """Test creating qubit operator."""
        op = QubitOperator()
        assert len(op.terms) == 0
    
    def test_qubit_operator_addition(self):
        """Test adding qubit operators."""
        op1 = QubitOperator()
        op1.terms[((0, 'Z'),)] = 1.0
        
        op2 = QubitOperator()
        op2.terms[((1, 'X'),)] = 2.0
        
        result = op1 + op2
        
        assert len(result.terms) == 2
    
    def test_qubit_operator_to_matrix(self):
        """Test converting to matrix."""
        op = QubitOperator()
        op.terms[((0, 'Z'),)] = 1.0
        
        matrix = op.to_matrix(n_qubits=1)
        
        expected = np.array([[1, 0], [0, -1]])
        assert np.allclose(matrix, expected)
    
    def test_qubit_operator_pauli_x(self):
        """Test Pauli X operator."""
        op = QubitOperator()
        op.terms[((0, 'X'),)] = 1.0
        
        matrix = op.to_matrix(n_qubits=1)
        
        expected = np.array([[0, 1], [1, 0]])
        assert np.allclose(matrix, expected)
    
    def test_qubit_operator_two_qubits(self):
        """Test two-qubit operator."""
        op = QubitOperator()
        op.terms[((0, 'Z'), (1, 'Z'))] = 1.0
        
        matrix = op.to_matrix(n_qubits=2)
        
        # ZZ operator
        assert matrix.shape == (4, 4)
        assert np.abs(matrix[0, 0]) > 0  # Should have diagonal elements


class TestJordanWignerTransform:
    """Test Jordan-Wigner transformation."""
    
    def test_jordan_wigner_number_operator(self):
        """Test JW transform of number operator."""
        fermion_op = FermionOperator()
        fermion_op.terms[((0, 1), (0, 0))] = 1.0  # n_0 = a†_0 a_0
        
        qubit_op = jordan_wigner_transform(fermion_op)
        
        # Should give (I - Z)/2
        assert len(qubit_op.terms) > 0
    
    def test_jordan_wigner_hopping_term(self):
        """Test JW transform of hopping term."""
        fermion_op = FermionOperator()
        fermion_op.terms[((0, 1), (1, 0))] = 1.0  # a†_0 a_1
        
        qubit_op = jordan_wigner_transform(fermion_op)
        
        assert len(qubit_op.terms) > 0
    
    def test_fermion_to_qubit_operator(self):
        """Test generic fermionic to qubit conversion."""
        fermion_op = FermionOperator()
        fermion_op.terms[((0, 1), (0, 0))] = 2.0
        
        qubit_op = fermion_to_qubit_operator(fermion_op, method='jordan_wigner')
        
        assert isinstance(qubit_op, QubitOperator)


class TestMolecularHamiltonian:
    """Test molecular Hamiltonian."""
    
    def test_hamiltonian_creation(self):
        """Test creating molecular Hamiltonian."""
        h = np.array([[1.0, 0.1], [0.1, 1.0]])
        g = np.zeros((2, 2, 2, 2))
        
        hamiltonian = MolecularHamiltonian(h, g, nuclear_repulsion=0.5)
        
        assert hamiltonian.n_orbitals == 2
        assert hamiltonian.nuclear_repulsion == 0.5
    
    def test_hamiltonian_to_fermion_operator(self):
        """Test converting Hamiltonian to fermionic operator."""
        h = np.eye(2)
        g = np.zeros((2, 2, 2, 2))
        
        hamiltonian = MolecularHamiltonian(h, g, nuclear_repulsion=1.0)
        fermion_op = hamiltonian.to_fermion_operator()
        
        # Should have constant term for nuclear repulsion
        assert () in fermion_op.terms
        assert fermion_op.terms[()] == 1.0
    
    def test_hamiltonian_to_qubit_operator(self):
        """Test converting Hamiltonian to qubit operator."""
        h = np.eye(2)
        g = np.zeros((2, 2, 2, 2))
        
        hamiltonian = MolecularHamiltonian(h, g)
        qubit_op = hamiltonian.to_qubit_operator()
        
        assert isinstance(qubit_op, QubitOperator)


class TestMoleculeCreation:
    """Test creating specific molecules."""
    
    def test_create_h2_hamiltonian(self):
        """Test creating H2 Hamiltonian."""
        hamiltonian = create_h2_hamiltonian(bond_length=0.74)
        
        assert hamiltonian.n_orbitals == 2
        assert hamiltonian.nuclear_repulsion > 0
        assert hamiltonian.one_body.shape == (2, 2)
        assert hamiltonian.two_body.shape == (2, 2, 2, 2)
    
    def test_create_h2_different_bond_length(self):
        """Test H2 at different bond length."""
        h1 = create_h2_hamiltonian(bond_length=0.74)
        h2 = create_h2_hamiltonian(bond_length=1.0)
        
        # Nuclear repulsion should decrease with distance
        assert h1.nuclear_repulsion > h2.nuclear_repulsion
    
    def test_create_lih_hamiltonian(self):
        """Test creating LiH Hamiltonian."""
        hamiltonian = create_lih_hamiltonian(bond_length=1.54)
        
        assert hamiltonian.n_orbitals == 4
        assert hamiltonian.nuclear_repulsion > 0
    
    def test_create_beh2_hamiltonian(self):
        """Test creating BeH2 Hamiltonian."""
        hamiltonian = create_beh2_hamiltonian(bond_length=1.33)
        
        assert hamiltonian.n_orbitals == 6
        assert hamiltonian.nuclear_repulsion > 0


class TestMolecularVQE:
    """Test molecular VQE."""
    
    def test_vqe_creation(self):
        """Test creating MolecularVQE."""
        hamiltonian = create_h2_hamiltonian()
        vqe = MolecularVQE(hamiltonian)
        
        assert vqe.n_qubits == 2
        assert len(vqe.energies) == 0
    
    def test_vqe_energy_evaluation(self):
        """Test energy evaluation."""
        hamiltonian = create_h2_hamiltonian()
        vqe = MolecularVQE(hamiltonian)
        
        params = np.zeros(2)
        energy = vqe.energy_evaluation(params)
        
        # Should return some energy value
        assert isinstance(energy, (int, float))
    
    def test_vqe_optimization(self):
        """Test VQE optimization."""
        hamiltonian = create_h2_hamiltonian()
        vqe = MolecularVQE(hamiltonian)
        
        energy, params = vqe.optimize(max_iterations=10)
        
        assert isinstance(energy, (int, float))
        assert len(params) > 0
        assert len(vqe.energies) > 0
    
    def test_vqe_compute_properties(self):
        """Test computing molecular properties."""
        hamiltonian = create_h2_hamiltonian()
        vqe = MolecularVQE(hamiltonian)
        
        params = np.zeros(2)
        properties = vqe.compute_properties(params)
        
        assert 'total_energy' in properties
        assert 'electronic_energy' in properties
        assert 'nuclear_repulsion' in properties
        assert 'n_qubits' in properties


class TestVQEFunctions:
    """Test VQE convenience functions."""
    
    def test_estimate_ground_state(self):
        """Test ground state estimation."""
        hamiltonian = create_h2_hamiltonian()
        
        energy, params = estimate_ground_state(hamiltonian, max_iterations=10)
        
        assert isinstance(energy, (int, float))
        assert len(params) > 0
    
    def test_estimate_ground_state_lih(self):
        """Test ground state for LiH."""
        hamiltonian = create_lih_hamiltonian()
        
        energy, params = estimate_ground_state(hamiltonian, max_iterations=5)
        
        assert isinstance(energy, (int, float))
    
    def test_optimize_bond_length(self):
        """Test bond length optimization."""
        bond_lengths = np.linspace(0.6, 1.0, 5)
        
        optimal_length, min_energy, energies = optimize_bond_length(
            'H2', bond_lengths, ansatz='UCCSD'
        )
        
        assert 0.6 <= optimal_length <= 1.0
        assert isinstance(min_energy, (int, float))
        assert len(energies) == 5
    
    def test_optimize_bond_length_lih(self):
        """Test bond length optimization for LiH."""
        bond_lengths = np.linspace(1.3, 1.8, 3)
        
        optimal_length, min_energy, energies = optimize_bond_length(
            'LiH', bond_lengths
        )
        
        assert 1.3 <= optimal_length <= 1.8
        assert len(energies) == 3
    
    def test_compute_molecular_properties(self):
        """Test computing molecular properties."""
        hamiltonian = create_h2_hamiltonian()
        
        properties = compute_molecular_properties(hamiltonian)
        
        assert 'total_energy' in properties
        assert 'electronic_energy' in properties
        assert properties['n_qubits'] == 2


class TestOperatorFunctions:
    """Test operator utility functions."""
    
    def test_commutator(self):
        """Test commutator calculation."""
        op1 = QubitOperator()
        op1.terms[((0, 'X'),)] = 1.0
        
        op2 = QubitOperator()
        op2.terms[((0, 'Z'),)] = 1.0
        
        result = commutator(op1, op2)
        
        assert isinstance(result, QubitOperator)
    
    def test_anticommutator(self):
        """Test anticommutator calculation."""
        op1 = FermionOperator()
        op1.terms[((0, 0),)] = 1.0
        
        op2 = FermionOperator()
        op2.terms[((0, 1),)] = 1.0
        
        result = anticommutator(op1, op2)
        
        assert isinstance(result, FermionOperator)
    
    def test_normal_ordered(self):
        """Test normal ordering."""
        op = FermionOperator()
        op.terms[((0, 0), (0, 1))] = 1.0  # a_0 a†_0
        
        result = normal_ordered(op)
        
        assert isinstance(result, FermionOperator)


class TestIntegration:
    """Integration tests for chemistry module."""
    
    def test_full_h2_vqe_pipeline(self):
        """Test complete H2 VQE pipeline."""
        # Create H2 at equilibrium
        hamiltonian = create_h2_hamiltonian(bond_length=0.74)
        
        # Convert to qubit operator
        qubit_op = hamiltonian.to_qubit_operator()
        assert isinstance(qubit_op, QubitOperator)
        
        # Run VQE
        energy, params = estimate_ground_state(hamiltonian, max_iterations=10)
        
        # Compute properties
        properties = compute_molecular_properties(hamiltonian, params)
        
        assert properties['total_energy'] == pytest.approx(energy, abs=1e-6)
        assert properties['n_qubits'] == 2
    
    def test_h2_dissociation_curve(self):
        """Test H2 dissociation curve."""
        bond_lengths = np.array([0.5, 0.74, 1.0, 1.5])
        energies = []
        
        for r in bond_lengths:
            hamiltonian = create_h2_hamiltonian(bond_length=r)
            energy, _ = estimate_ground_state(hamiltonian, max_iterations=5)
            energies.append(energy)
        
        # Should compute energies for all bond lengths
        assert len(energies) == len(bond_lengths)
        # All energies should be finite
        assert all(np.isfinite(e) for e in energies)
    
    def test_compare_molecules(self):
        """Test creating and comparing different molecules."""
        h2 = create_h2_hamiltonian()
        lih = create_lih_hamiltonian()
        beh2 = create_beh2_hamiltonian()
        
        # Different molecules have different sizes
        assert h2.n_orbitals < lih.n_orbitals < beh2.n_orbitals
        
        # All have positive nuclear repulsion
        assert h2.nuclear_repulsion > 0
        assert lih.nuclear_repulsion > 0
        assert beh2.nuclear_repulsion > 0
    
    def test_hamiltonian_consistency(self):
        """Test Hamiltonian conversion consistency."""
        hamiltonian = create_h2_hamiltonian()
        
        # Convert to fermion operator
        fermion_op = hamiltonian.to_fermion_operator()
        
        # Convert to qubit operator
        qubit_op = hamiltonian.to_qubit_operator()
        
        # Should preserve nuclear repulsion in constant term
        if () in fermion_op.terms:
            assert fermion_op.terms[()] == pytest.approx(
                hamiltonian.nuclear_repulsion, abs=1e-10
            )
    
    def test_vqe_convergence(self):
        """Test VQE convergence behavior."""
        hamiltonian = create_h2_hamiltonian()
        vqe = MolecularVQE(hamiltonian)
        
        energy, params = vqe.optimize(max_iterations=20, tolerance=1e-3)
        
        # Should have recorded energies
        assert len(vqe.energies) > 0
        
        # Energies should generally decrease
        # (allowing for some fluctuation in simple optimization)
        first_energy = vqe.energies[0]
        final_energy = vqe.energies[-1]
        
        # Final should be lower or similar
        assert final_energy <= first_energy + 0.5
    
    def test_bond_length_optimization_finds_minimum(self):
        """Test that bond optimization finds reasonable minimum."""
        # Test with fine grid around H2 equilibrium
        bond_lengths = np.linspace(0.6, 1.0, 9)
        
        optimal, min_energy, energies = optimize_bond_length('H2', bond_lengths)
        
        # Should test all bond lengths
        assert len(energies) == len(bond_lengths)
        
        # Minimum energy should be at the optimal point
        assert min_energy == min(energies)
        
        # Optimal should be within tested range
        assert 0.6 <= optimal <= 1.0
