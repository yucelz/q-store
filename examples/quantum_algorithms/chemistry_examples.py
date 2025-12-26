"""
Quantum chemistry examples using Q-Store.

Demonstrates molecular simulation and VQE calculations.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.chemistry import (
    MolecularHamiltonian,
    create_h2_hamiltonian,
    create_lih_hamiltonian,
    MolecularVQE,
    estimate_ground_state,
    QubitOperator,
    FermionOperator,
    jordan_wigner_transform
)
from q_store.visualization import visualize_circuit


def example_molecule_hamiltonian():
    """Demonstrate molecular Hamiltonian creation."""
    print("=" * 60)
    print("Example 1: Molecular Hamiltonian")
    print("=" * 60)

    # Create H2 molecule Hamiltonian
    hamiltonian = create_h2_hamiltonian(bond_length=0.74)

    print(f"H2 Molecule (bond length: 0.74 Å)")
    print(f"Number of orbitals: {hamiltonian.n_orbitals}")
    print(f"Number of qubits: {hamiltonian.n_qubits}")
    print(f"Nuclear repulsion: {hamiltonian.nuclear_repulsion:.4f} Hartree")

    # Get qubit operator
    qubit_op = hamiltonian.to_qubit_operator()
    print(f"\nQubit operator has {len(qubit_op.terms)} terms")

    # Show first few terms
    print("\nFirst 3 Hamiltonian terms:")
    for i, (term, coeff) in enumerate(list(qubit_op.terms.items())[:3]):
        pauli_str = ''.join([p[1] for p in sorted(term, key=lambda x: x[0])]) if term else 'I'
        print(f"  {i+1}. {np.real(coeff):.4f} * {pauli_str}")


def example_fermion_operators():
    """Demonstrate fermionic operator operations."""
    print("\n" + "=" * 60)
    print("Example 2: Fermionic Operators")
    print("=" * 60)

    # Create fermionic number operator: n_0 = a†_0 a_0
    fermion_op = FermionOperator()
    fermion_op.terms[((0, 1), (0, 0))] = 1.0  # creation then annihilation

    print("Fermionic number operator n_0 = a†_0 a_0")
    print(f"Number of terms: {len(fermion_op.terms)}")

    # Transform to qubit operator via Jordan-Wigner
    qubit_op = jordan_wigner_transform(fermion_op)

    print(f"\nAfter Jordan-Wigner transformation:")
    print(f"Number of Pauli terms: {len(qubit_op.terms)}")

    # Show terms
    print("\nQubit operator terms:")
    for i, (term, coeff) in enumerate(list(qubit_op.terms.items())[:3]):
        pauli_str = ''.join([p[1] for p in sorted(term, key=lambda x: x[0])]) if term else 'I'
        print(f"  {np.real(coeff):.4f} * {pauli_str}")


def example_vqe_ansatz():
    """Demonstrate VQE ansatz construction."""
    print("\n" + "=" * 60)
    print("Example 3: VQE Ansatz Circuit")
    print("=" * 60)

    # Create simple hardware-efficient ansatz
    def create_ansatz(n_qubits, params):
        circuit = UnifiedCircuit(n_qubits)

        # Initial state preparation (Hartree-Fock for H2)
        circuit.add_gate(GateType.X, [0])

        # Parameterized layer
        idx = 0
        for q in range(n_qubits):
            circuit.add_gate(GateType.RY, [q], parameters={'angle': params[idx]})
            idx += 1

        # Entangling layer
        for q in range(n_qubits - 1):
            circuit.add_gate(GateType.CNOT, [q, q + 1])

        # Second parameterized layer
        for q in range(n_qubits):
            circuit.add_gate(GateType.RZ, [q], parameters={'angle': params[idx]})
            idx += 1

        return circuit

    # Create ansatz for 2 qubits (H2 molecule)
    n_qubits = 2
    n_params = 2 * n_qubits
    params = np.array([0.1, 0.2, 0.3, 0.4])

    circuit = create_ansatz(n_qubits, params)

    print(f"Hardware-Efficient Ansatz ({n_qubits} qubits, {n_params} parameters):")
    print(visualize_circuit(circuit))
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")
    print(f"Parameters: {params}")


def example_vqe_energy():
    """Demonstrate VQE energy calculation."""
    print("\n" + "=" * 60)
    print("Example 4: VQE Energy Evaluation")
    print("=" * 60)

    # Create H2 Hamiltonian
    hamiltonian = create_h2_hamiltonian(bond_length=0.74)

    print(f"H2 molecule at 0.74 Å")
    print(f"Nuclear repulsion: {hamiltonian.nuclear_repulsion:.4f} Hartree")

    # Create VQE instance
    vqe = MolecularVQE(hamiltonian, ansatz='HardwareEfficient')

    print(f"VQE setup:")
    print(f"  Number of qubits: {vqe.n_qubits}")
    print(f"  Ansatz type: {vqe.ansatz}")

    # Test energy with different parameters
    params_list = [
        np.array([0.0, 0.0]),
        np.array([0.1, 0.2]),
        np.array([0.5, 0.3]),
    ]

    print("\nEnergy for different parameters:")
    for i, params in enumerate(params_list):
        energy = vqe.energy_evaluation(params)
        print(f"  {i+1}. params={params}, energy={energy:.4f} Hartree")


def example_ground_state_estimation():
    """Demonstrate ground state estimation."""
    print("\n" + "=" * 60)
    print("Example 5: Ground State Estimation")
    print("=" * 60)

    # Create H2 Hamiltonian
    hamiltonian = create_h2_hamiltonian(bond_length=0.74)

    print("Estimating H2 ground state...")
    print("Running VQE optimization (10 iterations)")

    # Estimate ground state
    energy, optimal_params = estimate_ground_state(hamiltonian, max_iterations=10)

    print(f"\nOptimization complete:")
    print(f"  Ground state energy: {energy:.6f} Hartree")
    print(f"  Optimal parameters: {optimal_params}")
    print(f"  Number of parameters: {len(optimal_params)}")


def example_different_molecules():
    """Demonstrate different molecule Hamiltonians."""
    print("\n" + "=" * 60)
    print("Example 6: Different Molecules")
    print("=" * 60)

    # H2 molecule
    h2 = create_h2_hamiltonian(bond_length=0.74)
    print("H2 molecule:")
    print(f"  Orbitals: {h2.n_orbitals}")
    print(f"  Qubits: {h2.n_qubits}")
    print(f"  Nuclear repulsion: {h2.nuclear_repulsion:.4f} Hartree")

    # LiH molecule
    lih = create_lih_hamiltonian(bond_length=1.54)
    print("\nLiH molecule:")
    print(f"  Orbitals: {lih.n_orbitals}")
    print(f"  Qubits: {lih.n_qubits}")
    print(f"  Nuclear repulsion: {lih.nuclear_repulsion:.4f} Hartree")

    print("\nNote: More complex molecules require more qubits")
    print("      and longer computation times.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE QUANTUM CHEMISTRY EXAMPLES")
    print("=" * 60)

    example_molecule_hamiltonian()
    example_fermion_operators()
    example_vqe_ansatz()
    example_vqe_energy()
    example_ground_state_estimation()
    example_different_molecules()

    print("\n" + "=" * 60)
    print("Chemistry examples completed!")
    print("=" * 60)
