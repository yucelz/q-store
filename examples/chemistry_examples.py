"""
Quantum chemistry examples using Q-Store.

Demonstrates molecular simulation and VQE calculations.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.chemistry import Molecule, VQE, PauliString
from q_store.visualization import visualize_circuit


def example_molecule_creation():
    """Demonstrate molecule creation and properties."""
    print("=" * 60)
    print("Example 1: Molecule Creation")
    print("=" * 60)

    # Create H2 molecule
    h2 = Molecule(
        atoms=[('H', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, 0.74])],
        charge=0,
        multiplicity=1
    )

    print(f"Molecule: {h2.name}")
    print(f"Atoms: {len(h2.atoms)}")
    print(f"Nuclear repulsion: {h2.nuclear_repulsion:.4f} Hartree")

    # Get qubit Hamiltonian
    hamiltonian = h2.get_qubit_hamiltonian()

    print(f"\nQubit Hamiltonian:")
    print(f"  Number of terms: {len(hamiltonian.terms)}")
    print(f"  First 3 terms:")
    for i, (pauli_string, coeff) in enumerate(hamiltonian.terms[:3]):
        print(f"    {i+1}. {coeff:.4f} * {pauli_string}")


def example_pauli_strings():
    """Demonstrate Pauli string operations."""
    print("\n" + "=" * 60)
    print("Example 2: Pauli Strings")
    print("=" * 60)

    # Create Pauli strings
    p1 = PauliString("XYZI")
    p2 = PauliString("IXYZ")

    print(f"P1: {p1}")
    print(f"P2: {p2}")

    # Multiplication
    p3 = p1 * p2
    print(f"\nP1 * P2 = {p3}")

    # Expectation value circuit
    circuit = UnifiedCircuit(4)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.CNOT, [0, 1])

    print(f"\nCircuit:")
    print(visualize_circuit(circuit))

    # Measurement circuit for Pauli string
    meas_circuit = p1.to_measurement_circuit()
    print(f"\nMeasurement circuit for {p1}:")
    print(visualize_circuit(meas_circuit))


def example_vqe_ansatz():
    """Demonstrate VQE ansatz construction."""
    print("\n" + "=" * 60)
    print("Example 3: VQE Ansatz")
    print("=" * 60)

    # Create simple ansatz
    def create_ansatz(n_qubits, params):
        circuit = UnifiedCircuit(n_qubits)

        # Initial state preparation
        circuit.add_gate(GateType.X, [0])

        # Parameterized layer
        idx = 0
        for q in range(n_qubits):
            circuit.add_gate(GateType.RY, [q], parameters=[params[idx]])
            idx += 1

        # Entangling layer
        for q in range(n_qubits - 1):
            circuit.add_gate(GateType.CNOT, [q, q + 1])

        # Second parameterized layer
        for q in range(n_qubits):
            circuit.add_gate(GateType.RZ, [q], parameters=[params[idx]])
            idx += 1

        return circuit

    # Create ansatz for 2 qubits
    n_qubits = 2
    n_params = 2 * n_qubits
    params = np.array([0.1, 0.2, 0.3, 0.4])

    circuit = create_ansatz(n_qubits, params)

    print(f"VQE Ansatz ({n_qubits} qubits, {n_params} parameters):")
    print(visualize_circuit(circuit))
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Parameters: {params}")


def example_vqe_energy():
    """Demonstrate VQE energy calculation."""
    print("\n" + "=" * 60)
    print("Example 4: VQE Energy Calculation")
    print("=" * 60)

    # Create simple Hamiltonian: H = Z0 + 0.5*Z1
    from q_store.chemistry import QubitHamiltonian

    hamiltonian = QubitHamiltonian()
    hamiltonian.add_term(PauliString("ZI"), 1.0)
    hamiltonian.add_term(PauliString("IZ"), 0.5)

    print("Hamiltonian:")
    for pauli_str, coeff in hamiltonian.terms:
        print(f"  {coeff:.2f} * {pauli_str}")

    # Create VQE instance
    vqe = VQE(hamiltonian, n_qubits=2, n_layers=1)

    # Test energy with different parameters
    params_list = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([np.pi, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi, 0.0, 0.0]),
    ]

    print("\nEnergy for different parameters:")
    for i, params in enumerate(params_list):
        energy = vqe.compute_energy(params)
        print(f"  {i+1}. params={params[:2]}, energy={energy:.4f}")


def example_uccsd_ansatz():
    """Demonstrate UCCSD ansatz."""
    print("\n" + "=" * 60)
    print("Example 5: UCCSD Ansatz")
    print("=" * 60)

    # UCCSD-like ansatz for 2 electrons in 4 orbitals (2 qubits)
    def create_uccsd_ansatz(params):
        circuit = UnifiedCircuit(4)

        # Hartree-Fock initial state: |1100>
        circuit.add_gate(GateType.X, [0])
        circuit.add_gate(GateType.X, [1])

        # Single excitations
        # 0->2 excitation
        circuit.add_gate(GateType.RY, [0], parameters=[params[0]])
        circuit.add_gate(GateType.CNOT, [0, 2])
        circuit.add_gate(GateType.RY, [2], parameters=[-params[0]])
        circuit.add_gate(GateType.CNOT, [0, 2])

        # 1->3 excitation
        circuit.add_gate(GateType.RY, [1], parameters=[params[1]])
        circuit.add_gate(GateType.CNOT, [1, 3])
        circuit.add_gate(GateType.RY, [3], parameters=[-params[1]])
        circuit.add_gate(GateType.CNOT, [1, 3])

        return circuit

    params = np.array([0.1, 0.2])
    circuit = create_uccsd_ansatz(params)

    print("UCCSD-inspired ansatz (4 qubits, 2 single excitations):")
    print(visualize_circuit(circuit))
    print(f"\nInitial state: |1100> (Hartree-Fock)")
    print(f"Excitation parameters: {params}")


def example_chemistry_workflow():
    """Demonstrate complete chemistry workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Chemistry Workflow")
    print("=" * 60)

    # 1. Define molecule
    print("Step 1: Define H2 molecule")
    h2 = Molecule(
        atoms=[('H', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, 0.74])],
        charge=0,
        multiplicity=1
    )
    print(f"  Molecule: {h2.name}")
    print(f"  Bond length: 0.74 Angstrom")

    # 2. Get Hamiltonian
    print("\nStep 2: Generate qubit Hamiltonian")
    hamiltonian = h2.get_qubit_hamiltonian()
    print(f"  Number of terms: {len(hamiltonian.terms)}")

    # 3. Setup VQE
    print("\nStep 3: Setup VQE")
    n_qubits = h2.n_qubits
    vqe = VQE(hamiltonian, n_qubits=n_qubits, n_layers=2)
    n_params = vqe.get_n_parameters()
    print(f"  Qubits: {n_qubits}")
    print(f"  Parameters: {n_params}")

    # 4. Initialize parameters
    print("\nStep 4: Initialize parameters")
    params = np.random.randn(n_params) * 0.1
    print(f"  Initial params: {params[:4]}...")

    # 5. Compute energy
    print("\nStep 5: Compute energy")
    energy = vqe.compute_energy(params)
    print(f"  Energy: {energy:.6f} Hartree")

    # 6. Get ansatz circuit
    print("\nStep 6: VQE ansatz circuit")
    circuit = vqe.get_circuit(params)
    print(visualize_circuit(circuit))

    print("\n(In practice, parameters would be optimized to minimize energy)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE QUANTUM CHEMISTRY EXAMPLES")
    print("=" * 60)

    example_molecule_creation()
    example_pauli_strings()
    example_vqe_ansatz()
    example_vqe_energy()
    example_uccsd_ansatz()
    example_chemistry_workflow()

    print("\n" + "=" * 60)
    print("Chemistry examples completed!")
    print("=" * 60)
