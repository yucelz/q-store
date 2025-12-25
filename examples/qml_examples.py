"""
Quantum Machine Learning examples using Q-Store.

Demonstrates QML features including feature maps, kernels, and training.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.kernels import QuantumKernel
from q_store.visualization import visualize_circuit


def example_feature_map():
    """Demonstrate quantum feature maps."""
    print("=" * 60)
    print("Example 1: Quantum Feature Maps")
    print("=" * 60)

    # Create a simple feature map function
    def angle_feature_map(x):
        """Angle encoding feature map."""
        n_qubits = len(x)
        circuit = UnifiedCircuit(n_qubits)
        for i in range(n_qubits):
            circuit.add_gate(GateType.RY, [i], parameters={'angle': x[i] * np.pi})
        return circuit

    # Sample data
    x = np.array([0.5, 0.3, 0.8])

    # Create circuit
    circuit = angle_feature_map(x)

    print("\nFeature map circuit for x=[0.5, 0.3, 0.8]:")
    print(visualize_circuit(circuit))
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")


def example_quantum_kernel():
    """Demonstrate quantum kernel computation."""
    print("\n" + "=" * 60)
    print("Example 2: Quantum Kernels")
    print("=" * 60)

    # Define a simple feature map
    def simple_feature_map(x):
        n_qubits = len(x)
        circuit = UnifiedCircuit(n_qubits)
        for i in range(n_qubits):
            circuit.add_gate(GateType.RY, [i], parameters={'angle': x[i]})
        return circuit

    # Create kernel
    kernel = QuantumKernel(simple_feature_map, n_qubits=2)

    # Sample data points
    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])
    x3 = np.array([0.9, 0.8])

    # Compute kernel values
    k11 = kernel.evaluate(x1, x1)
    k12 = kernel.evaluate(x1, x2)
    k13 = kernel.evaluate(x1, x3)

    print(f"\nKernel values:")
    print(f"  K(x1, x1) = {k11:.4f}  (should be ~1.0)")
    print(f"  K(x1, x2) = {k12:.4f}")
    print(f"  K(x1, x3) = {k13:.4f}")

    # Compute kernel matrix
    X = np.array([x1, x2, x3])
    K = kernel.compute_matrix(X)

    print(f"\nKernel matrix:")
    print(K)


def example_variational_circuit():
    """Demonstrate creating variational quantum circuits."""
    print("\n" + "=" * 60)
    print("Example 3: Variational Quantum Circuit")
    print("=" * 60)

    # Create a parameterized variational circuit
    def create_variational_ansatz(params):
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.RY, [0], parameters={'angle': params[0]})
        circuit.add_gate(GateType.RY, [1], parameters={'angle': params[1]})
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.RY, [0], parameters={'angle': params[2]})
        circuit.add_gate(GateType.RY, [1], parameters={'angle': params[3]})
        return circuit

    # Initial parameters
    params = np.array([0.1, 0.2, 0.3, 0.4])

    print("Variational ansatz:")
    circuit = create_variational_ansatz(params)
    print(visualize_circuit(circuit))

    print(f"\nParameters: {params}")
    print(f"Circuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")

def example_entangled_feature_map():
    """Demonstrate entangled feature maps."""
    print("\n" + "=" * 60)
    print("Example 4: Entangled Feature Maps")
    print("=" * 60)

    # ZZ-style feature map with entanglement
    def zz_feature_map(x):
        n_qubits = len(x)
        circuit = UnifiedCircuit(n_qubits)

        # Hadamard layer
        for i in range(n_qubits):
            circuit.add_gate(GateType.H, [i])

        # Encode data
        for i in range(n_qubits):
            circuit.add_gate(GateType.RZ, [i], parameters={'angle': x[i]})

        # Entanglement layer
        for i in range(n_qubits - 1):
            circuit.add_gate(GateType.CNOT, [i, i + 1])
            angle = x[i] * x[i + 1]  # Second-order interaction
            circuit.add_gate(GateType.RZ, [i + 1], parameters={'angle': angle})
            circuit.add_gate(GateType.CNOT, [i, i + 1])

        return circuit

    # Sample data
    x = np.array([0.5, 0.3])

    print(f"Input data: {x}")

    circuit = zz_feature_map(x)
    print("\nZZ Feature Map Circuit:")
    print(visualize_circuit(circuit))
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")


def example_data_encoding():
    """Demonstrate different data encoding strategies."""
    print("\n" + "=" * 60)
    print("Example 5: Data Encoding Strategies")
    print("=" * 60)

    # Sample data
    x = np.array([0.5, 0.3])

    print(f"Input data: {x}")

    # Angle encoding
    print("\n1. Angle Encoding (RY rotations):")
    circuit_angle = UnifiedCircuit(2)
    circuit_angle.add_gate(GateType.RY, [0], parameters={'angle': x[0] * np.pi})
    circuit_angle.add_gate(GateType.RY, [1], parameters={'angle': x[1] * np.pi})
    print(visualize_circuit(circuit_angle))

    # IQP encoding
    print("\n2. IQP-style Encoding:")
    circuit_iqp = UnifiedCircuit(2)
    circuit_iqp.add_gate(GateType.H, [0])
    circuit_iqp.add_gate(GateType.H, [1])
    circuit_iqp.add_gate(GateType.RZ, [0], parameters={'angle': x[0] * 2 * np.pi})
    circuit_iqp.add_gate(GateType.RZ, [1], parameters={'angle': x[1] * 2 * np.pi})
    circuit_iqp.add_gate(GateType.CNOT, [0, 1])
    print(visualize_circuit(circuit_iqp))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE QUANTUM ML EXAMPLES")
    print("=" * 60)

    example_feature_map()
    example_quantum_kernel()
    example_variational_circuit()
    example_entangled_feature_map()
    example_data_encoding()

    print("\n" + "=" * 60)
    print("QML examples completed!")
    print("=" * 60)
