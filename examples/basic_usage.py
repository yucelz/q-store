"""
Basic usage examples for Q-Store.

This module demonstrates fundamental Q-Store operations.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.visualization import visualize_circuit, visualize_state


def example_bell_state():
    """Create and visualize a Bell state circuit."""
    print("=" * 60)
    print("Example 1: Bell State Creation")
    print("=" * 60)

    # Create a 2-qubit circuit
    circuit = UnifiedCircuit(2)

    # Apply Hadamard to qubit 0
    circuit.add_gate(GateType.H, [0])

    # Apply CNOT with control=0, target=1
    circuit.add_gate(GateType.CNOT, [0, 1])

    # Visualize the circuit
    print("\nCircuit:")
    print(visualize_circuit(circuit))

    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")


def example_parameterized_circuit():
    """Create a parameterized circuit with rotation gates."""
    print("\n" + "=" * 60)
    print("Example 2: Parameterized Circuit")
    print("=" * 60)

    circuit = UnifiedCircuit(2)

    # Add parameterized rotation gates
    circuit.add_gate(GateType.RX, [0], parameters={'angle': np.pi/4})
    circuit.add_gate(GateType.RY, [1], parameters={'angle': np.pi/2})
    circuit.add_gate(GateType.CNOT, [0, 1])
    circuit.add_gate(GateType.RZ, [1], parameters={'angle': np.pi/3})

    print("\nCircuit:")
    print(visualize_circuit(circuit))

    print(f"\nGates with parameters:")
    for i, gate in enumerate(circuit.gates):
        if gate.parameters:
            angle_val = gate.parameters.get('angle', list(gate.parameters.values())[0] if gate.parameters else 0)
            print(f"  Gate {i}: {gate.gate_type.name} with θ={angle_val:.4f}")


def example_circuit_optimization():
    """Demonstrate circuit optimization."""
    print("\n" + "=" * 60)
    print("Example 3: Circuit Optimization")
    print("=" * 60)

    # Create circuit with redundant gates
    circuit = UnifiedCircuit(2)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.X, [0])
    circuit.add_gate(GateType.X, [0])  # Cancels with previous X
    circuit.add_gate(GateType.CNOT, [0, 1])

    print("\nOriginal circuit:")
    print(visualize_circuit(circuit))
    print(f"Gates: {len(circuit.gates)}, Depth: {circuit.depth}")

    # Optimize
    optimized = circuit.optimize()

    print("\nOptimized circuit:")
    print(visualize_circuit(optimized))
    print(f"Gates: {len(optimized.gates)}, Depth: {optimized.depth}")
    print(f"Reduction: {len(circuit.gates) - len(optimized.gates)} gates removed")


def example_backend_conversion():
    """Demonstrate conversion between different backends."""
    print("\n" + "=" * 60)
    print("Example 4: Backend Conversion")
    print("=" * 60)

    # Create Q-Store circuit
    circuit = UnifiedCircuit(2)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.CNOT, [0, 1])

    print("\nOriginal Q-Store circuit:")
    print(visualize_circuit(circuit))

    try:
        # Convert to Qiskit
        qiskit_circuit = circuit.to_qiskit()
        print("\n✓ Converted to Qiskit QuantumCircuit")

        # Convert back
        from_qiskit = UnifiedCircuit.from_qiskit(qiskit_circuit)
        print("✓ Converted back from Qiskit")
        print(f"  Gates preserved: {len(from_qiskit.gates)} == {len(circuit.gates)}")
    except ImportError:
        print("\n⚠ Qiskit not available for conversion demo")

    try:
        # Convert to Cirq
        cirq_circuit = circuit.to_cirq()
        print("✓ Converted to Cirq Circuit")

        # Convert back
        from_cirq = UnifiedCircuit.from_cirq(cirq_circuit)
        print("✓ Converted back from Cirq")
        print(f"  Gates preserved: {len(from_cirq.gates)} == {len(circuit.gates)}")
    except ImportError:
        print("⚠ Cirq not available for conversion demo")


def example_state_preparation():
    """Demonstrate state preparation and visualization."""
    print("\n" + "=" * 60)
    print("Example 5: State Visualization")
    print("=" * 60)

    # Create a superposition state
    state = np.array([1, 1, 0, 0]) / np.sqrt(2)

    print("\nBell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    print(visualize_state(state, mode='statevector'))

    print("\nMeasurement probabilities:")
    print(visualize_state(state, mode='probabilities'))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE BASIC EXAMPLES")
    print("=" * 60)

    example_bell_state()
    example_parameterized_circuit()
    example_circuit_optimization()
    example_backend_conversion()
    example_state_preparation()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
