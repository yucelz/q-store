"""
Error correction examples using Q-Store.

Demonstrates surface codes, stabilizer measurements, and error correction workflows.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.error_correction import (
    SurfaceCode,
    StabilizerMeasurement,
    ErrorSyndrome,
    Decoder
)
from q_store.visualization import visualize_circuit


def example_surface_code():
    """Demonstrate surface code creation."""
    print("=" * 60)
    print("Example 1: Surface Code")
    print("=" * 60)

    # Create surface code
    code = SurfaceCode(distance=3)

    print(f"Surface code distance: {code.distance}")
    print(f"Number of data qubits: {code.n_data_qubits}")
    print(f"Number of ancilla qubits: {code.n_ancilla_qubits}")
    print(f"Total qubits: {code.n_qubits}")

    # Get code layout
    layout = code.get_layout()
    print(f"\nCode layout:")
    print(layout)


def example_stabilizer_measurement():
    """Demonstrate stabilizer measurements."""
    print("\n" + "=" * 60)
    print("Example 2: Stabilizer Measurements")
    print("=" * 60)

    # Create stabilizer measurement
    # X-type stabilizer: X0 X1 X2 X3
    stabilizer = StabilizerMeasurement(
        qubits=[0, 1, 2, 3],
        ancilla=4,
        type='X'
    )

    print(f"Stabilizer type: {stabilizer.type}")
    print(f"Data qubits: {stabilizer.qubits}")
    print(f"Ancilla qubit: {stabilizer.ancilla}")

    # Get measurement circuit
    circuit = stabilizer.get_circuit()

    print(f"\nMeasurement circuit:")
    print(visualize_circuit(circuit))
    print(f"Circuit depth: {circuit.depth}")


def example_error_syndrome():
    """Demonstrate error syndrome extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Error Syndrome Extraction")
    print("=" * 60)

    # Create surface code
    code = SurfaceCode(distance=3)

    # Get syndrome extraction circuit
    syndrome_circuit = code.get_syndrome_circuit()

    print(f"Syndrome extraction circuit:")
    print(visualize_circuit(syndrome_circuit))

    # Simulate syndrome
    syndrome = ErrorSyndrome(
        x_syndromes=[0, 1, 0, 0],
        z_syndromes=[0, 0, 1, 0]
    )

    print(f"\nExample syndrome:")
    print(f"  X-syndromes: {syndrome.x_syndromes}")
    print(f"  Z-syndromes: {syndrome.z_syndromes}")
    print(f"  Total violated stabilizers: {syndrome.weight}")


def example_decoder():
    """Demonstrate syndrome decoding."""
    print("\n" + "=" * 60)
    print("Example 4: Syndrome Decoding")
    print("=" * 60)

    # Create decoder
    code = SurfaceCode(distance=3)
    decoder = Decoder(code, algorithm='mwpm')  # Minimum Weight Perfect Matching

    print(f"Decoder algorithm: {decoder.algorithm}")
    print(f"Code distance: {decoder.code.distance}")

    # Example syndrome
    syndrome = ErrorSyndrome(
        x_syndromes=[0, 1, 0, 1],
        z_syndromes=[1, 0, 0, 0]
    )

    print(f"\nInput syndrome:")
    print(f"  X-syndromes: {syndrome.x_syndromes}")
    print(f"  Z-syndromes: {syndrome.z_syndromes}")

    # Decode
    correction = decoder.decode(syndrome)

    print(f"\nDecoded correction:")
    print(f"  X errors on qubits: {correction.x_errors}")
    print(f"  Z errors on qubits: {correction.z_errors}")


def example_error_detection():
    """Demonstrate error detection."""
    print("\n" + "=" * 60)
    print("Example 5: Error Detection")
    print("=" * 60)

    # Create code
    code = SurfaceCode(distance=3)

    # Create logical state |0>
    circuit = code.create_logical_zero()

    print("Logical |0> state preparation:")
    print(visualize_circuit(circuit))

    # Inject error (X on qubit 0)
    circuit.add_gate(GateType.X, [0])

    print("\nError injected: X on qubit 0")

    # Measure syndrome
    syndrome_circuit = code.get_syndrome_circuit()
    combined = UnifiedCircuit(code.n_qubits)

    # Copy gates from both circuits
    for gate in circuit.gates:
        combined.add_gate(gate.gate_type, gate.targets,
                         controls=gate.controls, parameters=gate.parameters)
    for gate in syndrome_circuit.gates:
        combined.add_gate(gate.gate_type, gate.targets,
                         controls=gate.controls, parameters=gate.parameters)

    print("\nCombined circuit (state prep + error + syndrome):")
    print(visualize_circuit(combined))

    print("\n(In simulation, this would detect the X error via Z-stabilizers)")


def example_logical_operations():
    """Demonstrate logical operations on encoded qubits."""
    print("\n" + "=" * 60)
    print("Example 6: Logical Operations")
    print("=" * 60)

    # Create code
    code = SurfaceCode(distance=3)

    # Logical X operation
    print("Logical X gate:")
    logical_x = code.get_logical_x_circuit()
    print(visualize_circuit(logical_x))

    print(f"\nLogical X applies X on {len(logical_x.gates)} qubits")

    # Logical Z operation
    print("\nLogical Z gate:")
    logical_z = code.get_logical_z_circuit()
    print(visualize_circuit(logical_z))

    print(f"\nLogical Z applies Z on {len(logical_z.gates)} qubits")


def example_error_correction_workflow():
    """Demonstrate complete error correction workflow."""
    print("\n" + "=" * 60)
    print("Example 7: Complete Error Correction Workflow")
    print("=" * 60)

    # 1. Create surface code
    print("Step 1: Create surface code")
    code = SurfaceCode(distance=3)
    print(f"  Distance: {code.distance}")
    print(f"  Data qubits: {code.n_data_qubits}")
    print(f"  Ancilla qubits: {code.n_ancilla_qubits}")

    # 2. Prepare logical state
    print("\nStep 2: Prepare logical |0> state")
    circuit = code.create_logical_zero()
    print(f"  Preparation circuit depth: {circuit.depth}")

    # 3. Apply logical operation
    print("\nStep 3: Apply logical gate (Logical X)")
    logical_x = code.get_logical_x_circuit()
    for gate in logical_x.gates:
        circuit.add_gate(gate.gate_type, gate.targets,
                        controls=gate.controls, parameters=gate.parameters)

    # 4. Simulate error
    print("\nStep 4: Error occurs (X on qubit 1)")
    circuit.add_gate(GateType.X, [1])

    # 5. Syndrome measurement
    print("\nStep 5: Measure error syndrome")
    syndrome_circuit = code.get_syndrome_circuit()
    print(f"  Syndrome circuit depth: {syndrome_circuit.depth}")

    # 6. Decode and correct
    print("\nStep 6: Decode syndrome and apply correction")
    decoder = Decoder(code, algorithm='mwpm')

    # Simulate syndrome (in practice, from measurements)
    syndrome = ErrorSyndrome(
        x_syndromes=[0, 1, 1, 0],
        z_syndromes=[0, 0, 0, 0]
    )

    correction = decoder.decode(syndrome)
    print(f"  Detected errors: {correction.x_errors}")

    # Apply correction
    for qubit in correction.x_errors:
        circuit.add_gate(GateType.X, [qubit])

    print("\nStep 7: Verify logical state")
    print("  (In simulation, would verify logical state is preserved)")

    print(f"\nTotal workflow circuit depth: {circuit.depth}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE ERROR CORRECTION EXAMPLES")
    print("=" * 60)

    example_surface_code()
    example_stabilizer_measurement()
    example_error_syndrome()
    example_decoder()
    example_error_detection()
    example_logical_operations()
    example_error_correction_workflow()

    print("\n" + "=" * 60)
    print("Error correction examples completed!")
    print("=" * 60)
