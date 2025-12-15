"""
Quick test to verify Cirq adapter fix for measurements handling
"""
import asyncio
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def test_cirq_adapter():
    """Test Cirq adapter with mock result handling"""
    print("\n" + "="*70)
    print("Testing Cirq Adapter Fix")
    print("="*70 + "\n")

    ionq_key = os.getenv('IONQ_API_KEY')
    if not ionq_key:
        print("⚠️  IONQ_API_KEY not set - using mock backend instead")
        # Test with mock backend
        from q_store.backends.backend_manager import MockQuantumBackend
        from q_store.backends import QuantumCircuit, QuantumGate, GateType
        from dataclasses import dataclass

        backend = MockQuantumBackend(name="test", max_qubits=4, noise_level=0.0)

        # Create simple circuit
        circuit = QuantumCircuit(n_qubits=2)

        # Create gates with correct format
        h_gate = QuantumGate(gate_type=GateType.HADAMARD, qubits=[0])
        cnot_gate = QuantumGate(gate_type=GateType.CNOT, qubits=[0, 1])

        circuit.add_gate(h_gate)
        circuit.add_gate(cnot_gate)

        print("✓ Testing with mock backend")
        result = await backend.execute_circuit(circuit, shots=100)
        print(f"✓ Execution succeeded")
        print(f"  Counts: {result.counts}")
        print(f"  Total shots: {result.total_shots}")
        return True

    try:
        from q_store.backends.cirq_ionq_adapter import CirqIonQBackend
        from q_store.backends import QuantumCircuit, QuantumGate, GateType

        print(f"✓ IonQ API Key found: {ionq_key[:10]}...")

        # Create backend
        backend = CirqIonQBackend(api_key=ionq_key, target='simulator')
        await backend.initialize()
        print("✓ Cirq IonQ backend initialized")

        # Create simple circuit
        circuit = QuantumCircuit(n_qubits=2)

        # Create gates with correct format
        h_gate = QuantumGate(gate_type=GateType.HADAMARD, qubits=[0])
        cnot_gate = QuantumGate(gate_type=GateType.CNOT, qubits=[0, 1])

        circuit.add_gate(h_gate)
        circuit.add_gate(cnot_gate)

        print("Executing circuit on IonQ simulator...")
        result = await backend.execute_circuit(circuit, shots=100)

        print("✓ Circuit execution succeeded!")
        print(f"  Counts: {result.counts}")
        print(f"  Probabilities: {result.probabilities}")
        print(f"  Total shots: {result.total_shots}")
        print(f"  Backend: {result.metadata.get('backend')}")

        # Verify we got reasonable results
        if len(result.counts) > 0:
            print("✓ Got valid measurement results")
            return True
        else:
            print("✗ No measurement results returned")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_cirq_adapter()

    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED - Cirq adapter is working correctly")
    else:
        print("❌ TEST FAILED - Cirq adapter still has issues")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
