"""
Quick test of v3.1 features without requiring external services
"""

import asyncio
import numpy as np
from q_store.backends import (
    BackendManager,
    MockQuantumBackend,
    CircuitBuilder,
    create_bell_state_circuit,
    create_ghz_state_circuit
)

async def test_mock_backend():
    """Test mock quantum backend"""
    print("\n" + "="*60)
    print("Test 1: Mock Quantum Backend")
    print("="*60)

    backend = MockQuantumBackend(name="test", max_qubits=10, noise_level=0.0)
    await backend.initialize()

    # Test Bell state
    circuit = create_bell_state_circuit()
    print(f"\n✓ Created Bell state circuit: {circuit}")

    result = await backend.execute_circuit(circuit, shots=1000)
    print(f"\n✓ Execution results (1000 shots):")
    for bitstring, count in result.most_common(5):
        print(f"  {bitstring}: {count} ({count/10:.1f}%)")

    await backend.close()
    print("\n✓ Test passed!")

async def test_backend_manager():
    """Test backend manager with multiple backends"""
    print("\n" + "="*60)
    print("Test 2: Backend Manager")
    print("="*60)

    manager = BackendManager()

    # Register backends
    ideal = MockQuantumBackend("ideal", max_qubits=20, noise_level=0.0)
    noisy = MockQuantumBackend("noisy", max_qubits=15, noise_level=0.1)

    manager.register_backend("ideal", ideal, set_as_default=True,
                            metadata={"desc": "Ideal simulator"})
    manager.register_backend("noisy", noisy,
                            metadata={"desc": "Noisy simulator"})

    print(f"\n✓ Registered {len(manager.list_backends())} backends:")
    for backend in manager.list_backends():
        default = " (DEFAULT)" if backend['is_default'] else ""
        print(f"  - {backend['name']}: {backend['max_qubits']} qubits{default}")

    # Test backend selection
    circuit = create_ghz_state_circuit(10)
    best = manager.find_best_backend(circuit)
    print(f"\n✓ Best backend for 10-qubit circuit: {best}")

    await manager.close_all()
    print("\n✓ Test passed!")

async def test_circuit_builder():
    """Test circuit builder"""
    print("\n" + "="*60)
    print("Test 3: Circuit Builder")
    print("="*60)

    # Build custom circuit
    builder = CircuitBuilder(n_qubits=3)
    circuit = (builder
        .h(0)
        .cnot(0, 1)
        .cnot(1, 2)
        .rz(0, np.pi/4)
        .measure_all()
        .build())

    print(f"\n✓ Built circuit:")
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  Gates: {len(circuit.gates)}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Gate counts: {circuit.gate_count()}")

    # Execute
    backend = MockQuantumBackend()
    await backend.initialize()
    result = await backend.execute_circuit(circuit, shots=1000)

    print(f"\n✓ Top outcomes:")
    for bitstring, count in result.most_common(3):
        print(f"  {bitstring}: {count} shots")

    await backend.close()
    print("\n✓ Test passed!")

async def test_backend_capabilities():
    """Test backend capabilities"""
    print("\n" + "="*60)
    print("Test 4: Backend Capabilities")
    print("="*60)

    backend = MockQuantumBackend(max_qubits=25)
    await backend.initialize()

    caps = backend.get_capabilities()
    print(f"\n✓ Backend capabilities:")
    print(f"  Max qubits: {caps.max_qubits}")
    print(f"  Backend type: {caps.backend_type.value}")
    print(f"  Supported gates: {len(caps.supported_gates)} types")
    print(f"  Mid-circuit measurement: {caps.supports_mid_circuit_measurement}")

    info = backend.get_backend_info()
    print(f"\n✓ Backend info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    await backend.close()
    print("\n✓ Test passed!")

async def test_abstraction_layer():
    """Test that abstraction layer works correctly"""
    print("\n" + "="*60)
    print("Test 5: Abstraction Layer Validation")
    print("="*60)

    from q_store.backends import QuantumBackend

    # Verify MockQuantumBackend implements QuantumBackend
    backend = MockQuantumBackend()
    assert isinstance(backend, QuantumBackend)
    print("\n✓ MockQuantumBackend properly implements QuantumBackend")

    # Test all required methods exist
    required_methods = [
        'initialize', 'execute_circuit', 'get_capabilities',
        'get_backend_info', 'close', 'is_available'
    ]

    for method in required_methods:
        assert hasattr(backend, method)
        print(f"  ✓ {method}() exists")

    print("\n✓ Test passed!")

async def main():
    """Run all tests"""
    tests = [
        test_mock_backend,
        test_backend_manager,
        test_circuit_builder,
        test_backend_capabilities,
        test_abstraction_layer,
    ]

    print("\n" + "="*60)
    print("Q-Store v3.1 - Quick Tests")
    print("Testing hardware abstraction layer without external services")
    print("="*60)

    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nv3.1 hardware abstraction layer is working correctly!")
    print("You can now:")
    print("  - Use mock backends for testing")
    print("  - Switch between quantum SDKs (Cirq, Qiskit)")
    print("  - Add custom quantum backends")
    print("  - Test without API keys")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
