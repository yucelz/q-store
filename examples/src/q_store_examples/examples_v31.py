"""
Quantum Database v3.1 - Hardware-Agnostic Examples
Demonstrates using multiple backends and SDK flexibility
"""

import asyncio
import os
import numpy as np
from typing import List

# Import the new hardware-agnostic components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from q_store.core.quantum_database import QuantumDatabase, DatabaseConfig, QueryMode
from q_store.backends.backend_manager import BackendManager, MockQuantumBackend, setup_ionq_backends
from q_store.backends.quantum_backend_interface import (
    CircuitBuilder,
    create_bell_state_circuit,
    create_ghz_state_circuit
)


async def example_1_basic_usage_with_mock():
    """
    Example 1: Basic usage with mock backend (no API keys needed)
    Perfect for development and testing
    """
    print("\n" + "="*70)
    print("Example 1: Basic Usage with Mock Backend")
    print("="*70)

    # Configure database with mock backend
    config = DatabaseConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "mock-key"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        quantum_sdk="mock",  # No API key needed!
        enable_quantum=True
    )

    # Create database
    db = QuantumDatabase(config)

    async with db.connect():
        print("\n✓ Database connected with mock backend")

        # Insert some vectors
        test_vector = np.random.rand(768)
        await db.insert(
            id="test_vec_1",
            vector=test_vector,
            metadata={"type": "test"}
        )
        print("✓ Vector inserted")

        # List backends
        backends = db.list_backends()
        print(f"\n✓ Available backends: {len(backends)}")
        for backend in backends:
            print(f"  - {backend['name']}: {backend.get('metadata', {}).get('description', 'N/A')}")


async def example_2_multiple_backends():
    """
    Example 2: Using multiple backends and switching between them
    Demonstrates the plugin architecture
    """
    print("\n" + "="*70)
    print("Example 2: Multiple Backends")
    print("="*70)

    # Create backend manager
    manager = BackendManager()

    # Register multiple mock backends with different characteristics
    ideal_sim = MockQuantumBackend(
        name="ideal_simulator",
        max_qubits=20,
        noise_level=0.0
    )
    manager.register_backend(
        "ideal",
        ideal_sim,
        set_as_default=True,
        metadata={"description": "Ideal noiseless simulator"}
    )

    noisy_sim = MockQuantumBackend(
        name="noisy_simulator",
        max_qubits=20,
        noise_level=0.05
    )
    manager.register_backend(
        "noisy",
        noisy_sim,
        metadata={"description": "Simulator with 5% noise"}
    )

    # Create database with this manager
    config = DatabaseConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "mock-key"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        quantum_sdk="mock"
    )

    db = QuantumDatabase(config, backend_manager=manager)

    async with db.connect():
        print("\n✓ Database with multiple backends initialized")

        # List backends
        backends = db.list_backends()
        print(f"\n✓ Registered {len(backends)} backends:")
        for backend in backends:
            status = "✓ DEFAULT" if backend['is_default'] else "  "
            print(f"  {status} {backend['name']}: {backend['backend_type']} ({backend['max_qubits']} qubits)")

        # Switch backend
        print("\n✓ Switching to noisy backend...")
        db.switch_backend("noisy")

        backends = db.list_backends()
        for backend in backends:
            if backend['is_default']:
                print(f"  Now using: {backend['name']}")


async def example_3_cirq_vs_qiskit():
    """
    Example 3: Comparing Cirq and Qiskit adapters
    (Requires actual IonQ API key to run with real backends)
    """
    print("\n" + "="*70)
    print("Example 3: Cirq vs Qiskit Comparison")
    print("="*70)

    # Check if API keys available
    ionq_api_key = os.getenv("IONQ_API_KEY")

    if not ionq_api_key:
        print("\n⚠ IONQ_API_KEY not set. This example requires an IonQ API key.")
        print("  Set it with: export IONQ_API_KEY='your-key-here'")
        return

    # Create backend manager
    manager = BackendManager()

    # Set up both Cirq and Qiskit backends
    await setup_ionq_backends(
        manager,
        api_key=ionq_api_key,
        use_cirq=True,
        use_qiskit=True
    )

    print("\n✓ Registered backends:")
    for backend in manager.list_backends():
        print(f"  - {backend['name']}: {backend['sdk']}")

    print("\n✓ Both Cirq and Qiskit adapters ready!")
    print("  You can now compare performance across different SDKs")


async def example_4_circuit_building():
    """
    Example 4: Building circuits with hardware-agnostic interface
    """
    print("\n" + "="*70)
    print("Example 4: Hardware-Agnostic Circuit Building")
    print("="*70)

    # Build a circuit using CircuitBuilder
    print("\n✓ Building custom circuit...")
    builder = CircuitBuilder(n_qubits=3)

    # Create GHZ state
    circuit = (builder
        .h(0)
        .cnot(0, 1)
        .cnot(1, 2)
        .measure_all()
        .build())

    print(f"\n✓ Circuit info:")
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  Gates: {len(circuit.gates)}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Gate counts: {circuit.gate_count()}")

    # Execute on mock backend
    backend = MockQuantumBackend()
    await backend.initialize()

    result = await backend.execute_circuit(circuit, shots=1000)

    print(f"\n✓ Results:")
    for bitstring, count in result.most_common(5):
        prob = count / result.total_shots
        print(f"  {bitstring}: {count} ({prob:.1%})")


async def example_5_backend_selection():
    """
    Example 5: Automatic backend selection based on circuit requirements
    """
    print("\n" + "="*70)
    print("Example 5: Intelligent Backend Selection")
    print("="*70)

    # Create manager with multiple backends
    manager = BackendManager()

    # Register backends with different capabilities
    small_backend = MockQuantumBackend("small", max_qubits=5, noise_level=0.0)
    large_backend = MockQuantumBackend("large", max_qubits=20, noise_level=0.0)

    manager.register_backend("small", small_backend)
    manager.register_backend("large", large_backend)

    await manager.initialize_all()

    # Test with different sized circuits
    test_cases = [
        ("Small circuit", create_bell_state_circuit()),
        ("Medium circuit", create_ghz_state_circuit(5)),
        ("Large circuit", create_ghz_state_circuit(10)),
    ]

    for name, circuit in test_cases:
        best_backend = manager.find_best_backend(circuit)
        print(f"\n✓ {name} ({circuit.n_qubits} qubits)")
        print(f"  Best backend: {best_backend}")

    await manager.close_all()


async def example_6_production_deployment():
    """
    Example 6: Production-ready configuration
    Shows best practices for production deployment
    """
    print("\n" + "="*70)
    print("Example 6: Production Deployment Configuration")
    print("="*70)

    # Production config with fallback strategy
    config = DatabaseConfig(
        # Classical backend
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "required"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),

        # Quantum backend with fallback
        quantum_sdk=os.getenv("QUANTUM_SDK", "mock"),  # Falls back to mock if not set
        quantum_api_key=os.getenv("IONQ_API_KEY"),  # Optional
        quantum_target="simulator",

        # Feature flags
        enable_quantum=True,
        enable_superposition=True,
        enable_tunneling=True,

        # Performance tuning
        classical_candidate_pool=1000,
        quantum_batch_size=50,
        result_cache_ttl=300,

        # Reliability
        max_retries=3,
        quantum_job_timeout=120,
    )

    print("\n✓ Production configuration:")
    print(f"  Quantum SDK: {config.quantum_sdk}")
    print(f"  Quantum enabled: {config.enable_quantum}")
    print(f"  Superposition enabled: {config.enable_superposition}")
    print(f"  Cache TTL: {config.result_cache_ttl}s")

    # Create manager with fallback strategy
    manager = BackendManager()

    # Try to initialize real backend, fall back to mock
    if config.quantum_api_key and config.quantum_sdk == "cirq":
        try:
            from q_store.backends.cirq_ionq_adapter import CirqIonQBackend
            backend = CirqIonQBackend(config.quantum_api_key, config.quantum_target)
            manager.register_backend("production", backend, set_as_default=True)
            print("✓ Using real IonQ backend")
        except Exception as e:
            print(f"⚠ Failed to initialize IonQ backend: {e}")
            print("✓ Falling back to mock backend")
            mock = MockQuantumBackend("fallback", max_qubits=20)
            manager.register_backend("fallback", mock, set_as_default=True)
    else:
        print("✓ Using mock backend (no API key provided)")
        mock = MockQuantumBackend("production_mock", max_qubits=20)
        manager.register_backend("production_mock", mock, set_as_default=True)

    db = QuantumDatabase(config, backend_manager=manager)

    print("\n✓ Production database ready!")
    print("  This configuration provides:")
    print("  - Automatic fallback to mock if quantum backend unavailable")
    print("  - Caching for improved performance")
    print("  - Retry logic for reliability")
    print("  - Comprehensive monitoring via metrics")


async def main():
    """Run all examples"""
    examples = [
        example_1_basic_usage_with_mock,
        example_2_multiple_backends,
        example_3_cirq_vs_qiskit,
        example_4_circuit_building,
        example_5_backend_selection,
        example_6_production_deployment,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
