"""
Test script for IonQ parallel execution performance improvement.

This script compares sequential vs parallel execution to validate the speedup.

Usage:
    # Sequential execution (old behavior)
    python test_ionq_parallel.py --sequential

    # Parallel execution (new behavior, default)
    python test_ionq_parallel.py --parallel

    # Compare both
    python test_ionq_parallel.py --compare
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Load environment variables from examples/.env
try:
    from dotenv import load_dotenv
    examples_dir = Path(__file__).parent.parent
    env_path = examples_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"⚠️  .env file not found at {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

# Add src to path (go up two levels to get to repo root)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from q_store.backends import IonQHardwareBackend
from q_store.core import UnifiedCircuit, GateType


def create_test_circuit(n_qubits: int = 4, seed: int = 0) -> UnifiedCircuit:
    """
    Create a simple, robust test circuit.

    Uses basic gates that don't trigger IonQ/Cirq issues.
    """
    circuit = UnifiedCircuit(n_qubits=n_qubits)

    # Use a seed to create different circuits
    np.random.seed(seed)

    # Add simple gates that work reliably
    # Layer 1: Hadamards
    for i in range(n_qubits):
        circuit.add_gate(GateType.H, targets=[i])

    # Layer 2: Entanglement
    for i in range(n_qubits - 1):
        circuit.add_gate(GateType.CNOT, targets=[i, i + 1])

    # Layer 3: Rotations with safe angles (avoid 0 or very small values)
    for i in range(n_qubits):
        # Use angles between pi/8 and pi/2 to avoid edge cases
        angle = np.pi / 8 + (np.pi / 4) * np.random.rand()
        circuit.add_gate(GateType.RZ, targets=[i], parameters={'angle': float(angle)})

    return circuit


def test_sequential(backend: IonQHardwareBackend, num_circuits: int = 20):
    """Test sequential execution."""
    print(f"\n{'='*60}")
    print("SEQUENTIAL EXECUTION TEST")
    print(f"{'='*60}")

    # Create different circuits using seed
    circuits = [create_test_circuit(seed=i) for i in range(num_circuits)]

    print(f"Executing {num_circuits} circuits sequentially...")
    start_time = time.time()

    try:
        results = backend.execute_batch(
            circuits,
            shots=1024,
            use_parallel=False  # Force sequential
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Sequential execution completed:")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Circuits: {num_circuits}")
        print(f"   Time per circuit: {elapsed/num_circuits:.2f}s")
        print(f"   Throughput: {num_circuits/elapsed:.2f} circuits/s")

        return elapsed

    except Exception as e:
        print(f"\n❌ Sequential execution failed: {e}")
        raise


def test_parallel(backend: IonQHardwareBackend, num_circuits: int = 20, max_workers: int = 10):
    """Test parallel execution."""
    print(f"\n{'='*60}")
    print("PARALLEL EXECUTION TEST")
    print(f"{'='*60}")

    # Create different circuits using seed
    circuits = [create_test_circuit(seed=i) for i in range(num_circuits)]

    print(f"Executing {num_circuits} circuits in parallel (max_workers={max_workers})...")
    start_time = time.time()

    try:
        results = backend.execute_batch(
            circuits,
            shots=1024,
            use_parallel=True,
            max_workers=max_workers
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Parallel execution completed:")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Circuits: {num_circuits}")
        print(f"   Time per circuit: {elapsed/num_circuits:.2f}s")
        print(f"   Throughput: {num_circuits/elapsed:.2f} circuits/s")
        print(f"   Max workers: {max_workers}")

        return elapsed

    except Exception as e:
        print(f"\n❌ Parallel execution failed: {e}")
        raise


def compare_performance(backend: IonQHardwareBackend, num_circuits: int = 20):
    """Compare sequential vs parallel performance."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    # Test sequential
    sequential_time = test_sequential(backend, num_circuits)

    # Test parallel
    parallel_time = test_parallel(backend, num_circuits, max_workers=10)

    # Summary
    speedup = sequential_time / parallel_time
    time_saved = sequential_time - parallel_time

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sequential time:  {sequential_time:.2f}s")
    print(f"Parallel time:    {parallel_time:.2f}s")
    print(f"Time saved:       {time_saved:.2f}s ({time_saved/sequential_time*100:.1f}%)")
    print(f"Speedup:          {speedup:.2f}x")

    if speedup > 1.5:
        print(f"\n✅ SUCCESS: {speedup:.2f}x speedup achieved!")
    else:
        print(f"\n⚠️  WARNING: Speedup ({speedup:.2f}x) is lower than expected")


def main():
    parser = argparse.ArgumentParser(description='Test IonQ parallel execution')
    parser.add_argument('--sequential', action='store_true', help='Test sequential execution only')
    parser.add_argument('--parallel', action='store_true', help='Test parallel execution only')
    parser.add_argument('--compare', action='store_true', help='Compare sequential vs parallel')
    parser.add_argument('--num-circuits', type=int, default=20, help='Number of circuits to test')
    parser.add_argument('--max-workers', type=int, default=10, help='Max parallel workers')
    args = parser.parse_args()

    # Get IonQ API key
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        print("❌ ERROR: IONQ_API_KEY environment variable not set")
        print("   Please set it in examples/.env or export it")
        sys.exit(1)

    # Create backend
    print(f"\nCreating IonQ backend...")
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"   Target: simulator")

    try:
        backend = IonQHardwareBackend(
            api_key=api_key,
            target='simulator',
            timeout=600
        )
        print(f"✅ Backend created successfully")
    except Exception as e:
        print(f"❌ Failed to create backend: {e}")
        sys.exit(1)

    # Run tests
    if args.compare or (not args.sequential and not args.parallel):
        # Default: compare both
        compare_performance(backend, args.num_circuits)
    elif args.sequential:
        test_sequential(backend, args.num_circuits)
    elif args.parallel:
        test_parallel(backend, args.num_circuits, args.max_workers)


if __name__ == '__main__':
    main()
