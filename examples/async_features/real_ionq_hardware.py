#!/usr/bin/env python3
"""
Real IonQ Hardware Connection Example

This example uses the ACTUAL IonQ hardware backend (not the simulator mock).
It makes real API calls to cloud.ionq.com and uses your API credits.

Requirements:
- IONQ_API_KEY in .env
- cirq and cirq-ionq installed
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / "examples" / ".env")
    print("✓ Loaded .env file")
except ImportError:
    print("⚠ python-dotenv not available")

from q_store.backends import IonQHardwareBackend
from q_store.core import UnifiedCircuit, GateType


def create_bell_state_circuit():
    """Create a simple Bell state circuit."""
    circuit = UnifiedCircuit(n_qubits=2)

    # Create Bell state: (|00⟩ + |11⟩) / √2
    circuit.add_gate(GateType.H, targets=[0])
    circuit.add_gate(GateType.CNOT, targets=[0, 1])

    # Note: Measurements are implicit when executing on hardware

    return circuit


def main():
    """Test real IonQ hardware connection."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test real IonQ hardware connection')
    parser.add_argument('--skip-confirm', action='store_true',
                       help='Skip interactive confirmation (for testing)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("Real IonQ Hardware Backend Test")
    print("="*70 + "\n")

    # Check API key
    api_key = os.getenv('IONQ_API_KEY')
    if not api_key:
        print("❌ IONQ_API_KEY not found in environment")
        print("   Set it in examples/.env")
        return

    print(f"✓ API Key: {api_key[:10]}...{api_key[-4:]}")

    # Get target
    target = os.getenv('IONQ_TARGET', 'simulator')
    print(f"✓ Target: {target}")
    print()

    # Check for cirq-ionq
    try:
        import cirq
        import cirq_ionq
        print("✓ cirq and cirq-ionq installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install cirq cirq-ionq")
        return
    print()

    # Create backend
    print("Creating IonQ hardware backend...")
    try:
        backend = IonQHardwareBackend(
            api_key=api_key,
            target=target,
            use_native_gates=True,
            timeout=300
        )
        print("✓ Backend created")
        print(f"  Type: {type(backend).__name__}")
        print(f"  Target: {backend.target}")
        print(f"  Service: {backend.service}")
        print()
    except Exception as e:
        print(f"❌ Failed to create backend: {e}")
        return

    # Create circuit
    print("Creating Bell state circuit...")
    circuit = create_bell_state_circuit()
    print("✓ Circuit created")
    print(f"  Qubits: {circuit.n_qubits}")
    print(f"  Gates: {len(circuit.gates)}")
    print()

    print("Circuit structure:")
    for gate in circuit.gates:
        if gate.controls:
            print(f"  {gate.gate_type.value} on targets {gate.targets} controls {gate.controls}")
        else:
            print(f"  {gate.gate_type.value} on targets {gate.targets}")
    print()

    # Execute on real hardware
    print("⚠️  WARNING: Submitting to REAL IonQ backend")
    print("   This WILL consume API credits!")
    print()

    if not args.skip_confirm:
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    else:
        print("Skipping confirmation (--skip-confirm flag set)")

    print("\nSubmitting circuit to IonQ...")
    try:
        # Execute circuit
        result = backend.execute(circuit, shots=100)

        print("✓ Execution completed!")
        print()
        print("Results:")
        print(f"  Type: {type(result)}")

        # Display results based on structure
        if hasattr(result, 'measurements'):
            print(f"  Measurements: {result.measurements}")
        elif hasattr(result, 'histogram'):
            print(f"  Histogram: {result.histogram}")
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, dict) or isinstance(value, list):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result}")

        print()
        print("="*70)
        print("✅ Real IonQ execution completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
