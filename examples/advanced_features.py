"""
Advanced examples for Q-Store features.

Demonstrates verification, profiling, and advanced features.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.verification import (
    check_circuit_equivalence,
    is_unitary,
    PropertyVerifier,
    symbolic_circuit_analysis
)
from q_store.profiling import (
    profile_circuit,
    analyze_performance,
    profile_optimization
)
from q_store.visualization import visualize_circuit, BlochSphere


def example_circuit_verification():
    """Demonstrate circuit verification."""
    print("=" * 60)
    print("Example 1: Circuit Verification")
    print("=" * 60)
    
    # Create two equivalent circuits
    circuit1 = UnifiedCircuit(1)
    circuit1.add_gate(GateType.X, [0])
    
    circuit2 = UnifiedCircuit(1)
    circuit2.add_gate(GateType.H, [0])
    circuit2.add_gate(GateType.Z, [0])
    circuit2.add_gate(GateType.H, [0])  # HZH = X
    
    print("\nCircuit 1 (X gate):")
    print(visualize_circuit(circuit1))
    
    print("\nCircuit 2 (HZH sequence):")
    print(visualize_circuit(circuit2))
    
    # Verify equivalence
    is_equiv, details = check_circuit_equivalence(circuit1, circuit2)
    
    print(f"\nAre circuits equivalent? {is_equiv}")
    print(f"Unitary distance: {details['unitary_distance']:.10f}")


def example_property_verification():
    """Demonstrate property verification."""
    print("\n" + "=" * 60)
    print("Example 2: Property Verification")
    print("=" * 60)
    
    circuit = UnifiedCircuit(2)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.CNOT, [0, 1])
    
    print("\nCircuit:")
    print(visualize_circuit(circuit))
    
    # Verify properties
    verifier = PropertyVerifier()
    
    unitarity = verifier.verify_unitarity(circuit)
    print(f"\nIs unitary? {unitarity['is_unitary']}")
    print(f"Deviation: {unitarity['deviation']:.2e}")
    
    reversibility = verifier.verify_reversibility(circuit)
    print(f"\nIs reversible? {reversibility['is_reversible']}")
    
    # Symbolic analysis
    analysis = symbolic_circuit_analysis(circuit)
    print(f"\nSymbolic analysis:")
    print(f"  Gate counts: {analysis['gate_counts']}")
    print(f"  Is Hermitian? {analysis['is_hermitian']}")


def example_performance_profiling():
    """Demonstrate performance profiling."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Profiling")
    print("=" * 60)
    
    circuit = UnifiedCircuit(3)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.H, [1])
    circuit.add_gate(GateType.H, [2])
    circuit.add_gate(GateType.CNOT, [0, 1])
    circuit.add_gate(GateType.CNOT, [1, 2])
    
    print("\nCircuit:")
    print(visualize_circuit(circuit))
    
    # Profile circuit
    profile = profile_circuit(circuit)
    print(f"\nProfiling results:")
    print(f"  Total time: {profile.total_time:.6f}s")
    print(f"  Average gate time: {profile.avg_gate_time:.6f}s")
    print(f"  Gate counts: {profile.gate_counts}")


def example_performance_analysis():
    """Demonstrate performance analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Analysis")
    print("=" * 60)
    
    circuit = UnifiedCircuit(4)
    circuit.add_gate(GateType.H, [0])
    circuit.add_gate(GateType.CNOT, [0, 1])
    circuit.add_gate(GateType.CNOT, [1, 2])
    circuit.add_gate(GateType.CNOT, [2, 3])
    
    analysis = analyze_performance(circuit)
    
    print(f"\nPerformance metrics:")
    print(f"  Qubits: {analysis['n_qubits']}")
    print(f"  Gates: {analysis['n_gates']}")
    print(f"  Depth: {analysis['depth']}")
    print(f"  Parallelism score: {analysis['parallelism_score']:.3f}")
    print(f"  Memory estimate: {analysis['memory_estimate']/1024:.2f} KB")
    
    # Get optimization suggestions
    from q_store.profiling import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.suggest_optimizations(circuit)
    
    if suggestions:
        print(f"\nOptimization suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")


def example_optimization_profiling():
    """Demonstrate optimization profiling."""
    print("\n" + "=" * 60)
    print("Example 5: Optimization Profiling")
    print("=" * 60)
    
    # Create circuit with room for optimization
    original = UnifiedCircuit(2)
    original.add_gate(GateType.H, [0])
    original.add_gate(GateType.X, [0])
    original.add_gate(GateType.X, [0])  # Cancels
    original.add_gate(GateType.CNOT, [0, 1])
    original.add_gate(GateType.Z, [1])
    
    print("\nOriginal circuit:")
    print(visualize_circuit(original))
    
    # Optimize
    optimized = original.optimize()
    
    print("\nOptimized circuit:")
    print(visualize_circuit(optimized))
    
    # Profile the optimization
    result = profile_optimization(original, optimized)
    
    print(f"\nOptimization results:")
    print(f"  Gate reduction: {result.gate_reduction} ({result.gate_reduction_pct:.1f}%)")
    print(f"  Depth reduction: {result.depth_reduction} ({result.depth_reduction_pct:.1f}%)")
    print(f"  Preserves functionality: {result.preserves_functionality}")


def example_bloch_sphere():
    """Demonstrate Bloch sphere visualization."""
    print("\n" + "=" * 60)
    print("Example 6: Bloch Sphere Visualization")
    print("=" * 60)
    
    # Create Bloch sphere
    sphere = BlochSphere()
    
    # Add various states
    sphere.add_state(np.array([1, 0]), "|0⟩")
    sphere.add_state(np.array([0, 1]), "|1⟩")
    sphere.add_state(np.array([1, 1])/np.sqrt(2), "|+⟩")
    sphere.add_state(np.array([1, -1])/np.sqrt(2), "|-⟩")
    sphere.add_state(np.array([1, 1j])/np.sqrt(2), "|+i⟩")
    
    print("\nBloch Sphere:")
    print(sphere.get_ascii_representation())
    
    # Get state data
    data = sphere.get_state_data()
    print(f"\nTotal states: {len(data)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE ADVANCED EXAMPLES")
    print("=" * 60)
    
    example_circuit_verification()
    example_property_verification()
    example_performance_profiling()
    example_performance_analysis()
    example_optimization_profiling()
    example_bloch_sphere()
    
    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)
