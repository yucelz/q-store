"""
Performance Benchmarks for Q-Store v4.0

Benchmark suite for tracking performance across releases and detecting regressions.
"""

import pytest
import time
import numpy as np
from typing import Dict, List, Tuple
from q_store.core import UnifiedCircuit, GateType
from q_store.profiling import profile_circuit
from q_store.verification import check_circuit_equivalence, PropertyVerifier
from q_store.visualization import visualize_circuit
from q_store.tomography import reconstruct_state


# ============================================================================
# Benchmark Utilities
# ============================================================================

class BenchmarkResult:
    """Store benchmark results."""
    def __init__(self, name: str, duration: float, iterations: int):
        self.name = name
        self.duration = duration
        self.iterations = iterations
        self.avg_time = duration / iterations
    
    def __repr__(self):
        return f"{self.name}: {self.avg_time*1000:.3f}ms/iter ({self.iterations} iters)"


def benchmark(func, iterations: int = 100) -> BenchmarkResult:
    """Run a benchmark function multiple times and measure performance."""
    start = time.time()
    for _ in range(iterations):
        func()
    duration = time.time() - start
    return BenchmarkResult(func.__name__, duration, iterations)


# ============================================================================
# Circuit Creation Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestCircuitCreationBenchmarks:
    """Benchmark circuit creation operations."""

    def test_benchmark_bell_state_creation(self, benchmark_iterations=1000):
        """Benchmark Bell state creation."""
        def create_bell():
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.H, [0])
            circuit.add_gate(GateType.CNOT, [0, 1])
            return circuit
        
        result = benchmark(create_bell, benchmark_iterations)
        assert result.avg_time < 0.001  # Less than 1ms per circuit
        print(f"\n{result}")

    def test_benchmark_ghz_state_creation(self, benchmark_iterations=1000):
        """Benchmark GHZ state creation."""
        def create_ghz():
            circuit = UnifiedCircuit(3)
            circuit.add_gate(GateType.H, [0])
            circuit.add_gate(GateType.CNOT, [0, 1])
            circuit.add_gate(GateType.CNOT, [0, 2])
            return circuit
        
        result = benchmark(create_ghz, benchmark_iterations)
        assert result.avg_time < 0.001
        print(f"\n{result}")

    def test_benchmark_parameterized_circuit(self, benchmark_iterations=500):
        """Benchmark parameterized circuit creation."""
        def create_parameterized():
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.RY, [0], parameters={'theta': 0.5})
            circuit.add_gate(GateType.CNOT, [0, 1])
            circuit.add_gate(GateType.RZ, [1], parameters={'phi': 0.3})
            return circuit
        
        result = benchmark(create_parameterized, benchmark_iterations)
        assert result.avg_time < 0.002
        print(f"\n{result}")

    def test_benchmark_large_circuit_creation(self, benchmark_iterations=100):
        """Benchmark large circuit creation (10 qubits, 50 gates)."""
        def create_large():
            circuit = UnifiedCircuit(10)
            for i in range(10):
                circuit.add_gate(GateType.H, [i])
            for i in range(9):
                circuit.add_gate(GateType.CNOT, [i, i+1])
            for i in range(10):
                circuit.add_gate(GateType.RZ, [i], parameters={'theta': 0.1*i})
            for i in range(9):
                circuit.add_gate(GateType.CNOT, [i, i+1])
            for i in range(10):
                circuit.add_gate(GateType.H, [i])
            return circuit
        
        result = benchmark(create_large, benchmark_iterations)
        assert result.avg_time < 0.01  # Less than 10ms
        print(f"\n{result}")


# ============================================================================
# Profiling Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestProfilingBenchmarks:
    """Benchmark profiling operations."""

    def test_benchmark_profile_small_circuit(self, benchmark_iterations=500):
        """Benchmark profiling small circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        def profile_it():
            return profile_circuit(circuit)
        
        result = benchmark(profile_it, benchmark_iterations)
        assert result.avg_time < 0.002
        print(f"\n{result}")

    def test_benchmark_profile_medium_circuit(self, benchmark_iterations=200):
        """Benchmark profiling medium circuit (5 qubits, 20 gates)."""
        circuit = UnifiedCircuit(5)
        for i in range(5):
            circuit.add_gate(GateType.H, [i])
        for i in range(4):
            circuit.add_gate(GateType.CNOT, [i, i+1])
        for i in range(5):
            circuit.add_gate(GateType.RZ, [i], parameters={'theta': 0.1*i})
        for i in range(4):
            circuit.add_gate(GateType.CNOT, [i, i+1])
        
        def profile_it():
            return profile_circuit(circuit)
        
        result = benchmark(profile_it, benchmark_iterations)
        assert result.avg_time < 0.005
        print(f"\n{result}")

    def test_benchmark_batch_profiling(self, benchmark_iterations=50):
        """Benchmark profiling a batch of circuits."""
        circuits = []
        for i in range(10):
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.H, [0])
            for _ in range(i):
                circuit.add_gate(GateType.CNOT, [0, 1])
            circuits.append(circuit)
        
        def profile_batch():
            return [profile_circuit(c) for c in circuits]
        
        result = benchmark(profile_batch, benchmark_iterations)
        assert result.avg_time < 0.05  # 50ms for 10 circuits
        print(f"\n{result}")


# ============================================================================
# Verification Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestVerificationBenchmarks:
    """Benchmark verification operations."""

    def test_benchmark_unitary_check(self, benchmark_iterations=200):
        """Benchmark unitarity checking."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        verifier = PropertyVerifier()
        
        def check_it():
            return verifier.verify_unitarity(circuit)
        
        result = benchmark(check_it, benchmark_iterations)
        assert result.avg_time < 0.01
        print(f"\n{result}")

    def test_benchmark_circuit_equivalence(self, benchmark_iterations=100):
        """Benchmark circuit equivalence checking."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])
        
        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        circuit2.add_gate(GateType.CNOT, [0, 1])
        
        def check_equiv():
            return check_circuit_equivalence(circuit1, circuit2)
        
        result = benchmark(check_equiv, benchmark_iterations)
        assert result.avg_time < 0.02
        print(f"\n{result}")


# ============================================================================
# Visualization Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestVisualizationBenchmarks:
    """Benchmark visualization operations."""

    def test_benchmark_visualize_small_circuit(self, benchmark_iterations=500):
        """Benchmark visualizing small circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        def visualize_it():
            return visualize_circuit(circuit)
        
        result = benchmark(visualize_it, benchmark_iterations)
        assert result.avg_time < 0.005
        print(f"\n{result}")

    def test_benchmark_visualize_medium_circuit(self, benchmark_iterations=200):
        """Benchmark visualizing medium circuit."""
        circuit = UnifiedCircuit(4)
        for i in range(4):
            circuit.add_gate(GateType.H, [i])
        for i in range(3):
            circuit.add_gate(GateType.CNOT, [i, i+1])
        
        def visualize_it():
            return visualize_circuit(circuit)
        
        result = benchmark(visualize_it, benchmark_iterations)
        assert result.avg_time < 0.01
        print(f"\n{result}")


# ============================================================================
# Tomography Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestTomographyBenchmarks:
    """Benchmark tomography operations."""

    def test_benchmark_state_reconstruction(self, benchmark_iterations=50):
        """Benchmark state reconstruction."""
        measurements = {
            'Z': [0] * 900 + [1] * 100,
            'X': [0] * 500 + [1] * 500,
            'Y': [0] * 500 + [1] * 500,
        }
        
        def reconstruct_it():
            return reconstruct_state(measurements, n_qubits=1)
        
        result = benchmark(reconstruct_it, benchmark_iterations)
        assert result.avg_time < 0.1  # 100ms
        print(f"\n{result}")


# ============================================================================
# End-to-End Workflow Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestWorkflowBenchmarks:
    """Benchmark complete workflows."""

    def test_benchmark_create_profile_visualize(self, benchmark_iterations=100):
        """Benchmark complete workflow: create → profile → visualize."""
        def full_workflow():
            # Create
            circuit = UnifiedCircuit(3)
            circuit.add_gate(GateType.H, [0])
            circuit.add_gate(GateType.CNOT, [0, 1])
            circuit.add_gate(GateType.CNOT, [1, 2])
            
            # Profile
            profile = profile_circuit(circuit)
            
            # Visualize
            viz = visualize_circuit(circuit)
            
            return circuit, profile, viz
        
        result = benchmark(full_workflow, benchmark_iterations)
        assert result.avg_time < 0.02
        print(f"\n{result}")

    def test_benchmark_create_verify_profile(self, benchmark_iterations=100):
        """Benchmark workflow: create → verify → profile."""
        verifier = PropertyVerifier()
        
        def workflow():
            # Create
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.RY, [0], parameters={'theta': 0.5})
            circuit.add_gate(GateType.CNOT, [0, 1])
            
            # Verify
            valid = verifier.verify_unitarity(circuit)
            
            # Profile
            profile = profile_circuit(circuit)
            
            return circuit, valid, profile
        
        result = benchmark(workflow, benchmark_iterations)
        assert result.avg_time < 0.02
        print(f"\n{result}")

    def test_benchmark_batch_circuit_processing(self, benchmark_iterations=20):
        """Benchmark processing a batch of circuits."""
        def batch_workflow():
            results = []
            for i in range(10):
                # Create
                circuit = UnifiedCircuit(2)
                circuit.add_gate(GateType.H, [0])
                for _ in range(i):
                    circuit.add_gate(GateType.CNOT, [0, 1])
                
                # Profile
                profile = profile_circuit(circuit)
                
                # Visualize
                viz = visualize_circuit(circuit)
                
                results.append((circuit, profile, viz))
            return results
        
        result = benchmark(batch_workflow, benchmark_iterations)
        assert result.avg_time < 0.1  # 100ms for 10 circuits
        print(f"\n{result}")


# ============================================================================
# Scaling Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestScalingBenchmarks:
    """Benchmark performance scaling with circuit size."""

    def test_benchmark_scaling_qubit_count(self):
        """Benchmark performance vs qubit count."""
        results = []
        for n_qubits in [2, 4, 6, 8, 10]:
            circuit = UnifiedCircuit(n_qubits)
            for i in range(n_qubits):
                circuit.add_gate(GateType.H, [i])
            for i in range(n_qubits - 1):
                circuit.add_gate(GateType.CNOT, [i, i+1])
            
            # Profile
            start = time.time()
            profile = profile_circuit(circuit)
            duration = time.time() - start
            
            results.append((n_qubits, duration))
            print(f"\n{n_qubits} qubits: {duration*1000:.3f}ms")
        
        # Check scaling is reasonable (roughly linear or better)
        assert results[-1][1] < results[0][1] * 10  # 10x qubits shouldn't be >10x slower

    def test_benchmark_scaling_gate_count(self):
        """Benchmark performance vs gate count."""
        results = []
        for n_gates in [10, 20, 50, 100]:
            circuit = UnifiedCircuit(4)
            for i in range(n_gates):
                if i % 2 == 0:
                    circuit.add_gate(GateType.H, [i % 4])
                else:
                    circuit.add_gate(GateType.CNOT, [i % 3, (i % 3) + 1])
            
            # Profile
            start = time.time()
            profile = profile_circuit(circuit)
            duration = time.time() - start
            
            results.append((n_gates, duration))
            print(f"\n{n_gates} gates: {duration*1000:.3f}ms")
        
        # Check scaling is reasonable
        assert results[-1][1] < results[0][1] * 15  # 10x gates shouldn't be >15x slower


# ============================================================================
# Regression Test Baselines
# ============================================================================

@pytest.mark.benchmark
class TestRegressionBaselines:
    """Establish baseline performance metrics for regression testing."""

    def test_baseline_bell_state_workflow(self):
        """Baseline: Complete Bell state workflow."""
        iterations = 200
        
        start = time.time()
        for _ in range(iterations):
            circuit = UnifiedCircuit(2)
            circuit.add_gate(GateType.H, [0])
            circuit.add_gate(GateType.CNOT, [0, 1])
            profile = profile_circuit(circuit)
            viz = visualize_circuit(circuit)
        duration = time.time() - start
        
        avg_time = duration / iterations
        print(f"\nBaseline Bell state workflow: {avg_time*1000:.3f}ms/iter")
        
        # Baseline: Should complete in < 5ms
        assert avg_time < 0.005

    def test_baseline_verification_performance(self):
        """Baseline: Verification performance."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.CNOT, [1, 2])
        verifier = PropertyVerifier()
        
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            is_valid = verifier.verify_unitarity(circuit)
        duration = time.time() - start
        
        avg_time = duration / iterations
        print(f"\nBaseline verification: {avg_time*1000:.3f}ms/iter")
        
        # Baseline: Should complete in < 10ms
        assert avg_time < 0.01

    def test_baseline_profiling_performance(self):
        """Baseline: Profiling performance."""
        circuit = UnifiedCircuit(5)
        for i in range(5):
            circuit.add_gate(GateType.H, [i])
        for i in range(4):
            circuit.add_gate(GateType.CNOT, [i, i+1])
        
        iterations = 200
        start = time.time()
        for _ in range(iterations):
            profile = profile_circuit(circuit)
        duration = time.time() - start
        
        avg_time = duration / iterations
        print(f"\nBaseline profiling: {avg_time*1000:.3f}ms/iter")
        
        # Baseline: Should complete in < 5ms
        assert avg_time < 0.005


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-m", "benchmark", "-s"])
