"""
Tests for profiling module.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.profiling import (
    CircuitProfiler,
    profile_circuit,
    PerformanceAnalyzer,
    analyze_performance,
    OptimizationProfiler,
    profile_optimization
)


class TestCircuitProfiler:
    """Test circuit profiling."""
    
    def test_profiler_creation(self):
        """Test creating a profiler."""
        profiler = CircuitProfiler()
        assert profiler is not None
        assert len(profiler.profiles) == 0
    
    def test_profile_simple_circuit(self):
        """Test profiling a simple circuit."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        profiler = CircuitProfiler()
        profile = profiler.profile_circuit(circuit)
        
        assert profile.n_qubits == 2
        assert profile.n_gates == 2
        assert profile.depth == 2
        assert profile.total_time > 0
        assert len(profile.gate_profiles) == 2
        assert len(profile.gate_counts) > 0
    
    def test_gate_time_distribution(self):
        """Test gate time distribution analysis."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        profiler = CircuitProfiler()
        profile = profiler.profile_circuit(circuit)
        distribution = profiler.get_gate_time_distribution(profile)
        
        assert GateType.H in distribution
        assert GateType.CNOT in distribution
        assert distribution[GateType.H] > 0
        assert distribution[GateType.CNOT] > 0
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        profiler = CircuitProfiler()
        profile = profiler.profile_circuit(circuit)
        bottlenecks = profiler.get_bottlenecks(profile, threshold=0.01)
        
        # Should return list (possibly empty for fast gates)
        assert isinstance(bottlenecks, list)
    
    def test_profile_comparison(self):
        """Test comparing two profiles."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])
        
        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        
        profiler = CircuitProfiler()
        profile1 = profiler.profile_circuit(circuit1)
        profile2 = profiler.profile_circuit(circuit2)
        
        comparison = profiler.compare_profiles(profile1, profile2)
        
        assert 'time_diff' in comparison
        assert 'gate_count_diff' in comparison
        assert 'depth_diff' in comparison
        assert comparison['gate_count_diff'] == -1  # profile2 has fewer gates
    
    def test_profile_summary(self):
        """Test profile summary generation."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.X, [2])
        
        profiler = CircuitProfiler()
        profile = profiler.profile_circuit(circuit)
        summary = profiler.get_summary(profile)
        
        assert summary['n_qubits'] == 3
        assert summary['n_gates'] == 3
        assert summary['total_time'] > 0
        assert 'avg_gate_time' in summary
        assert 'gate_counts' in summary
    
    def test_convenience_function(self):
        """Test profile_circuit convenience function."""
        circuit = UnifiedCircuit(1)
        circuit.add_gate(GateType.H, [0])
        
        profile = profile_circuit(circuit)
        
        assert profile.n_qubits == 1
        assert profile.n_gates == 1


class TestPerformanceAnalyzer:
    """Test performance analysis."""
    
    def test_analyzer_creation(self):
        """Test creating an analyzer."""
        analyzer = PerformanceAnalyzer()
        assert analyzer is not None
        assert len(analyzer.metrics_history) == 0
    
    def test_analyze_circuit(self):
        """Test analyzing a circuit."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.X, [2])
        
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_circuit(circuit)
        
        assert analysis['n_qubits'] == 3
        assert analysis['n_gates'] == 3
        assert 'gate_distribution' in analysis
        assert 'qubit_utilization' in analysis
        assert 'parallelism_score' in analysis
        assert 'memory_estimate' in analysis
        assert 'efficiency' in analysis
    
    def test_gate_distribution_analysis(self):
        """Test gate distribution analysis."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [1])
        circuit.add_gate(GateType.CNOT, [0, 1])
        
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_circuit(circuit)
        dist = analysis['gate_distribution']
        
        assert dist['single_qubit_gates'] == 2
        assert dist['two_qubit_gates'] == 1
        assert dist['unique_gate_types'] == 2
    
    def test_qubit_utilization_analysis(self):
        """Test qubit utilization analysis."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.X, [1])
        # Qubit 2 is idle
        
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_circuit(circuit)
        util = analysis['qubit_utilization']
        
        assert util['idle_qubits'] == 1  # Qubit 2
        assert util['qubit_usage'][0] == 2
        assert util['qubit_usage'][1] == 1
        assert util['qubit_usage'][2] == 0
    
    def test_parallelism_estimation(self):
        """Test parallelism score calculation."""
        # High parallelism: gates on different qubits (depth 1)
        circuit_parallel = UnifiedCircuit(3)
        circuit_parallel.add_gate(GateType.H, [0])
        circuit_parallel.add_gate(GateType.H, [1])
        circuit_parallel.add_gate(GateType.H, [2])
        
        # Lower parallelism: gates with dependencies (depth > 1)
        circuit_sequential = UnifiedCircuit(3)
        circuit_sequential.add_gate(GateType.H, [0])
        circuit_sequential.add_gate(GateType.CNOT, [0, 1])  # Depends on qubit 0
        circuit_sequential.add_gate(GateType.CNOT, [1, 2])  # Depends on qubit 1
        
        analyzer = PerformanceAnalyzer()
        parallel_score = analyzer.analyze_circuit(circuit_parallel)['parallelism_score']
        sequential_score = analyzer.analyze_circuit(circuit_sequential)['parallelism_score']
        
        # Parallel circuit should have higher score (more gates can run simultaneously)
        assert parallel_score >= sequential_score
    
    def test_memory_estimation(self):
        """Test memory estimation."""
        circuit_small = UnifiedCircuit(2)
        circuit_small.add_gate(GateType.H, [0])
        
        circuit_large = UnifiedCircuit(10)
        circuit_large.add_gate(GateType.H, [0])
        
        analyzer = PerformanceAnalyzer()
        mem_small = analyzer.analyze_circuit(circuit_small)['memory_estimate']
        mem_large = analyzer.analyze_circuit(circuit_large)['memory_estimate']
        
        # Larger circuit should require more memory
        assert mem_large > mem_small
    
    def test_circuit_comparison(self):
        """Test comparing two circuits."""
        circuit1 = UnifiedCircuit(2)
        circuit1.add_gate(GateType.H, [0])
        circuit1.add_gate(GateType.CNOT, [0, 1])
        circuit1.add_gate(GateType.X, [1])
        
        circuit2 = UnifiedCircuit(2)
        circuit2.add_gate(GateType.H, [0])
        
        analyzer = PerformanceAnalyzer()
        comparison = analyzer.compare_circuits(circuit1, circuit2)
        
        assert 'circuit1' in comparison
        assert 'circuit2' in comparison
        assert 'differences' in comparison
        assert comparison['differences']['gate_count_diff'] < 0  # circuit2 has fewer gates
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        # Create circuit with idle qubits
        circuit = UnifiedCircuit(5)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.X, [0])
        
        analyzer = PerformanceAnalyzer()
        suggestions = analyzer.suggest_optimizations(circuit)
        
        assert isinstance(suggestions, list)
        # Should suggest reducing idle qubits
        assert any('idle' in s.lower() for s in suggestions)
    
    def test_convenience_function(self):
        """Test analyze_performance convenience function."""
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.H, [0])
        
        analysis = analyze_performance(circuit)
        
        assert analysis['n_qubits'] == 2
        assert 'gate_distribution' in analysis


class TestOptimizationProfiler:
    """Test optimization profiling."""
    
    def test_profiler_creation(self):
        """Test creating an optimization profiler."""
        profiler = OptimizationProfiler()
        assert profiler is not None
        assert len(profiler.results) == 0
    
    def test_profile_optimization(self):
        """Test profiling an optimization."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.H, [0])  # Cancels out
        original.add_gate(GateType.CNOT, [0, 1])
        
        optimized = UnifiedCircuit(2)
        optimized.add_gate(GateType.CNOT, [0, 1])
        
        profiler = OptimizationProfiler()
        result = profiler.profile_optimization(original, optimized)
        
        assert result.original_gates == 3
        assert result.optimized_gates == 1
        assert result.gate_reduction == 2
        assert result.gate_reduction_pct > 0
    
    def test_optimization_with_timing(self):
        """Test optimization profiling with timing."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        def optimize(circuit):
            # Simple identity optimization
            opt = UnifiedCircuit(circuit.n_qubits)
            opt.add_gate(GateType.CNOT, [0, 1])
            return opt
        
        profiler = OptimizationProfiler()
        result = profiler.profile_optimization_with_timing(original, optimize)
        
        assert result.optimization_time >= 0
        assert result.gate_reduction > 0
    
    def test_compare_optimizations(self):
        """Test comparing multiple optimizations."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.X, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        opt1 = UnifiedCircuit(2)
        opt1.add_gate(GateType.H, [0])
        opt1.add_gate(GateType.CNOT, [0, 1])
        
        opt2 = UnifiedCircuit(2)
        opt2.add_gate(GateType.CNOT, [0, 1])
        
        profiler = OptimizationProfiler()
        comparison = profiler.compare_optimizations(original, {
            'opt1': opt1,
            'opt2': opt2
        })
        
        assert 'opt1' in comparison
        assert 'opt2' in comparison
        assert comparison['opt2']['gate_reduction'] > comparison['opt1']['gate_reduction']
    
    def test_get_statistics(self):
        """Test getting statistics across optimizations."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        optimized = UnifiedCircuit(2)
        optimized.add_gate(GateType.CNOT, [0, 1])
        
        profiler = OptimizationProfiler()
        profiler.profile_optimization(original, optimized)
        profiler.profile_optimization(original, optimized)
        
        stats = profiler.get_statistics()
        
        assert stats['n_optimizations'] == 2
        assert 'avg_gate_reduction_pct' in stats
        assert 'avg_depth_reduction_pct' in stats
    
    def test_benchmark_optimization(self):
        """Test benchmarking an optimization."""
        circuits = [
            UnifiedCircuit(2),
            UnifiedCircuit(3),
            UnifiedCircuit(2)
        ]
        for c in circuits:
            c.add_gate(GateType.H, [0])
            c.add_gate(GateType.X, [0])
        
        def optimize(circuit):
            opt = UnifiedCircuit(circuit.n_qubits)
            opt.add_gate(GateType.X, [0])
            return opt
        
        profiler = OptimizationProfiler()
        benchmark = profiler.benchmark_optimization(circuits, optimize)
        
        assert benchmark['n_circuits'] == 3
        assert 'avg_gate_reduction_pct' in benchmark
        assert 'avg_time' in benchmark
        assert 'success_rate' in benchmark
    
    def test_generate_report(self):
        """Test report generation."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        optimized = UnifiedCircuit(2)
        optimized.add_gate(GateType.CNOT, [0, 1])
        
        profiler = OptimizationProfiler()
        result = profiler.profile_optimization(original, optimized)
        report = profiler.generate_report(result)
        
        assert isinstance(report, str)
        assert 'Original Gates' in report
        assert 'Gate Reduction' in report
    
    def test_convenience_function(self):
        """Test profile_optimization convenience function."""
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        optimized = UnifiedCircuit(2)
        optimized.add_gate(GateType.CNOT, [0, 1])
        
        result = profile_optimization(original, optimized)
        
        assert result.original_gates == 2
        assert result.optimized_gates == 1


class TestIntegration:
    """Test integration between profiling components."""
    
    def test_profile_and_analyze(self):
        """Test profiling and analyzing a circuit."""
        circuit = UnifiedCircuit(3)
        circuit.add_gate(GateType.H, [0])
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.X, [2])
        
        # Profile
        profile = profile_circuit(circuit)
        assert profile.n_gates == 3
        
        # Analyze
        analysis = analyze_performance(circuit)
        assert analysis['n_gates'] == 3
    
    def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create original circuit
        original = UnifiedCircuit(2)
        original.add_gate(GateType.H, [0])
        original.add_gate(GateType.X, [0])
        original.add_gate(GateType.CNOT, [0, 1])
        
        # Analyze original
        original_analysis = analyze_performance(original)
        
        # Create optimized version
        optimized = UnifiedCircuit(2)
        optimized.add_gate(GateType.H, [0])
        optimized.add_gate(GateType.CNOT, [0, 1])
        
        # Profile optimization
        opt_result = profile_optimization(original, optimized)
        
        # Analyze optimized
        optimized_analysis = analyze_performance(optimized)
        
        assert opt_result.gate_reduction > 0
        assert optimized_analysis['n_gates'] < original_analysis['n_gates']
