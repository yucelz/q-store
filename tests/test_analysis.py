"""
Tests for circuit analysis tools.
"""

import pytest
import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.analysis import (
    CircuitComplexity,
    compute_circuit_depth,
    compute_circuit_width,
    count_gates_by_type,
    compute_t_depth,
    compute_cnot_count,
    ResourceEstimator,
    estimate_resources,
    estimate_execution_time,
    estimate_hardware_cost,
    HardwareModel,
    CircuitMetrics,
    compute_entanglement_measure,
    compute_parallelism_score,
    compute_critical_path_length,
    analyze_circuit_structure
)


def create_simple_circuit():
    """Create a simple test circuit."""
    circuit = UnifiedCircuit(n_qubits=3)
    circuit.add_gate(GateType.H, targets=[0])
    circuit.add_gate(GateType.CNOT, targets=[0, 1])
    circuit.add_gate(GateType.CNOT, targets=[1, 2])
    circuit.add_gate(GateType.H, targets=[0])
    return circuit


def create_parallel_circuit():
    """Create a circuit with parallel gates."""
    circuit = UnifiedCircuit(n_qubits=4)
    # Layer 1: Independent gates
    circuit.add_gate(GateType.H, targets=[0])
    circuit.add_gate(GateType.H, targets=[1])
    circuit.add_gate(GateType.H, targets=[2])
    circuit.add_gate(GateType.H, targets=[3])
    # Layer 2: Two CNOTs
    circuit.add_gate(GateType.CNOT, targets=[0, 1])
    circuit.add_gate(GateType.CNOT, targets=[2, 3])
    return circuit


class TestCircuitComplexity:
    """Test circuit complexity analysis."""
    
    def test_complexity_creation(self):
        """Test creating complexity analyzer."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.circuit == circuit
    
    def test_total_gates(self):
        """Test total gate count."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.total_gates() == 4
    
    def test_gate_counts(self):
        """Test counting gates by type."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        counts = complexity.gate_counts()
        
        assert counts[GateType.H] == 2
        assert counts[GateType.CNOT] == 2
    
    def test_circuit_depth(self):
        """Test circuit depth computation."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        depth = complexity.depth()
        
        # H, CNOT, CNOT, H = depth 4 (sequential)
        assert depth >= 3
    
    def test_circuit_width(self):
        """Test circuit width."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.width() == 3
    
    def test_t_count(self):
        """Test T gate counting."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.T, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        circuit.add_gate(GateType.T, targets=[1])
        
        complexity = CircuitComplexity(circuit)
        assert complexity.t_count() == 2
    
    def test_t_depth(self):
        """Test T-depth computation."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.T, targets=[0])
        circuit.add_gate(GateType.T, targets=[1])  # Parallel with first
        circuit.add_gate(GateType.T, targets=[0])  # Sequential on qubit 0
        
        complexity = CircuitComplexity(circuit)
        t_depth = complexity.t_depth()
        assert t_depth == 2  # Two T layers on qubit 0
    
    def test_cnot_count(self):
        """Test CNOT counting."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.cnot_count() == 2
    
    def test_single_qubit_gate_count(self):
        """Test single-qubit gate counting."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.single_qubit_gate_count() == 2  # 2 H gates
    
    def test_two_qubit_gate_count(self):
        """Test two-qubit gate counting."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        assert complexity.two_qubit_gate_count() == 2  # 2 CNOTs
    
    def test_complexity_summary(self):
        """Test complexity summary."""
        circuit = create_simple_circuit()
        complexity = CircuitComplexity(circuit)
        summary = complexity.summary()
        
        assert 'total_gates' in summary
        assert 'depth' in summary
        assert 'width' in summary
        assert summary['total_gates'] == 4
        assert summary['width'] == 3


class TestComplexityFunctions:
    """Test standalone complexity functions."""
    
    def test_compute_circuit_depth(self):
        """Test circuit depth function."""
        circuit = create_simple_circuit()
        depth = compute_circuit_depth(circuit)
        assert depth >= 3
    
    def test_compute_circuit_width(self):
        """Test circuit width function."""
        circuit = create_simple_circuit()
        width = compute_circuit_width(circuit)
        assert width == 3
    
    def test_count_gates_by_type(self):
        """Test gate counting function."""
        circuit = create_simple_circuit()
        counts = count_gates_by_type(circuit)
        assert counts[GateType.H] == 2
        assert counts[GateType.CNOT] == 2
    
    def test_compute_t_depth(self):
        """Test T-depth function."""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.T, targets=[0])
        circuit.add_gate(GateType.T, targets=[0])
        
        t_depth = compute_t_depth(circuit)
        assert t_depth == 2
    
    def test_compute_cnot_count(self):
        """Test CNOT count function."""
        circuit = create_simple_circuit()
        cnot_count = compute_cnot_count(circuit)
        assert cnot_count == 2
    
    def test_empty_circuit(self):
        """Test functions on empty circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        assert compute_circuit_depth(circuit) == 0
        assert compute_t_depth(circuit) == 0
        assert compute_cnot_count(circuit) == 0


class TestResourceEstimator:
    """Test resource estimation."""
    
    def test_estimator_creation(self):
        """Test creating resource estimator."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        assert estimator.circuit == circuit
    
    def test_estimate_execution_time(self):
        """Test execution time estimation."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        time = estimator.estimate_execution_time()
        
        assert time > 0
        assert isinstance(time, float)
    
    def test_estimate_error_rate(self):
        """Test error rate estimation."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        error_rate = estimator.estimate_error_rate()
        
        assert 0 <= error_rate <= 1
    
    def test_estimate_decoherence_error(self):
        """Test decoherence error estimation."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        decoherence = estimator.estimate_decoherence_error()
        
        assert 0 <= decoherence <= 1
    
    def test_hardware_compatibility(self):
        """Test hardware compatibility check."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        compatibility = estimator.check_hardware_compatibility()
        
        assert 'compatible' in compatibility
        assert 'issues' in compatibility
        assert 'hardware' in compatibility
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        cost = estimator.estimate_cost(shots=1024)
        
        assert 'base_cost_per_shot' in cost
        assert 'total_cost' in cost
        assert cost['shots'] == 1024
    
    def test_different_hardware_models(self):
        """Test with different hardware models."""
        circuit = create_simple_circuit()
        
        generic_est = ResourceEstimator(circuit, HardwareModel(name='Generic'))
        time_generic = generic_est.estimate_execution_time()
        
        ionq_model = HardwareModel(
            name='IonQ',
            single_qubit_gate_time=10.0,
            two_qubit_gate_time=200.0
        )
        ionq_est = ResourceEstimator(circuit, ionq_model)
        time_ionq = ionq_est.estimate_execution_time()
        
        # IonQ should be slower (longer gate times)
        assert time_ionq > time_generic
    
    def test_resource_summary(self):
        """Test comprehensive resource summary."""
        circuit = create_simple_circuit()
        estimator = ResourceEstimator(circuit)
        summary = estimator.summary()
        
        assert 'complexity' in summary
        assert 'execution_time_us' in summary
        assert 'error_rate' in summary
        assert 'hardware_compatibility' in summary
        assert 'cost_estimate' in summary


class TestResourceFunctions:
    """Test standalone resource functions."""
    
    def test_estimate_resources(self):
        """Test estimate_resources function."""
        circuit = create_simple_circuit()
        resources = estimate_resources(circuit)
        
        assert 'complexity' in resources
        assert 'execution_time_us' in resources
    
    def test_estimate_execution_time_function(self):
        """Test execution time function."""
        circuit = create_simple_circuit()
        time = estimate_execution_time(circuit)
        assert time > 0
    
    def test_estimate_hardware_cost_function(self):
        """Test hardware cost function."""
        circuit = create_simple_circuit()
        cost = estimate_hardware_cost(circuit, shots=100)
        
        assert 'total_cost' in cost
        assert cost['shots'] == 100
    
    def test_different_hardware_names(self):
        """Test with different hardware names."""
        circuit = create_simple_circuit()
        
        generic_time = estimate_execution_time(circuit, 'generic')
        ibm_time = estimate_execution_time(circuit, 'ibm_quantum')
        ionq_time = estimate_execution_time(circuit, 'ionq')
        
        assert all(t > 0 for t in [generic_time, ibm_time, ionq_time])


class TestCircuitMetrics:
    """Test circuit metrics analysis."""
    
    def test_metrics_creation(self):
        """Test creating metrics analyzer."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        assert metrics.circuit == circuit
    
    def test_count_entangling_gates(self):
        """Test entangling gate counting."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        assert metrics.count_entangling_gates() == 2  # 2 CNOTs
    
    def test_entanglement_measure(self):
        """Test entanglement measure."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        measure = metrics.compute_entanglement_measure()
        
        assert 0 <= measure <= 1
        assert measure == 0.5  # 2 entangling / 4 total
    
    def test_parallelism_score(self):
        """Test parallelism score."""
        circuit = create_parallel_circuit()
        metrics = CircuitMetrics(circuit)
        score = metrics.compute_parallelism_score()
        
        assert 0 <= score <= 1
        # Parallel circuit should have high score
        assert score > 0.5
    
    def test_find_parallel_layers(self):
        """Test parallel layer identification."""
        circuit = create_parallel_circuit()
        metrics = CircuitMetrics(circuit)
        layers = metrics.find_parallel_layers()
        
        assert len(layers) > 0
        # First layer should have 4 H gates
        assert len(layers[0]) == 4
    
    def test_critical_path_length(self):
        """Test critical path computation."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        path_length = metrics.compute_critical_path_length()
        
        assert path_length >= 3
    
    def test_qubit_usage(self):
        """Test qubit usage analysis."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        usage = metrics.analyze_qubit_usage()
        
        assert 'gate_counts_per_qubit' in usage
        assert 'idle_time_per_qubit' in usage
        assert 'total_qubits' in usage
        assert usage['total_qubits'] == 3
    
    def test_connectivity_requirements(self):
        """Test connectivity analysis."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        connectivity = metrics.compute_connectivity_requirements()
        
        # Should have (0,1) and (1,2) connections
        assert len(connectivity) >= 2
    
    def test_metrics_summary(self):
        """Test metrics summary."""
        circuit = create_simple_circuit()
        metrics = CircuitMetrics(circuit)
        summary = metrics.summary()
        
        assert 'total_gates' in summary
        assert 'entangling_gates' in summary
        assert 'parallelism_score' in summary
        assert 'qubit_usage' in summary


class TestMetricsFunctions:
    """Test standalone metrics functions."""
    
    def test_compute_entanglement_measure(self):
        """Test entanglement measure function."""
        circuit = create_simple_circuit()
        measure = compute_entanglement_measure(circuit)
        assert 0 <= measure <= 1
    
    def test_compute_parallelism_score(self):
        """Test parallelism score function."""
        circuit = create_parallel_circuit()
        score = compute_parallelism_score(circuit)
        assert 0 <= score <= 1
    
    def test_compute_critical_path_length(self):
        """Test critical path function."""
        circuit = create_simple_circuit()
        length = compute_critical_path_length(circuit)
        assert length >= 3
    
    def test_analyze_circuit_structure(self):
        """Test structure analysis function."""
        circuit = create_simple_circuit()
        structure = analyze_circuit_structure(circuit)
        
        assert 'total_gates' in structure
        assert 'entanglement_measure' in structure
        assert 'parallelism_score' in structure


class TestIntegration:
    """Integration tests for analysis tools."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis workflow."""
        circuit = create_simple_circuit()
        
        # Complexity analysis
        complexity = CircuitComplexity(circuit)
        comp_summary = complexity.summary()
        
        # Resource estimation
        resources = estimate_resources(circuit)
        
        # Metrics analysis
        metrics = CircuitMetrics(circuit)
        metrics_summary = metrics.summary()
        
        # All should have meaningful data
        assert comp_summary['total_gates'] > 0
        assert resources['execution_time_us'] > 0
        assert metrics_summary['total_gates'] > 0
    
    def test_empty_circuit_analysis(self):
        """Test analysis of empty circuit."""
        circuit = UnifiedCircuit(n_qubits=2)
        
        complexity = CircuitComplexity(circuit)
        assert complexity.total_gates() == 0
        assert complexity.depth() == 0
        
        metrics = CircuitMetrics(circuit)
        assert metrics.count_entangling_gates() == 0
    
    def test_large_circuit_analysis(self):
        """Test analysis on larger circuit."""
        circuit = UnifiedCircuit(n_qubits=10)
        
        # Create a larger circuit
        for i in range(10):
            circuit.add_gate(GateType.H, targets=[i])
        for i in range(9):
            circuit.add_gate(GateType.CNOT, targets=[i, i+1])
        
        complexity = CircuitComplexity(circuit)
        assert complexity.total_gates() == 19
        assert complexity.width() == 10
        
        resources = estimate_resources(circuit)
        assert resources['execution_time_us'] > 0
