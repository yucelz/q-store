"""
Advanced circuit analysis tools for Q-Store.

This module provides comprehensive circuit analysis capabilities including:
- Complexity metrics (gate counts, depth, width)
- Resource estimation (T-gates, CNOTs, total gates)
- Circuit properties (entanglement, parallelism)
- Cost models for different quantum hardware
"""

from .complexity import (
    CircuitComplexity,
    compute_circuit_depth,
    compute_circuit_width,
    count_gates_by_type,
    compute_t_depth,
    compute_cnot_count
)

from .resource_estimator import (
    ResourceEstimator,
    estimate_resources,
    estimate_execution_time,
    estimate_hardware_cost,
    HardwareModel
)

from .circuit_metrics import (
    CircuitMetrics,
    compute_entanglement_measure,
    compute_parallelism_score,
    compute_critical_path_length,
    analyze_circuit_structure
)

from .quantum_metrics_computer import (
    QuantumMetricsComputer,
    compute_all_quantum_metrics
)

__all__ = [
    # Complexity analysis
    'CircuitComplexity',
    'compute_circuit_depth',
    'compute_circuit_width',
    'count_gates_by_type',
    'compute_t_depth',
    'compute_cnot_count',

    # Resource estimation
    'ResourceEstimator',
    'estimate_resources',
    'estimate_execution_time',
    'estimate_hardware_cost',
    'HardwareModel',

    # Circuit metrics
    'CircuitMetrics',
    'compute_entanglement_measure',
    'compute_parallelism_score',
    'compute_critical_path_length',
    'analyze_circuit_structure',

    # v4.1 Enhanced: Quantum-specific metrics
    'QuantumMetricsComputer',
    'compute_all_quantum_metrics'
]
