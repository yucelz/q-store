"""
Performance analyzer for quantum circuits.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from q_store.core import UnifiedCircuit, GateType


@dataclass
class PerformanceMetrics:
    """Performance metrics for circuit analysis."""
    execution_time: float
    memory_estimate: int  # In bytes
    gate_efficiency: float  # Gates per second
    qubit_utilization: float  # Fraction of qubits used
    parallelism_score: float  # Potential for parallel execution


class PerformanceAnalyzer:
    """
    Analyzer for quantum circuit performance characteristics.
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize performance analyzer.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.metrics_history: List[PerformanceMetrics] = []

    def analyze_circuit(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Analyze circuit performance characteristics.

        Args:
            circuit: Circuit to analyze

        Returns:
            Dictionary with performance analysis
        """
        analysis = {}

        # Basic metrics
        analysis['n_qubits'] = circuit.n_qubits
        analysis['n_gates'] = len(circuit.gates)
        analysis['depth'] = circuit.depth

        # Gate distribution
        analysis['gate_distribution'] = self._analyze_gate_distribution(circuit)

        # Qubit utilization
        analysis['qubit_utilization'] = self._analyze_qubit_utilization(circuit)

        # Parallelism potential
        analysis['parallelism_score'] = self._estimate_parallelism(circuit)

        # Memory estimate
        analysis['memory_estimate'] = self._estimate_memory(circuit)

        # Efficiency metrics
        analysis['efficiency'] = self._compute_efficiency_metrics(circuit)

        return analysis

    def _analyze_gate_distribution(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """Analyze distribution of gate types."""
        gate_counts = {}
        single_qubit_gates = 0
        two_qubit_gates = 0

        for gate in circuit.gates:
            gate_counts[gate.gate_type] = gate_counts.get(gate.gate_type, 0) + 1

            if len(gate.targets) == 1:
                single_qubit_gates += 1
            elif len(gate.targets) == 2:
                two_qubit_gates += 1

        return {
            'gate_counts': gate_counts,
            'single_qubit_gates': single_qubit_gates,
            'two_qubit_gates': two_qubit_gates,
            'single_to_two_ratio': single_qubit_gates / two_qubit_gates if two_qubit_gates > 0 else float('inf'),
            'most_common_gate': max(gate_counts.items(), key=lambda x: x[1])[0] if gate_counts else None,
            'unique_gate_types': len(gate_counts)
        }

    def _analyze_qubit_utilization(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """Analyze how qubits are utilized."""
        qubit_usage = {i: 0 for i in range(circuit.n_qubits)}

        for gate in circuit.gates:
            for target in gate.targets:
                qubit_usage[target] += 1

        usage_values = list(qubit_usage.values())
        total_usage = sum(usage_values)

        return {
            'qubit_usage': qubit_usage,
            'avg_gates_per_qubit': total_usage / circuit.n_qubits if circuit.n_qubits > 0 else 0,
            'max_gates_on_qubit': max(usage_values) if usage_values else 0,
            'min_gates_on_qubit': min(usage_values) if usage_values else 0,
            'idle_qubits': sum(1 for v in usage_values if v == 0),
            'utilization_uniformity': 1.0 - (np.std(usage_values) / np.mean(usage_values)) if np.mean(usage_values) > 0 else 0.0
        }

    def _estimate_parallelism(self, circuit: UnifiedCircuit) -> float:
        """
        Estimate parallelism potential.

        Higher score means more gates can potentially execute in parallel.
        """
        if circuit.depth == 0 or len(circuit.gates) == 0:
            return 0.0

        # Ideal parallelism would be all gates in one layer
        # Actual parallelism is gates / depth
        parallelism = len(circuit.gates) / circuit.depth

        # Normalize to [0, 1] based on number of qubits
        max_parallelism = circuit.n_qubits
        normalized = min(parallelism / max_parallelism, 1.0) if max_parallelism > 0 else 0.0

        return normalized

    def _estimate_memory(self, circuit: UnifiedCircuit) -> int:
        """
        Estimate memory requirements in bytes.

        State vector simulation requires 2^n complex numbers.
        """
        # Complex128 = 16 bytes per amplitude
        n_amplitudes = 2 ** circuit.n_qubits
        state_vector_size = n_amplitudes * 16

        # Add overhead for gates (rough estimate)
        gate_overhead = len(circuit.gates) * 100  # ~100 bytes per gate

        return state_vector_size + gate_overhead

    def _compute_efficiency_metrics(self, circuit: UnifiedCircuit) -> Dict[str, float]:
        """Compute various efficiency metrics."""
        return {
            'gates_per_qubit': len(circuit.gates) / circuit.n_qubits if circuit.n_qubits > 0 else 0.0,
            'gates_per_depth': len(circuit.gates) / circuit.depth if circuit.depth > 0 else 0.0,
            'depth_to_gates_ratio': circuit.depth / len(circuit.gates) if len(circuit.gates) > 0 else 0.0,
            'sparsity': 1.0 - (len(circuit.gates) / (circuit.n_qubits * circuit.depth)) if (circuit.n_qubits * circuit.depth) > 0 else 0.0
        }

    def compare_circuits(self, circuit1: UnifiedCircuit,
                        circuit2: UnifiedCircuit) -> Dict[str, Any]:
        """
        Compare performance characteristics of two circuits.

        Args:
            circuit1: First circuit
            circuit2: Second circuit

        Returns:
            Comparison results
        """
        analysis1 = self.analyze_circuit(circuit1)
        analysis2 = self.analyze_circuit(circuit2)

        return {
            'circuit1': analysis1,
            'circuit2': analysis2,
            'differences': {
                'gate_count_diff': analysis2['n_gates'] - analysis1['n_gates'],
                'depth_diff': analysis2['depth'] - analysis1['depth'],
                'memory_diff': analysis2['memory_estimate'] - analysis1['memory_estimate'],
                'parallelism_diff': analysis2['parallelism_score'] - analysis1['parallelism_score'],
                'gate_reduction_pct': 100 * (analysis1['n_gates'] - analysis2['n_gates']) / analysis1['n_gates'] if analysis1['n_gates'] > 0 else 0.0,
                'depth_reduction_pct': 100 * (analysis1['depth'] - analysis2['depth']) / analysis1['depth'] if analysis1['depth'] > 0 else 0.0
            }
        }

    def suggest_optimizations(self, circuit: UnifiedCircuit) -> List[str]:
        """
        Suggest potential optimizations based on analysis.

        Args:
            circuit: Circuit to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        analysis = self.analyze_circuit(circuit)

        # Check for low parallelism
        if analysis['parallelism_score'] < 0.3:
            suggestions.append("Low parallelism detected. Consider reordering gates to increase parallel execution.")

        # Check qubit utilization
        qubit_util = analysis['qubit_utilization']
        if qubit_util['idle_qubits'] > 0:
            suggestions.append(f"{qubit_util['idle_qubits']} idle qubits detected. Consider reducing circuit width.")

        # Check for unbalanced qubit usage
        if qubit_util['utilization_uniformity'] < 0.5:
            suggestions.append("Uneven qubit utilization. Some qubits are used much more than others.")

        # Check depth vs gates
        if analysis['efficiency']['depth_to_gates_ratio'] > 0.5:
            suggestions.append("High depth-to-gates ratio. Circuit may benefit from gate commutation optimizations.")

        # Check for two-qubit gate dominance
        gate_dist = analysis['gate_distribution']
        if gate_dist['two_qubit_gates'] > gate_dist['single_qubit_gates']:
            suggestions.append("High ratio of two-qubit gates. These are typically more expensive to execute.")

        # Memory warnings
        if analysis['memory_estimate'] > 1e9:  # > 1GB
            suggestions.append(f"Large memory requirement ({analysis['memory_estimate'] / 1e9:.2f} GB). Consider circuit splitting.")

        return suggestions


def analyze_performance(circuit: UnifiedCircuit) -> Dict[str, Any]:
    """
    Convenience function to analyze circuit performance.

    Args:
        circuit: Circuit to analyze

    Returns:
        Performance analysis
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_circuit(circuit)
