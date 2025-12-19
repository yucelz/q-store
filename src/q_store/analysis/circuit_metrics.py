"""
Advanced circuit metrics and analysis.

Provides structural analysis of quantum circuits including
entanglement measures, parallelism metrics, and circuit properties.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from ..core import UnifiedCircuit, GateType
from .complexity import compute_circuit_depth


class CircuitMetrics:
    """
    Analyze structural properties of quantum circuits.
    
    Computes advanced metrics including:
    - Entanglement measures
    - Parallelism scores
    - Critical path analysis
    - Gate dependency graphs
    """
    
    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize circuit metrics analyzer.
        
        Args:
            circuit: Quantum circuit to analyze
        """
        self.circuit = circuit
        self._entangling_gates = None
        self._parallel_layers = None
    
    def count_entangling_gates(self) -> int:
        """
        Count number of entangling (two-qubit) gates.
        
        Returns:
            Number of entangling gates
        """
        if self._entangling_gates is None:
            self._entangling_gates = 0
            for gate in self.circuit.gates:
                if len(gate.targets) > 1 or gate.controls:
                    self._entangling_gates += 1
        return self._entangling_gates
    
    def compute_entanglement_measure(self) -> float:
        """
        Compute normalized entanglement measure.
        
        Simple metric based on ratio of entangling gates to total gates.
        
        Returns:
            Entanglement measure (0 to 1)
        """
        total_gates = len(self.circuit.gates)
        if total_gates == 0:
            return 0.0
        
        entangling_count = self.count_entangling_gates()
        return entangling_count / total_gates
    
    def compute_parallelism_score(self) -> float:
        """
        Compute circuit parallelism score.
        
        Measures how much parallelism exists in the circuit.
        Score = 1 - (depth / total_gates) for circuits with gates.
        
        Returns:
            Parallelism score (0 to 1, higher = more parallel)
        """
        total_gates = len(self.circuit.gates)
        if total_gates == 0:
            return 0.0
        
        depth = compute_circuit_depth(self.circuit)
        if depth == 0:
            return 0.0
        
        # Perfect parallelism: depth = 1, score = 1
        # No parallelism: depth = total_gates, score = 0
        return 1.0 - (depth / total_gates)
    
    def find_parallel_layers(self) -> List[List[int]]:
        """
        Partition gates into parallel layers.
        
        Returns:
            List of layers, each containing gate indices that can execute in parallel
        """
        if self._parallel_layers is not None:
            return self._parallel_layers
        
        if not self.circuit.gates:
            self._parallel_layers = []
            return self._parallel_layers
        
        # Track when each qubit is available
        qubit_available_at = [0] * self.circuit.n_qubits
        gate_times = []
        
        for i, gate in enumerate(self.circuit.gates):
            targets = gate.targets
            controls = gate.controls or []
            all_qubits = targets + controls
            
            # Find when all qubits are available
            earliest_time = max(qubit_available_at[q] for q in all_qubits)
            gate_time = earliest_time + 1
            
            gate_times.append(gate_time)
            
            # Update qubit availability
            for q in all_qubits:
                qubit_available_at[q] = gate_time
        
        # Group gates by time layer
        max_time = max(gate_times) if gate_times else 0
        layers = [[] for _ in range(max_time)]
        
        for i, time in enumerate(gate_times):
            layers[time - 1].append(i)
        
        self._parallel_layers = layers
        return self._parallel_layers
    
    def compute_critical_path_length(self) -> int:
        """
        Compute critical path length (circuit depth).
        
        Returns:
            Length of critical path
        """
        return compute_circuit_depth(self.circuit)
    
    def analyze_qubit_usage(self) -> Dict:
        """
        Analyze how qubits are used in the circuit.
        
        Returns:
            Dictionary with per-qubit statistics
        """
        qubit_gate_counts = [0] * self.circuit.n_qubits
        qubit_idle_time = [0] * self.circuit.n_qubits
        
        # Track last usage time for each qubit
        last_used = [-1] * self.circuit.n_qubits
        
        for time_step, gate in enumerate(self.circuit.gates):
            targets = gate.targets
            controls = gate.controls or []
            all_qubits = targets + controls
            
            for q in all_qubits:
                qubit_gate_counts[q] += 1
                if last_used[q] >= 0:
                    qubit_idle_time[q] += time_step - last_used[q] - 1
                last_used[q] = time_step
        
        return {
            'gate_counts_per_qubit': qubit_gate_counts,
            'idle_time_per_qubit': qubit_idle_time,
            'total_qubits': self.circuit.n_qubits,
            'active_qubits': sum(1 for count in qubit_gate_counts if count > 0)
        }
    
    def compute_connectivity_requirements(self) -> Set[Tuple[int, int]]:
        """
        Determine required qubit connectivity.
        
        Returns:
            Set of qubit pairs that need to interact
        """
        connectivity = set()
        
        for gate in self.circuit.gates:
            targets = gate.targets
            controls = gate.controls or []
            
            # Two-qubit gates require connectivity
            if len(targets) == 2:
                q1, q2 = sorted(targets)
                connectivity.add((q1, q2))
            elif controls and targets:
                for ctrl in controls:
                    for tgt in targets:
                        q1, q2 = sorted([ctrl, tgt])
                        connectivity.add((q1, q2))
        
        return connectivity
    
    def summary(self) -> Dict:
        """
        Get comprehensive circuit metrics summary.
        
        Returns:
            Dictionary with all computed metrics
        """
        return {
            'total_gates': len(self.circuit.gates),
            'entangling_gates': self.count_entangling_gates(),
            'entanglement_measure': self.compute_entanglement_measure(),
            'parallelism_score': self.compute_parallelism_score(),
            'critical_path_length': self.compute_critical_path_length(),
            'qubit_usage': self.analyze_qubit_usage(),
            'connectivity_requirements': len(self.compute_connectivity_requirements()),
            'parallel_layers': len(self.find_parallel_layers())
        }


def compute_entanglement_measure(circuit: UnifiedCircuit) -> float:
    """
    Compute entanglement measure for circuit.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Entanglement measure (0 to 1)
    """
    metrics = CircuitMetrics(circuit)
    return metrics.compute_entanglement_measure()


def compute_parallelism_score(circuit: UnifiedCircuit) -> float:
    """
    Compute parallelism score for circuit.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Parallelism score (0 to 1)
    """
    metrics = CircuitMetrics(circuit)
    return metrics.compute_parallelism_score()


def compute_critical_path_length(circuit: UnifiedCircuit) -> int:
    """
    Compute critical path length.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Critical path length (circuit depth)
    """
    metrics = CircuitMetrics(circuit)
    return metrics.compute_critical_path_length()


def analyze_circuit_structure(circuit: UnifiedCircuit) -> Dict:
    """
    Perform comprehensive structural analysis.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Complete circuit metrics summary
    """
    metrics = CircuitMetrics(circuit)
    return metrics.summary()
