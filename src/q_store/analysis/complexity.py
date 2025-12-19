"""
Circuit complexity analysis tools.

Provides metrics for analyzing quantum circuit complexity including
gate counts, depth, width, and specialized gate metrics.
"""

from typing import Dict, List, Optional
import numpy as np
from ..core import UnifiedCircuit, GateType


class CircuitComplexity:
    """
    Analyze complexity metrics of a quantum circuit.
    
    Computes various complexity measures including:
    - Total gate count
    - Circuit depth (critical path length)
    - Circuit width (number of qubits)
    - Gate counts by type
    - T-depth (number of T-gate layers)
    - CNOT count and depth
    """
    
    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize complexity analyzer.
        
        Args:
            circuit: Quantum circuit to analyze
        """
        self.circuit = circuit
        self._gate_counts = None
        self._depth = None
        self._t_depth = None
        self._cnot_depth = None
    
    def total_gates(self) -> int:
        """Get total number of gates in circuit."""
        return len(self.circuit.gates)
    
    def gate_counts(self) -> Dict[GateType, int]:
        """
        Count gates by type.
        
        Returns:
            Dictionary mapping gate types to counts
        """
        if self._gate_counts is None:
            self._gate_counts = {}
            for gate in self.circuit.gates:
                gate_type = gate.gate_type
                self._gate_counts[gate_type] = self._gate_counts.get(gate_type, 0) + 1
        return self._gate_counts.copy()
    
    def depth(self) -> int:
        """
        Compute circuit depth (length of critical path).
        
        Returns:
            Circuit depth (number of layers)
        """
        if self._depth is None:
            self._depth = compute_circuit_depth(self.circuit)
        return self._depth
    
    def width(self) -> int:
        """Get circuit width (number of qubits)."""
        return self.circuit.n_qubits
    
    def t_count(self) -> int:
        """Count number of T gates."""
        counts = self.gate_counts()
        return counts.get(GateType.T, 0)
    
    def t_depth(self) -> int:
        """
        Compute T-depth (number of T-gate layers).
        
        Important for fault-tolerant quantum computing as
        T gates are typically the most expensive.
        
        Returns:
            T-depth of circuit
        """
        if self._t_depth is None:
            self._t_depth = compute_t_depth(self.circuit)
        return self._t_depth
    
    def cnot_count(self) -> int:
        """Count number of CNOT gates."""
        return compute_cnot_count(self.circuit)
    
    def cnot_depth(self) -> int:
        """
        Compute CNOT depth (number of CNOT layers).
        
        Returns:
            CNOT depth of circuit
        """
        if self._cnot_depth is None:
            self._cnot_depth = self._compute_gate_depth(GateType.CNOT)
        return self._cnot_depth
    
    def single_qubit_gate_count(self) -> int:
        """Count single-qubit gates."""
        counts = self.gate_counts()
        single_qubit_gates = [
            GateType.H, GateType.X, GateType.Y, GateType.Z,
            GateType.S, GateType.T, GateType.RX, GateType.RY, GateType.RZ
        ]
        return sum(counts.get(gate, 0) for gate in single_qubit_gates)
    
    def two_qubit_gate_count(self) -> int:
        """Count two-qubit gates."""
        counts = self.gate_counts()
        two_qubit_gates = [
            GateType.CNOT, GateType.CZ, GateType.SWAP
        ]
        return sum(counts.get(gate, 0) for gate in two_qubit_gates)
    
    def _compute_gate_depth(self, gate_type: GateType) -> int:
        """Compute depth for specific gate type."""
        # Get positions of all gates of this type
        gate_positions = []
        for i, gate in enumerate(self.circuit.gates):
            if gate.gate_type == gate_type:
                gate_positions.append((i, gate.targets))
        
        if not gate_positions:
            return 0
        
        # Track qubit availability
        qubit_available_at = [0] * self.circuit.n_qubits
        depth = 0
        
        for _, targets in gate_positions:
            # Find when all target qubits are available
            earliest_time = max(qubit_available_at[q] for q in targets)
            # Place gate at earliest_time + 1
            gate_time = earliest_time + 1
            # Update qubit availability
            for q in targets:
                qubit_available_at[q] = gate_time
            depth = max(depth, gate_time)
        
        return depth
    
    def summary(self) -> Dict:
        """
        Get comprehensive complexity summary.
        
        Returns:
            Dictionary with all complexity metrics
        """
        return {
            'total_gates': self.total_gates(),
            'depth': self.depth(),
            'width': self.width(),
            'gate_counts': self.gate_counts(),
            't_count': self.t_count(),
            't_depth': self.t_depth(),
            'cnot_count': self.cnot_count(),
            'cnot_depth': self.cnot_depth(),
            'single_qubit_gates': self.single_qubit_gate_count(),
            'two_qubit_gates': self.two_qubit_gate_count()
        }


def compute_circuit_depth(circuit: UnifiedCircuit) -> int:
    """
    Compute circuit depth (critical path length).
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Circuit depth
    """
    if not circuit.gates:
        return 0
    
    # Track when each qubit is available
    qubit_available_at = [0] * circuit.n_qubits
    
    for gate in circuit.gates:
        targets = gate.targets
        controls = gate.controls or []
        all_qubits = targets + controls
        
        # Find when all qubits for this gate are available
        earliest_time = max(qubit_available_at[q] for q in all_qubits)
        
        # Place gate at earliest_time + 1
        gate_time = earliest_time + 1
        
        # Update availability for all qubits involved
        for q in all_qubits:
            qubit_available_at[q] = gate_time
    
    return max(qubit_available_at)


def compute_circuit_width(circuit: UnifiedCircuit) -> int:
    """
    Compute circuit width (number of qubits).
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Number of qubits
    """
    return circuit.n_qubits


def count_gates_by_type(circuit: UnifiedCircuit) -> Dict[GateType, int]:
    """
    Count gates by type.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Dictionary mapping gate types to counts
    """
    counts = {}
    for gate in circuit.gates:
        gate_type = gate.gate_type
        counts[gate_type] = counts.get(gate_type, 0) + 1
    return counts


def compute_t_depth(circuit: UnifiedCircuit) -> int:
    """
    Compute T-depth (number of T-gate layers).
    
    T-depth is important for fault-tolerant quantum computing
    as T gates are typically the most expensive to implement.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        T-depth of circuit
    """
    # Filter for T gates only
    t_gates = []
    for i, gate in enumerate(circuit.gates):
        if gate.gate_type == GateType.T:
            t_gates.append((i, gate.targets[0]))
    
    if not t_gates:
        return 0
    
    # Track when each qubit's T gate can be executed
    qubit_t_time = {}
    depth = 0
    
    for _, qubit in t_gates:
        # T gates on different qubits can be parallel
        if qubit not in qubit_t_time:
            qubit_t_time[qubit] = 0
        
        qubit_t_time[qubit] += 1
        depth = max(depth, qubit_t_time[qubit])
    
    return depth


def compute_cnot_count(circuit: UnifiedCircuit) -> int:
    """
    Count CNOT gates in circuit.
    
    Args:
        circuit: Quantum circuit
    
    Returns:
        Number of CNOT gates
    """
    count = 0
    for gate in circuit.gates:
        if gate.gate_type == GateType.CNOT:
            count += 1
    return count
