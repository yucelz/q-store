"""
Circuit Optimizer for Q-Store v4.0

Provides circuit optimization strategies for different target backends:
- Gate cancellation and simplification
- Circuit depth reduction
- Native gate compilation (especially for IonQ)
- Parameter consolidation

The optimizer can automatically select the best strategy based on
the target backend, or manual strategies can be specified.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

from .unified_circuit import UnifiedCircuit, Gate, GateType


@dataclass
class OptimizationMetrics:
    """Metrics for circuit optimization"""
    original_gate_count: int
    optimized_gate_count: int
    original_depth: int
    optimized_depth: int
    gates_removed: int
    gates_added: int

    @property
    def gate_reduction_percent(self) -> float:
        """Calculate percentage of gates removed"""
        if self.original_gate_count == 0:
            return 0.0
        return 100 * (self.original_gate_count - self.optimized_gate_count) / self.original_gate_count

    @property
    def depth_reduction_percent(self) -> float:
        """Calculate percentage of depth reduced"""
        if self.original_depth == 0:
            return 0.0
        return 100 * (self.original_depth - self.optimized_depth) / self.original_depth

    def __repr__(self) -> str:
        return (
            f"OptimizationMetrics(\n"
            f"  Gates: {self.original_gate_count} → {self.optimized_gate_count} "
            f"({self.gate_reduction_percent:.1f}% reduction)\n"
            f"  Depth: {self.original_depth} → {self.optimized_depth} "
            f"({self.depth_reduction_percent:.1f}% reduction)\n"
            f"  Changes: -{self.gates_removed} gates, +{self.gates_added} gates\n"
            f")"
        )


class CircuitOptimizer:
    """
    Optimize quantum circuits for specific target backends

    Optimization strategies:
    - 'none': No optimization
    - 'basic': Gate cancellation and simple reductions
    - 'aggressive': Deep optimization with gate rewriting
    - 'ionq_native': Optimize for IonQ native gates
    - 'depth': Minimize circuit depth
    - 'gate_count': Minimize number of gates
    - 'auto': Automatically select based on backend

    Example:
        >>> optimizer = CircuitOptimizer(strategy='basic')
        >>> optimized_circuit = optimizer.optimize(circuit)
        >>> print(optimizer.get_metrics())
    """

    def __init__(self, strategy: str = 'auto', target_backend: Optional[str] = None):
        """
        Initialize optimizer

        Args:
            strategy: Optimization strategy to use
            target_backend: Target backend name (for 'auto' strategy)
        """
        self.strategy = strategy
        self.target_backend = target_backend
        self.metrics: Optional[OptimizationMetrics] = None

    def optimize(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Optimize a circuit

        Args:
            circuit: Circuit to optimize

        Returns:
            Optimized circuit
        """
        original_gates = len(circuit.gates)
        original_depth = circuit.depth

        # Select strategy
        if self.strategy == 'auto':
            strategy = self._auto_select_strategy()
        else:
            strategy = self.strategy

        # Apply optimization
        if strategy == 'none':
            optimized = circuit.copy()
        elif strategy == 'basic':
            optimized = self._basic_optimization(circuit)
        elif strategy == 'aggressive':
            optimized = self._aggressive_optimization(circuit)
        elif strategy == 'ionq_native':
            optimized = self._ionq_native_optimization(circuit)
        elif strategy == 'depth':
            optimized = self._depth_optimization(circuit)
        elif strategy == 'gate_count':
            optimized = self._gate_count_optimization(circuit)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")

        # Calculate metrics
        gates_removed = original_gates - len(optimized.gates)
        gates_added = max(0, len(optimized.gates) - original_gates)

        self.metrics = OptimizationMetrics(
            original_gate_count=original_gates,
            optimized_gate_count=len(optimized.gates),
            original_depth=original_depth,
            optimized_depth=optimized.depth,
            gates_removed=max(0, gates_removed),
            gates_added=gates_added
        )

        return optimized

    def get_metrics(self) -> Optional[OptimizationMetrics]:
        """Get optimization metrics from last optimization"""
        return self.metrics

    def _auto_select_strategy(self) -> str:
        """Automatically select optimization strategy based on backend"""
        if self.target_backend is None:
            return 'basic'

        backend = self.target_backend.lower()

        if 'ionq' in backend:
            return 'ionq_native'
        elif 'simulator' in backend or 'local' in backend:
            return 'aggressive'
        else:
            return 'basic'

    def _basic_optimization(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Basic optimization: gate cancellation and simple reductions

        Optimizations:
        - Cancel adjacent inverse gates (H-H, X-X, etc.)
        - Merge rotation gates (RX-RX, RY-RY, RZ-RZ)
        - Remove identity operations
        """
        optimized = UnifiedCircuit(n_qubits=circuit.n_qubits)
        optimized.parameters = circuit.parameters.copy()

        gates = list(circuit.gates)
        i = 0

        while i < len(gates):
            current_gate = gates[i]

            # Check if next gate cancels this one
            if i + 1 < len(gates):
                next_gate = gates[i + 1]

                # Check for cancellation
                if self._gates_cancel(current_gate, next_gate):
                    i += 2  # Skip both gates
                    continue

                # Check for rotation merging
                merged = self._try_merge_rotations(current_gate, next_gate)
                if merged is not None:
                    optimized.gates.append(merged)
                    i += 2
                    continue

            # No optimization, add gate
            optimized.gates.append(current_gate)
            i += 1

        return optimized

    def _gates_cancel(self, gate1: Gate, gate2: Gate) -> bool:
        """Check if two gates cancel each other"""
        # Must operate on same qubits
        if gate1.targets != gate2.targets:
            return False

        # Self-inverse gates
        self_inverse = {GateType.H, GateType.X, GateType.Y, GateType.Z,
                       GateType.CNOT, GateType.SWAP}

        if gate1.gate_type == gate2.gate_type and gate1.gate_type in self_inverse:
            return True

        return False

    def _try_merge_rotations(self, gate1: Gate, gate2: Gate) -> Optional[Gate]:
        """Try to merge two rotation gates"""
        # Must be same gate type and targets
        if gate1.gate_type != gate2.gate_type or gate1.targets != gate2.targets:
            return None

        # Must be rotation gates
        rotation_gates = {GateType.RX, GateType.RY, GateType.RZ}
        if gate1.gate_type not in rotation_gates:
            return None

        # Both must have numeric parameters (can't merge symbolic)
        if gate1.parameters is None or gate2.parameters is None:
            return None

        angle1 = gate1.parameters.get('angle')
        angle2 = gate2.parameters.get('angle')

        if isinstance(angle1, (int, float)) and isinstance(angle2, (int, float)):
            # Merge angles
            merged_angle = angle1 + angle2

            # Create merged gate
            return Gate(
                gate_type=gate1.gate_type,
                targets=gate1.targets,
                parameters={'angle': merged_angle}
            )

        return None

    def _aggressive_optimization(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Aggressive optimization with gate rewriting

        Additional optimizations beyond basic:
        - Gate commutation and reordering
        - Multi-gate patterns (e.g., H-CNOT-H = CZ)
        - Constant propagation
        """
        # Start with basic optimization
        optimized = self._basic_optimization(circuit)

        # Apply pattern-based rewriting
        optimized = self._apply_gate_patterns(optimized)

        # Try to commute gates to enable more cancellations
        optimized = self._commute_gates(optimized)

        # One more pass of basic optimization
        optimizer = CircuitOptimizer(strategy='basic')
        optimized = optimizer.optimize(optimized)

        return optimized

    def _apply_gate_patterns(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply known gate equivalence patterns"""
        optimized = UnifiedCircuit(n_qubits=circuit.n_qubits)
        optimized.parameters = circuit.parameters.copy()

        gates = list(circuit.gates)
        i = 0

        while i < len(gates):
            # Check for H-CNOT-H pattern (equivalent to CZ)
            if i + 2 < len(gates):
                if (gates[i].gate_type == GateType.H and
                    gates[i+1].gate_type == GateType.CNOT and
                    gates[i+2].gate_type == GateType.H):

                    # Check if H gates are on CNOT target
                    target = gates[i+1].targets[1]
                    if gates[i].targets == [target] and gates[i+2].targets == [target]:
                        # Replace with CZ
                        optimized.add_gate(GateType.CZ, targets=gates[i+1].targets)
                        i += 3
                        continue

            # No pattern matched, add gate
            optimized.gates.append(gates[i])
            i += 1

        return optimized

    def _commute_gates(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Try to commute gates to enable more optimization"""
        # Simplified commutation - only commute gates on different qubits
        optimized = UnifiedCircuit(n_qubits=circuit.n_qubits)
        optimized.parameters = circuit.parameters.copy()

        gates = list(circuit.gates)

        # Build dependency graph
        changed = True
        while changed:
            changed = False
            for i in range(len(gates) - 1):
                if self._can_commute(gates[i], gates[i+1]):
                    # Swap gates
                    gates[i], gates[i+1] = gates[i+1], gates[i]
                    changed = True

        optimized.gates = gates
        return optimized

    def _can_commute(self, gate1: Gate, gate2: Gate) -> bool:
        """Check if two gates can be commuted"""
        # Gates on completely different qubits can commute
        qubits1 = set(gate1.targets)
        if gate1.controls:
            qubits1.update(gate1.controls)

        qubits2 = set(gate2.targets)
        if gate2.controls:
            qubits2.update(gate2.controls)

        # No overlap = can commute
        if not qubits1.intersection(qubits2):
            return True

        # Same qubits: check if they're both Z-basis (Z, RZ, S, T)
        z_gates = {GateType.Z, GateType.RZ, GateType.S, GateType.T}
        if (qubits1 == qubits2 and
            gate1.gate_type in z_gates and
            gate2.gate_type in z_gates):
            return True

        return False

    def _ionq_native_optimization(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Optimize for IonQ native gates (GPI, GPI2, MS)

        This converts to native gates and applies IonQ-specific optimizations
        """
        from .circuit_converters import IonQNativeConverter

        # First do basic optimization
        optimized = self._basic_optimization(circuit)

        # Convert to native gates
        ionq_json = IonQNativeConverter.to_ionq_native(optimized, optimize=True)

        # Convert back to UnifiedCircuit (simplified - in practice would parse JSON)
        # For now, return the optimized circuit
        return optimized

    def _depth_optimization(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Optimize to minimize circuit depth"""
        # Start with aggressive optimization
        optimized = self._aggressive_optimization(circuit)

        # Try to parallelize gates
        optimized = self._parallelize_gates(optimized)

        return optimized

    def _parallelize_gates(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Reorder gates to maximize parallelization"""
        # Group gates by the qubits they act on
        # Then reorder to minimize depth

        optimized = UnifiedCircuit(n_qubits=circuit.n_qubits)
        optimized.parameters = circuit.parameters.copy()

        # For now, just return the circuit (full implementation is complex)
        # TODO: Implement proper gate scheduling algorithm
        optimized.gates = circuit.gates.copy()

        return optimized

    def _gate_count_optimization(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Optimize to minimize number of gates"""
        # This is similar to aggressive but prioritizes gate count over depth
        return self._aggressive_optimization(circuit)


def optimize(circuit: UnifiedCircuit,
            strategy: str = 'auto',
            target_backend: Optional[str] = None) -> Tuple[UnifiedCircuit, OptimizationMetrics]:
    """
    Convenience function to optimize a circuit

    Args:
        circuit: Circuit to optimize
        strategy: Optimization strategy
        target_backend: Target backend (for auto strategy)

    Returns:
        Tuple of (optimized_circuit, metrics)

    Example:
        >>> optimized, metrics = optimize(circuit, strategy='basic')
        >>> print(f"Reduced gates by {metrics.gate_reduction_percent:.1f}%")
    """
    optimizer = CircuitOptimizer(strategy=strategy, target_backend=target_backend)
    optimized_circuit = optimizer.optimize(circuit)
    metrics = optimizer.get_metrics()

    return optimized_circuit, metrics


# Add optimization method to UnifiedCircuit
def _add_optimize_method():
    """Add optimize method to UnifiedCircuit class"""

    def optimize_circuit(self, strategy: str = 'auto', target_backend: Optional[str] = None) -> 'UnifiedCircuit':
        """
        Optimize this circuit

        Args:
            strategy: Optimization strategy to use
            target_backend: Target backend (for auto strategy)

        Returns:
            Optimized circuit
        """
        optimizer = CircuitOptimizer(strategy=strategy, target_backend=target_backend)
        return optimizer.optimize(self)

    UnifiedCircuit.optimize = optimize_circuit


# Auto-register optimization method
_add_optimize_method()
