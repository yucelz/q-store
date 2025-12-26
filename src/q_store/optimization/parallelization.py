"""
Circuit parallelization analysis and optimization.

Identifies opportunities for parallel gate execution and optimizes
circuit depth by maximizing parallelism.
"""

from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from q_store.core import UnifiedCircuit, Gate, GateType


@dataclass
class ParallelLayer:
    """Represents a layer of gates that can execute in parallel."""
    layer_id: int
    gates: List[Tuple[int, Gate]]  # (original_index, gate)
    qubits_used: Set[int]
    depth_position: int


class ParallelizationAnalyzer:
    """
    Analyzes circuit parallelization opportunities.

    Identifies which gates can execute simultaneously and computes
    optimal circuit depth.
    """

    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize parallelization analyzer.

        Args:
            circuit: Circuit to analyze
        """
        self.circuit = circuit

    def analyze(self) -> List[ParallelLayer]:
        """
        Decompose circuit into parallel layers.

        Returns:
            List of parallel layers
        """
        return find_parallel_layers(self.circuit)

    def compute_depth(self) -> int:
        """
        Compute circuit depth (number of parallel layers).

        Returns:
            Circuit depth
        """
        layers = self.analyze()
        return len(layers)

    def get_parallelism_stats(self) -> Dict[str, float]:
        """
        Compute parallelization statistics.

        Returns:
            Statistics dictionary
        """
        layers = self.analyze()
        total_gates = len(self.circuit.gates)
        depth = len(layers)

        gates_per_layer = [len(layer.gates) for layer in layers]
        avg_parallelism = sum(gates_per_layer) / depth if depth > 0 else 0
        max_parallelism = max(gates_per_layer) if gates_per_layer else 0

        # Parallelization efficiency: actual depth vs sequential depth
        efficiency = total_gates / depth if depth > 0 else 0

        return {
            'total_gates': total_gates,
            'depth': depth,
            'avg_parallelism': avg_parallelism,
            'max_parallelism': max_parallelism,
            'efficiency': efficiency,
        }

    def visualize_layers(self) -> str:
        """
        Create text visualization of parallel layers.

        Returns:
            Text representation of layers
        """
        layers = self.analyze()
        lines = []
        lines.append(f"Circuit Depth: {len(layers)} layers")
        lines.append(f"Total Gates: {len(self.circuit.gates)}")
        lines.append("")

        for layer in layers:
            lines.append(f"Layer {layer.layer_id}:")
            for idx, gate in layer.gates:
                qubits_str = ",".join(map(str, gate.targets))
                lines.append(f"  Gate {idx}: {gate.gate_type} on qubits [{qubits_str}]")
            lines.append("")

        return "\n".join(lines)


def find_parallel_layers(circuit: UnifiedCircuit) -> List[ParallelLayer]:
    """
    Decompose circuit into parallel execution layers.

    Gates can execute in parallel if they act on disjoint qubit sets.

    Args:
        circuit: Circuit to analyze

    Returns:
        List of parallel layers
    """
    layers = []
    gates = circuit.gates
    remaining_gates = list(enumerate(gates))
    layer_id = 0

    while remaining_gates:
        # Build next parallel layer
        layer_gates = []
        used_qubits = set()
        gates_to_remove = []

        for idx, (orig_idx, gate) in enumerate(remaining_gates):
            # Get all qubits used by this gate
            gate_qubits = set(gate.targets)
            if gate.controls:
                gate_qubits.update(gate.controls)

            # Can add to layer if no qubit conflicts
            if gate_qubits.isdisjoint(used_qubits):
                layer_gates.append((orig_idx, gate))
                used_qubits.update(gate_qubits)
                gates_to_remove.append(idx)

        # Remove gates added to layer
        remaining_gates = [
            g for i, g in enumerate(remaining_gates)
            if i not in gates_to_remove
        ]

        # Add layer
        layers.append(ParallelLayer(
            layer_id=layer_id,
            gates=layer_gates,
            qubits_used=used_qubits,
            depth_position=layer_id
        ))
        layer_id += 1

    return layers


def compute_circuit_depth(circuit: UnifiedCircuit) -> int:
    """
    Compute minimum circuit depth with parallel execution.

    Args:
        circuit: Circuit to analyze

    Returns:
        Circuit depth (number of parallel layers)
    """
    layers = find_parallel_layers(circuit)
    return len(layers)


def optimize_for_parallelism(circuit: UnifiedCircuit) -> UnifiedCircuit:
    """
    Reorder gates to maximize parallelism.

    Args:
        circuit: Circuit to optimize

    Returns:
        Optimized circuit with improved parallelism
    """
    # Build dependency graph
    gates = circuit.gates
    dependencies = defaultdict(set)

    for i in range(len(gates)):
        gate_i = gates[i]
        qubits_i = set(gate_i.targets)
        if gate_i.controls:
            qubits_i.update(gate_i.controls)

        for j in range(i):
            gate_j = gates[j]
            qubits_j = set(gate_j.targets)
            if gate_j.controls:
                qubits_j.update(gate_j.controls)

            # Gate i depends on gate j if they share qubits
            if not qubits_i.isdisjoint(qubits_j):
                dependencies[i].add(j)

    # Topological sort with level-based scheduling
    # (This is a simplified scheduling algorithm)
    scheduled = []
    available = set(i for i in range(len(gates)) if not dependencies[i])

    while available:
        # Schedule all available gates in this "round"
        current_batch = sorted(available)
        scheduled.extend(current_batch)

        # Update available gates
        new_available = set()
        for i in range(len(gates)):
            if i in scheduled or i in available:
                continue
            # Check if all dependencies are satisfied
            if dependencies[i].issubset(set(scheduled)):
                new_available.add(i)

        available = new_available

    # Build optimized circuit
    new_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
    for idx in scheduled:
        gate = gates[idx]
        new_circuit.add_gate(
            gate.gate_type,
            targets=gate.targets,
            parameters=gate.parameters
        )

    return new_circuit


def analyze_critical_path(circuit: UnifiedCircuit) -> Dict[str, any]:
    """
    Analyze critical path through circuit.

    The critical path is the longest dependency chain that determines
    minimum circuit depth.

    Args:
        circuit: Circuit to analyze

    Returns:
        Critical path analysis
    """
    gates = circuit.gates

    # Build per-qubit gate chains
    qubit_chains = defaultdict(list)
    for idx, gate in enumerate(gates):
        for qubit in gate.targets:
            qubit_chains[qubit].append(idx)
        if gate.controls:
            for qubit in gate.controls:
                qubit_chains[qubit].append(idx)

    # Find longest chain
    longest_chain = []
    longest_qubit = -1

    for qubit, chain in qubit_chains.items():
        if len(chain) > len(longest_chain):
            longest_chain = chain
            longest_qubit = qubit

    return {
        'critical_qubit': longest_qubit,
        'critical_path_length': len(longest_chain),
        'critical_path_gates': longest_chain,
        'qubit_depths': {q: len(chain) for q, chain in qubit_chains.items()},
    }


def get_parallelism_profile(circuit: UnifiedCircuit) -> List[int]:
    """
    Get parallelism profile (gates per layer).

    Args:
        circuit: Circuit to analyze

    Returns:
        List where element i is number of gates in layer i
    """
    layers = find_parallel_layers(circuit)
    return [len(layer.gates) for layer in layers]


def estimate_execution_time(
    circuit: UnifiedCircuit,
    gate_times: Dict[GateType, float] = None
) -> float:
    """
    Estimate circuit execution time with parallel execution.

    Args:
        circuit: Circuit to analyze
        gate_times: Dictionary mapping gate types to execution times

    Returns:
        Estimated execution time
    """
    if gate_times is None:
        # Default gate times (arbitrary units)
        gate_times = {
            GateType.H: 1.0,
            GateType.X: 1.0,
            GateType.Y: 1.0,
            GateType.Z: 1.0,
            GateType.RX: 1.5,
            GateType.RY: 1.5,
            GateType.RZ: 1.5,
            GateType.CNOT: 2.0,
            GateType.CZ: 2.0,
            GateType.SWAP: 3.0,
            GateType.CCX: 4.0,
        }

    layers = find_parallel_layers(circuit)
    total_time = 0.0

    for layer in layers:
        # Time for layer is maximum gate time in that layer
        layer_time = max(
            gate_times.get(gate.gate_type, 1.0)
            for _, gate in layer.gates
        )
        total_time += layer_time

    return total_time
