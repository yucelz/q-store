"""
Advanced Circuit Compiler with Routing and SWAP Insertion.

Compiles circuits for hardware constraints including:
- Qubit mapping to device topology
- SWAP insertion for limited connectivity
- Gate decomposition to native sets
- Optimization passes
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from ..core import UnifiedCircuit, GateType, Gate
from .topology import DeviceTopology, create_topology
from .gate_decomposition import GateDecomposer

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result from circuit compilation."""
    compiled_circuit: UnifiedCircuit
    initial_mapping: Dict[int, int]  # logical -> physical
    final_mapping: Dict[int, int]
    swap_count: int
    original_depth: int
    compiled_depth: int
    gate_count_original: int
    gate_count_compiled: int
    metadata: Dict


class CircuitCompiler:
    """
    Advanced circuit compiler for hardware-constrained devices.

    Performs:
    1. Initial qubit placement
    2. Routing with SWAP insertion
    3. Gate decomposition
    4. Optimization

    Args:
        topology: Device topology
        native_gates: Native gate set
        optimization_level: 0-3 (0=none, 3=aggressive)

    Example:
        >>> topology = create_topology('grid', rows=3, cols=3)
        >>> compiler = CircuitCompiler(topology)
        >>> result = compiler.compile(circuit)
        >>> print(f"Added {result.swap_count} SWAPs")
    """

    def __init__(
        self,
        topology: DeviceTopology,
        native_gates: Optional[set] = None,
        optimization_level: int = 2
    ):
        self.topology = topology
        self.decomposer = GateDecomposer(native_gates)
        self.optimization_level = optimization_level

    def compile(self, circuit: UnifiedCircuit) -> CompilationResult:
        """
        Compile circuit for device topology.

        Args:
            circuit: Input logical circuit

        Returns:
            CompilationResult with compiled circuit and statistics
        """
        if circuit.n_qubits > self.topology.n_qubits:
            raise ValueError(
                f"Circuit has {circuit.n_qubits} qubits but device only has {self.topology.n_qubits}"
            )

        logger.info(f"Compiling circuit with {len(circuit.gates)} gates")

        # Step 1: Initial qubit placement
        initial_mapping = self._initial_placement(circuit)
        logger.debug(f"Initial placement: {initial_mapping}")

        # Step 2: Route circuit with SWAP insertion
        routed_circuit, final_mapping, swap_count = self._route_circuit(
            circuit, initial_mapping
        )
        logger.info(f"Routing added {swap_count} SWAP gates")

        # Step 3: Decompose to native gates
        decomposed_circuit = self.decomposer.decompose(routed_circuit)

        # Step 4: Optimization
        if self.optimization_level > 0:
            from ..core import CircuitOptimizer
            optimizer = CircuitOptimizer()
            optimized_circuit = optimizer.optimize(decomposed_circuit)
        else:
            optimized_circuit = decomposed_circuit

        # Calculate depths
        original_depth = self._calculate_depth(circuit)
        compiled_depth = self._calculate_depth(optimized_circuit)

        return CompilationResult(
            compiled_circuit=optimized_circuit,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            swap_count=swap_count,
            original_depth=original_depth,
            compiled_depth=compiled_depth,
            gate_count_original=len(circuit.gates),
            gate_count_compiled=len(optimized_circuit.gates),
            metadata={
                'topology': self.topology.name,
                'optimization_level': self.optimization_level,
            }
        )

    def _initial_placement(self, circuit: UnifiedCircuit) -> Dict[int, int]:
        """
        Determine initial logical to physical qubit mapping.

        Strategy: Place frequently interacting qubits close together.
        """
        # Count two-qubit gate interactions
        interactions = {}
        for gate in circuit.gates:
            if len(gate.targets) == 2:
                q1, q2 = sorted(gate.targets)
                key = (q1, q2)
                interactions[key] = interactions.get(key, 0) + 1

        # Sort by interaction frequency
        sorted_pairs = sorted(interactions.items(), key=lambda x: x[1], reverse=True)

        # Greedy placement: start with most interacting pair
        mapping = {}
        used_physical = set()

        if sorted_pairs:
            # Place first pair on connected physical qubits
            (log_q1, log_q2), _ = sorted_pairs[0]
            phys_q1 = 0
            phys_q2 = self._find_connected_qubit(phys_q1, used_physical)

            mapping[log_q1] = phys_q1
            mapping[log_q2] = phys_q2
            used_physical.add(phys_q1)
            used_physical.add(phys_q2)

        # Place remaining qubits
        for log_qubit in range(circuit.n_qubits):
            if log_qubit not in mapping:
                # Find unused physical qubit
                for phys_qubit in range(self.topology.n_qubits):
                    if phys_qubit not in used_physical:
                        mapping[log_qubit] = phys_qubit
                        used_physical.add(phys_qubit)
                        break

        return mapping

    def _find_connected_qubit(self, qubit: int, used: set) -> int:
        """Find unused qubit connected to given qubit."""
        neighbors = self.topology.get_neighbors(qubit)
        for neighbor in neighbors:
            if neighbor not in used:
                return neighbor

        # No unused neighbor, return any unused qubit
        for q in range(self.topology.n_qubits):
            if q not in used:
                return q

        return qubit + 1  # Fallback

    def _route_circuit(
        self,
        circuit: UnifiedCircuit,
        initial_mapping: Dict[int, int]
    ) -> Tuple[UnifiedCircuit, Dict[int, int], int]:
        """
        Route circuit with SWAP insertion.

        Returns:
            (routed_circuit, final_mapping, swap_count)
        """
        routed_circuit = UnifiedCircuit(n_qubits=self.topology.n_qubits)
        current_mapping = initial_mapping.copy()
        reverse_mapping = {v: k for k, v in current_mapping.items()}
        swap_count = 0

        for gate in circuit.gates:
            if len(gate.targets) == 1:
                # Single-qubit gate - just apply to physical qubit
                logical_q = gate.targets[0]
                physical_q = current_mapping[logical_q]

                new_gate = Gate(
                    gate.gate_type,
                    targets=[physical_q],
                    parameters=gate.parameters,
                    name=gate.name
                )
                routed_circuit.gates.append(new_gate)

            elif len(gate.targets) == 2:
                # Two-qubit gate - check connectivity
                log_q1, log_q2 = gate.targets
                phys_q1 = current_mapping[log_q1]
                phys_q2 = current_mapping[log_q2]

                if self.topology.is_connected(phys_q1, phys_q2):
                    # Qubits are connected - can apply gate directly
                    new_gate = Gate(
                        gate.gate_type,
                        targets=[phys_q1, phys_q2],
                        parameters=gate.parameters,
                        name=gate.name
                    )
                    routed_circuit.gates.append(new_gate)
                else:
                    # Need to insert SWAPs
                    path = self.topology.shortest_path(phys_q1, phys_q2)

                    # Insert SWAPs to move qubits together
                    for i in range(len(path) - 2):
                        swap_q1 = path[i]
                        swap_q2 = path[i + 1]

                        # Add SWAP gate
                        swap_gate = Gate(GateType.SWAP, targets=[swap_q1, swap_q2])
                        routed_circuit.gates.append(swap_gate)
                        swap_count += 1

                        # Update mapping
                        log_at_q1 = reverse_mapping.get(swap_q1)
                        log_at_q2 = reverse_mapping.get(swap_q2)

                        if log_at_q1 is not None and log_at_q2 is not None:
                            current_mapping[log_at_q1] = swap_q2
                            current_mapping[log_at_q2] = swap_q1
                            reverse_mapping[swap_q1] = log_at_q2
                            reverse_mapping[swap_q2] = log_at_q1

                    # Now qubits should be adjacent
                    phys_q1 = current_mapping[log_q1]
                    phys_q2 = current_mapping[log_q2]

                    new_gate = Gate(
                        gate.gate_type,
                        targets=[phys_q1, phys_q2],
                        parameters=gate.parameters,
                        name=gate.name
                    )
                    routed_circuit.gates.append(new_gate)

            else:
                # Multi-qubit gate (e.g., Toffoli)
                # For now, just map to physical qubits
                physical_targets = [current_mapping[q] for q in gate.targets]
                new_gate = Gate(
                    gate.gate_type,
                    targets=physical_targets,
                    parameters=gate.parameters,
                    name=gate.name
                )
                routed_circuit.gates.append(new_gate)

        return routed_circuit, current_mapping, swap_count

    def _calculate_depth(self, circuit: UnifiedCircuit) -> int:
        """
        Calculate circuit depth (critical path length).

        Simplified: assumes all gates have unit depth.
        """
        if not circuit.gates:
            return 0

        # Track when each qubit is last used
        qubit_times = [0] * circuit.n_qubits

        for gate in circuit.gates:
            # Gate starts after all involved qubits are free
            start_time = max(qubit_times[q] for q in gate.targets)
            end_time = start_time + 1

            # Update qubit times
            for q in gate.targets:
                qubit_times[q] = end_time

        return max(qubit_times)


def compile_circuit(
    circuit: UnifiedCircuit,
    topology: Optional[DeviceTopology] = None,
    topology_type: str = 'all_to_all',
    native_gates: Optional[set] = None,
    optimization_level: int = 2
) -> CompilationResult:
    """
    Compile circuit for hardware constraints.

    Args:
        circuit: Input circuit
        topology: Device topology (or create from topology_type)
        topology_type: Type of topology to create if topology is None
        native_gates: Native gate set
        optimization_level: 0-3

    Returns:
        CompilationResult

    Example:
        >>> # Compile for linear topology
        >>> result = compile_circuit(
        ...     circuit,
        ...     topology_type='linear',
        ...     native_gates={GateType.RZ, GateType.SX, GateType.CNOT}
        ... )
        >>> print(f"Overhead: {result.gate_count_compiled / result.gate_count_original:.2f}x")
    """
    if topology is None:
        topology = create_topology(topology_type, n_qubits=circuit.n_qubits)

    compiler = CircuitCompiler(topology, native_gates, optimization_level)
    return compiler.compile(circuit)
