"""
Gate commutation analysis for circuit optimization.

Analyzes which gates commute and can be reordered to improve
circuit structure and enable further optimizations.
"""

from typing import List, Set, Tuple, Dict
from dataclasses import dataclass
from q_store.core import UnifiedCircuit, Gate, GateType


@dataclass
class CommutationRelation:
    """Represents commutation relationship between gates."""
    gate1_idx: int
    gate2_idx: int
    commute: bool
    reason: str = ""


class CommutationAnalyzer:
    """
    Analyzes gate commutation relationships in quantum circuits.

    Gates commute if they can be applied in any order without
    affecting the final result.
    """

    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize commutation analyzer.

        Args:
            circuit: Circuit to analyze
        """
        self.circuit = circuit
        self._commutation_cache: Dict[Tuple[int, int], bool] = {}

    def analyze(self) -> List[CommutationRelation]:
        """
        Analyze all pairwise gate commutations.

        Returns:
            List of commutation relationships
        """
        relations = []
        gates = self.circuit.gates

        for i in range(len(gates)):
            for j in range(i + 1, len(gates)):
                commutes = self.check_commutation(i, j)
                reason = self._get_commutation_reason(gates[i], gates[j], commutes)
                relations.append(CommutationRelation(i, j, commutes, reason))

        return relations

    def check_commutation(self, idx1: int, idx2: int) -> bool:
        """
        Check if two gates commute.

        Args:
            idx1: Index of first gate
            idx2: Index of second gate

        Returns:
            True if gates commute
        """
        key = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
        if key in self._commutation_cache:
            return self._commutation_cache[key]

        gate1 = self.circuit.gates[idx1]
        gate2 = self.circuit.gates[idx2]

        result = can_commute(gate1, gate2)
        self._commutation_cache[key] = result
        return result

    def find_commuting_window(self, gate_idx: int, direction: str = "forward") -> List[int]:
        """
        Find window of gates that commute with given gate.

        Args:
            gate_idx: Index of gate to analyze
            direction: "forward" or "backward"

        Returns:
            Indices of commuting gates
        """
        commuting = []
        gates = self.circuit.gates

        if direction == "forward":
            for i in range(gate_idx + 1, len(gates)):
                if self.check_commutation(gate_idx, i):
                    commuting.append(i)
                else:
                    break
        else:  # backward
            for i in range(gate_idx - 1, -1, -1):
                if self.check_commutation(i, gate_idx):
                    commuting.append(i)
                else:
                    break

        return commuting

    def _get_commutation_reason(self, gate1: Gate, gate2: Gate, commutes: bool) -> str:
        """Get human-readable reason for commutation result."""
        if commutes:
            qubits1 = set(gate1.targets)
            qubits2 = set(gate2.targets)
            if qubits1.isdisjoint(qubits2):
                return "Disjoint qubit sets"
            elif gate1.gate_type == gate2.gate_type:
                return "Same gate type on same qubits"
            else:
                return "Gates commute algebraically"
        else:
            return "Gates do not commute"


def can_commute(gate1: Gate, gate2: Gate) -> bool:
    """
    Check if two gates commute.

    Gates commute if:
    1. They act on disjoint qubits
    2. They are both diagonal in the same basis
    3. They satisfy specific algebraic commutation relations

    Args:
        gate1: First gate
        gate2: Second gate

    Returns:
        True if gates commute
    """
    # Get all qubits involved (targets + controls)
    qubits1 = set(gate1.targets)
    if gate1.controls:
        qubits1.update(gate1.controls)

    qubits2 = set(gate2.targets)
    if gate2.controls:
        qubits2.update(gate2.controls)

    # Disjoint qubits always commute
    if qubits1.isdisjoint(qubits2):
        return True

    # Same single-qubit gate on same qubit commutes
    if gate1.targets == gate2.targets and gate1.gate_type == gate2.gate_type:
        if len(gate1.targets) == 1 and not gate1.controls and not gate2.controls:
            return True

    # Pauli gates commute with themselves
    pauli_gates = {GateType.X, GateType.Y, GateType.Z}
    if gate1.gate_type in pauli_gates and gate1.gate_type == gate2.gate_type:
        if gate1.targets == gate2.targets:
            return True

    # Diagonal gates commute with each other on same qubits
    diagonal_gates = {GateType.Z, GateType.S, GateType.T, GateType.RZ, GateType.CZ}
    if gate1.gate_type in diagonal_gates and gate2.gate_type in diagonal_gates:
        if set(gate1.targets) == set(gate2.targets):
            return True

    # CNOT commutes with Z on control or target
    if gate1.gate_type == GateType.CNOT and gate2.gate_type == GateType.Z:
        # Z on control commutes with CNOT
        if gate2.targets[0] == gate1.targets[0]:
            return True
        # Z on target commutes with CNOT
        if gate2.targets[0] == gate1.targets[1]:
            return True

    if gate2.gate_type == GateType.CNOT and gate1.gate_type == GateType.Z:
        if gate1.targets[0] == gate2.targets[0]:
            return True
        if gate1.targets[0] == gate2.targets[1]:
            return True

    # CNOT with CNOT on disjoint controls/targets
    if gate1.gate_type == GateType.CNOT and gate2.gate_type == GateType.CNOT:
        # Same CNOT commutes
        if gate1.targets == gate2.targets:
            return True

    return False


def commute_gates(circuit: UnifiedCircuit, idx1: int, idx2: int) -> UnifiedCircuit:
    """
    Swap order of two commuting gates.

    Args:
        circuit: Circuit containing gates
        idx1: Index of first gate
        idx2: Index of second gate

    Returns:
        New circuit with gates swapped

    Raises:
        ValueError: If gates don't commute
    """
    if not can_commute(circuit.gates[idx1], circuit.gates[idx2]):
        raise ValueError(f"Gates at indices {idx1} and {idx2} do not commute")

    # Create new circuit with swapped gates
    new_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
    gates = list(circuit.gates)
    gates[idx1], gates[idx2] = gates[idx2], gates[idx1]

    for gate in gates:
        new_circuit.add_gate(
            gate.gate_type,
            targets=gate.targets,
            parameters=gate.parameters
        )

    return new_circuit


def reorder_commuting_gates(
    circuit: UnifiedCircuit,
    optimize_for: str = "depth"
) -> UnifiedCircuit:
    """
    Reorder commuting gates to optimize circuit.

    Args:
        circuit: Circuit to optimize
        optimize_for: Optimization target ("depth", "gate_count", "locality")

    Returns:
        Optimized circuit
    """
    analyzer = CommutationAnalyzer(circuit)
    new_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)

    gates = list(circuit.gates)
    processed = [False] * len(gates)

    if optimize_for == "depth":
        # Try to parallelize gates by grouping commuting operations
        for i in range(len(gates)):
            if processed[i]:
                continue

            # Find all gates that commute with this one
            commuting_group = [i]
            for j in range(i + 1, len(gates)):
                if processed[j]:
                    continue
                # Check if gate j commutes with all in group
                if all(can_commute(gates[j], gates[k]) for k in commuting_group):
                    commuting_group.append(j)

            # Add commuting gates
            for idx in commuting_group:
                gate = gates[idx]
                new_circuit.add_gate(
                    gate.gate_type,
                    targets=gate.targets,
                    parameters=gate.parameters
                )
                processed[idx] = True

    elif optimize_for == "locality":
        # Group gates by qubit to improve locality
        qubit_last_use = {}

        for i in range(len(gates)):
            if processed[i]:
                continue

            gate = gates[i]
            qubits = set(gate.targets)

            # Find next gate on same qubits
            new_circuit.add_gate(
                gate.gate_type,
                targets=gate.targets,
                parameters=gate.parameters
            )
            processed[i] = True

    else:  # gate_count - just copy as-is
        for gate in gates:
            new_circuit.add_gate(
                gate.gate_type,
                targets=gate.targets,
                parameters=gate.parameters
            )

    return new_circuit
