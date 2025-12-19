"""
Gate fusion for circuit optimization.

Combines multiple gates into single operations to reduce circuit depth
and improve execution efficiency.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
from q_store.core import UnifiedCircuit, Gate, GateType


@dataclass
class FusionOpportunity:
    """Represents an opportunity to fuse gates."""
    start_idx: int
    end_idx: int
    gate_types: List[GateType]
    qubits: List[int]
    description: str
    savings: int  # Number of gates saved


class GateFuser:
    """
    Identifies and applies gate fusion optimizations.

    Gate fusion combines sequences of gates into equivalent
    but more efficient representations.
    """

    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize gate fuser.

        Args:
            circuit: Circuit to optimize
        """
        self.circuit = circuit

    def find_fusion_opportunities(self) -> List[FusionOpportunity]:
        """
        Find all gate fusion opportunities.

        Returns:
            List of fusion opportunities
        """
        opportunities = []
        gates = self.circuit.gates
        
        i = 0
        while i < len(gates):
            # Check for inverse gate cancellation first
            opp = self._find_inverse_pair(i)
            if opp:
                opportunities.append(opp)
                i = opp.end_idx + 1
                continue
            
            # Check for rotation gate fusion
            opp = self._find_rotation_sequence(i)
            if opp:
                opportunities.append(opp)
                i = opp.end_idx + 1
                continue
            
            # Check for single-qubit gate sequences
            opp = self._find_single_qubit_sequence(i)
            if opp:
                opportunities.append(opp)
                i = opp.end_idx + 1
                continue
            
            i += 1
        
        return opportunities

    def apply_fusions(self, opportunities: List[FusionOpportunity] = None) -> UnifiedCircuit:
        """
        Apply gate fusion optimizations.

        Args:
            opportunities: Specific opportunities to apply (None = all)

        Returns:
            Optimized circuit
        """
        if opportunities is None:
            opportunities = self.find_fusion_opportunities()

        if not opportunities:
            return self.circuit

        # Sort by start index (reverse to handle indices correctly)
        opportunities = sorted(opportunities, key=lambda x: x.start_idx, reverse=True)

        new_circuit = UnifiedCircuit(n_qubits=self.circuit.n_qubits)
        gates = list(self.circuit.gates)
        fused_indices = set()

        # Mark indices that will be fused
        for opp in opportunities:
            for idx in range(opp.start_idx, opp.end_idx + 1):
                fused_indices.add(idx)

        # Build new circuit
        for i, gate in enumerate(gates):
            if i in fused_indices:
                # Check if this is the start of a fusion
                opp = next((o for o in opportunities if o.start_idx == i), None)
                if opp:
                    # Apply fusion
                    fused_gate = self._create_fused_gate(opp)
                    if fused_gate:
                        new_circuit.add_gate(
                            fused_gate.gate_type,
                            targets=fused_gate.targets,
                            parameters=fused_gate.parameters
                        )
            else:
                # Keep original gate
                new_circuit.add_gate(
                    gate.gate_type,
                    targets=gate.targets,
                    parameters=gate.parameters
                )

        return new_circuit

    def _find_single_qubit_sequence(self, start_idx: int) -> Optional[FusionOpportunity]:
        """Find sequence of single-qubit gates on same qubit."""
        gates = self.circuit.gates
        if start_idx >= len(gates):
            return None

        first_gate = gates[start_idx]
        if len(first_gate.targets) != 1:
            return None

        qubit = first_gate.targets[0]
        sequence = [start_idx]

        for i in range(start_idx + 1, len(gates)):
            gate = gates[i]
            if len(gate.targets) == 1 and gate.targets[0] == qubit:
                sequence.append(i)
            elif qubit in gate.targets or (gate.controls and qubit in gate.controls):
                # Qubit is used by multi-qubit gate
                break

        if len(sequence) >= 2:
            gate_types = [gates[i].gate_type for i in sequence]
            return FusionOpportunity(
                start_idx=sequence[0],
                end_idx=sequence[-1],
                gate_types=gate_types,
                qubits=[qubit],
                description=f"Fuse {len(sequence)} single-qubit gates",
                savings=len(sequence) - 1
            )

        return None

    def _find_rotation_sequence(self, start_idx: int) -> Optional[FusionOpportunity]:
        """Find sequence of rotation gates that can be combined."""
        gates = self.circuit.gates
        if start_idx >= len(gates):
            return None

        first_gate = gates[start_idx]
        if first_gate.gate_type not in [GateType.RX, GateType.RY, GateType.RZ]:
            return None

        rotation_type = first_gate.gate_type
        qubit = first_gate.targets[0]
        sequence = [start_idx]

        for i in range(start_idx + 1, len(gates)):
            gate = gates[i]
            if (gate.gate_type == rotation_type and
                len(gate.targets) == 1 and
                gate.targets[0] == qubit):
                sequence.append(i)
            elif qubit in gate.targets or (gate.controls and qubit in gate.controls):
                break

        if len(sequence) >= 2:
            return FusionOpportunity(
                start_idx=sequence[0],
                end_idx=sequence[-1],
                gate_types=[rotation_type] * len(sequence),
                qubits=[qubit],
                description=f"Combine {len(sequence)} {rotation_type} rotations",
                savings=len(sequence) - 1
            )

        return None

    def _find_inverse_pair(self, start_idx: int) -> Optional[FusionOpportunity]:
        """Find adjacent inverse gate pairs that cancel."""
        gates = self.circuit.gates
        if start_idx >= len(gates) - 1:
            return None

        gate1 = gates[start_idx]
        gate2 = gates[start_idx + 1]

        # Check if gates are inverses
        inverse_pairs = [
            (GateType.X, GateType.X),
            (GateType.Y, GateType.Y),
            (GateType.Z, GateType.Z),
            (GateType.H, GateType.H),
            (GateType.CNOT, GateType.CNOT),
            (GateType.SWAP, GateType.SWAP),
        ]

        for type1, type2 in inverse_pairs:
            if (gate1.gate_type == type1 and gate2.gate_type == type2 and
                gate1.targets == gate2.targets):
                return FusionOpportunity(
                    start_idx=start_idx,
                    end_idx=start_idx + 1,
                    gate_types=[type1, type2],
                    qubits=gate1.targets,
                    description="Cancel inverse gates",
                    savings=2
                )

        return None

    def _create_fused_gate(self, opportunity: FusionOpportunity) -> Optional[Gate]:
        """Create fused gate from opportunity."""
        gates = [self.circuit.gates[i] for i in range(
            opportunity.start_idx, opportunity.end_idx + 1
        )]

        # Handle rotation fusion
        if all(g.gate_type in [GateType.RX, GateType.RY, GateType.RZ] for g in gates):
            rotation_type = gates[0].gate_type
            total_angle = sum(g.parameters.get('angle', 0) for g in gates)

            return Gate(
                gate_type=rotation_type,
                targets=gates[0].targets,
                parameters={'angle': total_angle}
            )

        # Handle inverse cancellation (return None = remove both)
        if opportunity.description == "Cancel inverse gates":
            return None

        # For other fusions, keep first gate as approximation
        return gates[0]


def fuse_single_qubit_gates(circuit: UnifiedCircuit) -> UnifiedCircuit:
    """
    Fuse consecutive single-qubit gates.

    Args:
        circuit: Circuit to optimize

    Returns:
        Circuit with fused single-qubit gates
    """
    fuser = GateFuser(circuit)
    opportunities = [
        opp for opp in fuser.find_fusion_opportunities()
        if "single-qubit" in opp.description
    ]
    return fuser.apply_fusions(opportunities)


def fuse_rotation_gates(circuit: UnifiedCircuit) -> UnifiedCircuit:
    """
    Combine rotation gates of same type.

    Args:
        circuit: Circuit to optimize

    Returns:
        Circuit with combined rotations
    """
    fuser = GateFuser(circuit)
    opportunities = [
        opp for opp in fuser.find_fusion_opportunities()
        if "rotation" in opp.description.lower()
    ]
    return fuser.apply_fusions(opportunities)


def identify_fusion_opportunities(circuit: UnifiedCircuit) -> List[FusionOpportunity]:
    """
    Identify all fusion opportunities without applying them.

    Args:
        circuit: Circuit to analyze

    Returns:
        List of fusion opportunities
    """
    fuser = GateFuser(circuit)
    return fuser.find_fusion_opportunities()
