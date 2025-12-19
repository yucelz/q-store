"""
Template-based circuit optimization.

Identifies patterns in circuits and replaces them with optimized
equivalents using template matching.
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from q_store.core import UnifiedCircuit, Gate, GateType


@dataclass
class OptimizationTemplate:
    """Template for pattern matching and replacement."""
    name: str
    pattern: List[GateType]
    replacement: List[GateType]
    pattern_qubits: int
    description: str
    savings: int  # Gates saved


class TemplateOptimizer:
    """
    Template-based circuit optimization.

    Matches circuit patterns and replaces them with optimized equivalents.
    """

    def __init__(self, circuit: UnifiedCircuit):
        """
        Initialize template optimizer.

        Args:
            circuit: Circuit to optimize
        """
        self.circuit = circuit
        self.templates = standard_templates()

    def add_template(self, template: OptimizationTemplate):
        """Add custom optimization template."""
        self.templates.append(template)

    def find_matches(self) -> List[Tuple[int, OptimizationTemplate]]:
        """
        Find all template matches in circuit.

        Returns:
            List of (start_index, template) tuples
        """
        matches = []
        gates = self.circuit.gates

        for template in self.templates:
            i = 0
            while i <= len(gates) - len(template.pattern):
                if self._matches_template(i, template):
                    matches.append((i, template))
                    i += len(template.pattern)
                else:
                    i += 1

        return matches

    def apply_templates(self) -> UnifiedCircuit:
        """
        Apply all template optimizations.

        Returns:
            Optimized circuit
        """
        matches = self.find_matches()
        if not matches:
            return self.circuit

        # Sort by start index (reverse)
        matches = sorted(matches, key=lambda x: x[0], reverse=True)

        new_circuit = UnifiedCircuit(n_qubits=self.circuit.n_qubits)
        gates = list(self.circuit.gates)
        replaced_indices = set()

        # Mark indices to be replaced
        for start_idx, template in matches:
            for i in range(start_idx, start_idx + len(template.pattern)):
                replaced_indices.add(i)

        # Build optimized circuit
        i = 0
        while i < len(gates):
            if i in replaced_indices:
                # Find matching template
                match = next((m for m in matches if m[0] == i), None)
                if match:
                    start_idx, template = match
                    # Apply replacement
                    replacement_gates = self._create_replacement_gates(
                        start_idx, template
                    )
                    for gate in replacement_gates:
                        new_circuit.add_gate(
                            gate.gate_type,
                            targets=gate.targets,
                            parameters=gate.parameters
                        )
                    i += len(template.pattern)
                    continue

            # Keep original gate
            gate = gates[i]
            new_circuit.add_gate(
                gate.gate_type,
                targets=gate.targets,
                parameters=gate.parameters
            )
            i += 1

        return new_circuit

    def _matches_template(self, start_idx: int, template: OptimizationTemplate) -> bool:
        """Check if template matches at given position."""
        gates = self.circuit.gates

        if start_idx + len(template.pattern) > len(gates):
            return False

        # Check if gate types match
        for i, expected_type in enumerate(template.pattern):
            gate = gates[start_idx + i]
            if gate.gate_type != expected_type:
                return False

        # Check if gates act on same qubits (for single-qubit templates)
        if template.pattern_qubits == 1:
            first_qubit = gates[start_idx].targets[0]
            for i in range(1, len(template.pattern)):
                gate = gates[start_idx + i]
                if len(gate.targets) != 1 or gate.targets[0] != first_qubit:
                    return False

        return True

    def _create_replacement_gates(
        self,
        start_idx: int,
        template: OptimizationTemplate
    ) -> List[Gate]:
        """Create replacement gates from template."""
        gates = []
        original_gates = self.circuit.gates[start_idx:start_idx + len(template.pattern)]

        # Get qubits from original pattern
        qubits = original_gates[0].targets

        # Create replacement gates
        for gate_type in template.replacement:
            gates.append(Gate(
                gate_type=gate_type,
                targets=qubits
            ))

        return gates


def standard_templates() -> List[OptimizationTemplate]:
    """
    Get standard optimization templates.

    Returns:
        List of optimization templates
    """
    templates = []

    # Hadamard cancellation: H H -> I
    templates.append(OptimizationTemplate(
        name="hadamard_cancel",
        pattern=[GateType.H, GateType.H],
        replacement=[],
        pattern_qubits=1,
        description="Cancel adjacent Hadamard gates",
        savings=2
    ))

    # Pauli cancellation: X X -> I, Y Y -> I, Z Z -> I
    for pauli in [GateType.X, GateType.Y, GateType.Z]:
        templates.append(OptimizationTemplate(
            name=f"{pauli}_cancel",
            pattern=[pauli, pauli],
            replacement=[],
            pattern_qubits=1,
            description=f"Cancel adjacent {pauli} gates",
            savings=2
        ))

    # S^2 = Z
    templates.append(OptimizationTemplate(
        name="s_to_z",
        pattern=[GateType.S, GateType.S],
        replacement=[GateType.Z],
        pattern_qubits=1,
        description="Two S gates = Z gate",
        savings=1
    ))

    # T^4 = Z
    templates.append(OptimizationTemplate(
        name="t4_to_z",
        pattern=[GateType.T, GateType.T, GateType.T, GateType.T],
        replacement=[GateType.Z],
        pattern_qubits=1,
        description="Four T gates = Z gate",
        savings=3
    ))

    # X Z X = -Z (but we ignore global phase)
    templates.append(OptimizationTemplate(
        name="xzx_simplify",
        pattern=[GateType.X, GateType.Z, GateType.X],
        replacement=[GateType.Z],
        pattern_qubits=1,
        description="Simplify X Z X pattern",
        savings=2
    ))

    # H X H = Z
    templates.append(OptimizationTemplate(
        name="hxh_to_z",
        pattern=[GateType.H, GateType.X, GateType.H],
        replacement=[GateType.Z],
        pattern_qubits=1,
        description="H X H = Z",
        savings=2
    ))

    # H Z H = X
    templates.append(OptimizationTemplate(
        name="hzh_to_x",
        pattern=[GateType.H, GateType.Z, GateType.H],
        replacement=[GateType.X],
        pattern_qubits=1,
        description="H Z H = X",
        savings=2
    ))

    return templates


def match_and_replace_templates(
    circuit: UnifiedCircuit,
    templates: List[OptimizationTemplate] = None
) -> UnifiedCircuit:
    """
    Apply template-based optimization.

    Args:
        circuit: Circuit to optimize
        templates: Templates to use (None = standard templates)

    Returns:
        Optimized circuit
    """
    optimizer = TemplateOptimizer(circuit)

    if templates is not None:
        optimizer.templates = templates

    return optimizer.apply_templates()


def create_optimization_template(
    name: str,
    pattern_gates: List[GateType],
    replacement_gates: List[GateType],
    description: str = ""
) -> OptimizationTemplate:
    """
    Create custom optimization template.

    Args:
        name: Template name
        pattern_gates: Gate pattern to match
        replacement_gates: Replacement gates
        description: Human-readable description

    Returns:
        Optimization template
    """
    return OptimizationTemplate(
        name=name,
        pattern=pattern_gates,
        replacement=replacement_gates,
        pattern_qubits=1,  # Simplified - assumes single qubit
        description=description,
        savings=len(pattern_gates) - len(replacement_gates)
    )


def apply_peephole_optimization(circuit: UnifiedCircuit) -> UnifiedCircuit:
    """
    Apply peephole optimization (small window pattern matching).

    Args:
        circuit: Circuit to optimize

    Returns:
        Optimized circuit
    """
    # This is a simplified peephole optimizer
    # that applies standard templates
    return match_and_replace_templates(circuit)


def optimize_rotation_sequences(circuit: UnifiedCircuit) -> UnifiedCircuit:
    """
    Optimize sequences of rotation gates.

    Args:
        circuit: Circuit to optimize

    Returns:
        Optimized circuit
    """
    new_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
    gates = circuit.gates

    i = 0
    while i < len(gates):
        gate = gates[i]

        # Check for rotation sequences
        if gate.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            rotation_type = gate.gate_type
            qubit = gate.targets[0]
            total_angle = gate.parameters.get('angle', 0)

            # Look ahead for same rotation on same qubit
            j = i + 1
            while j < len(gates):
                next_gate = gates[j]
                if (next_gate.gate_type == rotation_type and
                    len(next_gate.targets) == 1 and
                    next_gate.targets[0] == qubit):
                    total_angle += next_gate.parameters.get('angle', 0)
                    j += 1
                else:
                    break

            # Add combined rotation
            if abs(total_angle) > 1e-10:  # Skip near-zero rotations
                new_circuit.add_gate(
                    rotation_type,
                    targets=[qubit],
                    parameters={'angle': total_angle}
                )

            i = j
        else:
            # Keep non-rotation gates
            new_circuit.add_gate(
                gate.gate_type,
                targets=gate.targets,
                parameters=gate.parameters
            )
            i += 1

    return new_circuit
