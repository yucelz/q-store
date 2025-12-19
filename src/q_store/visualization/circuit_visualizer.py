"""
Circuit visualization tools.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from q_store.core import UnifiedCircuit, GateType


@dataclass
class VisualizationConfig:
    """Configuration for circuit visualization."""
    width: int = 80
    show_measurements: bool = True
    show_barriers: bool = True
    gate_symbols: Dict[GateType, str] = None
    
    def __post_init__(self):
        """Initialize default gate symbols."""
        if self.gate_symbols is None:
            self.gate_symbols = {
                GateType.H: 'H',
                GateType.X: 'X',
                GateType.Y: 'Y',
                GateType.Z: 'Z',
                GateType.S: 'S',
                GateType.T: 'T',
                GateType.CNOT: '⊕',
                GateType.CZ: '●',
                GateType.SWAP: '×',
                GateType.RX: 'Rx',
                GateType.RY: 'Ry',
                GateType.RZ: 'Rz'
            }


class CircuitVisualizer:
    """
    Visualizer for quantum circuits.
    
    Creates text-based and structured representations of circuits.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize circuit visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
    
    def visualize(self, circuit: UnifiedCircuit) -> str:
        """
        Generate ASCII visualization of circuit.
        
        Args:
            circuit: Circuit to visualize
            
        Returns:
            ASCII art representation
        """
        lines = self._create_qubit_lines(circuit.n_qubits)
        
        # Process gates
        for gate in circuit.gates:
            self._add_gate_to_lines(lines, gate, circuit.n_qubits)
        
        # Add qubit labels
        result = []
        for i, line in enumerate(lines):
            result.append(f"q{i}: {line}")
        
        return "\n".join(result)
    
    def _create_qubit_lines(self, n_qubits: int) -> List[str]:
        """Create initial qubit lines."""
        return ["─" * 4 for _ in range(n_qubits)]
    
    def _add_gate_to_lines(self, lines: List[str], gate: Any, n_qubits: int):
        """Add a gate to the visualization lines."""
        symbol = self.config.gate_symbols.get(gate.gate_type, str(gate.gate_type.name))
        
        if gate.gate_type == GateType.CNOT:
            # Two-qubit gate
            control, target = gate.targets[0], gate.targets[1]
            
            # Add control
            lines[control] += "─●─"
            
            # Add vertical line
            start, end = min(control, target), max(control, target)
            for i in range(start + 1, end):
                lines[i] += "─│─"
            
            # Add target
            lines[target] += "─⊕─"
            
            # Align other qubits
            for i in range(n_qubits):
                if i not in [control, target] and i < start or i > end:
                    lines[i] += "───"
        
        elif len(gate.targets) == 1:
            # Single-qubit gate
            target = gate.targets[0]
            gate_str = f"─{symbol}─"
            lines[target] += gate_str
            
            # Align other qubits
            padding = "─" * len(gate_str)
            for i in range(n_qubits):
                if i != target:
                    lines[i] += padding
        
        else:
            # Multi-qubit gate (general case)
            for i in range(n_qubits):
                if i in gate.targets:
                    lines[i] += f"─{symbol}─"
                else:
                    lines[i] += "───"
    
    def get_circuit_diagram(self, circuit: UnifiedCircuit) -> Dict[str, Any]:
        """
        Get structured circuit diagram data.
        
        Args:
            circuit: Circuit to visualize
            
        Returns:
            Dictionary with diagram data
        """
        layers = self._compute_circuit_layers(circuit)
        
        return {
            'n_qubits': circuit.n_qubits,
            'n_gates': len(circuit.gates),
            'depth': circuit.depth,
            'layers': layers,
            'ascii': self.visualize(circuit)
        }
    
    def _compute_circuit_layers(self, circuit: UnifiedCircuit) -> List[List[Dict[str, Any]]]:
        """Compute circuit layers for visualization."""
        layers = []
        current_layer = []
        used_qubits = set()
        
        for gate in circuit.gates:
            gate_qubits = set(gate.targets)
            
            # Check if gate conflicts with current layer
            if gate_qubits & used_qubits:
                # Start new layer
                if current_layer:
                    layers.append(current_layer)
                current_layer = []
                used_qubits = set()
            
            # Add gate to current layer
            current_layer.append({
                'gate_type': gate.gate_type.name,
                'targets': gate.targets,
                'parameters': gate.parameters
            })
            used_qubits.update(gate_qubits)
        
        # Add final layer
        if current_layer:
            layers.append(current_layer)
        
        return layers
    
    def compare_circuits(self, circuit1: UnifiedCircuit, 
                        circuit2: UnifiedCircuit) -> Dict[str, Any]:
        """
        Compare visualizations of two circuits.
        
        Args:
            circuit1: First circuit
            circuit2: Second circuit
            
        Returns:
            Comparison data
        """
        return {
            'circuit1': {
                'ascii': self.visualize(circuit1),
                'n_gates': len(circuit1.gates),
                'depth': circuit1.depth
            },
            'circuit2': {
                'ascii': self.visualize(circuit2),
                'n_gates': len(circuit2.gates),
                'depth': circuit2.depth
            },
            'differences': {
                'gate_diff': len(circuit2.gates) - len(circuit1.gates),
                'depth_diff': circuit2.depth - circuit1.depth
            }
        }
    
    def export_latex(self, circuit: UnifiedCircuit) -> str:
        """
        Export circuit to LaTeX (Qcircuit) format.
        
        Args:
            circuit: Circuit to export
            
        Returns:
            LaTeX code
        """
        lines = []
        lines.append(r"\begin{quantikz}")
        
        for i in range(circuit.n_qubits):
            qubit_gates = []
            for gate in circuit.gates:
                if i in gate.targets:
                    symbol = self.config.gate_symbols.get(gate.gate_type, gate.gate_type.name)
                    qubit_gates.append(f"\\gate{{{symbol}}}")
                else:
                    qubit_gates.append("\\qw")
            
            line = " & ".join(qubit_gates)
            lines.append(f"\\ket{{0}} & {line} \\\\")
        
        lines.append(r"\end{quantikz}")
        return "\n".join(lines)


def visualize_circuit(circuit: UnifiedCircuit, 
                     config: Optional[VisualizationConfig] = None) -> str:
    """
    Convenience function to visualize a circuit.
    
    Args:
        circuit: Circuit to visualize
        config: Optional visualization configuration
        
    Returns:
        ASCII visualization
    """
    visualizer = CircuitVisualizer(config)
    return visualizer.visualize(circuit)
