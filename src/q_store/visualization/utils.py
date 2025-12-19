"""
Utility functions for visualization.
"""

from typing import List, Dict, Any
import numpy as np
from q_store.core import UnifiedCircuit, GateType


def generate_ascii_circuit(circuit: UnifiedCircuit, width: int = 80) -> str:
    """
    Generate simple ASCII representation of a circuit.
    
    Args:
        circuit: Circuit to visualize
        width: Maximum line width
        
    Returns:
        ASCII string
    """
    lines = []
    
    # Header
    lines.append("=" * min(width, 60))
    lines.append(f"Circuit: {circuit.n_qubits} qubits, {len(circuit.gates)} gates, depth {circuit.depth}")
    lines.append("=" * min(width, 60))
    
    # Gate list
    for i, gate in enumerate(circuit.gates):
        targets_str = ",".join(map(str, gate.targets))
        params_str = ""
        if gate.parameters:
            params_str = f" [{', '.join(f'{p:.3f}' for p in gate.parameters)}]"
        
        lines.append(f"{i+1:3d}. {gate.gate_type.name:8s} q[{targets_str}]{params_str}")
    
    return "\n".join(lines)


def circuit_to_text(circuit: UnifiedCircuit, detailed: bool = False) -> str:
    """
    Convert circuit to detailed text representation.
    
    Args:
        circuit: Circuit to convert
        detailed: Include detailed gate information
        
    Returns:
        Text representation
    """
    lines = []
    
    lines.append(f"UnifiedCircuit(n_qubits={circuit.n_qubits})")
    
    if detailed:
        lines.append(f"Total gates: {len(circuit.gates)}")
        lines.append(f"Circuit depth: {circuit.depth}")
        lines.append("")
        
        # Gate statistics
        gate_counts = {}
        for gate in circuit.gates:
            gate_counts[gate.gate_type] = gate_counts.get(gate.gate_type, 0) + 1
        
        lines.append("Gate counts:")
        for gate_type, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {gate_type.name}: {count}")
        
        lines.append("")
        lines.append("Gate sequence:")
        for i, gate in enumerate(circuit.gates):
            lines.append(f"  {i}: {gate.gate_type.name} {gate.targets}")
    else:
        lines.append(f"Gates: {len(circuit.gates)}, Depth: {circuit.depth}")
    
    return "\n".join(lines)


def format_complex(value: complex, precision: int = 3) -> str:
    """
    Format complex number for display.
    
    Args:
        value: Complex number
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    real = np.real(value)
    imag = np.imag(value)
    
    threshold = 10 ** (-precision)
    
    if abs(imag) < threshold:
        return f"{real:.{precision}f}"
    elif abs(real) < threshold:
        return f"{imag:.{precision}f}i"
    else:
        sign = "+" if imag >= 0 else ""
        return f"{real:.{precision}f}{sign}{imag:.{precision}f}i"


def create_bar_chart(values: List[float], labels: List[str],
                    width: int = 40, symbol: str = "█") -> str:
    """
    Create ASCII bar chart.
    
    Args:
        values: Values to plot
        labels: Labels for each bar
        width: Maximum bar width
        symbol: Character for bars
        
    Returns:
        ASCII bar chart
    """
    if not values:
        return ""
    
    max_value = max(values)
    lines = []
    
    for label, value in zip(labels, values):
        bar_length = int(width * value / max_value) if max_value > 0 else 0
        bar = symbol * bar_length
        lines.append(f"{label:10s} {value:.4f} {bar}")
    
    return "\n".join(lines)


def state_to_basis_str(state_vector: np.ndarray, threshold: float = 1e-10) -> str:
    """
    Convert state vector to basis state string.
    
    Args:
        state_vector: State vector
        threshold: Threshold for including basis states
        
    Returns:
        String like "0.707|00⟩ + 0.707|11⟩"
    """
    n_qubits = int(np.log2(len(state_vector)))
    terms = []
    
    for i, amplitude in enumerate(state_vector):
        if np.abs(amplitude) > threshold:
            basis_state = format(i, f'0{n_qubits}b')
            coeff = format_complex(amplitude)
            terms.append(f"{coeff}|{basis_state}⟩")
    
    if not terms:
        return "|0⟩"
    
    return " + ".join(terms).replace("+ -", "- ")


def density_matrix_to_text(rho: np.ndarray, threshold: float = 1e-10) -> str:
    """
    Convert density matrix to text representation.
    
    Args:
        rho: Density matrix
        threshold: Threshold for displaying elements
        
    Returns:
        Text representation
    """
    lines = []
    n = rho.shape[0]
    
    for i in range(n):
        row_elements = []
        for j in range(n):
            element = rho[i, j]
            if np.abs(element) > threshold:
                row_elements.append(format_complex(element))
            else:
                row_elements.append("0")
        lines.append("  ".join(row_elements))
    
    return "\n".join(lines)
