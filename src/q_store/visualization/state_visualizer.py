"""
State visualization tools including Bloch sphere.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BlochVector:
    """Bloch vector representation of a qubit state."""
    x: float
    y: float
    z: float

    @classmethod
    def from_state(cls, state: np.ndarray) -> 'BlochVector':
        """
        Compute Bloch vector from state vector.

        Args:
            state: State vector [α, β] for |ψ⟩ = α|0⟩ + β|1⟩

        Returns:
            BlochVector
        """
        if len(state) != 2:
            raise ValueError("State must be a 2-element vector for single qubit")

        # Normalize
        state = state / np.linalg.norm(state)

        # Compute Bloch coordinates
        # ρ = 1/2(I + r·σ) where r = (x, y, z) is Bloch vector
        x = 2 * np.real(state[0] * np.conj(state[1]))
        y = 2 * np.imag(state[0] * np.conj(state[1]))
        z = np.abs(state[0])**2 - np.abs(state[1])**2

        return cls(x=float(x), y=float(y), z=float(z))

    def to_angles(self) -> Tuple[float, float]:
        """
        Convert to spherical angles (θ, φ).

        Returns:
            (theta, phi) in radians
        """
        theta = np.arccos(self.z)
        phi = np.arctan2(self.y, self.x)
        return float(theta), float(phi)


class BlochSphere:
    """
    Bloch sphere representation for single-qubit states.
    """

    def __init__(self):
        """Initialize Bloch sphere."""
        self.states: List[Tuple[str, BlochVector]] = []

    def add_state(self, state: np.ndarray, label: str = ""):
        """
        Add a state to the Bloch sphere.

        Args:
            state: State vector
            label: Optional label for the state
        """
        vector = BlochVector.from_state(state)
        self.states.append((label, vector))

    def clear(self):
        """Clear all states."""
        self.states = []

    def get_ascii_representation(self) -> str:
        """
        Get ASCII art representation of the Bloch sphere.

        Returns:
            ASCII art string
        """
        lines = []
        lines.append("        |z⟩")
        lines.append("         │")
        lines.append("         ●")
        lines.append("        ╱│╲")
        lines.append("       ╱ │ ╲")
        lines.append("      ╱  │  ╲")
        lines.append("  |y⟩───●───|x⟩")
        lines.append("      ╲  │  ╱")
        lines.append("       ╲ │ ╱")
        lines.append("        ╲│╱")
        lines.append("         ●")
        lines.append("         │")
        lines.append("       |-z⟩")

        if self.states:
            lines.append("\nStates:")
            for label, vec in self.states:
                theta, phi = vec.to_angles()
                lines.append(f"  {label}: θ={theta:.3f}, φ={phi:.3f}")
                lines.append(f"    (x={vec.x:.3f}, y={vec.y:.3f}, z={vec.z:.3f})")

        return "\n".join(lines)

    def get_state_data(self) -> List[Dict[str, Any]]:
        """
        Get structured data for all states.

        Returns:
            List of state dictionaries
        """
        data = []
        for label, vec in self.states:
            theta, phi = vec.to_angles()
            data.append({
                'label': label,
                'x': vec.x,
                'y': vec.y,
                'z': vec.z,
                'theta': theta,
                'phi': phi
            })
        return data


class StateVisualizer:
    """
    Visualizer for quantum states.
    """

    def __init__(self):
        """Initialize state visualizer."""
        pass

    def visualize_statevector(self, state: np.ndarray,
                            threshold: float = 1e-10) -> str:
        """
        Visualize a state vector.

        Args:
            state: State vector
            threshold: Amplitude threshold for display

        Returns:
            String representation
        """
        lines = []
        n_qubits = int(np.log2(len(state)))

        lines.append(f"State vector (n_qubits={n_qubits}):")
        lines.append("=" * 50)

        for i, amplitude in enumerate(state):
            if np.abs(amplitude) > threshold:
                basis_state = format(i, f'0{n_qubits}b')
                real = np.real(amplitude)
                imag = np.imag(amplitude)
                prob = np.abs(amplitude)**2

                # Format amplitude
                if np.abs(imag) < threshold:
                    amp_str = f"{real:+.4f}"
                elif np.abs(real) < threshold:
                    amp_str = f"{imag:+.4f}i"
                else:
                    amp_str = f"{real:+.4f}{imag:+.4f}i"

                lines.append(f"|{basis_state}⟩: {amp_str}  (prob: {prob:.4f})")

        return "\n".join(lines)

    def visualize_density_matrix(self, rho: np.ndarray,
                                threshold: float = 1e-10) -> str:
        """
        Visualize a density matrix.

        Args:
            rho: Density matrix
            threshold: Element threshold for display

        Returns:
            String representation
        """
        lines = []
        n = rho.shape[0]
        n_qubits = int(np.log2(n))

        lines.append(f"Density matrix (n_qubits={n_qubits}, dim={n}×{n}):")
        lines.append("=" * 50)

        # Show matrix elements
        for i in range(n):
            row = []
            for j in range(n):
                element = rho[i, j]
                if np.abs(element) > threshold:
                    real = np.real(element)
                    imag = np.imag(element)

                    if np.abs(imag) < threshold:
                        row.append(f"{real:+.3f}")
                    elif np.abs(real) < threshold:
                        row.append(f"{imag:+.3f}i")
                    else:
                        row.append(f"{real:.2f}{imag:+.2f}i")
                else:
                    row.append("  0  ")

            lines.append(" ".join(row))

        # Compute properties
        trace = np.trace(rho)
        purity = np.trace(rho @ rho)

        lines.append("")
        lines.append(f"Trace: {trace:.4f}")
        lines.append(f"Purity: {purity:.4f}")

        return "\n".join(lines)

    def visualize_probabilities(self, state: np.ndarray,
                               top_k: Optional[int] = None) -> str:
        """
        Visualize measurement probabilities.

        Args:
            state: State vector
            top_k: Show only top k probabilities

        Returns:
            String representation with bar chart
        """
        n_qubits = int(np.log2(len(state)))
        probabilities = np.abs(state)**2

        # Sort by probability
        indices = np.argsort(probabilities)[::-1]

        if top_k:
            indices = indices[:top_k]

        lines = []
        lines.append(f"Measurement probabilities (n_qubits={n_qubits}):")
        lines.append("=" * 60)

        max_prob = probabilities[indices[0]] if len(indices) > 0 else 1.0

        for idx in indices:
            basis_state = format(idx, f'0{n_qubits}b')
            prob = probabilities[idx]

            # Create bar
            bar_length = int(40 * prob / max_prob) if max_prob > 0 else 0
            bar = "█" * bar_length

            lines.append(f"|{basis_state}⟩ {prob:.4f} {bar}")

        return "\n".join(lines)

    def create_bloch_sphere(self, state: np.ndarray) -> BlochSphere:
        """
        Create Bloch sphere representation for a single-qubit state.

        Args:
            state: Single-qubit state vector

        Returns:
            BlochSphere object
        """
        if len(state) != 2:
            raise ValueError("Bloch sphere only for single-qubit states")

        sphere = BlochSphere()
        sphere.add_state(state, "ψ")
        return sphere

    def compare_states(self, state1: np.ndarray, state2: np.ndarray) -> Dict[str, Any]:
        """
        Compare two quantum states.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Comparison dictionary
        """
        # Normalize
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)

        # Compute overlap
        overlap = np.abs(np.vdot(state1, state2))
        fidelity = overlap ** 2

        # Trace distance (for pure states)
        trace_distance = np.sqrt(1 - fidelity**2)

        return {
            'overlap': float(overlap),
            'fidelity': float(fidelity),
            'trace_distance': float(trace_distance),
            'are_orthogonal': fidelity < 1e-10
        }


def visualize_state(state: np.ndarray,
                   mode: str = 'statevector') -> str:
    """
    Convenience function to visualize a quantum state.

    Args:
        state: State vector or density matrix
        mode: 'statevector', 'density', or 'probabilities'

    Returns:
        Visualization string
    """
    visualizer = StateVisualizer()

    if mode == 'statevector':
        return visualizer.visualize_statevector(state)
    elif mode == 'density':
        return visualizer.visualize_density_matrix(state)
    elif mode == 'probabilities':
        return visualizer.visualize_probabilities(state)
    else:
        raise ValueError(f"Unknown mode: {mode}")
