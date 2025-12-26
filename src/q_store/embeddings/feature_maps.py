"""
Feature map embeddings for quantum machine learning.

Feature maps encode classical data into quantum states using
parameterized quantum circuits.
"""

from typing import List, Callable, Optional
import numpy as np

from q_store.core import UnifiedCircuit, GateType


class ZZFeatureMap:
    """
    ZZ Feature Map for quantum machine learning.

    Encodes classical features using ZZ interactions between qubits.
    Common in quantum kernel methods and QML.

    Args:
        n_features: Number of classical features (= number of qubits)
        reps: Number of repetitions (depth)
        entanglement: Entanglement pattern ('full', 'linear', 'circular')
        data_map_func: Function to map data (default: lambda x: x)
    """

    def __init__(
        self,
        n_features: int,
        reps: int = 2,
        entanglement: str = 'full',
        data_map_func: Optional[Callable] = None
    ):
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.data_map_func = data_map_func or (lambda x: x)

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode classical data into quantum circuit.

        Args:
            data: Classical data vector (length = n_features)

        Returns:
            Quantum circuit encoding the data
        """
        if len(data) != self.n_features:
            raise ValueError(f"Data dimension {len(data)} != n_features {self.n_features}")

        circuit = UnifiedCircuit(n_qubits=self.n_features)

        # Initial Hadamard layer
        for q in range(self.n_features):
            circuit.add_gate(GateType.H, targets=[q])

        # Repeated layers
        for rep in range(self.reps):
            # Single-qubit rotations
            for q in range(self.n_features):
                angle = self.data_map_func(data[q])
                circuit.add_gate(GateType.RZ, targets=[q], parameters={'angle': 2 * angle})

            # Entangling layer
            if rep < self.reps - 1 or self.reps == 1:
                self._add_entangling_layer(circuit, data)

        return circuit

    def _add_entangling_layer(self, circuit: UnifiedCircuit, data: np.ndarray):
        """Add ZZ entangling gates."""
        pairs = self._get_entangling_pairs()

        for i, j in pairs:
            # ZZ(φ) = exp(-i φ Z_i Z_j) implemented using CNOTs
            angle = self.data_map_func((np.pi - data[i]) * (np.pi - data[j]))

            circuit.add_gate(GateType.CNOT, targets=[i, j])
            circuit.add_gate(GateType.RZ, targets=[j], parameters={'angle': 2 * angle})
            circuit.add_gate(GateType.CNOT, targets=[i, j])

    def _get_entangling_pairs(self) -> List[tuple]:
        """Get qubit pairs for entanglement."""
        pairs = []

        if self.entanglement == 'full':
            # All-to-all connectivity
            for i in range(self.n_features):
                for j in range(i + 1, self.n_features):
                    pairs.append((i, j))

        elif self.entanglement == 'linear':
            # Linear chain
            for i in range(self.n_features - 1):
                pairs.append((i, i + 1))

        elif self.entanglement == 'circular':
            # Ring topology
            for i in range(self.n_features - 1):
                pairs.append((i, i + 1))
            if self.n_features > 2:
                pairs.append((self.n_features - 1, 0))

        else:
            raise ValueError(f"Unknown entanglement: {self.entanglement}")

        return pairs


class PauliFeatureMap:
    """
    Pauli Feature Map.

    General feature map using Pauli rotations (X, Y, Z).

    Args:
        n_features: Number of features
        paulis: List of Pauli strings for rotations
        reps: Number of repetitions
        entanglement: Entanglement pattern
    """

    def __init__(
        self,
        n_features: int,
        paulis: List[str] = ['Z', 'ZZ'],
        reps: int = 2,
        entanglement: str = 'linear'
    ):
        self.n_features = n_features
        self.paulis = paulis
        self.reps = reps
        self.entanglement = entanglement

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """Encode data using Pauli rotations."""
        if len(data) != self.n_features:
            raise ValueError(f"Data dimension {len(data)} != n_features {self.n_features}")

        circuit = UnifiedCircuit(n_qubits=self.n_features)

        # Initial layer
        for q in range(self.n_features):
            circuit.add_gate(GateType.H, targets=[q])

        # Repeated layers
        for _ in range(self.reps):
            for pauli in self.paulis:
                self._apply_pauli_rotation(circuit, pauli, data)

        return circuit

    def _apply_pauli_rotation(self, circuit: UnifiedCircuit, pauli: str, data: np.ndarray):
        """Apply Pauli rotation based on data."""
        if pauli == 'Z':
            # Single-qubit Z rotations
            for q in range(self.n_features):
                circuit.add_gate(GateType.RZ, targets=[q], parameters={'angle': 2 * data[q]})

        elif pauli == 'ZZ':
            # Two-qubit ZZ rotations
            pairs = self._get_pairs()
            for i, j in pairs:
                angle = data[i] * data[j]
                circuit.add_gate(GateType.CNOT, targets=[i, j])
                circuit.add_gate(GateType.RZ, targets=[j], parameters={'angle': 2 * angle})
                circuit.add_gate(GateType.CNOT, targets=[i, j])

        elif pauli == 'Y':
            for q in range(self.n_features):
                circuit.add_gate(GateType.RY, targets=[q], parameters={'angle': 2 * data[q]})

        elif pauli == 'X':
            for q in range(self.n_features):
                circuit.add_gate(GateType.RX, targets=[q], parameters={'angle': 2 * data[q]})

    def _get_pairs(self) -> List[tuple]:
        """Get qubit pairs for two-qubit gates."""
        if self.entanglement == 'linear':
            return [(i, i + 1) for i in range(self.n_features - 1)]
        elif self.entanglement == 'full':
            return [(i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)]
        else:
            return []


class IQPFeatureMap:
    """
    Instantaneous Quantum Polynomial (IQP) Feature Map.

    Diagonal feature map that is classically hard to simulate.
    Uses only diagonal gates (Z rotations and CZ gates).

    Args:
        n_features: Number of features
        reps: Number of repetitions
    """

    def __init__(self, n_features: int, reps: int = 2):
        self.n_features = n_features
        self.reps = reps

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode data using IQP circuit.

        Args:
            data: Classical data vector

        Returns:
            IQP circuit
        """
        if len(data) != self.n_features:
            raise ValueError(f"Data dimension {len(data)} != n_features {self.n_features}")

        circuit = UnifiedCircuit(n_qubits=self.n_features)

        # Initial Hadamard layer
        for q in range(self.n_features):
            circuit.add_gate(GateType.H, targets=[q])

        # Repeated IQP layers
        for _ in range(self.reps):
            # Single-qubit Z rotations
            for q in range(self.n_features):
                circuit.add_gate(GateType.RZ, targets=[q], parameters={'angle': 2 * data[q]})

            # Two-qubit diagonal interactions
            for i in range(self.n_features):
                for j in range(i + 1, self.n_features):
                    # CZ rotation with data-dependent angle
                    angle = data[i] * data[j]

                    # CZ(θ) = diag(1, 1, 1, exp(iθ))
                    circuit.add_gate(GateType.H, targets=[j])
                    circuit.add_gate(GateType.CNOT, targets=[i, j])
                    circuit.add_gate(GateType.RZ, targets=[j], parameters={'angle': 2 * angle})
                    circuit.add_gate(GateType.CNOT, targets=[i, j])
                    circuit.add_gate(GateType.H, targets=[j])

        # Final Hadamard layer (optional, for kernel evaluation)
        for q in range(self.n_features):
            circuit.add_gate(GateType.H, targets=[q])

        return circuit
