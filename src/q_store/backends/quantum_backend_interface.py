"""
Quantum Backend Abstraction Layer
Provides hardware-agnostic interface for quantum operations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Type of quantum backend"""
    SIMULATOR = "simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    QPU = "qpu"
    MOCK = "mock"


class GateType(Enum):
    """Hardware-agnostic gate types"""
    # Single-qubit gates
    HADAMARD = "h"
    PAULI_X = "x"
    PAULI_Y = "y"
    PAULI_Z = "z"
    PHASE = "p"
    S = "s"
    T = "t"

    # Rotation gates
    RX = "rx"
    RY = "ry"
    RZ = "rz"

    # Two-qubit gates
    CNOT = "cnot"
    CZ = "cz"
    SWAP = "swap"

    # Multi-qubit gates
    TOFFOLI = "toffoli"

    # Measurement
    MEASURE = "measure"


@dataclass
class QuantumGate:
    """Hardware-agnostic gate representation"""
    gate_type: GateType
    qubits: List[int]
    parameters: Optional[Dict[str, float]] = None
    label: Optional[str] = None

    def __repr__(self) -> str:
        qubits_str = ','.join(map(str, self.qubits))
        params_str = f", params={self.parameters}" if self.parameters else ""
        return f"{self.gate_type.value}({qubits_str}{params_str})"


@dataclass
class QuantumCircuit:
    """
    Internal circuit representation (hardware-agnostic)
    Acts as intermediate representation (IR) between user code and backends
    """
    n_qubits: int
    gates: List[QuantumGate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_gate(self, gate: QuantumGate):
        """Add a gate to the circuit"""
        if max(gate.qubits, default=-1) >= self.n_qubits:
            raise ValueError(
                f"Gate acts on qubit {max(gate.qubits)} but circuit has only {self.n_qubits} qubits"
            )
        self.gates.append(gate)

    def depth(self) -> int:
        """Calculate circuit depth"""
        return len(self.gates)

    def gate_count(self) -> Dict[GateType, int]:
        """Count gates by type"""
        from collections import Counter
        return dict(Counter(gate.gate_type for gate in self.gates))

    def __repr__(self) -> str:
        return f"QuantumCircuit(n_qubits={self.n_qubits}, depth={self.depth()}, gates={len(self.gates)})"


@dataclass
class ExecutionResult:
    """Normalized execution result from any backend"""
    counts: Dict[str, int]
    probabilities: Dict[str, float]
    total_shots: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def most_common(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get n most common measurement outcomes"""
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:n]


@dataclass
class BackendCapabilities:
    """Describes capabilities of a quantum backend"""
    max_qubits: int
    supported_gates: List[GateType]
    backend_type: BackendType
    supports_mid_circuit_measurement: bool = False
    supports_reset: bool = False
    max_shots: int = 10000
    native_gate_set: Optional[List[GateType]] = None
    connectivity: Optional[List[Tuple[int, int]]] = None

    def supports_gate(self, gate_type: GateType) -> bool:
        """Check if backend supports a gate type"""
        return gate_type in self.supported_gates


class QuantumBackend(ABC):
    """
    Abstract base class for all quantum backends

    This provides a common interface for different quantum hardware/simulators,
    allowing the database to be hardware-agnostic.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend (connect to service, authenticate, etc.)

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a quantum circuit

        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
            **kwargs: Backend-specific options

        Returns:
            ExecutionResult with measurement outcomes

        Raises:
            Exception: If execution fails
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """
        Get backend capabilities

        Returns:
            BackendCapabilities describing what this backend can do
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get detailed backend information

        Returns:
            Dict with backend metadata (name, version, status, etc.)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up resources, close connections
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if backend is currently available

        Returns:
            True if backend can execute circuits
        """
        pass

    # Optional methods with default implementations

    async def validate_circuit(self, circuit: QuantumCircuit) -> bool:
        """
        Validate that circuit can run on this backend

        Args:
            circuit: Circuit to validate

        Returns:
            True if circuit is valid for this backend
        """
        caps = self.get_capabilities()

        if circuit.n_qubits > caps.max_qubits:
            return False

        for gate in circuit.gates:
            if not caps.supports_gate(gate.gate_type):
                return False

        return True

    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize circuit for this backend (optional)

        Args:
            circuit: Circuit to optimize

        Returns:
            Optimized circuit (default: returns original)
        """
        return circuit

    def estimate_cost(self, circuit: QuantumCircuit, shots: int) -> float:
        """
        Estimate cost to run circuit (optional)

        Args:
            circuit: Circuit to estimate
            shots: Number of shots

        Returns:
            Estimated cost in USD (0.0 for simulators)
        """
        return 0.0


class CircuitBuilder:
    """
    Helper class to build QuantumCircuit objects
    Provides convenient methods for common operations
    """

    def __init__(self, n_qubits: int):
        self.circuit = QuantumCircuit(n_qubits=n_qubits)

    def h(self, qubit: int) -> 'CircuitBuilder':
        """Add Hadamard gate"""
        self.circuit.add_gate(QuantumGate(GateType.HADAMARD, [qubit]))
        return self

    def x(self, qubit: int) -> 'CircuitBuilder':
        """Add Pauli-X gate"""
        self.circuit.add_gate(QuantumGate(GateType.PAULI_X, [qubit]))
        return self

    def y(self, qubit: int) -> 'CircuitBuilder':
        """Add Pauli-Y gate"""
        self.circuit.add_gate(QuantumGate(GateType.PAULI_Y, [qubit]))
        return self

    def z(self, qubit: int) -> 'CircuitBuilder':
        """Add Pauli-Z gate"""
        self.circuit.add_gate(QuantumGate(GateType.PAULI_Z, [qubit]))
        return self

    def rx(self, qubit: int, angle: float) -> 'CircuitBuilder':
        """Add RX rotation gate"""
        self.circuit.add_gate(QuantumGate(
            GateType.RX, [qubit], parameters={'angle': angle}
        ))
        return self

    def ry(self, qubit: int, angle: float) -> 'CircuitBuilder':
        """Add RY rotation gate"""
        self.circuit.add_gate(QuantumGate(
            GateType.RY, [qubit], parameters={'angle': angle}
        ))
        return self

    def rz(self, qubit: int, angle: float) -> 'CircuitBuilder':
        """Add RZ rotation gate"""
        self.circuit.add_gate(QuantumGate(
            GateType.RZ, [qubit], parameters={'angle': angle}
        ))
        return self

    def cnot(self, control: int, target: int) -> 'CircuitBuilder':
        """Add CNOT gate"""
        self.circuit.add_gate(QuantumGate(GateType.CNOT, [control, target]))
        return self

    def cz(self, control: int, target: int) -> 'CircuitBuilder':
        """Add CZ gate"""
        self.circuit.add_gate(QuantumGate(GateType.CZ, [control, target]))
        return self

    def swap(self, qubit1: int, qubit2: int) -> 'CircuitBuilder':
        """Add SWAP gate"""
        self.circuit.add_gate(QuantumGate(GateType.SWAP, [qubit1, qubit2]))
        return self

    def measure_all(self) -> 'CircuitBuilder':
        """Add measurement to all qubits"""
        for i in range(self.circuit.n_qubits):
            self.circuit.add_gate(QuantumGate(GateType.MEASURE, [i]))
        return self

    def measure(self, *qubits: int) -> 'CircuitBuilder':
        """Add measurement to specific qubits"""
        for qubit in qubits:
            self.circuit.add_gate(QuantumGate(GateType.MEASURE, [qubit]))
        return self

    def build(self) -> QuantumCircuit:
        """Return the constructed circuit"""
        return self.circuit


# Utility functions

def amplitude_encode_to_circuit(vector: np.ndarray) -> QuantumCircuit:
    """
    Convert classical vector to amplitude-encoded quantum circuit

    Args:
        vector: Classical vector to encode

    Returns:
        QuantumCircuit with amplitude encoding
    """
    # Normalize
    normalized = vector / np.linalg.norm(vector)

    # Calculate qubits needed
    n_qubits = int(np.ceil(np.log2(len(normalized))))

    # Create circuit
    builder = CircuitBuilder(n_qubits)

    # Simplified amplitude encoding using rotations
    # Full implementation would use recursive decomposition
    for i in range(min(n_qubits, len(normalized))):
        if abs(normalized[i]) > 1e-10:
            angle = 2 * np.arcsin(min(abs(normalized[i]), 1.0))
            builder.ry(i, angle)

    # Add entanglement
    for i in range(n_qubits - 1):
        builder.cnot(i, i + 1)

    return builder.build()


def create_bell_state_circuit() -> QuantumCircuit:
    """Create a simple Bell state circuit for testing"""
    return (CircuitBuilder(2)
            .h(0)
            .cnot(0, 1)
            .measure_all()
            .build())


def create_ghz_state_circuit(n_qubits: int) -> QuantumCircuit:
    """Create a GHZ state circuit for n qubits"""
    builder = CircuitBuilder(n_qubits)
    builder.h(0)
    for i in range(n_qubits - 1):
        builder.cnot(i, i + 1)
    builder.measure_all()
    return builder.build()
