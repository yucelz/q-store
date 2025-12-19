"""
Unified Circuit Representation for Q-Store v4.0

This module provides a framework-agnostic quantum circuit representation
that can be converted to and from various quantum computing frameworks
(Cirq, Qiskit, IonQ native gates, etc.).

Key Features:
- Framework-agnostic gate representation
- Parameterized circuits for machine learning
- Automatic optimization and compilation
- Serialization for storage and transmission
- Conversion to/from multiple frameworks
"""

from __future__ import annotations

import json
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


class GateType(str, Enum):
    """Supported quantum gate types"""
    # Single-qubit gates
    H = "H"  # Hadamard
    X = "X"  # Pauli-X (NOT)
    Y = "Y"  # Pauli-Y
    Z = "Z"  # Pauli-Z
    S = "S"  # S gate (√Z)
    T = "T"  # T gate (√S)
    RX = "RX"  # Rotation around X-axis
    RY = "RY"  # Rotation around Y-axis
    RZ = "RZ"  # Rotation around Z-axis

    # Two-qubit gates
    CNOT = "CNOT"  # Controlled-NOT
    CZ = "CZ"  # Controlled-Z
    SWAP = "SWAP"  # SWAP gate

    # IonQ Native Gates
    GPI = "GPI"  # IonQ GPI gate
    GPI2 = "GPI2"  # IonQ GPI2 gate
    MS = "MS"  # IonQ Mølmer-Sørensen gate

    # Multi-qubit gates
    CCX = "CCX"  # Toffoli (controlled-CNOT)
    CSWAP = "CSWAP"  # Fredkin (controlled-SWAP)


@dataclass
class Parameter:
    """
    Represents a parameterized value in a quantum circuit

    Can be either:
    - A fixed value (numeric)
    - A symbolic parameter (string name) for training
    """
    name: str
    value: Optional[float] = None
    is_symbolic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'is_symbolic': self.is_symbolic
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Parameter:
        """Create from dictionary"""
        return cls(**data)


@dataclass
class Gate:
    """
    Represents a quantum gate in the circuit

    Attributes:
        gate_type: Type of quantum gate
        targets: List of target qubit indices
        controls: Optional list of control qubit indices
        parameters: Optional parameters (for parameterized gates)
        name: Optional custom name for the gate
    """
    gate_type: Union[str, GateType]
    targets: List[int]
    controls: Optional[List[int]] = None
    parameters: Optional[Dict[str, Union[float, Parameter]]] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Validate gate construction"""
        if isinstance(self.gate_type, str):
            self.gate_type = GateType(self.gate_type)

        # Ensure targets is a list
        if isinstance(self.targets, int):
            self.targets = [self.targets]

        # Validate target count
        self._validate_gate()

    def _validate_gate(self):
        """Validate gate has correct number of targets"""
        single_qubit_gates = {GateType.H, GateType.X, GateType.Y, GateType.Z,
                             GateType.S, GateType.T, GateType.RX, GateType.RY,
                             GateType.RZ, GateType.GPI, GateType.GPI2}

        two_qubit_gates = {GateType.CNOT, GateType.CZ, GateType.SWAP, GateType.MS}

        if self.gate_type in single_qubit_gates:
            if len(self.targets) != 1:
                raise ValueError(f"{self.gate_type} requires exactly 1 target qubit")
        elif self.gate_type in two_qubit_gates:
            if len(self.targets) != 2:
                raise ValueError(f"{self.gate_type} requires exactly 2 target qubits")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            'gate_type': self.gate_type.value if isinstance(self.gate_type, GateType) else self.gate_type,
            'targets': self.targets,
        }

        if self.controls is not None:
            data['controls'] = self.controls

        if self.parameters is not None:
            data['parameters'] = {
                k: v.to_dict() if isinstance(v, Parameter) else v
                for k, v in self.parameters.items()
            }

        if self.name is not None:
            data['name'] = self.name

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Gate:
        """Create from dictionary"""
        data = data.copy()

        # Convert parameters if present
        if 'parameters' in data and data['parameters'] is not None:
            params = {}
            for k, v in data['parameters'].items():
                if isinstance(v, dict) and 'name' in v:
                    params[k] = Parameter.from_dict(v)
                else:
                    params[k] = v
            data['parameters'] = params

        return cls(**data)


class UnifiedCircuit:
    """
    Framework-agnostic quantum circuit representation

    This is the core component of Q-Store v4.0 that enables seamless
    integration with multiple quantum frameworks (Cirq, Qiskit, IonQ)
    and ML frameworks (TensorFlow, PyTorch).

    Features:
    - Convert to/from Cirq, Qiskit, native backends
    - Parameterized circuits with symbolic parameters
    - Automatic optimization and compilation
    - Serialization for storage and transmission
    - Circuit composition and manipulation

    Example:
        >>> circuit = UnifiedCircuit(n_qubits=4)
        >>> circuit.add_gate(GateType.H, targets=[0])
        >>> circuit.add_gate(GateType.CNOT, targets=[0, 1])
        >>> circuit.add_parameterized_layer(GateType.RY, 'theta')
        >>> cirq_circuit = circuit.to_cirq()
    """

    def __init__(self, n_qubits: int):
        """
        Initialize a new quantum circuit

        Args:
            n_qubits: Number of qubits in the circuit
        """
        if n_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")

        self.n_qubits = n_qubits
        self.gates: List[Gate] = []
        self.parameters: Dict[str, Parameter] = {}
        self._metadata: Dict[str, Any] = {}
        self._depth: Optional[int] = None  # Cache for circuit depth

    def add_gate(
        self,
        gate_type: Union[str, GateType],
        targets: Union[int, List[int]],
        controls: Optional[Union[int, List[int]]] = None,
        parameters: Optional[Dict[str, Union[float, str]]] = None,
        name: Optional[str] = None
    ) -> UnifiedCircuit:
        """
        Add a gate to the circuit

        Args:
            gate_type: Type of gate to add
            targets: Target qubit index or indices
            controls: Optional control qubit index or indices
            parameters: Optional gate parameters (for parameterized gates)
            name: Optional custom name for the gate

        Returns:
            Self for method chaining

        Example:
            >>> circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': 'theta_0'})
        """
        # Normalize inputs
        if isinstance(targets, int):
            targets = [targets]
        if controls is not None and isinstance(controls, int):
            controls = [controls]

        # Validate qubit indices
        all_qubits = targets + (controls if controls else [])
        if any(q >= self.n_qubits or q < 0 for q in all_qubits):
            raise ValueError(f"Qubit indices must be in range [0, {self.n_qubits})")

        # Handle parameters
        processed_params = None
        if parameters is not None:
            processed_params = {}
            for key, value in parameters.items():
                if isinstance(value, str):
                    # Symbolic parameter
                    if value not in self.parameters:
                        self.parameters[value] = Parameter(name=value, is_symbolic=True)
                    processed_params[key] = self.parameters[value]
                elif isinstance(value, (int, float)):
                    # Numeric parameter
                    processed_params[key] = value
                elif isinstance(value, Parameter):
                    processed_params[key] = value
                    if value.is_symbolic and value.name not in self.parameters:
                        self.parameters[value.name] = value

        # Create and add gate
        gate = Gate(
            gate_type=gate_type,
            targets=targets,
            controls=controls,
            parameters=processed_params,
            name=name
        )
        self.gates.append(gate)

        # Invalidate depth cache
        self._depth = None

        return self

    def add_parameterized_layer(
        self,
        gate_type: Union[str, GateType],
        param_prefix: str,
        qubits: Optional[List[int]] = None
    ) -> UnifiedCircuit:
        """
        Add a parameterized layer to all (or specified) qubits

        This is useful for creating variational quantum circuits where
        each qubit gets a parameterized rotation gate.

        Args:
            gate_type: Type of parameterized gate (RX, RY, RZ, etc.)
            param_prefix: Prefix for parameter names (will add _0, _1, etc.)
            qubits: Optional list of qubits (default: all qubits)

        Returns:
            Self for method chaining

        Example:
            >>> circuit.add_parameterized_layer(GateType.RY, 'theta')
            # Adds RY gates with parameters theta_0, theta_1, theta_2, ...
        """
        if qubits is None:
            qubits = list(range(self.n_qubits))

        for qubit in qubits:
            param_name = f"{param_prefix}_{qubit}"
            self.add_gate(
                gate_type=gate_type,
                targets=[qubit],
                parameters={'angle': param_name}
            )

        return self

    def add_entangling_layer(
        self,
        gate_type: Union[str, GateType] = GateType.CNOT,
        pattern: str = 'linear'
    ) -> UnifiedCircuit:
        """
        Add an entangling layer to the circuit

        Args:
            gate_type: Type of entangling gate (CNOT, CZ, etc.)
            pattern: Entanglement pattern:
                - 'linear': Connect adjacent qubits (0-1, 1-2, 2-3, ...)
                - 'circular': Linear + connect last to first
                - 'full': All-to-all connectivity

        Returns:
            Self for method chaining
        """
        if pattern == 'linear':
            for i in range(self.n_qubits - 1):
                self.add_gate(gate_type, targets=[i, i + 1])

        elif pattern == 'circular':
            for i in range(self.n_qubits - 1):
                self.add_gate(gate_type, targets=[i, i + 1])
            if self.n_qubits > 2:
                self.add_gate(gate_type, targets=[self.n_qubits - 1, 0])

        elif pattern == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    self.add_gate(gate_type, targets=[i, j])

        else:
            raise ValueError(f"Unknown entanglement pattern: {pattern}")

        return self

    @property
    def depth(self) -> int:
        """
        Calculate circuit depth (number of time steps)

        Returns:
            Circuit depth
        """
        if self._depth is not None:
            return self._depth

        # Track when each qubit becomes available
        qubit_times = [0] * self.n_qubits

        for gate in self.gates:
            # Find when all involved qubits are available
            involved_qubits = gate.targets + (gate.controls if gate.controls else [])
            ready_time = max(qubit_times[q] for q in involved_qubits)

            # Update all involved qubits
            for q in involved_qubits:
                qubit_times[q] = ready_time + 1

        self._depth = max(qubit_times) if qubit_times else 0
        return self._depth

    @property
    def n_parameters(self) -> int:
        """Get number of trainable parameters"""
        return len(self.parameters)

    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names"""
        return list(self.parameters.keys())

    def bind_parameters(self, values: Dict[str, float]) -> UnifiedCircuit:
        """
        Bind concrete values to symbolic parameters

        Args:
            values: Dictionary mapping parameter names to values

        Returns:
            New circuit with bound parameters
        """
        new_circuit = self.copy()

        for param_name, value in values.items():
            if param_name in new_circuit.parameters:
                new_circuit.parameters[param_name].value = value
                new_circuit.parameters[param_name].is_symbolic = False

        return new_circuit

    def copy(self) -> UnifiedCircuit:
        """Create a deep copy of the circuit"""
        return copy.deepcopy(self)

    def to_json(self) -> str:
        """
        Serialize circuit to JSON string

        Returns:
            JSON string representation
        """
        data = {
            'n_qubits': self.n_qubits,
            'gates': [g.to_dict() for g in self.gates],
            'parameters': {k: v.to_dict() for k, v in self.parameters.items()},
            'metadata': self._metadata
        }
        return json.dumps(data, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert circuit to dictionary

        Returns:
            Dictionary representation
        """
        return {
            'n_qubits': self.n_qubits,
            'gates': [g.to_dict() for g in self.gates],
            'parameters': {k: v.to_dict() for k, v in self.parameters.items()},
            'metadata': self._metadata
        }

    @classmethod
    def from_json(cls, json_str: str) -> UnifiedCircuit:
        """
        Deserialize circuit from JSON string

        Args:
            json_str: JSON string representation

        Returns:
            UnifiedCircuit instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedCircuit:
        """
        Create circuit from dictionary

        Args:
            data: Dictionary representation

        Returns:
            UnifiedCircuit instance
        """
        circuit = cls(n_qubits=data['n_qubits'])

        # Restore parameters
        circuit.parameters = {
            k: Parameter.from_dict(v) for k, v in data['parameters'].items()
        }

        # Restore gates
        circuit.gates = [Gate.from_dict(g) for g in data['gates']]

        # Restore metadata
        circuit._metadata = data.get('metadata', {})

        return circuit

    def __len__(self) -> int:
        """Get number of gates in circuit"""
        return len(self.gates)

    def __repr__(self) -> str:
        """String representation"""
        return (f"UnifiedCircuit(n_qubits={self.n_qubits}, "
                f"n_gates={len(self.gates)}, "
                f"n_parameters={self.n_parameters}, "
                f"depth={self.depth})")

    def __str__(self) -> str:
        """Detailed string representation"""
        lines = [
            f"UnifiedCircuit:",
            f"  Qubits: {self.n_qubits}",
            f"  Gates: {len(self.gates)}",
            f"  Depth: {self.depth}",
            f"  Parameters: {self.n_parameters}",
        ]

        if self.parameters:
            lines.append("  Parameter names:")
            for name in sorted(self.parameters.keys()):
                lines.append(f"    - {name}")

        return "\n".join(lines)
