"""
Qiskit-based IonQ Backend Adapter
Implements the QuantumBackend interface using Qiskit and qiskit-ionq
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .quantum_backend_interface import (
    BackendCapabilities,
    BackendType,
    ExecutionResult,
    GateType,
    QuantumBackend,
    QuantumCircuit,
    QuantumGate,
)

logger = logging.getLogger(__name__)


class QiskitIonQBackend(QuantumBackend):
    """
    IonQ quantum backend using Qiskit SDK

    This adapter provides Qiskit compatibility for users who prefer
    the Qiskit ecosystem while maintaining hardware abstraction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        target: str = "simulator",
        noise_model: Optional[str] = None,
    ):
        """
        Initialize Qiskit-IonQ backend

        Args:
            api_key: IonQ API key (or use IONQ_API_KEY env var)
            target: Backend target ('simulator', 'qpu.aria-1', etc.)
            noise_model: Optional noise model for simulator
        """
        self.api_key = api_key or os.getenv("IONQ_API_KEY")
        self.target = target
        self.noise_model = noise_model
        self._provider = None
        self._backend = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qiskit-IonQ provider"""
        if self._initialized:
            return

        try:
            from qiskit_ionq import IonQProvider

            if not self.api_key:
                raise ValueError("IonQ API key required (pass api_key or set IONQ_API_KEY env var)")

            # Create provider
            self._provider = IonQProvider(self.api_key)

            # Get backend
            if self.target == "simulator":
                self._backend = self._provider.get_backend("ionq_simulator")
            elif "qpu" in self.target:
                # Extract QPU name (e.g., 'qpu.aria-1' -> 'ionq_qpu.aria-1')
                qpu_name = self.target.replace("qpu.", "ionq_qpu.")
                self._backend = self._provider.get_backend(qpu_name)
            else:
                self._backend = self._provider.get_backend(self.target)

            self._initialized = True
            logger.info(f"Initialized Qiskit-IonQ backend: {self.target}")

        except ImportError:
            raise ImportError("qiskit-ionq not installed. Install with: pip install qiskit-ionq")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qiskit-IonQ backend: {e}")

    def _convert_to_qiskit(self, circuit: QuantumCircuit) -> "QuantumCircuit":
        """
        Convert internal QuantumCircuit to Qiskit QuantumCircuit

        Args:
            circuit: Internal circuit representation

        Returns:
            Qiskit QuantumCircuit object
        """
        from qiskit import QuantumCircuit as QiskitCircuit

        # Create Qiskit circuit
        qiskit_circuit = QiskitCircuit(circuit.n_qubits, circuit.n_qubits)

        # Track which qubits need measurement
        measured_qubits = set()

        # Convert gates
        for gate in circuit.gates:
            if gate.gate_type == GateType.MEASURE:
                measured_qubits.update(gate.qubits)
            else:
                self._add_gate_to_qiskit(qiskit_circuit, gate)

        # Add measurements at the end
        if measured_qubits:
            for qubit in measured_qubits:
                qiskit_circuit.measure(qubit, qubit)

        return qiskit_circuit

    def _add_gate_to_qiskit(self, qiskit_circuit: "QuantumCircuit", gate: QuantumGate) -> None:
        """
        Add a gate to Qiskit circuit

        Args:
            qiskit_circuit: Qiskit circuit to modify
            gate: Gate to add
        """
        qubits = gate.qubits

        # Single-qubit gates
        if gate.gate_type == GateType.HADAMARD:
            qiskit_circuit.h(qubits[0])
        elif gate.gate_type == GateType.PAULI_X:
            qiskit_circuit.x(qubits[0])
        elif gate.gate_type == GateType.PAULI_Y:
            qiskit_circuit.y(qubits[0])
        elif gate.gate_type == GateType.PAULI_Z:
            qiskit_circuit.z(qubits[0])
        elif gate.gate_type == GateType.S:
            qiskit_circuit.s(qubits[0])
        elif gate.gate_type == GateType.T:
            qiskit_circuit.t(qubits[0])

        # Rotation gates
        elif gate.gate_type == GateType.RX:
            angle = gate.parameters["angle"]
            qiskit_circuit.rx(angle, qubits[0])
        elif gate.gate_type == GateType.RY:
            angle = gate.parameters["angle"]
            qiskit_circuit.ry(angle, qubits[0])
        elif gate.gate_type == GateType.RZ:
            angle = gate.parameters["angle"]
            qiskit_circuit.rz(angle, qubits[0])
        elif gate.gate_type == GateType.PHASE:
            angle = gate.parameters["angle"]
            qiskit_circuit.p(angle, qubits[0])

        # Two-qubit gates
        elif gate.gate_type == GateType.CNOT:
            qiskit_circuit.cx(qubits[0], qubits[1])
        elif gate.gate_type == GateType.CZ:
            qiskit_circuit.cz(qubits[0], qubits[1])
        elif gate.gate_type == GateType.SWAP:
            qiskit_circuit.swap(qubits[0], qubits[1])

        # Multi-qubit gates
        elif gate.gate_type == GateType.TOFFOLI:
            qiskit_circuit.ccx(qubits[0], qubits[1], qubits[2])

        else:
            logger.warning(f"Unsupported gate type: {gate.gate_type}")

    async def execute_circuit(
        self, circuit: QuantumCircuit, shots: int = 1000, **kwargs
    ) -> ExecutionResult:
        """
        Execute circuit on IonQ hardware via Qiskit

        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
            **kwargs: Additional options

        Returns:
            ExecutionResult with measurement outcomes
        """
        if not self._initialized:
            await self.initialize()

        # Convert to Qiskit circuit
        qiskit_circuit = self._convert_to_qiskit(circuit)

        try:
            # Execute on backend
            job = self._backend.run(qiskit_circuit, shots=shots)
            result = job.result()

            # Convert to our format
            return self._convert_result(result, shots, circuit)

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

    def _convert_result(
        self, qiskit_result, total_shots: int, original_circuit: QuantumCircuit
    ) -> ExecutionResult:
        """Convert Qiskit result to ExecutionResult"""
        # Get counts from Qiskit result
        counts_dict = qiskit_result.get_counts()

        # Qiskit uses different bit ordering, may need to reverse
        counts = {}
        for bitstring, count in counts_dict.items():
            # Remove spaces if present
            clean_bitstring = bitstring.replace(" ", "")
            counts[clean_bitstring] = count

        # Calculate probabilities
        probabilities = {k: v / total_shots for k, v in counts.items()}

        return ExecutionResult(
            counts=counts,
            probabilities=probabilities,
            total_shots=total_shots,
            metadata={
                "backend": "qiskit_ionq",
                "target": self.target,
                "job_id": qiskit_result.job_id if hasattr(qiskit_result, "job_id") else None,
            },
        )

    def get_capabilities(self) -> BackendCapabilities:
        """Get IonQ backend capabilities"""
        # IonQ capabilities vary by target
        if "simulator" in self.target:
            max_qubits = 29
            backend_type = BackendType.SIMULATOR
        elif "aria" in self.target:
            max_qubits = 25
            backend_type = BackendType.QPU
        elif "forte" in self.target:
            max_qubits = 32
            backend_type = BackendType.QPU
        else:
            max_qubits = 11
            backend_type = BackendType.QPU

        return BackendCapabilities(
            max_qubits=max_qubits,
            supported_gates=[
                GateType.HADAMARD,
                GateType.PAULI_X,
                GateType.PAULI_Y,
                GateType.PAULI_Z,
                GateType.RX,
                GateType.RY,
                GateType.RZ,
                GateType.PHASE,
                GateType.CNOT,
                GateType.CZ,
                GateType.SWAP,
                GateType.S,
                GateType.T,
                GateType.TOFFOLI,
                GateType.MEASURE,
            ],
            backend_type=backend_type,
            supports_mid_circuit_measurement=False,
            supports_reset=False,
            max_shots=10000,
            native_gate_set=[GateType.RX, GateType.RY, GateType.RZ, GateType.CNOT],
            connectivity=None,  # All-to-all connectivity
        )

    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "provider": "IonQ",
            "sdk": "qiskit",
            "target": self.target,
            "version": self._get_version(),
            "backend_type": self._get_backend_type().value,
        }

    def _get_backend_type(self) -> BackendType:
        """Determine backend type from target"""
        if "simulator" in self.target:
            return BackendType.SIMULATOR
        else:
            return BackendType.QPU

    def _get_version(self) -> str:
        """Get Qiskit version"""
        try:
            import qiskit

            return qiskit.__version__
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not determine Qiskit version: {e}")
            return "unknown"

    async def close(self) -> None:
        """Close connection"""
        self._provider = None
        self._backend = None
        self._initialized = False
        logger.info("Closed Qiskit-IonQ backend")

    def is_available(self) -> bool:
        """Check if backend is available"""
        return self._initialized and self._backend is not None

    def estimate_cost(self, circuit: QuantumCircuit, shots: int) -> float:
        """Estimate execution cost"""
        if "simulator" in self.target:
            return 0.0
        else:
            # IonQ pricing: ~$0.01 per gate-shot
            n_gates = len([g for g in circuit.gates if g.gate_type != GateType.MEASURE])
            return (n_gates * shots) * 0.00001  # $0.01 per 1000 gate-shots
