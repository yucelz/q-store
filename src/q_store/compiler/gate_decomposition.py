"""
Gate Decomposition to Native Gate Sets.

Decomposes arbitrary gates into device-native gate sets.
"""

from typing import List, Dict, Optional
import logging
import numpy as np

from ..core import UnifiedCircuit, GateType, Gate

logger = logging.getLogger(__name__)


class GateDecomposer:
    """
    Decomposes gates into native gate sets.

    Supports decomposition to:
    - Single-qubit + CNOT (universal set)
    - Clifford+T
    - Native hardware gates (SX, RZ, CZ, etc.)

    Args:
        native_gates: Set of gate types supported natively

    Example:
        >>> decomposer = GateDecomposer(native_gates={GateType.RZ, GateType.SX, GateType.CNOT})
        >>> decomposed = decomposer.decompose(circuit)
    """

    def __init__(self, native_gates: Optional[set] = None):
        if native_gates is None:
            # Default universal set: single-qubit rotations + CNOT
            native_gates = {
                GateType.RX, GateType.RY, GateType.RZ,
                GateType.CNOT
            }
        self.native_gates = native_gates

    def is_native(self, gate_type: GateType) -> bool:
        """Check if gate is in native set."""
        return gate_type in self.native_gates

    def decompose_gate(self, gate: Gate) -> List[Gate]:
        """
        Decompose a single gate into native gates.

        Args:
            gate: Gate to decompose

        Returns:
            List of gates in native set
        """
        if self.is_native(gate.gate_type):
            return [gate]

        # Decomposition rules
        decomposed = []

        if gate.gate_type == GateType.H:
            decomposed.extend(self._decompose_hadamard(gate))

        elif gate.gate_type == GateType.X:
            decomposed.extend(self._decompose_x(gate))

        elif gate.gate_type == GateType.Y:
            decomposed.extend(self._decompose_y(gate))

        elif gate.gate_type == GateType.Z:
            decomposed.extend(self._decompose_z(gate))

        elif gate.gate_type == GateType.S:
            decomposed.extend(self._decompose_s(gate))

        elif gate.gate_type == GateType.T:
            decomposed.extend(self._decompose_t(gate))

        elif gate.gate_type == GateType.CZ:
            decomposed.extend(self._decompose_cz(gate))

        elif gate.gate_type == GateType.SWAP:
            decomposed.extend(self._decompose_swap(gate))

        elif gate.gate_type == GateType.TOFFOLI:
            decomposed.extend(self._decompose_toffoli(gate))

        else:
            logger.warning(f"No decomposition rule for {gate.gate_type}, keeping as-is")
            decomposed.append(gate)

        return decomposed

    def _decompose_hadamard(self, gate: Gate) -> List[Gate]:
        """H = RZ(π) @ RY(π/2) or H = SX @ RZ(π)"""
        q = gate.targets[0]

        if GateType.RZ in self.native_gates and GateType.RY in self.native_gates:
            return [
                Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi}),
                Gate(GateType.RY, targets=[q], parameters={'angle': np.pi/2}),
            ]
        elif GateType.SX in self.native_gates and GateType.RZ in self.native_gates:
            return [
                Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi/2}),
                Gate(GateType.SX, targets=[q]),
                Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi/2}),
            ]
        else:
            return [gate]  # Keep as-is

    def _decompose_x(self, gate: Gate) -> List[Gate]:
        """X = RX(π)"""
        q = gate.targets[0]

        if GateType.RX in self.native_gates:
            return [Gate(GateType.RX, targets=[q], parameters={'angle': np.pi})]
        elif GateType.SX in self.native_gates:
            # X = SX @ SX
            return [
                Gate(GateType.SX, targets=[q]),
                Gate(GateType.SX, targets=[q]),
            ]
        else:
            return [gate]

    def _decompose_y(self, gate: Gate) -> List[Gate]:
        """Y = RY(π)"""
        q = gate.targets[0]

        if GateType.RY in self.native_gates:
            return [Gate(GateType.RY, targets=[q], parameters={'angle': np.pi})]
        elif GateType.RZ in self.native_gates and GateType.RX in self.native_gates:
            # Y = RZ(π) @ RX(π) @ RZ(-π)
            return [
                Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi}),
                Gate(GateType.RX, targets=[q], parameters={'angle': np.pi}),
            ]
        else:
            return [gate]

    def _decompose_z(self, gate: Gate) -> List[Gate]:
        """Z = RZ(π)"""
        q = gate.targets[0]

        if GateType.RZ in self.native_gates:
            return [Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi})]
        else:
            return [gate]

    def _decompose_s(self, gate: Gate) -> List[Gate]:
        """S = RZ(π/2)"""
        q = gate.targets[0]

        if GateType.RZ in self.native_gates:
            return [Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi/2})]
        else:
            return [gate]

    def _decompose_t(self, gate: Gate) -> List[Gate]:
        """T = RZ(π/4)"""
        q = gate.targets[0]

        if GateType.RZ in self.native_gates:
            return [Gate(GateType.RZ, targets=[q], parameters={'angle': np.pi/4})]
        else:
            return [gate]

    def _decompose_cz(self, gate: Gate) -> List[Gate]:
        """CZ = H @ CNOT @ H"""
        q0, q1 = gate.targets

        if GateType.CNOT in self.native_gates and GateType.H in self.native_gates:
            return [
                Gate(GateType.H, targets=[q1]),
                Gate(GateType.CNOT, targets=[q0, q1]),
                Gate(GateType.H, targets=[q1]),
            ]
        else:
            return [gate]

    def _decompose_swap(self, gate: Gate) -> List[Gate]:
        """SWAP = CNOT @ CNOT @ CNOT"""
        q0, q1 = gate.targets

        if GateType.CNOT in self.native_gates:
            return [
                Gate(GateType.CNOT, targets=[q0, q1]),
                Gate(GateType.CNOT, targets=[q1, q0]),
                Gate(GateType.CNOT, targets=[q0, q1]),
            ]
        else:
            return [gate]

    def _decompose_toffoli(self, gate: Gate) -> List[Gate]:
        """
        Toffoli (CCX) decomposition.

        Uses ~6 CNOTs + single-qubit gates.
        """
        c1, c2, t = gate.targets

        if GateType.CNOT not in self.native_gates:
            return [gate]

        # Standard Toffoli decomposition
        return [
            Gate(GateType.H, targets=[t]),
            Gate(GateType.CNOT, targets=[c2, t]),
            Gate(GateType.T, targets=[t]),  # T†
            Gate(GateType.CNOT, targets=[c1, t]),
            Gate(GateType.T, targets=[t]),
            Gate(GateType.CNOT, targets=[c2, t]),
            Gate(GateType.T, targets=[t]),  # T†
            Gate(GateType.CNOT, targets=[c1, t]),
            Gate(GateType.T, targets=[c2]),
            Gate(GateType.T, targets=[t]),
            Gate(GateType.H, targets=[t]),
            Gate(GateType.CNOT, targets=[c1, c2]),
            Gate(GateType.T, targets=[c1]),
            Gate(GateType.T, targets=[c2]),  # T†
            Gate(GateType.CNOT, targets=[c1, c2]),
        ]

    def decompose(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Decompose all gates in circuit to native set.

        Args:
            circuit: Input circuit

        Returns:
            Circuit with only native gates
        """
        decomposed_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)

        for gate in circuit.gates:
            decomposed_gates = self.decompose_gate(gate)
            for dgate in decomposed_gates:
                decomposed_circuit.gates.append(dgate)

        logger.info(
            f"Decomposed {len(circuit.gates)} gates to {len(decomposed_circuit.gates)} native gates"
        )

        return decomposed_circuit


def decompose_to_native_gates(
    circuit: UnifiedCircuit,
    native_gates: Optional[set] = None
) -> UnifiedCircuit:
    """
    Decompose circuit to native gate set.

    Args:
        circuit: Input circuit
        native_gates: Set of native gate types

    Returns:
        Decomposed circuit

    Example:
        >>> # Decompose to IBM gate set
        >>> native = {GateType.RZ, GateType.SX, GateType.CNOT}
        >>> decomposed = decompose_to_native_gates(circuit, native)
    """
    decomposer = GateDecomposer(native_gates=native_gates)
    return decomposer.decompose(circuit)
