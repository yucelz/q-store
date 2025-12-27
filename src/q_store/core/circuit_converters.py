"""
Circuit Converters for Q-Store v4.0

Provides conversion between UnifiedCircuit and various quantum frameworks:
- Google Cirq
- IBM Qiskit
- IonQ Native Gates (GPi, GPi2, MS)

This enables Q-Store to work seamlessly with multiple quantum ecosystems.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np

from .unified_circuit import UnifiedCircuit, Gate, GateType, Parameter

# Optional imports with graceful degradation
try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter as QiskitParameter
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None
    QiskitParameter = None


class CirqConverter:
    """Convert between UnifiedCircuit and Google Cirq"""
    
    # Gate mapping: UnifiedCircuit -> Cirq
    GATE_MAP = {
        GateType.H: lambda q: cirq.H(q),
        GateType.X: lambda q: cirq.X(q),
        GateType.Y: lambda q: cirq.Y(q),
        GateType.Z: lambda q: cirq.Z(q),
        GateType.S: lambda q: cirq.S(q),
        GateType.T: lambda q: cirq.T(q),
        GateType.CNOT: lambda q0, q1: cirq.CNOT(q0, q1),
        GateType.CZ: lambda q0, q1: cirq.CZ(q0, q1),
        GateType.SWAP: lambda q0, q1: cirq.SWAP(q0, q1),
    }
    
    @staticmethod
    def to_cirq(unified_circuit: UnifiedCircuit) -> 'cirq.Circuit':
        """
        Convert UnifiedCircuit to Cirq circuit
        
        Args:
            unified_circuit: Circuit to convert
        
        Returns:
            Cirq Circuit
        
        Raises:
            ImportError: If Cirq is not installed
        """
        if not HAS_CIRQ:
            raise ImportError("Cirq is required for this conversion. Install with: pip install cirq")
        
        qubits = cirq.LineQubit.range(unified_circuit.n_qubits)
        circuit = cirq.Circuit()
        
        # Create Cirq parameter objects for symbolic parameters
        cirq_params = {}
        for param_name, param in unified_circuit.parameters.items():
            if param.is_symbolic:
                cirq_params[param_name] = cirq.Parameter(param_name)
        
        # Convert each gate
        for gate in unified_circuit.gates:
            cirq_gate = CirqConverter._convert_gate(gate, qubits, cirq_params)
            if cirq_gate is not None:
                # Handle both single gates and lists of gates (for GPI/GPI2 decomposition)
                if isinstance(cirq_gate, list):
                    for g in cirq_gate:
                        circuit.append(g)
                else:
                    circuit.append(cirq_gate)
        
        return circuit
    
    @staticmethod
    def _convert_gate(gate: Gate, qubits: List, cirq_params: Dict) -> Optional[Any]:
        """Convert a single gate to Cirq format"""
        targets = [qubits[i] for i in gate.targets]
        
        # Handle parameterized gates
        if gate.parameters:
            angle_param = gate.parameters.get('angle')
            if angle_param is not None:
                # Get the angle value or parameter
                if isinstance(angle_param, Parameter):
                    angle = cirq_params.get(angle_param.name, angle_param.value)
                else:
                    angle = angle_param
                
                # Parameterized rotation gates
                if gate.gate_type == GateType.RX:
                    return cirq.rx(angle)(targets[0])
                elif gate.gate_type == GateType.RY:
                    return cirq.ry(angle)(targets[0])
                elif gate.gate_type == GateType.RZ:
                    return cirq.rz(angle)(targets[0])
        
        # Non-parameterized gates
        if gate.gate_type in CirqConverter.GATE_MAP:
            return CirqConverter.GATE_MAP[gate.gate_type](*targets)
        
        # Handle IonQ native gates
        if gate.gate_type == GateType.GPI:
            # GPI(φ) gate: rotation by π around axis at angle φ in XY plane
            # Implemented as: Rz(-φ) · X · Rz(φ)
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            # Return as a sequence: Rz(-phi), X, Rz(phi)
            return [
                cirq.rz(-phi)(targets[0]),
                cirq.X(targets[0]),
                cirq.rz(phi)(targets[0])
            ]

        elif gate.gate_type == GateType.GPI2:
            # GPI2(φ) gate: rotation by π/2 around axis at angle φ in XY plane
            # Implemented as: Rz(-φ) · √X · Rz(φ)
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            return [
                cirq.rz(-phi)(targets[0]),
                cirq.XPowGate(exponent=0.5)(targets[0]),
                cirq.rz(phi)(targets[0])
            ]
        
        elif gate.gate_type == GateType.MS:
            # Mølmer-Sørensen gate (XX interaction)
            angle = gate.parameters.get('angle', np.pi/4) if gate.parameters else np.pi/4
            return cirq.XXPowGate(exponent=angle/np.pi)(targets[0], targets[1])
        
        return None
    
    @staticmethod
    def from_cirq(cirq_circuit: 'cirq.Circuit') -> UnifiedCircuit:
        """
        Convert Cirq circuit to UnifiedCircuit
        
        Args:
            cirq_circuit: Cirq circuit to convert
        
        Returns:
            UnifiedCircuit
        
        Raises:
            ImportError: If Cirq is not installed
        """
        if not HAS_CIRQ:
            raise ImportError("Cirq is required for this conversion. Install with: pip install cirq")
        
        # Get number of qubits
        qubits = sorted(cirq_circuit.all_qubits())
        n_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        
        unified = UnifiedCircuit(n_qubits=n_qubits)
        
        # Convert each operation
        for moment in cirq_circuit:
            for op in moment:
                gate_qubits = [qubit_map[q] for q in op.qubits]
                CirqConverter._add_cirq_operation(unified, op, gate_qubits)
        
        return unified
    
    @staticmethod
    def _add_cirq_operation(circuit: UnifiedCircuit, op: Any, qubits: List[int]):
        """Add a Cirq operation to UnifiedCircuit"""
        gate = op.gate
        
        # Common gates
        if isinstance(gate, cirq.HPowGate):
            circuit.add_gate(GateType.H, targets=qubits)
        elif isinstance(gate, cirq.XPowGate):
            circuit.add_gate(GateType.X, targets=qubits)
        elif isinstance(gate, cirq.YPowGate):
            circuit.add_gate(GateType.Y, targets=qubits)
        elif isinstance(gate, cirq.ZPowGate):
            circuit.add_gate(GateType.Z, targets=qubits)
        elif isinstance(gate, cirq.CNotPowGate):
            circuit.add_gate(GateType.CNOT, targets=qubits)
        elif isinstance(gate, cirq.CZPowGate):
            circuit.add_gate(GateType.CZ, targets=qubits)
        elif isinstance(gate, cirq.SwapPowGate):
            circuit.add_gate(GateType.SWAP, targets=qubits)
        
        # Rotation gates (check for Rx, Ry, Rz)
        elif hasattr(gate, '_gate_name'):
            name = gate._gate_name
            if 'Rx' in name or isinstance(gate, (cirq.ops.common_gates.Rx, cirq.ops.phased_x_gate.PhasedXPowGate)):
                # Extract angle if available
                angle = getattr(gate, 'exponent', 1.0) * np.pi
                circuit.add_gate(GateType.RX, targets=qubits, parameters={'angle': angle})
            elif 'Ry' in name:
                angle = getattr(gate, 'exponent', 1.0) * np.pi
                circuit.add_gate(GateType.RY, targets=qubits, parameters={'angle': angle})
            elif 'Rz' in name:
                angle = getattr(gate, 'exponent', 1.0) * np.pi
                circuit.add_gate(GateType.RZ, targets=qubits, parameters={'angle': angle})


class QiskitConverter:
    """Convert between UnifiedCircuit and IBM Qiskit"""
    
    @staticmethod
    def to_qiskit(unified_circuit: UnifiedCircuit) -> 'QuantumCircuit':
        """
        Convert UnifiedCircuit to Qiskit circuit
        
        Args:
            unified_circuit: Circuit to convert
        
        Returns:
            Qiskit QuantumCircuit
        
        Raises:
            ImportError: If Qiskit is not installed
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for this conversion. Install with: pip install qiskit")
        
        circuit = QuantumCircuit(unified_circuit.n_qubits)
        
        # Create Qiskit parameter objects for symbolic parameters
        qiskit_params = {}
        for param_name, param in unified_circuit.parameters.items():
            if param.is_symbolic:
                qiskit_params[param_name] = QiskitParameter(param_name)
        
        # Convert each gate
        for gate in unified_circuit.gates:
            QiskitConverter._convert_gate(circuit, gate, qiskit_params)
        
        return circuit
    
    @staticmethod
    def _convert_gate(qiskit_circuit: 'QuantumCircuit', gate: Gate, qiskit_params: Dict):
        """Convert a single gate to Qiskit format"""
        targets = gate.targets
        
        # Handle parameterized gates
        if gate.parameters:
            angle_param = gate.parameters.get('angle')
            if angle_param is not None:
                # Get the angle value or parameter
                if isinstance(angle_param, Parameter):
                    angle = qiskit_params.get(angle_param.name, angle_param.value)
                else:
                    angle = angle_param
                
                # Parameterized rotation gates
                if gate.gate_type == GateType.RX:
                    qiskit_circuit.rx(angle, targets[0])
                    return
                elif gate.gate_type == GateType.RY:
                    qiskit_circuit.ry(angle, targets[0])
                    return
                elif gate.gate_type == GateType.RZ:
                    qiskit_circuit.rz(angle, targets[0])
                    return
        
        # Single-qubit gates
        if gate.gate_type == GateType.H:
            qiskit_circuit.h(targets[0])
        elif gate.gate_type == GateType.X:
            qiskit_circuit.x(targets[0])
        elif gate.gate_type == GateType.Y:
            qiskit_circuit.y(targets[0])
        elif gate.gate_type == GateType.Z:
            qiskit_circuit.z(targets[0])
        elif gate.gate_type == GateType.S:
            qiskit_circuit.s(targets[0])
        elif gate.gate_type == GateType.T:
            qiskit_circuit.t(targets[0])
        
        # Two-qubit gates
        elif gate.gate_type == GateType.CNOT:
            qiskit_circuit.cx(targets[0], targets[1])
        elif gate.gate_type == GateType.CZ:
            qiskit_circuit.cz(targets[0], targets[1])
        elif gate.gate_type == GateType.SWAP:
            qiskit_circuit.swap(targets[0], targets[1])
        
        # IonQ native gates - decompose to standard gates
        elif gate.gate_type == GateType.GPI:
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            qiskit_circuit.rx(np.pi, targets[0])
            qiskit_circuit.rz(phi, targets[0])
        
        elif gate.gate_type == GateType.GPI2:
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            qiskit_circuit.rx(np.pi/2, targets[0])
            qiskit_circuit.rz(phi, targets[0])
        
        elif gate.gate_type == GateType.MS:
            angle = gate.parameters.get('angle', np.pi/4) if gate.parameters else np.pi/4
            # MS gate decomposition (simplified)
            qiskit_circuit.rxx(angle, targets[0], targets[1])
    
    @staticmethod
    def from_qiskit(qiskit_circuit: 'QuantumCircuit') -> UnifiedCircuit:
        """
        Convert Qiskit circuit to UnifiedCircuit
        
        Args:
            qiskit_circuit: Qiskit circuit to convert
        
        Returns:
            UnifiedCircuit
        
        Raises:
            ImportError: If Qiskit is not installed
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for this conversion. Install with: pip install qiskit")
        
        unified = UnifiedCircuit(n_qubits=qiskit_circuit.num_qubits)
        
        # Convert each instruction
        for instruction, qubits, _ in qiskit_circuit.data:
            qubit_indices = [qiskit_circuit.find_bit(q).index for q in qubits]
            QiskitConverter._add_qiskit_instruction(unified, instruction, qubit_indices)
        
        return unified
    
    @staticmethod
    def _add_qiskit_instruction(circuit: UnifiedCircuit, instruction: Any, qubits: List[int]):
        """Add a Qiskit instruction to UnifiedCircuit"""
        name = instruction.name.upper()
        
        # Map Qiskit gate names to GateTypes
        gate_map = {
            'H': GateType.H,
            'X': GateType.X,
            'Y': GateType.Y,
            'Z': GateType.Z,
            'S': GateType.S,
            'T': GateType.T,
            'CX': GateType.CNOT,
            'CNOT': GateType.CNOT,
            'CZ': GateType.CZ,
            'SWAP': GateType.SWAP,
        }
        
        if name in gate_map:
            circuit.add_gate(gate_map[name], targets=qubits)
        
        # Rotation gates
        elif name == 'RX':
            angle = instruction.params[0] if instruction.params else 0
            circuit.add_gate(GateType.RX, targets=qubits, parameters={'angle': angle})
        elif name == 'RY':
            angle = instruction.params[0] if instruction.params else 0
            circuit.add_gate(GateType.RY, targets=qubits, parameters={'angle': angle})
        elif name == 'RZ':
            angle = instruction.params[0] if instruction.params else 0
            circuit.add_gate(GateType.RZ, targets=qubits, parameters={'angle': angle})


class IonQNativeConverter:
    """
    Convert UnifiedCircuit to IonQ native gate format
    
    IonQ uses native gates:
    - GPI(φ): Single-qubit rotation
    - GPI2(φ): Half rotation
    - MS(φ₀, φ₁): Two-qubit Mølmer-Sørensen gate
    
    This converter can decompose standard gates into native gates
    for optimal performance on IonQ hardware.
    """
    
    @staticmethod
    def to_ionq_native(
        unified_circuit: UnifiedCircuit,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Convert UnifiedCircuit to IonQ native gate JSON format
        
        Args:
            unified_circuit: Circuit to convert
            optimize: Whether to optimize for native gates
        
        Returns:
            Dictionary in IonQ JSON format
        """
        if optimize:
            # Decompose standard gates to native gates
            native_circuit = IonQNativeConverter._decompose_to_native(unified_circuit)
        else:
            native_circuit = unified_circuit
        
        # Build IonQ JSON format
        ionq_json = {
            "qubits": native_circuit.n_qubits,
            "circuit": []
        }
        
        for gate in native_circuit.gates:
            ionq_gate = IonQNativeConverter._gate_to_ionq_json(gate)
            if ionq_gate:
                ionq_json["circuit"].append(ionq_gate)
        
        return ionq_json
    
    @staticmethod
    def _decompose_to_native(circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Decompose standard gates to IonQ native gates
        
        Decomposition rules:
        - H = GPI2(0) + GPI(π/2)
        - X = GPI(0)
        - Y = GPI(π/2)
        - Z = GPI(0) + GPI(π)
        - RX(θ) = GPI2(0) + GPI(θ) + GPI2(0)
        - RY(θ) = GPI2(π/2) + GPI(θ) + GPI2(π/2)
        - RZ(θ) = ... (phase rotation)
        - CNOT = MS + single qubit corrections
        """
        native = UnifiedCircuit(n_qubits=circuit.n_qubits)
        
        for gate in circuit.gates:
            if gate.gate_type in [GateType.GPI, GateType.GPI2, GateType.MS]:
                # Already native
                native.gates.append(gate)
            
            elif gate.gate_type == GateType.H:
                # H = GPI2(0) + GPI(π/2)
                native.add_gate(GateType.GPI2, targets=gate.targets, parameters={'phi': 0})
                native.add_gate(GateType.GPI, targets=gate.targets, parameters={'phi': np.pi/2})
            
            elif gate.gate_type == GateType.X:
                # X = GPI(0)
                native.add_gate(GateType.GPI, targets=gate.targets, parameters={'phi': 0})
            
            elif gate.gate_type == GateType.Y:
                # Y = GPI(π/2)
                native.add_gate(GateType.GPI, targets=gate.targets, parameters={'phi': np.pi/2})
            
            elif gate.gate_type == GateType.CNOT:
                # CNOT decomposition using MS gate
                q0, q1 = gate.targets
                # Simplified decomposition (actual decomposition is more complex)
                native.add_gate(GateType.GPI2, targets=[q0], parameters={'phi': 0})
                native.add_gate(GateType.MS, targets=[q0, q1], parameters={'angle': np.pi/4})
                native.add_gate(GateType.GPI2, targets=[q0], parameters={'phi': 0})
            
            else:
                # For other gates, keep as-is (will be handled by IonQ)
                native.gates.append(gate)
        
        return native
    
    @staticmethod
    def _gate_to_ionq_json(gate: Gate) -> Optional[Dict[str, Any]]:
        """Convert a gate to IonQ JSON format"""
        if gate.gate_type == GateType.GPI:
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            return {
                "gate": "gpi",
                "targets": gate.targets,
                "phase": float(phi)
            }
        
        elif gate.gate_type == GateType.GPI2:
            phi = gate.parameters.get('phi', 0) if gate.parameters else 0
            return {
                "gate": "gpi2",
                "targets": gate.targets,
                "phase": float(phi)
            }
        
        elif gate.gate_type == GateType.MS:
            angle = gate.parameters.get('angle', np.pi/4) if gate.parameters else np.pi/4
            return {
                "gate": "ms",
                "targets": gate.targets,
                "angle": float(angle)
            }
        
        # For non-native gates, use standard format
        elif gate.gate_type == GateType.CNOT:
            return {
                "gate": "cnot",
                "control": gate.targets[0],
                "target": gate.targets[1]
            }
        
        return None


# Convenience functions for the UnifiedCircuit class
def add_converters_to_unified_circuit():
    """Add conversion methods to UnifiedCircuit class"""
    
    def to_cirq(self) -> 'cirq.Circuit':
        """Convert to Cirq circuit"""
        return CirqConverter.to_cirq(self)
    
    def to_qiskit(self) -> 'QuantumCircuit':
        """Convert to Qiskit circuit"""
        return QiskitConverter.to_qiskit(self)
    
    def to_ionq_native(self, optimize: bool = True) -> Dict[str, Any]:
        """Convert to IonQ native gates"""
        return IonQNativeConverter.to_ionq_native(self, optimize=optimize)
    
    @classmethod
    def from_cirq(cls, circuit: 'cirq.Circuit') -> 'UnifiedCircuit':
        """Create from Cirq circuit"""
        return CirqConverter.from_cirq(circuit)
    
    @classmethod
    def from_qiskit(cls, circuit: 'QuantumCircuit') -> 'UnifiedCircuit':
        """Create from Qiskit circuit"""
        return QiskitConverter.from_qiskit(circuit)
    
    # Add methods to UnifiedCircuit
    UnifiedCircuit.to_cirq = to_cirq
    UnifiedCircuit.to_qiskit = to_qiskit
    UnifiedCircuit.to_ionq_native = to_ionq_native
    UnifiedCircuit.from_cirq = from_cirq
    UnifiedCircuit.from_qiskit = from_qiskit


# Auto-register converters when module is imported
add_converters_to_unified_circuit()
