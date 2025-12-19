"""
PennyLane Lightning GPU-Accelerated Quantum Simulator Backend.

This module integrates PennyLane's Lightning simulator for high-performance
quantum circuit simulation with GPU acceleration support.

Lightning features:
- GPU-accelerated state vector simulation (CUDA)
- Highly optimized C++ backend
- Automatic CPU fallback when GPU unavailable
- Integration with PennyLane ecosystem
- Target: 70-90% GPU utilization
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

from ..core import UnifiedCircuit, GateType
from .quantum_backend_interface import (
    QuantumBackend,
    BackendType,
    BackendCapabilities,
    ExecutionResult
)

logger = logging.getLogger(__name__)


class LightningBackend(QuantumBackend):
    """
    GPU-accelerated quantum simulator using PennyLane Lightning.

    Lightning provides high-performance state vector simulation with:
    - CUDA GPU acceleration for massive speedups
    - Highly optimized C++ implementation
    - Automatic fallback to CPU when GPU unavailable
    - Efficient memory management
    - Support for large-scale circuits

    Performance characteristics:
    - Best for: 10-30 qubit circuits with GPU
    - GPU Speedup: 10-100x vs CPU for large circuits
    - Memory: 2^n complex numbers (requires GPU memory)
    - Target GPU utilization: 70-90%

    Args:
        device: Device to use ('lightning.gpu' or 'lightning.qubit')
        wires: Number of qubits/wires (auto-detected from circuit if None)
        shots: Default number of shots for measurements

    Example:
        >>> backend = LightningBackend(device='lightning.gpu')
        >>> result = backend.execute(circuit, shots=1000)
        >>> print(f"GPU utilized: {result.metadata['gpu_used']}")
    """

    def __init__(
        self,
        device: str = 'lightning.qubit',  # Default to CPU, will try GPU
        wires: Optional[int] = None,
        shots: int = 1000
    ):
        if not HAS_PENNYLANE:
            raise ImportError(
                "PennyLane is required for LightningBackend. Install with: "
                "pip install pennylane pennylane-lightning"
            )

        self.device_name = device
        self.n_wires = wires
        self.default_shots = shots
        self.dev = None
        self._gpu_available = False

        # Try to detect GPU availability
        if device == 'lightning.gpu' or device == 'auto':
            try:
                # Attempt to create GPU device
                test_dev = qml.device('lightning.gpu', wires=2)
                self._gpu_available = True
                self.device_name = 'lightning.gpu'
                logger.info("Lightning GPU device available")
            except Exception as e:
                logger.warning(f"GPU device unavailable ({e}), falling back to CPU")
                self.device_name = 'lightning.qubit'
                self._gpu_available = False

        self._initialized = True
        logger.info(f"LightningBackend initialized (device={self.device_name})")

    def _create_device(self, n_qubits: int, shots: Optional[int] = None) -> Any:
        """
        Create PennyLane device for given number of qubits.

        Args:
            n_qubits: Number of qubits
            shots: Number of shots (None for exact simulation)

        Returns:
            PennyLane device
        """
        return qml.device(
            self.device_name,
            wires=n_qubits,
            shots=shots
        )

    def get_capabilities(self) -> BackendCapabilities:
        """Get Lightning backend capabilities."""
        return BackendCapabilities(
            backend_type=BackendType.SIMULATOR,
            max_qubits=30 if self._gpu_available else 25,
            supports_batching=True,
            supports_shots=True,
            supports_state_vector=True,
            supports_density_matrix=False,
            native_gates=[
                GateType.HADAMARD,
                GateType.PAULI_X,
                GateType.PAULI_Y,
                GateType.PAULI_Z,
                GateType.S,
                GateType.T,
                GateType.RX,
                GateType.RY,
                GateType.RZ,
                GateType.CNOT,
                GateType.CZ,
                GateType.SWAP,
            ]
        )

    def _unified_to_pennylane(self, circuit: UnifiedCircuit) -> List:
        """
        Convert UnifiedCircuit to PennyLane operations.

        Args:
            circuit: UnifiedCircuit to convert

        Returns:
            List of PennyLane operations
        """
        ops = []

        # Gate mapping from UnifiedCircuit to PennyLane
        gate_map = {
            GateType.HADAMARD: qml.Hadamard,
            GateType.PAULI_X: qml.PauliX,
            GateType.PAULI_Y: qml.PauliY,
            GateType.PAULI_Z: qml.PauliZ,
            GateType.S: qml.S,
            GateType.T: qml.T,
            GateType.RX: qml.RX,
            GateType.RY: qml.RY,
            GateType.RZ: qml.RZ,
            GateType.CNOT: qml.CNOT,
            GateType.CZ: qml.CZ,
            GateType.SWAP: qml.SWAP,
        }

        for gate in circuit.gates:
            gate_type = gate.gate_type
            qubits = gate.qubits
            params = gate.parameters if gate.parameters else []

            if gate_type in gate_map:
                pl_gate = gate_map[gate_type]

                if params:
                    # Parameterized gate
                    ops.append(pl_gate(params[0], wires=qubits[0]))
                elif len(qubits) == 1:
                    # Single-qubit gate
                    ops.append(pl_gate(wires=qubits[0]))
                elif len(qubits) == 2:
                    # Two-qubit gate
                    ops.append(pl_gate(wires=qubits))
                else:
                    logger.warning(f"Unsupported gate configuration: {gate}")
            else:
                logger.warning(f"Gate type {gate_type} not supported in PennyLane conversion")

        return ops

    def execute(
        self,
        circuit: UnifiedCircuit,
        shots: int = 1000,
        parameters: Optional[Dict[str, float]] = None
    ) -> ExecutionResult:
        """
        Execute quantum circuit using Lightning simulator.

        Args:
            circuit: UnifiedCircuit to execute
            shots: Number of measurement shots
            parameters: Parameter values for parameterized circuits

        Returns:
            ExecutionResult containing samples and metadata
        """
        if not self._initialized:
            raise RuntimeError("LightningBackend not initialized")

        # Bind parameters if provided
        if parameters:
            circuit = circuit.bind_parameters(parameters)

        # Create device
        dev = self._create_device(circuit.n_qubits, shots=shots)

        # Convert circuit to PennyLane operations
        pl_ops = self._unified_to_pennylane(circuit)

        # Create QNode for execution
        @qml.qnode(dev)
        def circuit_func():
            for op in pl_ops:
                qml.apply(op)
            return qml.sample()

        # Execute
        try:
            samples_raw = circuit_func()

            # Convert samples to numpy array
            if shots == 1:
                samples = np.array([samples_raw])
            else:
                samples = np.array(samples_raw)

            # Ensure correct shape
            if len(samples.shape) == 1:
                samples = samples.reshape(shots, -1)

            # Calculate statistics
            unique_states, counts = np.unique(samples, axis=0, return_counts=True)
            probabilities = counts / shots

            # Convert to bitstring format
            bitstring_counts = {}
            bitstring_probs = {}
            for state, count, prob in zip(unique_states, counts, probabilities):
                bitstring = ''.join(map(str, state.astype(int)))
                bitstring_counts[bitstring] = int(count)
                bitstring_probs[bitstring] = float(prob)

            return ExecutionResult(
                samples=samples,
                counts=bitstring_counts,
                probabilities=bitstring_probs,
                metadata={
                    'backend': 'lightning',
                    'device': self.device_name,
                    'shots': shots,
                    'num_qubits': circuit.n_qubits,
                    'gpu_used': self._gpu_available
                }
            )
        except Exception as e:
            logger.error(f"Lightning execution failed: {e}")
            raise

    def execute_batch(
        self,
        circuits: List[UnifiedCircuit],
        shots: int = 1000,
        parameters: Optional[List[Dict[str, float]]] = None
    ) -> List[ExecutionResult]:
        """
        Execute multiple circuits in batch.

        Lightning supports efficient sequential execution.

        Args:
            circuits: List of UnifiedCircuits to execute
            shots: Number of shots per circuit
            parameters: Optional list of parameter dictionaries

        Returns:
            List of ExecutionResults
        """
        if parameters is None:
            parameters = [None] * len(circuits)

        results = []
        for circuit, params in zip(circuits, parameters):
            result = self.execute(circuit, shots=shots, parameters=params)
            results.append(result)

        return results

    def get_state_vector(self, circuit: UnifiedCircuit) -> np.ndarray:
        """
        Get the state vector after circuit execution.

        Args:
            circuit: UnifiedCircuit to execute

        Returns:
            Complex state vector of shape (2^n,)
        """
        if not self._initialized:
            raise RuntimeError("LightningBackend not initialized")

        # Create device without shots for exact simulation
        dev = self._create_device(circuit.n_qubits, shots=None)

        # Convert circuit
        pl_ops = self._unified_to_pennylane(circuit)

        # Create QNode
        @qml.qnode(dev)
        def circuit_func():
            for op in pl_ops:
                qml.apply(op)
            return qml.state()

        # Execute and get state
        state = circuit_func()
        return np.array(state)

    def compute_expectation(
        self,
        circuit: UnifiedCircuit,
        observable: Any,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute expectation value of an observable.

        Args:
            circuit: UnifiedCircuit to execute
            observable: Observable (PauliZ, PauliX, etc.)
            parameters: Optional parameter values

        Returns:
            Expectation value
        """
        if parameters:
            circuit = circuit.bind_parameters(parameters)

        # Create device
        dev = self._create_device(circuit.n_qubits, shots=None)

        # Convert circuit
        pl_ops = self._unified_to_pennylane(circuit)

        # Default to Z observable on first qubit
        if isinstance(observable, str) and observable == 'Z':
            obs = qml.PauliZ(0)
        else:
            obs = observable

        # Create QNode
        @qml.qnode(dev)
        def circuit_func():
            for op in pl_ops:
                qml.apply(op)
            return qml.expval(obs)

        # Execute
        expectation = circuit_func()
        return float(expectation)

    def reset(self):
        """Reset the backend state."""
        # Lightning is stateless
        pass

    def close(self):
        """Close backend resources."""
        self._initialized = False
        self.dev = None
        logger.info("Lightning backend closed")


def create_lightning_backend(use_gpu: bool = True) -> LightningBackend:
    """
    Factory function to create a Lightning backend.

    Args:
        use_gpu: Whether to attempt GPU acceleration

    Returns:
        Configured LightningBackend instance

    Example:
        >>> backend = create_lightning_backend(use_gpu=True)
        >>> result = backend.execute(circuit, shots=1000)
    """
    device = 'lightning.gpu' if use_gpu else 'lightning.qubit'
    return LightningBackend(device=device)
