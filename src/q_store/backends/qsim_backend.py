"""
Google qsim High-Performance Quantum Simulator Backend.

This module integrates Google's qsim simulator for high-performance quantum
circuit simulation with multi-threading support.

qsim features:
- Optimized C++ state vector simulator
- OpenMP multi-threading support
- 3-5x faster than standard simulators on moderate circuits
- Integration with Cirq ecosystem
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

try:
    import cirq
    import qsimcirq
    HAS_QSIM = True
except ImportError:
    HAS_QSIM = False

from ..core import UnifiedCircuit, GateType
from .quantum_backend_interface import (
    QuantumBackend,
    BackendType,
    BackendCapabilities,
    ExecutionResult
)

logger = logging.getLogger(__name__)


class QsimBackend(QuantumBackend):
    """
    High-performance quantum simulator using Google's qsim.

    qsim is a state vector simulator optimized for speed using:
    - AVX/AVX2/AVX-512 vectorization
    - OpenMP parallel execution
    - Optimized gate fusion
    - Cache-efficient data structures

    Performance characteristics:
    - Best for: 4-25 qubit circuits with moderate depth
    - Speedup: 3-5x vs standard Cirq simulator
    - Memory: 2^n complex numbers (16 bytes per amplitude)
    - Threads: Configurable via num_threads parameter

    Args:
        num_threads: Number of OpenMP threads (default: auto-detect)
        use_gpu: Whether to use GPU acceleration (requires qsim-cuda)
        verbosity: Logging verbosity (0=silent, 1=normal, 2=verbose)

    Example:
        >>> backend = QsimBackend(num_threads=4)
        >>> result = backend.execute(circuit, shots=1000)
        >>> print(f"Samples: {result.samples}")
    """

    def __init__(
        self,
        num_threads: Optional[int] = None,
        use_gpu: bool = False,
        verbosity: int = 0
    ):
        if not HAS_QSIM:
            raise ImportError(
                "qsim is required for QsimBackend. Install with: "
                "pip install qsimcirq"
            )

        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.verbosity = verbosity

        # Create qsim simulator options
        self.qsim_options = qsimcirq.QSimOptions(
            cpu_threads=num_threads if num_threads else 0,  # 0 = auto-detect
            verbosity=verbosity
        )

        # Create simulator instance
        if use_gpu:
            try:
                self.simulator = qsimcirq.QSimSimulator(
                    qsim_options=self.qsim_options,
                    use_gpu=True
                )
                logger.info("qsim GPU simulator initialized")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
                self.simulator = qsimcirq.QSimSimulator(qsim_options=self.qsim_options)
        else:
            self.simulator = qsimcirq.QSimSimulator(qsim_options=self.qsim_options)
            logger.info(f"qsim CPU simulator initialized (threads={self.num_threads or 'auto'})")

        self._initialized = True

    def get_capabilities(self) -> BackendCapabilities:
        """Get qsim backend capabilities."""
        return BackendCapabilities(
            backend_type=BackendType.SIMULATOR,
            max_qubits=30,  # qsim can handle up to ~30 qubits on standard hardware
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

    def _unified_to_cirq(self, circuit: UnifiedCircuit) -> cirq.Circuit:
        """
        Convert UnifiedCircuit to Cirq circuit for qsim execution.

        Args:
            circuit: UnifiedCircuit to convert

        Returns:
            Cirq circuit compatible with qsim
        """
        # Use the built-in conversion method from UnifiedCircuit
        return circuit.to_cirq()

    def execute(
        self,
        circuit: UnifiedCircuit,
        shots: int = 1000,
        parameters: Optional[Dict[str, float]] = None
    ) -> ExecutionResult:
        """
        Execute quantum circuit using qsim simulator.

        Args:
            circuit: UnifiedCircuit to execute
            shots: Number of measurement shots
            parameters: Parameter values for parameterized circuits

        Returns:
            ExecutionResult containing samples and metadata
        """
        if not self._initialized:
            raise RuntimeError("QsimBackend not initialized")

        # Bind parameters if provided
        if parameters:
            circuit = circuit.bind_parameters(parameters)

        # Convert to Cirq circuit
        cirq_circuit = self._unified_to_cirq(circuit)

        # Add measurements if not present
        qubits = sorted(cirq_circuit.all_qubits())
        if not any(isinstance(op.gate, cirq.MeasurementGate)
                  for moment in cirq_circuit for op in moment):
            cirq_circuit.append(cirq.measure(*qubits, key='result'))

        # Execute with qsim
        try:
            result = self.simulator.run(cirq_circuit, repetitions=shots)

            # Extract measurement results
            samples = result.measurements['result']

            # Calculate measurement statistics
            unique_states, counts = np.unique(samples, axis=0, return_counts=True)
            probabilities = counts / shots

            # Convert to bitstring format
            bitstrings = [''.join(map(str, state)) for state in samples]

            return ExecutionResult(
                samples=samples,
                counts=dict(zip(
                    [''.join(map(str, state)) for state in unique_states],
                    counts.tolist()
                )),
                probabilities=dict(zip(
                    [''.join(map(str, state)) for state in unique_states],
                    probabilities.tolist()
                )),
                metadata={
                    'backend': 'qsim',
                    'shots': shots,
                    'num_qubits': len(qubits),
                    'num_threads': self.num_threads,
                    'use_gpu': self.use_gpu
                }
            )
        except Exception as e:
            logger.error(f"qsim execution failed: {e}")
            raise

    def execute_batch(
        self,
        circuits: List[UnifiedCircuit],
        shots: int = 1000,
        parameters: Optional[List[Dict[str, float]]] = None
    ) -> List[ExecutionResult]:
        """
        Execute multiple circuits in batch.

        Note: qsim doesn't have native batch support, so we execute sequentially.
        However, multi-threading within each circuit provides good performance.

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
            raise RuntimeError("QsimBackend not initialized")

        # Convert to Cirq circuit (without measurements)
        cirq_circuit = self._unified_to_cirq(circuit)

        # Remove any measurement gates
        cirq_circuit = cirq.Circuit([
            moment for moment in cirq_circuit
            if not any(isinstance(op.gate, cirq.MeasurementGate) for op in moment)
        ])

        # Simulate to get state vector
        result = self.simulator.simulate(cirq_circuit)

        return result.state_vector()

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
            observable: Observable operator (Cirq-compatible)
            parameters: Optional parameter values

        Returns:
            Expectation value
        """
        if parameters:
            circuit = circuit.bind_parameters(parameters)

        # Get state vector
        state = self.get_state_vector(circuit)

        # If observable is a Pauli string, compute expectation
        if isinstance(observable, str):
            # Simple Z measurement on all qubits
            if observable == 'Z':
                # Compute <ψ|Z|ψ> for first qubit
                n_qubits = circuit.n_qubits
                z_matrix = np.array([[1, 0], [0, -1]])
                identity = np.eye(2)

                # Build Z ⊗ I ⊗ I ⊗ ... operator
                operator = z_matrix
                for _ in range(n_qubits - 1):
                    operator = np.kron(operator, identity)

                expectation = np.real(np.vdot(state, operator @ state))
                return float(expectation)

        # For more complex observables, would need proper implementation
        raise NotImplementedError(f"Observable type {type(observable)} not yet supported")

    def reset(self):
        """Reset the backend state."""
        # qsim is stateless, no reset needed
        pass

    def close(self):
        """Close backend resources."""
        self._initialized = False
        logger.info("qsim backend closed")


def create_qsim_backend(num_threads: Optional[int] = None, use_gpu: bool = False) -> QsimBackend:
    """
    Factory function to create a qsim backend.

    Args:
        num_threads: Number of OpenMP threads (None = auto-detect)
        use_gpu: Whether to attempt GPU acceleration

    Returns:
        Configured QsimBackend instance

    Example:
        >>> backend = create_qsim_backend(num_threads=4)
        >>> result = backend.execute(circuit, shots=1000)
    """
    return QsimBackend(num_threads=num_threads, use_gpu=use_gpu)
