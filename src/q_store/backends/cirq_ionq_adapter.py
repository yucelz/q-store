"""
Cirq-based IonQ Backend Adapter
Implements the QuantumBackend interface using Cirq and cirq-ionq
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import os

from .quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    QuantumGate,
    ExecutionResult,
    BackendCapabilities,
    BackendType,
    GateType
)

logger = logging.getLogger(__name__)


class CirqIonQBackend(QuantumBackend):
    """
    IonQ quantum backend using Cirq SDK

    This adapter converts our internal circuit representation to Cirq circuits
    and executes them on IonQ hardware via cirq-ionq.
    """

    def __init__(
        self,
        api_key: str,
        target: str = 'simulator',
        noise_model: Optional[str] = None
    ):
        """
        Initialize Cirq-IonQ backend

        Args:
            api_key: IonQ API key from cloud.ionq.com
            target: Backend target ('simulator', 'qpu.aria-1', 'qpu.forte-1', etc.)
            noise_model: Optional noise model for simulator ('aria-1', 'forte-1', etc.)
        """
        self.api_key = api_key
        self.target = target
        self.noise_model = noise_model
        self._service = None
        self._initialized = False

        # Gate mapping
        self._gate_map = {
            GateType.HADAMARD: 'H',
            GateType.PAULI_X: 'X',
            GateType.PAULI_Y: 'Y',
            GateType.PAULI_Z: 'Z',
            GateType.RX: 'rx',
            GateType.RY: 'ry',
            GateType.RZ: 'rz',
            GateType.CNOT: 'CNOT',
            GateType.CZ: 'CZ',
            GateType.SWAP: 'SWAP',
            GateType.S: 'S',
            GateType.T: 'T',
        }

    async def initialize(self) -> None:
        """Initialize Cirq-IonQ service"""
        if self._initialized:
            return

        try:
            import cirq
            import cirq_ionq as ionq

            self._service = ionq.Service(api_key=self.api_key)
            self._initialized = True
            logger.info(f"Initialized Cirq-IonQ backend: {self.target}")

        except ImportError:
            raise ImportError(
                "cirq-ionq not installed. Install with: pip install cirq-ionq"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cirq-IonQ backend: {e}")

    def _convert_to_cirq(self, circuit: QuantumCircuit) -> 'cirq.Circuit':
        """
        Convert internal QuantumCircuit to Cirq circuit

        Args:
            circuit: Internal circuit representation

        Returns:
            Cirq Circuit object
        """
        import cirq

        # Create qubits
        qubits = cirq.LineQubit.range(circuit.n_qubits)
        cirq_circuit = cirq.Circuit()

        # Convert gates
        for gate in circuit.gates:
            cirq_op = self._gate_to_cirq(gate, qubits)
            if cirq_op is not None:
                cirq_circuit.append(cirq_op)

        return cirq_circuit

    def _gate_to_cirq(
        self,
        gate: QuantumGate,
        qubits: List['cirq.LineQubit']
    ) -> Optional['cirq.Operation']:
        """
        Convert a single gate to Cirq operation

        Args:
            gate: Internal gate representation
            qubits: List of Cirq qubits

        Returns:
            Cirq Operation or None for measurements
        """
        import cirq

        gate_qubits = [qubits[i] for i in gate.qubits]

        # Handle measurement separately
        if gate.gate_type == GateType.MEASURE:
            # Cirq measurements added at the end
            return None

        # Single-qubit gates
        if gate.gate_type == GateType.HADAMARD:
            return cirq.H(*gate_qubits)
        elif gate.gate_type == GateType.PAULI_X:
            return cirq.X(*gate_qubits)
        elif gate.gate_type == GateType.PAULI_Y:
            return cirq.Y(*gate_qubits)
        elif gate.gate_type == GateType.PAULI_Z:
            return cirq.Z(*gate_qubits)
        elif gate.gate_type == GateType.S:
            return cirq.S(*gate_qubits)
        elif gate.gate_type == GateType.T:
            return cirq.T(*gate_qubits)

        # Rotation gates
        elif gate.gate_type == GateType.RX:
            angle = gate.parameters['angle']
            return cirq.rx(angle)(*gate_qubits)
        elif gate.gate_type == GateType.RY:
            angle = gate.parameters['angle']
            return cirq.ry(angle)(*gate_qubits)
        elif gate.gate_type == GateType.RZ:
            angle = gate.parameters['angle']
            return cirq.rz(angle)(*gate_qubits)

        # Two-qubit gates
        elif gate.gate_type == GateType.CNOT:
            return cirq.CNOT(*gate_qubits)
        elif gate.gate_type == GateType.CZ:
            return cirq.CZ(*gate_qubits)
        elif gate.gate_type == GateType.SWAP:
            return cirq.SWAP(*gate_qubits)

        # Multi-qubit gates
        else:
            logger.warning(f"Unsupported gate type: {gate.gate_type}")
            return None

    async def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute circuit on IonQ hardware

        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
            **kwargs: Additional options (name, etc.)

        Returns:
            ExecutionResult with measurement outcomes
        """
        if not self._initialized:
            await self.initialize()

        import cirq

        # Convert to Cirq circuit
        cirq_circuit = self._convert_to_cirq(circuit)

        # Add measurements
        qubits = cirq.LineQubit.range(circuit.n_qubits)
        cirq_circuit.append(cirq.measure(*qubits, key='result'))

        try:
            # Create job
            job = self._service.create_job(
                circuit=cirq_circuit,
                repetitions=shots,
                target=self.target,
                name=kwargs.get('name', 'quantum_db_query')
            )

            # Wait for results
            results = job.results()

            # Handle both list and single result
            # IonQ returns a list of results, take the first one
            if isinstance(results, list):
                if len(results) == 0:
                    raise ValueError("No results returned from IonQ job")
                cirq_result = results[0]
            else:
                cirq_result = results

            # Convert to our format
            return self._convert_result(cirq_result, shots, circuit)

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

    def _convert_result(
        self,
        cirq_result,
        total_shots: int,
        original_circuit: QuantumCircuit
    ) -> ExecutionResult:
        """Convert Cirq result to ExecutionResult"""
        import cirq
        import numpy as np

        # Handle different result formats
        measurements = None

        # Try to get measurements in various ways
        if hasattr(cirq_result, 'measurement_dict'):
            # IonQ SimulatorResult format - has measurement_dict
            measurement_dict = cirq_result.measurement_dict
            if isinstance(measurement_dict, dict) and 'result' in measurement_dict:
                measurements = measurement_dict['result']
            elif hasattr(cirq_result, 'probabilities') and callable(cirq_result.probabilities):
                # probabilities is a method, not a property
                try:
                    probs = cirq_result.probabilities()
                    # Convert probabilities to counts
                    counts = {}
                    for bitstring, prob in probs.items():
                        counts[bitstring] = int(prob * total_shots)
                    probabilities = probs

                    return ExecutionResult(
                        counts=counts,
                        probabilities=probabilities,
                        total_shots=total_shots,
                        metadata={
                            'backend': 'cirq_ionq',
                            'target': self.target,
                            'noise_model': self.noise_model
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not call probabilities() method: {e}")
                    measurements = None
        elif hasattr(cirq_result, 'measurements'):
            # Standard Cirq result object
            measurements_dict = cirq_result.measurements
            if isinstance(measurements_dict, dict):
                measurements = measurements_dict.get('result', None)
            else:
                measurements = measurements_dict
        elif hasattr(cirq_result, 'data'):
            # Alternative format
            measurements = cirq_result.data.get('result', None)
        elif isinstance(cirq_result, dict):
            # Direct dictionary
            measurements = cirq_result.get('result', None)

        if measurements is None:
            # Fallback: try to extract from any available attribute
            logger.warning(f"Unexpected result format: {type(cirq_result)}, using uniform distribution fallback")
            # Create uniform distribution as fallback
            n_qubits = original_circuit.n_qubits
            counts = {format(i, f'0{n_qubits}b'): total_shots // (2**n_qubits) for i in range(2**n_qubits)}
            probabilities = {k: v / total_shots for k, v in counts.items()}
        else:
            # Count outcomes from measurements
            counts = {}
            measurements_array = np.array(measurements) if not isinstance(measurements, np.ndarray) else measurements

            for measurement in measurements_array:
                bitstring = ''.join(str(int(b)) for b in measurement)
                counts[bitstring] = counts.get(bitstring, 0) + 1

            # Calculate probabilities
            probabilities = {k: v / total_shots for k, v in counts.items()}

        return ExecutionResult(
            counts=counts,
            probabilities=probabilities,
            total_shots=total_shots,
            metadata={
                'backend': 'cirq_ionq',
                'target': self.target,
                'noise_model': self.noise_model
            }
        )

    def get_capabilities(self) -> BackendCapabilities:
        """Get IonQ backend capabilities"""
        # IonQ capabilities vary by target
        if 'simulator' in self.target:
            max_qubits = 29
            backend_type = BackendType.SIMULATOR
        elif 'aria' in self.target:
            max_qubits = 25
            backend_type = BackendType.QPU
        elif 'forte' in self.target:
            max_qubits = 32
            backend_type = BackendType.QPU
        else:
            max_qubits = 11
            backend_type = BackendType.QPU

        return BackendCapabilities(
            max_qubits=max_qubits,
            supported_gates=[
                GateType.HADAMARD, GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z,
                GateType.RX, GateType.RY, GateType.RZ,
                GateType.CNOT, GateType.CZ, GateType.SWAP,
                GateType.S, GateType.T, GateType.MEASURE
            ],
            backend_type=backend_type,
            supports_mid_circuit_measurement=False,
            supports_reset=False,
            max_shots=10000,
            native_gate_set=[GateType.RX, GateType.RY, GateType.RZ, GateType.CNOT],
            connectivity=None  # All-to-all connectivity
        )

    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            'provider': 'IonQ',
            'sdk': 'cirq',
            'target': self.target,
            'version': self._get_version(),
            'backend_type': self._get_backend_type().value
        }

    def _get_backend_type(self) -> BackendType:
        """Determine backend type from target"""
        if 'simulator' in self.target:
            return BackendType.SIMULATOR
        else:
            return BackendType.QPU

    def _get_version(self) -> str:
        """Get Cirq version"""
        try:
            import cirq
            return cirq.__version__
        except:
            return "unknown"

    async def close(self) -> None:
        """Close connection"""
        self._service = None
        self._initialized = False
        logger.info("Closed Cirq-IonQ backend")

    def is_available(self) -> bool:
        """Check if backend is available"""
        # Would need to query IonQ API for actual status
        return self._initialized

    def estimate_cost(self, circuit: QuantumCircuit, shots: int) -> float:
        """Estimate execution cost"""
        if 'simulator' in self.target:
            return 0.0
        else:
            # IonQ pricing: ~$0.01 per gate-shot
            # Simplified estimate
            n_gates = len([g for g in circuit.gates if g.gate_type != GateType.MEASURE])
            return (n_gates * shots) * 0.00001  # $0.01 per 1000 gate-shots

    # v3.3 NEW: Async job submission methods
    async def submit_job_async(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        **kwargs
    ) -> str:
        """
        Submit job without waiting for completion (v3.3)

        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
            **kwargs: Additional options

        Returns:
            Job ID for later retrieval
        """
        if not self._initialized:
            await self.initialize()

        import cirq

        # Convert to Cirq circuit
        cirq_circuit = self._convert_to_cirq(circuit)

        # Add measurements
        qubits = cirq.LineQubit.range(circuit.n_qubits)
        cirq_circuit.append(cirq.measure(*qubits, key='result'))

        try:
            # Create job (non-blocking)
            job = self._service.create_job(
                circuit=cirq_circuit,
                repetitions=shots,
                target=self.target,
                name=kwargs.get('name', f'quantum_ml_{asyncio.current_task().get_name()}')
            )

            # Return job ID immediately (don't wait)
            job_id = job.job_id()
            logger.debug(f"Submitted job: {job_id}")

            return job_id

        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            raise

    async def check_job_status(self, job_id: str) -> str:
        """
        Check job status without fetching results (v3.3)

        Args:
            job_id: Job identifier

        Returns:
            Status string ('submitted', 'running', 'completed', 'failed')
        """
        if not self._initialized:
            await self.initialize()

        try:
            job = self._service.get_job(job_id)

            # Get status (should be fast)
            # IonQ API changed: use status() instead of execution_status()
            status = job.status()

            # Map IonQ status to our status
            # status() returns a string directly (e.g., 'completed', 'running', 'failed')
            ionq_status = str(status).lower()

            if 'complete' in ionq_status or 'success' in ionq_status:
                return 'completed'
            elif 'fail' in ionq_status or 'error' in ionq_status or 'cancel' in ionq_status:
                return 'failed'
            elif 'running' in ionq_status:
                return 'running'
            else:
                return 'submitted'

        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return 'failed'

    async def get_job_result(
        self,
        job_id: str,
        original_circuit: Optional[QuantumCircuit] = None
    ) -> ExecutionResult:
        """
        Fetch result for a completed job (v3.3)

        Args:
            job_id: Job identifier
            original_circuit: Original circuit (for metadata)

        Returns:
            ExecutionResult with measurements
        """
        if not self._initialized:
            await self.initialize()

        try:
            job = self._service.get_job(job_id)

            # This will wait if job not complete
            results = job.results()

            # Handle both list and single result
            if isinstance(results, list):
                if len(results) == 0:
                    raise ValueError("No results returned from IonQ job")
                cirq_result = results[0]
            else:
                cirq_result = results

            # Get shots from job metadata if available
            shots = 1000  # default
            if hasattr(job, 'repetitions'):
                shots = job.repetitions()

            # Convert to our format
            return self._convert_result(cirq_result, shots, original_circuit)

        except Exception as e:
            logger.error(f"Error fetching job result: {e}")
            raise

    def execute_circuit_sync(
        self,
        circuit,
        shots: int = 1000
    ) -> ExecutionResult:
        """
        Synchronous circuit execution (for batch manager fallback)

        Args:
            circuit: Native Cirq circuit or QuantumCircuit
            shots: Number of shots

        Returns:
            ExecutionResult
        """
        # Run async method in sync context
        import asyncio
        loop = asyncio.get_event_loop()
        if isinstance(circuit, QuantumCircuit):
            return loop.run_until_complete(self.execute_circuit(circuit, shots))
        else:
            # Already native format - convert back
            # This is a simplified fallback
            logger.warning("Sync execution with native circuit - may not work correctly")
            raise NotImplementedError("Sync execution with native circuit not fully supported")
