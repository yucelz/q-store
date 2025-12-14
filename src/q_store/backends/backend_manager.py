"""
Backend Manager and Mock Backend
Manages multiple quantum backends and provides testing infrastructure
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
from collections import Counter

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


class BackendManager:
    """
    Manages multiple quantum backends with plugin architecture
    Allows dynamic registration and selection of backends
    """

    def __init__(self):
        self._backends: Dict[str, QuantumBackend] = {}
        self._default_backend: Optional[str] = None
        self._backend_metadata: Dict[str, Dict[str, Any]] = {}

    def register_backend(
        self,
        name: str,
        backend: QuantumBackend,
        set_as_default: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new backend

        Args:
            name: Unique name for this backend
            backend: QuantumBackend instance
            set_as_default: Whether to set as default backend
            metadata: Optional metadata about the backend
        """
        if name in self._backends:
            logger.warning(f"Backend {name} already registered, overwriting")

        self._backends[name] = backend
        self._backend_metadata[name] = metadata or {}

        if set_as_default or self._default_backend is None:
            self._default_backend = name

        logger.info(f"Registered backend: {name}")

    def unregister_backend(self, name: str):
        """
        Unregister a backend

        Args:
            name: Name of backend to remove
        """
        if name in self._backends:
            del self._backends[name]
            del self._backend_metadata[name]

            if self._default_backend == name:
                self._default_backend = next(iter(self._backends.keys()), None)

            logger.info(f"Unregistered backend: {name}")

    def get_backend(self, name: Optional[str] = None) -> QuantumBackend:
        """
        Get backend by name or default

        Args:
            name: Backend name (uses default if None)

        Returns:
            QuantumBackend instance

        Raises:
            ValueError: If backend not found
        """
        backend_name = name or self._default_backend

        if backend_name is None:
            raise ValueError("No backends registered and no default set")

        if backend_name not in self._backends:
            available = ", ".join(self._backends.keys())
            raise ValueError(
                f"Backend '{backend_name}' not found. "
                f"Available backends: {available}"
            )

        return self._backends[backend_name]

    def list_backends(self) -> List[Dict[str, Any]]:
        """
        List all available backends with info

        Returns:
            List of backend information dicts
        """
        backends_info = []

        for name, backend in self._backends.items():
            info = backend.get_backend_info()
            info['name'] = name
            info['is_default'] = (name == self._default_backend)
            info['metadata'] = self._backend_metadata.get(name, {})
            info['available'] = backend.is_available()

            # Add capabilities summary
            caps = backend.get_capabilities()
            info['max_qubits'] = caps.max_qubits
            info['backend_type'] = caps.backend_type.value

            backends_info.append(info)

        return backends_info

    def set_default_backend(self, name: str):
        """
        Set the default backend

        Args:
            name: Backend name

        Raises:
            ValueError: If backend not found
        """
        if name not in self._backends:
            raise ValueError(f"Backend '{name}' not registered")

        self._default_backend = name
        logger.info(f"Set default backend to: {name}")

    def get_default_backend_name(self) -> Optional[str]:
        """Get name of default backend"""
        return self._default_backend

    async def initialize_all(self):
        """Initialize all registered backends"""
        for name, backend in self._backends.items():
            try:
                await backend.initialize()
                logger.info(f"Initialized backend: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize backend {name}: {e}")

    async def close_all(self):
        """Close all registered backends"""
        for name, backend in self._backends.items():
            try:
                await backend.close()
                logger.info(f"Closed backend: {name}")
            except Exception as e:
                logger.error(f"Failed to close backend {name}: {e}")

    def find_best_backend(
        self,
        circuit: QuantumCircuit,
        prefer_qpu: bool = False,
        max_cost: Optional[float] = None
    ) -> Optional[str]:
        """
        Find best backend for a circuit

        Args:
            circuit: Circuit to execute
            prefer_qpu: Prefer real QPU over simulator
            max_cost: Maximum cost constraint

        Returns:
            Name of best backend, or None if none suitable
        """
        candidates = []

        for name, backend in self._backends.items():
            if not backend.is_available():
                continue

            caps = backend.get_capabilities()

            # Check if backend can handle circuit
            if circuit.n_qubits > caps.max_qubits:
                continue

            # Check cost constraint
            if max_cost is not None:
                cost = backend.estimate_cost(circuit, 1000)
                if cost > max_cost:
                    continue

            # Score backend
            score = 0
            if caps.backend_type == BackendType.QPU:
                score += 100 if prefer_qpu else 10
            elif caps.backend_type == BackendType.SIMULATOR:
                score += 50
            elif caps.backend_type == BackendType.NOISY_SIMULATOR:
                score += 30
            else:  # MOCK
                score += 5

            # Prefer backends with more qubits
            score += caps.max_qubits

            candidates.append((name, score))

        if not candidates:
            return None

        # Return backend with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


class MockQuantumBackend(QuantumBackend):
    """
    Mock quantum backend for testing
    Simulates quantum behavior without actual quantum hardware
    """

    def __init__(
        self,
        name: str = "mock",
        max_qubits: int = 10,
        noise_level: float = 0.0
    ):
        """
        Initialize mock backend

        Args:
            name: Backend name
            max_qubits: Maximum number of qubits
            noise_level: Noise level (0.0 = ideal, 1.0 = maximum noise)
        """
        self.name = name
        self.max_qubits = max_qubits
        self.noise_level = noise_level
        self._initialized = False
        self._execution_count = 0

    async def initialize(self) -> None:
        """Initialize mock backend"""
        self._initialized = True
        logger.info(f"Mock backend '{self.name}' initialized")

    async def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute circuit with mock simulation

        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
            **kwargs: Additional options (ignored)

        Returns:
            ExecutionResult with simulated outcomes
        """
        if not self._initialized:
            await self.initialize()

        self._execution_count += 1

        # Simple simulation: generate random bitstrings weighted by circuit
        n_qubits = circuit.n_qubits

        # Count measurement gates
        measurement_qubits = set()
        for gate in circuit.gates:
            if gate.gate_type == GateType.MEASURE:
                measurement_qubits.update(gate.qubits)

        if not measurement_qubits:
            # No measurements, measure all qubits
            measurement_qubits = set(range(n_qubits))

        n_measured = len(measurement_qubits)

        # Generate outcomes
        # For mock: create somewhat realistic distribution
        # Bell state-like: more correlation than pure random
        counts = {}

        # Determine if circuit has entanglement (CNOTs)
        has_entanglement = any(
            gate.gate_type in [GateType.CNOT, GateType.CZ, GateType.SWAP]
            for gate in circuit.gates
        )

        if has_entanglement:
            # Correlated outcomes (e.g., 00 and 11 more likely)
            for _ in range(shots):
                if np.random.random() < (1 - self.noise_level) * 0.5:
                    outcome = '0' * n_measured
                elif np.random.random() < (1 - self.noise_level) * 0.5:
                    outcome = '1' * n_measured
                else:
                    # Random outcome
                    outcome = ''.join(str(np.random.randint(2)) for _ in range(n_measured))

                counts[outcome] = counts.get(outcome, 0) + 1
        else:
            # Random outcomes
            for _ in range(shots):
                outcome = ''.join(str(np.random.randint(2)) for _ in range(n_measured))
                counts[outcome] = counts.get(outcome, 0) + 1

        # Add noise
        if self.noise_level > 0:
            # Flip some bits
            noisy_counts = {}
            for outcome, count in counts.items():
                bits = list(outcome)
                for i in range(len(bits)):
                    if np.random.random() < self.noise_level * 0.1:
                        bits[i] = '1' if bits[i] == '0' else '0'
                noisy_outcome = ''.join(bits)
                noisy_counts[noisy_outcome] = noisy_counts.get(noisy_outcome, 0) + count
            counts = noisy_counts

        # Calculate probabilities
        probabilities = {k: v / shots for k, v in counts.items()}

        return ExecutionResult(
            counts=counts,
            probabilities=probabilities,
            total_shots=shots,
            metadata={
                'backend': self.name,
                'noise_level': self.noise_level,
                'execution_count': self._execution_count
            }
        )

    def get_capabilities(self) -> BackendCapabilities:
        """Get mock backend capabilities"""
        return BackendCapabilities(
            max_qubits=self.max_qubits,
            supported_gates=[gate for gate in GateType],
            backend_type=BackendType.MOCK,
            supports_mid_circuit_measurement=True,
            supports_reset=True,
            max_shots=100000,
            native_gate_set=[GateType.RX, GateType.RY, GateType.RZ, GateType.CNOT]
        )

    def get_backend_info(self) -> Dict[str, Any]:
        """Get mock backend info"""
        return {
            'name': self.name,
            'type': 'mock',
            'version': '1.0.0',
            'max_qubits': self.max_qubits,
            'noise_level': self.noise_level,
            'status': 'available',
            'queue_depth': 0,
            'executions': self._execution_count
        }

    async def close(self) -> None:
        """Close mock backend"""
        self._initialized = False
        logger.info(f"Mock backend '{self.name}' closed")

    def is_available(self) -> bool:
        """Mock backend is always available"""
        return True

    def estimate_cost(self, circuit: QuantumCircuit, shots: int) -> float:
        """Mock backend is free"""
        return 0.0


# Convenience functions

def create_default_backend_manager() -> BackendManager:
    """
    Create a backend manager with common backends

    Returns:
        BackendManager with mock backend registered
    """
    manager = BackendManager()

    # Register mock backend as default
    mock_backend = MockQuantumBackend(
        name="mock_ideal",
        max_qubits=20,
        noise_level=0.0
    )
    manager.register_backend(
        "mock_ideal",
        mock_backend,
        set_as_default=True,
        metadata={'description': 'Ideal mock simulator for testing'}
    )

    return manager


async def setup_ionq_backends(
    manager: BackendManager,
    api_key: str,
    use_cirq: bool = True,
    use_qiskit: bool = False
) -> BackendManager:
    """
    Set up IonQ backends in the manager

    Args:
        manager: BackendManager to register backends to
        api_key: IonQ API key
        use_cirq: Whether to register Cirq adapter
        use_qiskit: Whether to register Qiskit adapter

    Returns:
        Updated BackendManager
    """
    # Register Cirq-based backends
    if use_cirq:
        try:
            from .cirq_ionq_adapter import CirqIonQBackend

            # Simulator
            cirq_sim = CirqIonQBackend(api_key=api_key, target='simulator')
            manager.register_backend(
                'ionq_sim_cirq',
                cirq_sim,
                metadata={'description': 'IonQ simulator via Cirq'}
            )

            # QPU (if available)
            cirq_qpu = CirqIonQBackend(api_key=api_key, target='qpu.aria-1')
            manager.register_backend(
                'ionq_aria_cirq',
                cirq_qpu,
                metadata={'description': 'IonQ Aria QPU via Cirq'}
            )
        except ImportError:
            logger.warning("cirq-ionq not installed, skipping Cirq backends")

    # Register Qiskit-based backends
    if use_qiskit:
        try:
            from .qiskit_ionq_adapter import QiskitIonQBackend

            qiskit_sim = QiskitIonQBackend(api_key=api_key, target='simulator')
            manager.register_backend(
                'ionq_sim_qiskit',
                qiskit_sim,
                metadata={'description': 'IonQ simulator via Qiskit'}
            )
        except ImportError:
            logger.warning("qiskit-ionq not installed, skipping Qiskit backends")

    return manager
