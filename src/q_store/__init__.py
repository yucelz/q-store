"""
Quantum-Native (Q-Store) Database Architecture v3.1
A novel database architecture leveraging quantum mechanical properties with hardware abstraction.

Key Features:
- Hardware-agnostic quantum backend interface
- Support for multiple quantum SDKs (Cirq, Qiskit, Mock)
- Plugin architecture for easy backend addition
- Production-ready with comprehensive error handling
- Backward compatible with v2.0
"""

# Core database components
from .core.quantum_database import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    QueryResult,
    Metrics,
    ConnectionPool,
    MockPineconeIndex
)

# Quantum backend abstraction layer (v3.1)
from .backends.quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    QuantumGate,
    ExecutionResult,
    BackendCapabilities,
    BackendType,
    GateType,
    CircuitBuilder,
    amplitude_encode_to_circuit,
    create_bell_state_circuit,
    create_ghz_state_circuit
)

from .backends.backend_manager import (
    BackendManager,
    MockQuantumBackend,
    create_default_backend_manager,
    setup_ionq_backends
)

# Legacy backend (backward compatibility)
from .backends.ionq_backend import IonQQuantumBackend

# Core quantum components
from .core.state_manager import StateManager, QuantumState, StateStatus
from .core.entanglement_registry import EntanglementRegistry
from .core.tunneling_engine import TunnelingEngine

__version__ = "3.1.0"
__all__ = [
    # Core database
    "QuantumDatabase",
    "DatabaseConfig",
    "QueryMode",
    "QueryResult",
    "Metrics",
    "ConnectionPool",
    "MockPineconeIndex",

    # Backend abstraction (v3.1)
    "QuantumBackend",
    "QuantumCircuit",
    "QuantumGate",
    "ExecutionResult",
    "BackendCapabilities",
    "BackendType",
    "GateType",
    "CircuitBuilder",
    "BackendManager",
    "MockQuantumBackend",
    "create_default_backend_manager",
    "setup_ionq_backends",

    # Utility functions
    "amplitude_encode_to_circuit",
    "create_bell_state_circuit",
    "create_ghz_state_circuit",

    # Legacy (backward compatibility)
    "IonQQuantumBackend",

    # Core components
    "StateManager",
    "QuantumState",
    "StateStatus",
    "EntanglementRegistry",
    "TunnelingEngine",
]
