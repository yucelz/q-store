"""
Quantum backend implementations.
Hardware-agnostic abstraction layer with multiple SDK support.
"""

from .backend_manager import (
    BackendManager,
    MockQuantumBackend,
    create_default_backend_manager,
    setup_ionq_backends,
)

# Keep legacy import for backward compatibility
from .ionq_backend import IonQQuantumBackend
from .quantum_backend_interface import (
    BackendCapabilities,
    BackendType,
    CircuitBuilder,
    ExecutionResult,
    GateType,
    QuantumBackend,
    QuantumCircuit,
    QuantumGate,
    amplitude_encode_to_circuit,
    create_bell_state_circuit,
    create_ghz_state_circuit,
)

__all__ = [
    # Core abstraction
    "QuantumBackend",
    "QuantumCircuit",
    "QuantumGate",
    "ExecutionResult",
    "BackendCapabilities",
    "BackendType",
    "GateType",
    "CircuitBuilder",
    # Backend management
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
]
