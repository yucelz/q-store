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

# High-performance simulators (optional dependencies)
try:
    from .qsim_backend import QsimBackend, create_qsim_backend
    HAS_QSIM = True
except ImportError:
    HAS_QSIM = False
    QsimBackend = None
    create_qsim_backend = None

try:
    from .lightning_backend import LightningBackend, create_lightning_backend
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    LightningBackend = None
    create_lightning_backend = None

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
    # High-performance backends
    "QsimBackend",
    "create_qsim_backend",
    "HAS_QSIM",
    "LightningBackend",
    "create_lightning_backend",
    "HAS_LIGHTNING",
    # Utility functions
    "amplitude_encode_to_circuit",
    "create_bell_state_circuit",
    "create_ghz_state_circuit",
    # Legacy (backward compatibility)
    "IonQQuantumBackend",
]
