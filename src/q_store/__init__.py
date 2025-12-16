"""
Quantum-Native (Q-Store) Database Architecture v3.4
A novel database architecture leveraging quantum mechanical properties with hardware abstraction
and complete ML training capabilities.

Key Features:
- Hardware-agnostic quantum backend interface
- Support for multiple quantum SDKs (Cirq, Qiskit, Mock)
- Plugin architecture for easy backend addition
- Quantum ML training with gradient computation
- Production-ready with comprehensive error handling
- Backward compatible with v3.1, v3.2, and v3.3

v3.4 Enhancements:
- IonQ Batch API integration (8-12x performance improvement)
- Smart circuit caching with template-based parameter binding
- IonQ native gate compilation (GPi, GPi2, MS gates)
- Connection pooling for persistent HTTP connections
- Training time reduced from 30min to 3-4min
- Throughput increased to 5-8 circuits/second
"""

# Exceptions and constants
from . import constants, exceptions
from .backends.backend_manager import (
    BackendManager,
    MockQuantumBackend,
    create_default_backend_manager,
    setup_ionq_backends,
)

# Legacy backend (backward compatibility)
from .backends.ionq_backend import IonQQuantumBackend

# Quantum backend abstraction layer (v3.1)
from .backends.quantum_backend_interface import (
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
from .core.entanglement_registry import EntanglementRegistry

# Core database components
from .core.quantum_database import (
    ConnectionPool,
    DatabaseConfig,
    Metrics,
    MockPineconeIndex,
    QuantumDatabase,
    QueryMode,
    QueryResult,
)

# Core quantum components
from .core.state_manager import QuantumState, StateManager, StateStatus
from .core.tunneling_engine import TunnelingEngine

# ML Training Components (v3.2)
from .ml import (
    FiniteDifferenceGradient,
    GradientResult,
    LayerConfig,
    NaturalGradientComputer,
    QuantumConvolutionalLayer,
    QuantumDataEncoder,
    QuantumFeatureMap,
    QuantumGradientComputer,
    QuantumLayer,
    QuantumModel,
    QuantumPoolingLayer,
    QuantumTrainer,
    TrainingConfig,
    TrainingMetrics,
)

__version__ = "3.4.0"
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
    # ML Training (v3.2)
    "QuantumLayer",
    "QuantumConvolutionalLayer",
    "QuantumPoolingLayer",
    "LayerConfig",
    "QuantumGradientComputer",
    "FiniteDifferenceGradient",
    "NaturalGradientComputer",
    "GradientResult",
    "QuantumDataEncoder",
    "QuantumFeatureMap",
    "QuantumTrainer",
    "QuantumModel",
    "TrainingConfig",
    "TrainingMetrics",
]
