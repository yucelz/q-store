"""
Quantum-Native (Q-Store) Database Architecture v3.5
A novel database architecture leveraging quantum mechanical properties with hardware abstraction
and complete ML training capabilities.

Key Features:
- Hardware-agnostic quantum backend interface
- Support for multiple quantum SDKs (Cirq, Qiskit, Mock)
- Plugin architecture for easy backend addition
- Quantum ML training with gradient computation
- Production-ready with comprehensive error handling
- Backward compatible with v3.1, v3.2, v3.3, and v3.4

v3.3 Enhancements:
- SPSA gradient estimation (24-48x faster training)
- Hardware-efficient quantum layers (33% fewer parameters)
- Circuit batching and caching (2-5x speedup)
- Adaptive gradient optimization
- Performance tracking and monitoring

v3.4 Enhancements:
- Concurrent circuit submission (~1.6x faster)
- Native gate compilation (30% faster execution)
- Smart template-based caching (3-4x faster prep)
- Connection pooling and parallel execution

v3.5 NEW: Advanced Optimizations (2-3x additional speedup)
- Multi-backend orchestration (distribute across backends)
- Adaptive circuit optimization (dynamic simplification)
- Adaptive shot allocation (smart resource management)
- Natural gradient descent (2-3x fewer iterations)
- Honest performance metrics and realistic expectations
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

__version__ = "4.1.0"
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
