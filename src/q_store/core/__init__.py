"""
Core quantum database components and v4.0 circuit primitives.
"""

from .entanglement_registry import EntanglementRegistry
from .quantum_database import (
    ConnectionPool,
    DatabaseConfig,
    Metrics,
    MockPineconeIndex,
    QuantumDatabase,
    QueryMode,
    QueryResult,
)
from .state_manager import QuantumState, StateManager, StateStatus
from .tunneling_engine import TunnelingEngine

# v4.0 Core Components
from .unified_circuit import UnifiedCircuit, Gate, GateType, Parameter
from .circuit_converters import CirqConverter, QiskitConverter, IonQNativeConverter
from .circuit_optimizer import CircuitOptimizer, OptimizationMetrics, optimize

__all__ = [
    # Quantum Database (v3.x)
    "QuantumDatabase",
    "DatabaseConfig",
    "QueryMode",
    "QueryResult",
    "Metrics",
    "ConnectionPool",
    "MockPineconeIndex",
    "StateManager",
    "QuantumState",
    "StateStatus",
    "EntanglementRegistry",
    "TunnelingEngine",
    # v4.0 Core Components
    "UnifiedCircuit",
    "Gate",
    "GateType",
    "Parameter",
    "CirqConverter",
    "QiskitConverter",
    "IonQNativeConverter",
    "CircuitOptimizer",
    "OptimizationMetrics",
    "optimize",
]
