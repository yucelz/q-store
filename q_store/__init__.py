"""
Quantum-Native (Q-Store) Database Architecture
A novel database architecture leveraging quantum mechanical properties.
"""

from .quantum_database import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    QueryResult,
    Metrics,
    ConnectionPool,
    MockPineconeIndex
)
from .ionq_backend import IonQQuantumBackend
from .state_manager import StateManager, QuantumState, StateStatus
from .entanglement_registry import EntanglementRegistry
from .tunneling_engine import TunnelingEngine

__version__ = "2.0.0"
__all__ = [
    "QuantumDatabase",
    "DatabaseConfig",
    "QueryMode",
    "QueryResult",
    "Metrics",
    "ConnectionPool",
    "MockPineconeIndex",
    "IonQQuantumBackend",
    "StateManager",
    "QuantumState",
    "StateStatus",
    "EntanglementRegistry",
    "TunnelingEngine",
]
