"""
Quantum-Native (Q-Store) Database Architecture
A novel database architecture leveraging quantum mechanical properties.
"""

from .core.quantum_database import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    QueryResult,
    Metrics,
    ConnectionPool,
    MockPineconeIndex
)
from .backends.ionq_backend import IonQQuantumBackend
from .core.state_manager import StateManager, QuantumState, StateStatus
from .core.entanglement_registry import EntanglementRegistry
from .core.tunneling_engine import TunnelingEngine

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
