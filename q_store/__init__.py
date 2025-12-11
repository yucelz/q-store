"""
Quantum-Native (Q-Store) Database Architecture
A novel database architecture leveraging quantum mechanical properties.
"""

from .quantum_database import QuantumDatabase, QuantumDatabaseConfig
from .ionq_backend import IonQQuantumBackend
from .state_manager import StateManager, QuantumState
from .entanglement_registry import EntanglementRegistry
from .tunneling_engine import TunnelingEngine

__version__ = "1.0.0"
__all__ = [
    "QuantumDatabase",
    "QuantumDatabaseConfig",
    "IonQQuantumBackend",
    "StateManager",
    "QuantumState",
    "EntanglementRegistry",
    "TunnelingEngine",
]
