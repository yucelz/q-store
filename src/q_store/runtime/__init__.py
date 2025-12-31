"""
Q-Store v4.1 Runtime - Async Execution System

This module provides the async execution infrastructure for quantum-first ML models.

Components:
- AsyncQuantumExecutor: Non-blocking quantum circuit submission and execution
- ResultCache: LRU cache for quantum measurement results
- BackendClient: Connection pooling and rate limiting for quantum backends

Key Features:
- Never blocks on quantum hardware latency
- Parallel circuit execution
- Result caching for identical circuits
- Connection pooling and rate limiting
- Background job polling
"""

from q_store.runtime.async_executor import AsyncQuantumExecutor
from q_store.runtime.result_cache import ResultCache
from q_store.runtime.backend_client import BackendClient, SimulatorClient, IonQClient
from q_store.runtime.ionq_adapter import IonQBackendClientAdapter
from q_store.runtime.gradient_strategies import (
    GradientStrategy,
    SPSAGradientEstimator,
    AdaptiveGradientEstimator,
    GradientEstimate,
)

__all__ = [
    'AsyncQuantumExecutor',
    'ResultCache',
    'BackendClient',
    'SimulatorClient',
    'IonQClient',
    'IonQBackendClientAdapter',
    # v4.1 Enhanced
    'GradientStrategy',
    'SPSAGradientEstimator',
    'AdaptiveGradientEstimator',
    'GradientEstimate',
]
