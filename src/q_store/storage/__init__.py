"""
Q-Store v4.1 Storage - Async Production Storage System

Battle-tested async storage architecture following best practices:
- Never block training loop
- In-memory first (RAM/GPU)
- Async writes to disk
- Zarr for checkpoints (binary, compressed)
- Parquet for metrics (columnar, analytics-ready)

Design Principles:
1. Never store raw quantum states (too large)
2. In-memory first - disk is backup
3. Async everything - zero blocking
4. Append-only where possible
5. Atomic operations for consistency

Components:
- AsyncBuffer: Non-blocking ring buffer for pending writes
- AsyncMetricsWriter: Background Parquet writer
- CheckpointManager: Zarr-based model checkpointing
- MetricsLogger: High-level async metrics API
"""

from q_store.storage.async_buffer import AsyncBuffer
from q_store.storage.async_writer import AsyncMetricsWriter
from q_store.storage.checkpoint_manager import CheckpointManager
from q_store.storage.metrics_schema import TrainingMetrics, QuantumMetrics, AsyncMetricsLogger
from q_store.storage.adaptive_measurement import (
    AdaptiveMeasurementPolicy,
    EarlyStoppingMeasurement,
    CombinedAdaptiveMeasurement,
    MeasurementResult,
)

__all__ = [
    'AsyncBuffer',
    'AsyncMetricsWriter',
    'CheckpointManager',
    'TrainingMetrics',
    'QuantumMetrics',  # v4.1 Enhanced
    'AsyncMetricsLogger',
    # v4.1 Enhanced: Adaptive measurement (75% cost savings)
    'AdaptiveMeasurementPolicy',
    'EarlyStoppingMeasurement',
    'CombinedAdaptiveMeasurement',
    'MeasurementResult',
]
