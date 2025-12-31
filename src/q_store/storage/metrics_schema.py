"""
Training Metrics Schema and Async Logger - v4.1 Enhanced

Defines standard schema for training metrics and provides async logging.

Features:
- Strongly typed metrics schema
- Async non-blocking logging
- Automatic timestamping
- Parquet backend
- Query support
- Quantum-specific metrics (v4.1 Enhanced)

Design:
- Dataclass for type safety
- AsyncBuffer for non-blocking
- AsyncMetricsWriter for persistence
- Pandas for queries
- TrainingMetrics: Basic metrics (backward compatible)
- QuantumMetrics: Extended metrics with quantum-specific fields
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time
from pathlib import Path

from q_store.storage.async_buffer import AsyncBuffer
from q_store.storage.async_writer import AsyncMetricsWriter


@dataclass
class TrainingMetrics:
    """
    Schema for training metrics.

    Attributes
    ----------
    epoch : int
        Training epoch
    step : int
        Global step number
    timestamp : float
        Unix timestamp
    train_loss : float
        Training loss
    val_loss : float, optional
        Validation loss
    grad_norm : float
        Gradient norm
    grad_max : float
        Maximum gradient value
    grad_min : float
        Minimum gradient value
    circuit_execution_time_ms : float
        Quantum circuit execution time
    circuits_executed : int
        Number of circuits executed
    qubits_used : int
        Number of qubits used
    shots_per_circuit : int
        Shots per circuit
    backend : str
        Backend name
    queue_time_ms : float, optional
        Queue time on backend
    cost_usd : float, optional
        Cost in USD
    credits_used : float, optional
        Credits used
    batch_time_ms : float
        Batch processing time
    throughput_samples_per_sec : float
        Throughput in samples/sec

    Examples
    --------
    >>> metrics = TrainingMetrics(
    ...     epoch=1,
    ...     step=100,
    ...     timestamp=time.time(),
    ...     train_loss=0.5,
    ...     grad_norm=0.1,
    ...     # ... other fields
    ... )
    """

    # Step info
    epoch: int
    step: int
    timestamp: float

    # Loss
    train_loss: float
    val_loss: Optional[float] = None

    # Gradients
    grad_norm: float = 0.0
    grad_max: float = 0.0
    grad_min: float = 0.0

    # Quantum metrics
    circuit_execution_time_ms: float = 0.0
    circuits_executed: int = 0
    qubits_used: int = 0
    shots_per_circuit: int = 1024

    # Backend info
    backend: str = 'simulator'
    queue_time_ms: Optional[float] = None

    # Cost tracking
    cost_usd: Optional[float] = None
    credits_used: Optional[float] = None

    # Performance
    batch_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        return asdict(self)


@dataclass
class QuantumMetrics:
    """
    Extended metrics schema with quantum-specific fields (v4.1 Enhanced).

    Extends TrainingMetrics with quantum computing metrics for better
    understanding of training dynamics and quantum advantage.

    Attributes
    ----------
    # Standard metrics (from TrainingMetrics)
    epoch : int
        Training epoch
    step : int
        Global step number
    timestamp : float
        Unix timestamp
    train_loss : float
        Training loss
    val_loss : float, optional
        Validation loss

    # Gradient metrics (enhanced)
    grad_norm : float
        Gradient L2 norm
    gradient_variance : float
        Variance across gradient dimensions
    gradient_snr : float
        Signal-to-noise ratio of gradients

    # Quantum circuit metrics
    circuit_depth : int
        Depth of quantum circuit
    entangling_gates : int
        Number of two-qubit gates
    measurement_bases_used : int
        Number of measurement bases
    shots_per_circuit : int
        Shots per circuit execution

    # Quantum-specific advanced metrics
    expressibility_score : float, optional
        Circuit expressibility [0, 1] (higher is better)
    entanglement_entropy : float, optional
        Von Neumann entropy of entanglement

    # Performance metrics
    circuit_execution_time_ms : float
        Quantum circuit execution time
    measurement_efficiency : float
        Useful shots / total shots
    cache_hit_rate : float
        Circuit cache hit rate

    # Cost tracking
    shots_used : int
        Total shots used this step
    estimated_cost_usd : float
        Estimated cost in USD

    # Backend info
    circuits_executed : int
        Number of circuits executed
    qubits_used : int
        Number of qubits used
    backend : str
        Backend name
    queue_time_ms : float, optional
        Queue time on backend

    # Other
    batch_time_ms : float
        Batch processing time
    throughput_samples_per_sec : float
        Throughput in samples/sec

    Examples
    --------
    >>> metrics = QuantumMetrics(
    ...     epoch=1,
    ...     step=100,
    ...     timestamp=time.time(),
    ...     train_loss=0.5,
    ...     grad_norm=0.1,
    ...     gradient_variance=0.05,
    ...     gradient_snr=2.5,
    ...     circuit_depth=10,
    ...     entangling_gates=15,
    ...     expressibility_score=0.75,
    ...     measurement_efficiency=0.85,
    ...     shots_used=750,  # Reduced from 3072 baseline!
    ...     estimated_cost_usd=0.025
    ... )
    """

    # Step info
    epoch: int
    step: int
    timestamp: float

    # Loss
    train_loss: float
    val_loss: Optional[float] = None

    # Gradients (enhanced)
    grad_norm: float = 0.0
    gradient_variance: float = 0.0  # NEW v4.1
    gradient_snr: float = 0.0  # NEW v4.1

    # Quantum circuit metrics (enhanced)
    circuit_depth: int = 0  # NEW v4.1
    entangling_gates: int = 0  # NEW v4.1
    measurement_bases_used: int = 1  # NEW v4.1
    shots_per_circuit: int = 1024

    # Quantum-specific advanced metrics (NEW v4.1)
    expressibility_score: Optional[float] = None
    entanglement_entropy: Optional[float] = None

    # Performance (enhanced)
    circuit_execution_time_ms: float = 0.0
    measurement_efficiency: float = 1.0  # NEW v4.1
    cache_hit_rate: float = 0.0  # NEW v4.1

    # Cost tracking (enhanced)
    shots_used: int = 0  # NEW v4.1
    estimated_cost_usd: float = 0.0  # NEW v4.1

    # Backend info
    circuits_executed: int = 0
    qubits_used: int = 0
    backend: str = 'simulator'
    queue_time_ms: Optional[float] = None

    # Legacy fields for backward compatibility
    grad_max: float = 0.0
    grad_min: float = 0.0
    cost_usd: Optional[float] = None
    credits_used: Optional[float] = None
    batch_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        return asdict(self)

    def to_training_metrics(self) -> TrainingMetrics:
        """
        Convert to TrainingMetrics for backward compatibility.

        Returns
        -------
        TrainingMetrics
            Basic training metrics
        """
        return TrainingMetrics(
            epoch=self.epoch,
            step=self.step,
            timestamp=self.timestamp,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            grad_norm=self.grad_norm,
            grad_max=self.grad_max,
            grad_min=self.grad_min,
            circuit_execution_time_ms=self.circuit_execution_time_ms,
            circuits_executed=self.circuits_executed,
            qubits_used=self.qubits_used,
            shots_per_circuit=self.shots_per_circuit,
            backend=self.backend,
            queue_time_ms=self.queue_time_ms,
            cost_usd=self.cost_usd or self.estimated_cost_usd,
            credits_used=self.credits_used,
            batch_time_ms=self.batch_time_ms,
            throughput_samples_per_sec=self.throughput_samples_per_sec
        )


class AsyncMetricsLogger:
    """
    Async metrics logger with Parquet backend.

    Never blocks training loop!

    Parameters
    ----------
    output_path : Path or str
        Output Parquet file path
    buffer_size : int, default=1000
        Buffer size
    flush_interval : int, default=100
        Flush interval

    Examples
    --------
    >>> logger = AsyncMetricsLogger('metrics.parquet')
    >>>
    >>> # Log step metrics
    >>> await logger.log_step(
    ...     epoch=1,
    ...     step=100,
    ...     train_loss=0.5,
    ...     grad_norm=0.1
    ... )
    >>>
    >>> # Log epoch summary
    >>> await logger.log_epoch(1, train_metrics={...}, val_metrics={...})
    >>>
    >>> # Shutdown
    >>> logger.stop()
    """

    def __init__(
        self,
        output_path: Path,
        buffer_size: int = 1000,
        flush_interval: int = 100,
    ):
        self.output_path = Path(output_path)

        # Create async buffer and writer
        self.buffer = AsyncBuffer(maxsize=buffer_size, name='metrics')
        self.writer = AsyncMetricsWriter(
            buffer=self.buffer,
            output_path=self.output_path,
            flush_interval=flush_interval,
        )

        # Start writer
        self.writer.start()

    async def log_step(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        grad_norm: float = 0.0,
        **kwargs
    ):
        """
        Log training step metrics.

        Parameters
        ----------
        epoch : int
            Epoch number
        step : int
            Step number
        train_loss : float
            Training loss
        grad_norm : float
            Gradient norm
        **kwargs
            Additional metrics
        """
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            timestamp=time.time(),
            train_loss=train_loss,
            grad_norm=grad_norm,
            **kwargs
        )

        # Push to buffer (non-blocking!)
        self.buffer.push(metrics.to_dict())

    async def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log epoch summary.

        Parameters
        ----------
        epoch : int
            Epoch number
        train_metrics : dict
            Training metrics
        val_metrics : dict, optional
            Validation metrics
        """
        metrics = TrainingMetrics(
            epoch=epoch,
            step=-1,  # Epoch summary
            timestamp=time.time(),
            train_loss=train_metrics.get('loss', 0.0),
            val_loss=val_metrics.get('loss') if val_metrics else None,
            grad_norm=train_metrics.get('grad_norm', 0.0),
            circuit_execution_time_ms=train_metrics.get('circuit_time_ms', 0.0),
            circuits_executed=train_metrics.get('circuits_executed', 0),
            qubits_used=train_metrics.get('qubits_used', 0),
            backend=train_metrics.get('backend', 'simulator'),
        )

        self.buffer.push(metrics.to_dict())

    async def log(self, metrics: TrainingMetrics):
        """
        Log metrics directly.

        Parameters
        ----------
        metrics : TrainingMetrics
            Metrics to log
        """
        self.buffer.push(metrics.to_dict())

    def stop(self):
        """Stop async writer."""
        self.writer.stop()

    def stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            'buffer': self.buffer.stats(),
            'writer': self.writer.stats(),
        }
