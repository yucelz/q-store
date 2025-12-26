"""
Training Metrics Schema and Async Logger

Defines standard schema for training metrics and provides async logging.

Features:
- Strongly typed metrics schema
- Async non-blocking logging
- Automatic timestamping
- Parquet backend
- Query support

Design:
- Dataclass for type safety
- AsyncBuffer for non-blocking
- AsyncMetricsWriter for persistence
- Pandas for queries
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
