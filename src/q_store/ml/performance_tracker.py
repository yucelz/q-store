"""
Performance Tracker - v3.3
Monitors and logs training performance metrics

Key Innovation: Comprehensive performance monitoring for optimization analysis
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a single batch"""

    batch_idx: int
    epoch: int
    loss: float
    gradient_norm: float
    n_circuits: int
    time_ms: float
    learning_rate: float
    timestamp: float

    # Optional detailed metrics
    cache_hit_rate: Optional[float] = None
    method_used: Optional[str] = None
    param_variance: Optional[float] = None


@dataclass
class EpochMetrics:
    """Metrics for a full epoch"""

    epoch: int
    avg_loss: float
    min_loss: float
    max_loss: float
    total_time_ms: float
    total_circuits: int
    avg_gradient_norm: float
    timestamp: float
    cache_hit_rate: Optional[float] = None


class PerformanceTracker:
    """
    Track and analyze training performance

    Features:
    - Real-time metrics logging
    - Performance statistics
    - Bottleneck identification
    - Training progress visualization data
    """

    def __init__(self, log_dir: Optional[str] = None, save_interval: int = 10):
        """
        Initialize performance tracker

        Args:
            log_dir: Directory for saving metrics
            save_interval: Save metrics every N batches
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.save_interval = save_interval

        # Metrics storage
        self.batch_metrics: List[BatchMetrics] = []
        self.epoch_metrics: List[EpochMetrics] = []

        # Running statistics
        self.total_circuits_executed = 0
        self.total_time_ms = 0.0
        self.start_time = time.time()

        # Current epoch tracking
        self.current_epoch_batches: List[BatchMetrics] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Performance metrics will be saved to: {self.log_dir}")

    def log_batch(
        self,
        batch_idx: int,
        epoch: int,
        loss: float,
        gradient_norm: float,
        n_circuits: int,
        time_ms: float,
        learning_rate: float,
        cache_stats: Optional[Dict[str, Any]] = None,
        method_used: Optional[str] = None,
    ):
        """
        Log metrics for a single batch

        Args:
            batch_idx: Batch index in epoch
            epoch: Current epoch number
            loss: Batch loss value
            gradient_norm: L2 norm of gradients
            n_circuits: Circuits executed
            time_ms: Batch execution time
            learning_rate: Current learning rate
            cache_stats: Optional cache statistics
            method_used: Gradient method used
        """
        # Extract cache hit rate if available
        cache_hit_rate = None
        if cache_stats:
            cache_hit_rate = cache_stats.get("hit_rate")

        # Create metrics object
        metrics = BatchMetrics(
            batch_idx=batch_idx,
            epoch=epoch,
            loss=loss,
            gradient_norm=gradient_norm,
            n_circuits=n_circuits,
            time_ms=time_ms,
            learning_rate=learning_rate,
            timestamp=time.time(),
            cache_hit_rate=cache_hit_rate,
            method_used=method_used,
        )

        # Store metrics
        self.batch_metrics.append(metrics)
        self.current_epoch_batches.append(metrics)

        # Update totals
        self.total_circuits_executed += n_circuits
        self.total_time_ms += time_ms

        # Periodic save
        if len(self.batch_metrics) % self.save_interval == 0:
            self._save_metrics()

        logger.debug(
            f"Batch {batch_idx} epoch {epoch}: "
            f"loss={loss:.4f}, "
            f"||∇||={gradient_norm:.4f}, "
            f"circuits={n_circuits}, "
            f"time={time_ms:.2f}ms"
        )

    def log_epoch(self, epoch: int):
        """
        Log metrics for completed epoch

        Args:
            epoch: Epoch number
        """
        if not self.current_epoch_batches:
            logger.warning(f"No batches recorded for epoch {epoch}")
            return

        # Compute epoch statistics
        losses = [b.loss for b in self.current_epoch_batches]
        gradient_norms = [b.gradient_norm for b in self.current_epoch_batches]
        times = [b.time_ms for b in self.current_epoch_batches]
        circuits = [b.n_circuits for b in self.current_epoch_batches]

        cache_hit_rates = [
            b.cache_hit_rate for b in self.current_epoch_batches if b.cache_hit_rate is not None
        ]

        metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=np.mean(losses),
            min_loss=np.min(losses),
            max_loss=np.max(losses),
            total_time_ms=np.sum(times),
            total_circuits=np.sum(circuits),
            avg_gradient_norm=np.mean(gradient_norms),
            cache_hit_rate=np.mean(cache_hit_rates) if cache_hit_rates else None,
            timestamp=time.time(),
        )

        self.epoch_metrics.append(metrics)

        logger.info(
            f"Epoch {epoch} completed: "
            f"avg_loss={metrics.avg_loss:.4f}, "
            f"time={metrics.total_time_ms/1000:.2f}s, "
            f"circuits={metrics.total_circuits}"
        )

        # Clear current epoch batches
        self.current_epoch_batches.clear()

        # Save metrics
        self._save_metrics()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics

        Returns:
            Dictionary with performance stats
        """
        if not self.batch_metrics:
            return {}

        # Overall statistics
        all_losses = [b.loss for b in self.batch_metrics]
        all_times = [b.time_ms for b in self.batch_metrics]
        all_circuits = [b.n_circuits for b in self.batch_metrics]

        total_runtime = time.time() - self.start_time

        stats = {
            "total_batches": len(self.batch_metrics),
            "total_epochs": len(self.epoch_metrics),
            "total_circuits": self.total_circuits_executed,
            "total_time_ms": self.total_time_ms,
            "total_runtime_s": total_runtime,
            # Loss statistics
            "final_loss": all_losses[-1] if all_losses else None,
            "min_loss": np.min(all_losses) if all_losses else None,
            "max_loss": np.max(all_losses) if all_losses else None,
            "avg_loss": np.mean(all_losses) if all_losses else None,
            # Time statistics
            "avg_batch_time_ms": np.mean(all_times) if all_times else None,
            "total_batch_time_s": np.sum(all_times) / 1000 if all_times else None,
            # Circuit statistics
            "avg_circuits_per_batch": np.mean(all_circuits) if all_circuits else None,
            "circuits_per_second": (
                self.total_circuits_executed / total_runtime if total_runtime > 0 else 0
            ),
            # Efficiency
            "ms_per_circuit": (
                self.total_time_ms / self.total_circuits_executed
                if self.total_circuits_executed > 0
                else 0
            ),
        }

        # Add cache statistics if available
        cache_metrics = [b for b in self.batch_metrics if b.cache_hit_rate is not None]
        if cache_metrics:
            stats["avg_cache_hit_rate"] = np.mean([b.cache_hit_rate for b in cache_metrics])

        return stats

    def estimate_speedup(self, baseline_circuits_per_batch: int = 96) -> Dict[str, float]:
        """
        Estimate speedup compared to baseline

        Args:
            baseline_circuits_per_batch: Expected circuits without optimization

        Returns:
            Speedup metrics
        """
        if not self.batch_metrics:
            return {}

        avg_circuits = np.mean([b.n_circuits for b in self.batch_metrics])

        # Circuit reduction
        circuit_speedup = baseline_circuits_per_batch / avg_circuits

        # Time per batch
        avg_time = np.mean([b.time_ms for b in self.batch_metrics])

        # Assume baseline would take proportionally more time
        baseline_time = avg_time * circuit_speedup
        time_speedup = baseline_time / avg_time

        return {
            "circuit_reduction_factor": circuit_speedup,
            "estimated_time_speedup": time_speedup,
            "avg_circuits_actual": avg_circuits,
            "avg_circuits_baseline": baseline_circuits_per_batch,
            "circuits_saved_per_batch": baseline_circuits_per_batch - avg_circuits,
        }

    def get_convergence_data(self) -> Dict[str, List]:
        """
        Get data for convergence plots

        Returns:
            Dictionary with plot data
        """
        return {
            "batch_losses": [b.loss for b in self.batch_metrics],
            "batch_times": [b.time_ms for b in self.batch_metrics],
            "batch_circuits": [b.n_circuits for b in self.batch_metrics],
            "gradient_norms": [b.gradient_norm for b in self.batch_metrics],
            "epoch_avg_losses": [e.avg_loss for e in self.epoch_metrics],
            "epoch_times": [e.total_time_ms for e in self.epoch_metrics],
        }

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify performance bottlenecks

        Returns:
            Dictionary with bottleneck analysis
        """
        if not self.batch_metrics:
            return {}

        times = [b.time_ms for b in self.batch_metrics]
        circuits = [b.n_circuits for b in self.batch_metrics]

        # Find slowest batches
        sorted_by_time = sorted(enumerate(times), key=lambda x: x[1], reverse=True)

        slowest_batches = sorted_by_time[:5]

        # Analyze circuit execution efficiency
        time_per_circuit = [t / c if c > 0 else 0 for t, c in zip(times, circuits)]

        avg_efficiency = np.mean(time_per_circuit)

        bottlenecks = {
            "slowest_batches": [
                {
                    "batch_idx": idx,
                    "time_ms": times[idx],
                    "circuits": circuits[idx],
                    "efficiency_ms_per_circuit": (
                        times[idx] / circuits[idx] if circuits[idx] > 0 else 0
                    ),
                }
                for idx, _ in slowest_batches
            ],
            "avg_ms_per_circuit": avg_efficiency,
            "max_ms_per_circuit": np.max(time_per_circuit),
            "min_ms_per_circuit": np.min(time_per_circuit),
        }

        return bottlenecks

    def _save_metrics(self):
        """Save metrics to disk"""
        if not self.log_dir:
            return

        try:
            # Save batch metrics
            batch_file = self.log_dir / "batch_metrics.json"
            with open(batch_file, "w") as f:
                json.dump([asdict(m) for m in self.batch_metrics], f, indent=2)

            # Save epoch metrics
            epoch_file = self.log_dir / "epoch_metrics.json"
            with open(epoch_file, "w") as f:
                json.dump([asdict(m) for m in self.epoch_metrics], f, indent=2)

            # Save statistics
            stats_file = self.log_dir / "statistics.json"
            with open(stats_file, "w") as f:
                json.dump(self.get_statistics(), f, indent=2)

            logger.debug(f"Saved metrics to {self.log_dir}")

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def reset(self):
        """Reset all metrics"""
        self.batch_metrics.clear()
        self.epoch_metrics.clear()
        self.current_epoch_batches.clear()
        self.total_circuits_executed = 0
        self.total_time_ms = 0.0
        self.start_time = time.time()

        logger.info("Reset performance tracker")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"PerformanceTracker("
            f"batches={stats.get('total_batches', 0)}, "
            f"epochs={stats.get('total_epochs', 0)}, "
            f"circuits={stats.get('total_circuits', 0)}, "
            f"time={stats.get('total_runtime_s', 0):.1f}s)"
        )
