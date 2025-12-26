"""
Comprehensive Tests for Q-Store v4.1 Phase 3: Storage Architecture

Tests all Phase 3 components:
- AsyncBuffer with ring buffer and overflow handling
- AsyncMetricsWriter with Parquet format
- CheckpointManager with Zarr compression
- TrainingMetrics schema
- AsyncMetricsLogger integration
"""

import pytest
import asyncio
import numpy as np
import time
import shutil
from pathlib import Path
import pandas as pd
import zarr

from q_store.storage import (
    AsyncBuffer,
    AsyncMetricsWriter,
    CheckpointManager,
    TrainingMetrics,
    AsyncMetricsLogger,
)


# ============================================================================
# Test AsyncBuffer
# ============================================================================

class TestAsyncBuffer:
    """Test AsyncBuffer ring buffer implementation."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = AsyncBuffer(maxsize=100)

        assert buffer.maxsize == 100
        assert len(buffer) == 0
        assert buffer.is_empty()

    def test_push_pop(self):
        """Test basic push/pop operations."""
        buffer = AsyncBuffer(maxsize=10)

        # Push items
        for i in range(5):
            success = buffer.push(i)
            assert success == True

        assert len(buffer) == 5

        # Pop items in FIFO order
        for i in range(5):
            value = buffer.pop(timeout=0.1)
            assert value == i

        assert buffer.is_empty()

    def test_overflow_handling(self):
        """Test buffer overflow handling."""
        buffer = AsyncBuffer(maxsize=3, drop_on_full=True)

        # Fill buffer
        buffer.push(1)
        buffer.push(2)
        buffer.push(3)

        assert buffer.is_full()

        # Try to push when full
        success = buffer.push(4)

        # Item should be dropped
        assert success == False
        assert buffer.dropped_items == 1

    def test_blocking_push(self):
        """Test blocking push when buffer is full."""
        buffer = AsyncBuffer(maxsize=3, drop_on_full=False)

        # Fill buffer
        buffer.push(1)
        buffer.push(2)
        buffer.push(3)

        # Push with timeout should block and fail
        start = time.time()
        success = buffer.push(4, timeout=0.1)
        duration = time.time() - start

        assert success == False
        assert duration >= 0.1

    def test_pop_timeout(self):
        """Test pop timeout on empty buffer."""
        buffer = AsyncBuffer(maxsize=10)

        start = time.time()
        value = buffer.pop(timeout=0.1)
        duration = time.time() - start

        assert value is None
        assert duration >= 0.1

    def test_queue_statistics(self):
        """Test buffer statistics tracking."""
        buffer = AsyncBuffer(maxsize=10)

        # Push/pop items
        for i in range(5):
            buffer.push(i)

        for _ in range(3):
            buffer.pop()

        stats = buffer.stats()

        assert stats['current_size'] == 2
        assert stats['max_size'] == 10
        assert stats['dropped_items'] == 0

    def test_thread_safety(self):
        """Test buffer is thread-safe."""
        import threading

        buffer = AsyncBuffer(maxsize=100)

        def producer():
            for i in range(50):
                buffer.push(i)
                time.sleep(0.001)

        def consumer():
            count = 0
            while count < 50:
                value = buffer.pop(timeout=0.1)
                if value is not None:
                    count += 1
                time.sleep(0.001)

        # Start threads
        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)

        t1.start()
        t2.start()

        t1.join(timeout=10)
        t2.join(timeout=10)

        assert buffer.is_empty()


# ============================================================================
# Test AsyncMetricsWriter
# ============================================================================

class TestAsyncMetricsWriter:
    """Test AsyncMetricsWriter with Parquet format."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "metrics"

    def test_initialization(self, temp_dir):
        """Test writer initialization."""
        writer = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=100,
            flush_interval=10.0
        )

        assert writer.buffer_size == 100
        assert writer.flush_interval == 10.0

        writer.close()

    def test_write_metrics(self, temp_dir):
        """Test writing metrics to Parquet."""
        writer = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=5,
            flush_interval=1.0
        )

        # Write metrics
        for i in range(10):
            writer.write({
                'epoch': i,
                'loss': 1.0 / (i + 1),
                'accuracy': 0.5 + i * 0.05,
                'timestamp': time.time()
            })

        # Force flush
        writer.flush()
        writer.close()

        # Verify Parquet file created
        parquet_files = list(temp_dir.glob('*.parquet'))
        assert len(parquet_files) > 0

        # Read and verify data
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 10
        assert 'epoch' in df.columns
        assert 'loss' in df.columns
        assert 'accuracy' in df.columns

    def test_buffered_writes(self, temp_dir):
        """Test buffered writing with flush interval."""
        writer = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=100,
            flush_interval=0.5
        )

        # Write metrics
        for i in range(10):
            writer.write({'epoch': i, 'loss': 1.0})

        # Should not flush immediately
        time.sleep(0.1)
        parquet_files = list(temp_dir.glob('*.parquet'))
        assert len(parquet_files) == 0

        # Wait for flush interval
        time.sleep(0.5)

        # Force flush to ensure write
        writer.flush()
        writer.close()

        parquet_files = list(temp_dir.glob('*.parquet'))
        assert len(parquet_files) > 0

    def test_append_mode(self, temp_dir):
        """Test append mode for continuous writing."""
        writer1 = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=5
        )

        # Write first batch
        for i in range(5):
            writer1.write({'epoch': i, 'loss': 1.0})

        writer1.flush()
        writer1.close()

        # Write second batch with new writer
        writer2 = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=5
        )

        for i in range(5, 10):
            writer2.write({'epoch': i, 'loss': 0.5})

        writer2.flush()
        writer2.close()

        # Verify all data is present
        parquet_files = list(temp_dir.glob('*.parquet'))

        all_data = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        assert len(combined) == 10

    def test_storage_overhead(self, temp_dir):
        """Test storage overhead is minimal."""
        writer = AsyncMetricsWriter(
            output_dir=str(temp_dir),
            buffer_size=10
        )

        # Write 1000 metrics
        for i in range(1000):
            writer.write({
                'epoch': i,
                'loss': np.random.rand(),
                'accuracy': np.random.rand(),
                'timestamp': time.time()
            })

        writer.flush()
        writer.close()

        # Check file size
        parquet_files = list(temp_dir.glob('*.parquet'))
        total_size = sum(f.stat().st_size for f in parquet_files)

        # Should be small (< 100KB for 1000 records)
        assert total_size < 100 * 1024
        print(f"Storage overhead: {total_size / 1024:.2f} KB for 1000 records")


# ============================================================================
# Test CheckpointManager
# ============================================================================

class TestCheckpointManager:
    """Test CheckpointManager with Zarr compression."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "checkpoints"

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            max_to_keep=3,
            compression='zstd'
        )

        assert manager.max_to_keep == 3
        assert manager.compression == 'zstd'

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            compression='zstd'
        )

        # Create fake model weights
        weights = {
            'layer1': np.random.randn(100, 50),
            'layer2': np.random.randn(50, 10),
        }

        # Save checkpoint
        checkpoint_path = manager.save(
            epoch=1,
            weights=weights,
            metrics={'loss': 0.5, 'accuracy': 0.85}
        )

        assert checkpoint_path.exists()

        # Verify Zarr directory created
        zarr_dirs = list(temp_dir.glob('checkpoint_*'))
        assert len(zarr_dirs) > 0

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save checkpoint
        original_weights = {
            'layer1': np.random.randn(100, 50),
            'layer2': np.random.randn(50, 10),
        }

        manager.save(epoch=1, weights=original_weights)

        # Load checkpoint
        loaded = manager.load_latest()

        assert loaded is not None
        assert 'epoch' in loaded
        assert 'weights' in loaded
        assert 'metrics' in loaded

        # Verify weights match
        for key in original_weights:
            assert np.allclose(loaded['weights'][key], original_weights[key])

    def test_max_to_keep(self, temp_dir):
        """Test max_to_keep limits checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            max_to_keep=3
        )

        weights = {'layer1': np.random.randn(10, 10)}

        # Save 5 checkpoints
        for epoch in range(1, 6):
            manager.save(
                epoch=epoch,
                weights=weights,
                metrics={'loss': 1.0 / epoch}
            )

        # Should only keep 3 best
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3

    def test_compression_effectiveness(self, temp_dir):
        """Test Zarr compression reduces size."""
        # Save without compression
        manager_no_comp = CheckpointManager(
            checkpoint_dir=str(temp_dir / "no_comp"),
            compression=None
        )

        # Save with compression
        manager_comp = CheckpointManager(
            checkpoint_dir=str(temp_dir / "comp"),
            compression='zstd'
        )

        # Create large weights
        weights = {
            'layer1': np.random.randn(1000, 1000),
        }

        manager_no_comp.save(epoch=1, weights=weights)
        manager_comp.save(epoch=1, weights=weights)

        # Check sizes
        def dir_size(path):
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        size_no_comp = dir_size(temp_dir / "no_comp")
        size_comp = dir_size(temp_dir / "comp")

        # Compression should reduce size
        compression_ratio = size_no_comp / size_comp
        print(f"Compression ratio: {compression_ratio:.2f}x")
        assert compression_ratio > 1.5

    def test_atomic_saves(self, temp_dir):
        """Test atomic saves prevent corruption."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        weights = {'layer1': np.random.randn(100, 100)}

        # Save multiple checkpoints rapidly
        for i in range(5):
            manager.save(epoch=i, weights=weights)

        # All checkpoints should be valid
        checkpoints = manager.list_checkpoints()
        for checkpoint_path in checkpoints:
            loaded = manager.load(checkpoint_path)
            assert loaded is not None

    def test_metadata_tracking(self, temp_dir):
        """Test metadata is tracked with checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        weights = {'layer1': np.random.randn(10, 10)}
        metadata = {
            'loss': 0.123,
            'accuracy': 0.876,
            'learning_rate': 0.001,
            'batch_size': 32
        }

        manager.save(epoch=10, weights=weights, metrics=metadata)

        loaded = manager.load_latest()

        assert loaded['epoch'] == 10
        for key, value in metadata.items():
            assert loaded['metrics'][key] == value


# ============================================================================
# Test TrainingMetrics
# ============================================================================

class TestTrainingMetrics:
    """Test TrainingMetrics dataclass schema."""

    def test_schema_definition(self):
        """Test metrics schema is well-defined."""
        from dataclasses import fields

        metric_fields = [f.name for f in fields(TrainingMetrics)]

        # Should have essential fields
        assert 'epoch' in metric_fields
        assert 'loss' in metric_fields
        assert 'accuracy' in metric_fields
        assert 'timestamp' in metric_fields

    def test_create_metric(self):
        """Test creating metric instance."""
        metric = TrainingMetrics(
            epoch=1,
            loss=0.5,
            accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            timestamp=time.time()
        )

        assert metric.epoch == 1
        assert metric.loss == 0.5
        assert metric.accuracy == 0.85

    def test_to_dict(self):
        """Test converting metric to dictionary."""
        metric = TrainingMetrics(
            epoch=1,
            loss=0.5,
            accuracy=0.85,
            timestamp=time.time()
        )

        metric_dict = metric.to_dict()

        assert isinstance(metric_dict, dict)
        assert metric_dict['epoch'] == 1
        assert metric_dict['loss'] == 0.5


# ============================================================================
# Test AsyncMetricsLogger
# ============================================================================

class TestAsyncMetricsLogger:
    """Test AsyncMetricsLogger integration."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "metrics"

    def test_initialization(self, temp_dir):
        """Test logger initialization."""
        logger = AsyncMetricsLogger(
            metrics_dir=str(temp_dir),
            buffer_size=100,
            flush_interval=10.0
        )

        assert logger is not None

        logger.close()

    def test_log_metrics(self, temp_dir):
        """Test logging metrics."""
        logger = AsyncMetricsLogger(
            metrics_dir=str(temp_dir),
            buffer_size=5
        )

        # Log metrics
        for i in range(10):
            logger.log({
                'epoch': i,
                'loss': 1.0 / (i + 1),
                'accuracy': 0.5 + i * 0.05
            })

        logger.close()

        # Verify metrics saved
        parquet_files = list(temp_dir.glob('*.parquet'))
        assert len(parquet_files) > 0

    def test_non_blocking_logging(self, temp_dir):
        """Test logging is non-blocking."""
        logger = AsyncMetricsLogger(
            metrics_dir=str(temp_dir),
            buffer_size=1000
        )

        # Log many metrics
        start = time.time()
        for i in range(100):
            logger.log({'epoch': i, 'loss': 1.0})
        log_time = time.time() - start

        # Should be very fast (< 10ms)
        assert log_time < 0.01

        logger.close()


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase3Integration:
    """Integration tests for Phase 3 storage."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path

    def test_full_storage_pipeline(self, temp_dir):
        """Test complete storage pipeline."""
        # Initialize components
        metrics_logger = AsyncMetricsLogger(
            metrics_dir=str(temp_dir / "metrics"),
            buffer_size=10
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(temp_dir / "checkpoints"),
            max_to_keep=2
        )

        # Simulate training loop
        for epoch in range(5):
            # Log metrics
            metrics_logger.log({
                'epoch': epoch,
                'loss': 1.0 / (epoch + 1),
                'accuracy': 0.5 + epoch * 0.1,
                'timestamp': time.time()
            })

            # Save checkpoint
            if epoch % 2 == 0:
                weights = {
                    'layer1': np.random.randn(10, 10),
                }
                checkpoint_manager.save(
                    epoch=epoch,
                    weights=weights,
                    metrics={'loss': 1.0 / (epoch + 1)}
                )

        # Close logger
        metrics_logger.close()

        # Verify storage
        metrics_files = list((temp_dir / "metrics").glob('*.parquet'))
        assert len(metrics_files) > 0

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0

    def test_storage_overhead(self, temp_dir):
        """Test storage overhead is minimal (<1%)."""
        metrics_logger = AsyncMetricsLogger(
            metrics_dir=str(temp_dir / "metrics")
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(temp_dir / "checkpoints"),
            compression='zstd'
        )

        # Large model weights
        weights = {
            'layer1': np.random.randn(1000, 1000),
            'layer2': np.random.randn(1000, 500),
        }

        # Save checkpoint
        checkpoint_manager.save(epoch=1, weights=weights)

        # Log many metrics
        for i in range(1000):
            metrics_logger.log({'epoch': i, 'loss': 1.0, 'accuracy': 0.8})

        metrics_logger.close()

        # Calculate overhead
        def dir_size(path):
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

        weights_size = sum(w.nbytes for w in weights.values())
        checkpoint_size = dir_size(temp_dir / "checkpoints")
        metrics_size = dir_size(temp_dir / "metrics")

        total_overhead = checkpoint_size + metrics_size
        overhead_pct = (total_overhead / weights_size) * 100

        print(f"Storage overhead: {overhead_pct:.2f}%")
        print(f"Checkpoint size: {checkpoint_size / 1024 / 1024:.2f} MB")
        print(f"Metrics size: {metrics_size / 1024:.2f} KB")

        # Should be < 1% overhead
        assert overhead_pct < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
