"""
Tests for Q-Store v4.1 Storage System

Tests Phase 3 components:
- AsyncBuffer
- AsyncMetricsWriter
- CheckpointManager
- AsyncMetricsLogger
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import time
import shutil
import pandas as pd

from q_store.storage import (
    AsyncBuffer,
    AsyncMetricsWriter,
    CheckpointManager,
    TrainingMetrics,
    AsyncMetricsLogger,
)


# ============================================================================
# AsyncBuffer Tests
# ============================================================================

class TestAsyncBuffer:
    """Test AsyncBuffer."""

    def test_push_pop(self):
        """Test basic push/pop."""
        buffer = AsyncBuffer(maxsize=10)

        # Push items
        for i in range(5):
            assert buffer.push(i) == True

        assert len(buffer) == 5

        # Pop items
        for i in range(5):
            assert buffer.pop(timeout=0.1) == i

        assert len(buffer) == 0

    def test_overflow(self):
        """Test overflow handling."""
        buffer = AsyncBuffer(maxsize=5)

        # Fill buffer
        for i in range(5):
            assert buffer.push(i) == True

        # Overflow (should drop)
        assert buffer.push(999) == False
        assert buffer.dropped_count == 1

    def test_timeout(self):
        """Test pop timeout."""
        buffer = AsyncBuffer(maxsize=10)

        # Pop from empty buffer
        item = buffer.pop(timeout=0.1)
        assert item is None

    def test_stats(self):
        """Test statistics."""
        buffer = AsyncBuffer(maxsize=10, name='test')

        buffer.push(1)
        buffer.push(2)
        buffer.pop()

        stats = buffer.stats()
        assert stats['name'] == 'test'
        assert stats['pushed'] == 2
        assert stats['popped'] == 1
        assert stats['dropped'] == 0


# ============================================================================
# AsyncMetricsWriter Tests
# ============================================================================

class TestAsyncMetricsWriter:
    """Test AsyncMetricsWriter."""

    @pytest.fixture
    def temp_output(self, tmp_path):
        """Temporary output path."""
        return tmp_path / 'metrics.parquet'

    def test_basic_write(self, temp_output):
        """Test basic metrics writing."""
        buffer = AsyncBuffer()
        writer = AsyncMetricsWriter(
            buffer=buffer,
            output_path=temp_output,
            flush_interval=5,
        )

        writer.start()

        # Push metrics
        for i in range(10):
            buffer.push({'step': i, 'loss': i * 0.1})

        # Wait for writes
        time.sleep(2.0)

        writer.stop()

        # Verify file exists
        assert temp_output.exists()

        # Read back
        df = pd.read_parquet(temp_output)
        assert len(df) == 10
        assert 'step' in df.columns
        assert 'loss' in df.columns

    def test_append_mode(self, temp_output):
        """Test append mode."""
        buffer = AsyncBuffer()

        # First write
        writer1 = AsyncMetricsWriter(buffer, temp_output, flush_interval=5)
        writer1.start()
        for i in range(5):
            buffer.push({'step': i})
        time.sleep(1.5)
        writer1.stop()

        # Second write (append)
        writer2 = AsyncMetricsWriter(buffer, temp_output, flush_interval=5)
        writer2.start()
        for i in range(5, 10):
            buffer.push({'step': i})
        time.sleep(1.5)
        writer2.stop()

        # Should have 10 rows
        df = pd.read_parquet(temp_output)
        assert len(df) == 10


# ============================================================================
# CheckpointManager Tests
# ============================================================================

class TestCheckpointManager:
    """Test CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Temporary checkpoint directory."""
        return tmp_path / 'checkpoints'

    @pytest.mark.asyncio
    async def test_save_load(self, temp_checkpoint_dir):
        """Test save and load."""
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save checkpoint
        model_state = {
            'weights': np.random.randn(10, 10),
            'bias': np.random.randn(10),
        }

        await manager.save(
            epoch=1,
            model_state=model_state,
            metadata={'loss': 0.5},
        )

        # Load checkpoint
        state = await manager.load(epoch=1)

        assert state['epoch'] == 1
        assert 'weights' in state['model_state']
        assert 'bias' in state['model_state']
        assert state['metadata']['loss'] == 0.5

        # Check shape preserved
        np.testing.assert_array_equal(
            model_state['weights'],
            state['model_state']['weights']
        )

    @pytest.mark.asyncio
    async def test_cleanup(self, temp_checkpoint_dir):
        """Test old checkpoint cleanup."""
        manager = CheckpointManager(temp_checkpoint_dir, keep_last=2)

        # Save 5 checkpoints
        for epoch in range(1, 6):
            await manager.save(
                epoch=epoch,
                model_state={'data': np.array([epoch])},
            )

        # Should keep only last 2
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints == [4, 5]

    @pytest.mark.asyncio
    async def test_compression(self, temp_checkpoint_dir):
        """Test compression."""
        manager = CheckpointManager(
            temp_checkpoint_dir,
            compression='zstd',
            compression_level=5,
        )

        # Large array
        large_data = np.random.randn(1000, 1000)

        await manager.save(
            epoch=1,
            model_state={'large': large_data},
        )

        # Load and verify
        state = await manager.load(epoch=1)
        np.testing.assert_array_almost_equal(
            large_data,
            state['model_state']['large']
        )

    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(temp_checkpoint_dir)

        # Initially empty
        assert manager.list_checkpoints() == []
        assert manager.latest_checkpoint() is None


# ============================================================================
# TrainingMetrics Tests
# ============================================================================

class TestTrainingMetrics:
    """Test TrainingMetrics schema."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            timestamp=time.time(),
            train_loss=0.5,
            grad_norm=0.1,
        )

        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.train_loss == 0.5

    def test_to_dict(self):
        """Test conversion to dict."""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            timestamp=time.time(),
            train_loss=0.5,
            grad_norm=0.1,
            circuits_executed=32,
        )

        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data['epoch'] == 1
        assert data['circuits_executed'] == 32


# ============================================================================
# AsyncMetricsLogger Tests
# ============================================================================

class TestAsyncMetricsLogger:
    """Test AsyncMetricsLogger."""

    @pytest.fixture
    def temp_output(self, tmp_path):
        """Temporary output path."""
        return tmp_path / 'training_metrics.parquet'

    @pytest.mark.asyncio
    async def test_log_step(self, temp_output):
        """Test step logging."""
        logger = AsyncMetricsLogger(temp_output, buffer_size=100, flush_interval=10)

        # Log steps
        for step in range(20):
            await logger.log_step(
                epoch=1,
                step=step,
                train_loss=1.0 / (step + 1),
                grad_norm=0.1,
            )

        # Wait for writes
        await asyncio.sleep(2.0)
        logger.stop()

        # Verify
        df = pd.read_parquet(temp_output)
        assert len(df) == 20
        assert all(df['epoch'] == 1)

    @pytest.mark.asyncio
    async def test_log_epoch(self, temp_output):
        """Test epoch logging."""
        logger = AsyncMetricsLogger(temp_output)

        await logger.log_epoch(
            epoch=1,
            train_metrics={'loss': 0.5, 'grad_norm': 0.1},
            val_metrics={'loss': 0.55, 'accuracy': 0.9},
        )

        await asyncio.sleep(2.0)
        logger.stop()

        # Verify
        df = pd.read_parquet(temp_output)
        assert len(df) == 1
        assert df.iloc[0]['epoch'] == 1
        assert df.iloc[0]['step'] == -1  # Epoch marker
        assert df.iloc[0]['train_loss'] == 0.5
        assert df.iloc[0]['val_loss'] == 0.55

    @pytest.mark.asyncio
    async def test_stats(self, temp_output):
        """Test statistics."""
        logger = AsyncMetricsLogger(temp_output)

        for i in range(5):
            await logger.log_step(epoch=1, step=i, train_loss=0.5, grad_norm=0.1)

        stats = logger.stats()
        assert 'buffer' in stats
        assert 'writer' in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestStorageIntegration:
    """Integration tests for storage system."""

    @pytest.mark.asyncio
    async def test_full_training_workflow(self, tmp_path):
        """Test full training workflow with all components."""
        # Setup
        logger = AsyncMetricsLogger(
            tmp_path / 'metrics.parquet',
            buffer_size=100,
            flush_interval=20,
        )

        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=tmp_path / 'checkpoints',
            keep_last=2,
        )

        # Training loop
        for epoch in range(1, 4):
            for step in range(1, 11):
                # Log step
                await logger.log_step(
                    epoch=epoch,
                    step=step,
                    train_loss=1.0 / (epoch * step),
                    grad_norm=0.1,
                    circuits_executed=32,
                )

            # Log epoch
            await logger.log_epoch(
                epoch=epoch,
                train_metrics={'loss': 0.5, 'grad_norm': 0.1},
                val_metrics={'loss': 0.55, 'accuracy': 0.8 + epoch * 0.05},
            )

            # Save checkpoint
            await checkpoint_mgr.save(
                epoch=epoch,
                model_state={'weights': np.random.randn(10, 10)},
                metadata={'loss': 0.5},
            )

        # Wait for writes
        await asyncio.sleep(2.0)
        logger.stop()

        # Verify metrics
        df = pd.read_parquet(tmp_path / 'metrics.parquet')
        assert len(df) > 0

        # Verify checkpoints
        checkpoints = checkpoint_mgr.list_checkpoints()
        assert len(checkpoints) == 2  # Only last 2 kept
        assert checkpoints == [2, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
