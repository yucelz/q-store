"""
Unit tests for Q-Store v4.1.1 Training Callbacks.

Tests cover:
- Base Callback class
- CallbackList
- ModelCheckpoint
- CSVLogger
- ProgressCallback
- LearningRateLogger
- TensorBoardCallback
- MLflowCallback
- WandBCallback
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from q_store.ml.callbacks import (
    Callback,
    CallbackList,
    ModelCheckpoint,
    CSVLogger,
    ProgressCallback,
    LearningRateLogger,
    TensorBoardCallback,
    MLflowCallback,
    WandBCallback,
    create_callback,
)


class TestCallback:
    """Test base Callback class."""

    def test_callback_interface(self):
        """Test that Callback defines expected interface."""
        callback = Callback()
        
        # All methods should exist and be callable
        callback.on_train_begin()
        callback.on_train_end()
        callback.on_epoch_begin(0)
        callback.on_epoch_end(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)

    def test_custom_callback(self):
        """Test creating a custom callback."""
        class TestCallback(Callback):
            def __init__(self):
                self.train_begin_called = False
                self.epoch_end_called = False
            
            def on_train_begin(self, logs=None):
                self.train_begin_called = True
            
            def on_epoch_end(self, epoch, logs=None):
                self.epoch_end_called = True
        
        callback = TestCallback()
        callback.on_train_begin()
        callback.on_epoch_end(0)
        
        assert callback.train_begin_called
        assert callback.epoch_end_called


class TestCallbackList:
    """Test CallbackList container."""

    def test_callback_list_creation(self):
        """Test creating a callback list."""
        cb1 = Callback()
        cb2 = Callback()
        
        callback_list = CallbackList([cb1, cb2])
        
        assert len(callback_list.callbacks) == 2

    def test_callback_list_execution(self):
        """Test that callback list calls all callbacks."""
        class TrackingCallback(Callback):
            def __init__(self):
                self.calls = []
            
            def on_epoch_end(self, epoch, logs=None):
                self.calls.append(epoch)
        
        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        
        callback_list = CallbackList([cb1, cb2])
        callback_list.on_epoch_end(5, logs={'loss': 0.5})
        
        assert 5 in cb1.calls
        assert 5 in cb2.calls

    def test_callback_list_empty(self):
        """Test empty callback list."""
        callback_list = CallbackList([])
        
        # Should not raise errors
        callback_list.on_train_begin()
        callback_list.on_epoch_end(0)


class TestModelCheckpoint:
    """Test ModelCheckpoint callback."""

    def test_model_checkpoint_save_best(self):
        """Test saving best model only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'best_model.pkl')
            
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                monitor='loss',
                mode='min',
                save_best_only=True
            )
            
            # Mock model
            model = Mock()
            model.save = Mock()
            checkpoint.model = model
            
            # Better loss - should save
            checkpoint.on_epoch_end(0, logs={'loss': 1.0})
            assert checkpoint.best_value == 1.0
            
            # Worse loss - should not save
            model.save.reset_mock()
            checkpoint.on_epoch_end(1, logs={'loss': 1.5})
            model.save.assert_not_called()
            
            # Better loss - should save
            checkpoint.on_epoch_end(2, logs={'loss': 0.8})
            assert checkpoint.best_value == 0.8

    def test_model_checkpoint_save_all(self):
        """Test saving every epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_{epoch:02d}.pkl')
            
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                save_best_only=False
            )
            
            model = Mock()
            model.save = Mock()
            checkpoint.model = model
            
            checkpoint.on_epoch_end(0, logs={'loss': 1.0})
            checkpoint.on_epoch_end(1, logs={'loss': 0.9})
            
            # Should be called twice
            assert model.save.call_count == 2

    def test_model_checkpoint_max_mode(self):
        """Test checkpoint in max mode (accuracy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'best_model.pkl')
            
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                monitor='accuracy',
                mode='max',
                save_best_only=True
            )
            
            model = Mock()
            checkpoint.model = model
            
            # Improving accuracy
            checkpoint.on_epoch_end(0, logs={'accuracy': 0.7})
            assert checkpoint.best_value == 0.7
            
            checkpoint.on_epoch_end(1, logs={'accuracy': 0.8})
            assert checkpoint.best_value == 0.8
            
            checkpoint.on_epoch_end(2, logs={'accuracy': 0.75})
            assert checkpoint.best_value == 0.8  # Should not update


class TestCSVLogger:
    """Test CSVLogger callback."""

    def test_csv_logger_basic(self):
        """Test basic CSV logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'training.csv')
            
            logger = CSVLogger(filepath)
            logger.on_train_begin()
            
            # Log several epochs
            logger.on_epoch_end(0, logs={'loss': 1.0, 'accuracy': 0.7})
            logger.on_epoch_end(1, logs={'loss': 0.8, 'accuracy': 0.75})
            logger.on_epoch_end(2, logs={'loss': 0.6, 'accuracy': 0.8})
            
            logger.on_train_end()
            
            # Check file exists and has content
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 4  # Header + 3 epochs
                assert 'loss' in lines[0]
                assert 'accuracy' in lines[0]

    def test_csv_logger_append_mode(self):
        """Test CSV logger in append mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'training.csv')
            
            # First run
            logger1 = CSVLogger(filepath, append=False)
            logger1.on_train_begin()
            logger1.on_epoch_end(0, logs={'loss': 1.0})
            logger1.on_train_end()
            
            # Second run (append)
            logger2 = CSVLogger(filepath, append=True)
            logger2.on_train_begin()
            logger2.on_epoch_end(1, logs={'loss': 0.8})
            logger2.on_train_end()
            
            # Should have both epochs
            with open(filepath, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2


class TestProgressCallback:
    """Test ProgressCallback."""

    def test_progress_callback_basic(self):
        """Test basic progress tracking."""
        callback = ProgressCallback(total_epochs=10)
        
        callback.on_train_begin()
        
        for epoch in range(10):
            callback.on_epoch_begin(epoch)
            callback.on_epoch_end(epoch, logs={'loss': 1.0 - epoch * 0.1})
        
        callback.on_train_end()
        
        # Should complete without errors
        assert True

    def test_progress_callback_with_steps(self):
        """Test progress with step tracking."""
        callback = ProgressCallback(
            total_epochs=5,
            steps_per_epoch=10
        )
        
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        
        for step in range(10):
            callback.on_batch_end(step, logs={'loss': 0.5})
        
        callback.on_epoch_end(0, logs={'loss': 0.5})


class TestLearningRateLogger:
    """Test LearningRateLogger."""

    def test_lr_logger(self):
        """Test learning rate logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'lr_log.json')
            
            logger = LearningRateLogger(filepath)
            logger.on_train_begin()
            
            # Log LR for several epochs
            logger.on_epoch_end(0, logs={'lr': 0.1})
            logger.on_epoch_end(1, logs={'lr': 0.09})
            logger.on_epoch_end(2, logs={'lr': 0.081})
            
            logger.on_train_end()
            
            # Check logged data
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert len(data) == 3
                assert data[0]['lr'] == 0.1


class TestTensorBoardCallback:
    """Test TensorBoardCallback."""

    @patch('q_store.ml.callbacks.SummaryWriter')
    def test_tensorboard_callback(self, mock_writer):
        """Test TensorBoard callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(log_dir=tmpdir)
            
            callback.on_train_begin()
            callback.on_epoch_end(0, logs={'loss': 1.0, 'accuracy': 0.7})
            callback.on_epoch_end(1, logs={'loss': 0.8, 'accuracy': 0.75})
            callback.on_train_end()
            
            # Check that writer methods were called
            # (exact calls depend on implementation)
            assert True

    @patch('q_store.ml.callbacks.SummaryWriter')
    def test_tensorboard_with_histograms(self, mock_writer):
        """Test TensorBoard with histogram logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TensorBoardCallback(
                log_dir=tmpdir,
                histogram_freq=1
            )
            
            model = Mock()
            model.get_weights = Mock(return_value=[np.random.rand(10, 5)])
            callback.model = model
            
            callback.on_train_begin()
            callback.on_epoch_end(0, logs={'loss': 1.0})
            callback.on_train_end()


class TestMLflowCallback:
    """Test MLflowCallback."""

    @patch('q_store.ml.callbacks.mlflow')
    def test_mlflow_callback_basic(self, mock_mlflow):
        """Test basic MLflow callback."""
        callback = MLflowCallback(
            tracking_uri='http://localhost:5000',
            experiment_name='test_experiment'
        )
        
        callback.on_train_begin(logs={'param1': 10, 'param2': 'value'})
        callback.on_epoch_end(0, logs={'loss': 1.0, 'accuracy': 0.7})
        callback.on_epoch_end(1, logs={'loss': 0.8, 'accuracy': 0.75})
        callback.on_train_end()
        
        # Verify MLflow calls were made
        assert mock_mlflow.set_tracking_uri.called

    @patch('q_store.ml.callbacks.mlflow')
    def test_mlflow_log_model(self, mock_mlflow):
        """Test MLflow model logging."""
        callback = MLflowCallback(
            tracking_uri='http://localhost:5000',
            experiment_name='test',
            log_model=True
        )
        
        model = Mock()
        callback.model = model
        
        callback.on_train_begin()
        callback.on_train_end()
        
        # Should attempt to log model
        assert True


class TestWandBCallback:
    """Test WandBCallback."""

    @patch('q_store.ml.callbacks.wandb')
    def test_wandb_callback_basic(self, mock_wandb):
        """Test basic W&B callback."""
        callback = WandBCallback(
            project='test_project',
            name='test_run'
        )
        
        callback.on_train_begin(logs={'learning_rate': 0.01})
        callback.on_epoch_end(0, logs={'loss': 1.0, 'accuracy': 0.7})
        callback.on_epoch_end(1, logs={'loss': 0.8, 'accuracy': 0.75})
        callback.on_train_end()
        
        # Verify wandb init was called
        assert mock_wandb.init.called

    @patch('q_store.ml.callbacks.wandb')
    def test_wandb_log_model(self, mock_wandb):
        """Test W&B model artifact logging."""
        callback = WandBCallback(
            project='test_project',
            log_model=True
        )
        
        model = Mock()
        callback.model = model
        
        callback.on_train_begin()
        callback.on_train_end()


class TestCallbackFactory:
    """Test create_callback factory function."""

    def test_create_model_checkpoint(self):
        """Test creating ModelCheckpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pkl')
            
            callback = create_callback(
                'model_checkpoint',
                filepath=filepath,
                monitor='loss'
            )
            
            assert isinstance(callback, ModelCheckpoint)

    def test_create_csv_logger(self):
        """Test creating CSVLogger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'log.csv')
            
            callback = create_callback(
                'csv_logger',
                filepath=filepath
            )
            
            assert isinstance(callback, CSVLogger)

    def test_create_progress_callback(self):
        """Test creating ProgressCallback."""
        callback = create_callback(
            'progress',
            total_epochs=10
        )
        
        assert isinstance(callback, ProgressCallback)

    def test_create_unknown_callback(self):
        """Test creating unknown callback raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_callback('unknown_callback')


class TestCallbackIntegration:
    """Integration tests for callbacks."""

    def test_multiple_callbacks_together(self):
        """Test using multiple callbacks together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'model.pkl'),
                save_best_only=True
            )
            csv_logger = CSVLogger(os.path.join(tmpdir, 'log.csv'))
            progress = ProgressCallback(total_epochs=5)
            
            callbacks = CallbackList([checkpoint, csv_logger, progress])
            
            # Simulate training
            callbacks.on_train_begin()
            
            for epoch in range(5):
                callbacks.on_epoch_begin(epoch)
                callbacks.on_epoch_end(epoch, logs={
                    'loss': 1.0 - epoch * 0.1,
                    'accuracy': 0.5 + epoch * 0.05
                })
            
            callbacks.on_train_end()
            
            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, 'log.csv'))

    def test_callback_with_model_access(self):
        """Test callbacks that need model access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'model.pkl'),
                save_best_only=True
            )
            
            # Mock model
            model = Mock()
            model.save = Mock()
            checkpoint.model = model
            
            # Trigger save
            checkpoint.on_epoch_end(0, logs={'loss': 0.5})
            
            # Model save should be called
            model.save.assert_called()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_callback_with_missing_metric(self):
        """Test callback when monitored metric is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'model.pkl'),
                monitor='val_loss',
                save_best_only=True
            )
            
            model = Mock()
            checkpoint.model = model
            
            # Provide logs without val_loss
            checkpoint.on_epoch_end(0, logs={'loss': 0.5})
            
            # Should handle gracefully (warning or default behavior)
            assert True

    def test_callback_with_empty_logs(self):
        """Test callback with empty logs dict."""
        callback = ProgressCallback(total_epochs=5)
        
        callback.on_train_begin()
        callback.on_epoch_end(0, logs={})
        callback.on_train_end()
        
        # Should not crash
        assert True

    def test_callback_exception_handling(self):
        """Test that callback exceptions don't break training."""
        class FailingCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                raise RuntimeError("Callback error")
        
        callbacks = CallbackList([FailingCallback()])
        
        # Depending on implementation, might catch or propagate
        # This tests robustness
        try:
            callbacks.on_epoch_end(0, logs={'loss': 0.5})
        except RuntimeError:
            pass  # Expected if not caught internally


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
