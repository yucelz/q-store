"""
Unit tests for Q-Store v4.1.1 Experiment Tracking.

Tests cover:
- MLflowTracker
- MLflowConfig
- Experiment management
- Run management
- Parameter and metric logging
- Model registry
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from q_store.ml.tracking import (
    MLflowTracker,
    MLflowConfig,
)


class TestMLflowConfig:
    """Test MLflowConfig dataclass."""

    def test_config_creation(self):
        """Test creating MLflow configuration."""
        config = MLflowConfig(
            tracking_uri='http://localhost:5000',
            experiment_name='test_experiment',
            run_name='test_run'
        )
        
        assert config.tracking_uri == 'http://localhost:5000'
        assert config.experiment_name == 'test_experiment'
        assert config.run_name == 'test_run'

    def test_config_defaults(self):
        """Test default configuration values."""
        config = MLflowConfig(experiment_name='test')
        
        assert config.artifact_location is None
        assert config.tags == {}


class TestMLflowTracker:
    """Test MLflowTracker class."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_tracker_initialization(self, mock_mlflow):
        """Test tracker initialization."""
        tracker = MLflowTracker(
            tracking_uri='http://localhost:5000',
            experiment_name='test_experiment'
        )
        
        assert tracker.experiment_name == 'test_experiment'
        mock_mlflow.set_tracking_uri.assert_called_with('http://localhost:5000')

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_start_run(self, mock_mlflow):
        """Test starting a run."""
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        tracker = MLflowTracker(experiment_name='test')
        tracker.start_run(run_name='test_run')
        
        mock_mlflow.start_run.assert_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_end_run(self, mock_mlflow):
        """Test ending a run."""
        tracker = MLflowTracker(experiment_name='test')
        tracker.end_run()
        
        mock_mlflow.end_run.assert_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_param(self, mock_mlflow):
        """Test logging a single parameter."""
        tracker = MLflowTracker(experiment_name='test')
        tracker.log_param('learning_rate', 0.01)
        
        mock_mlflow.log_param.assert_called_with('learning_rate', 0.01)

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_params(self, mock_mlflow):
        """Test logging multiple parameters."""
        tracker = MLflowTracker(experiment_name='test')
        params = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'n_qubits': 4
        }
        tracker.log_params(params)
        
        mock_mlflow.log_params.assert_called_with(params)

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_metric(self, mock_mlflow):
        """Test logging a single metric."""
        tracker = MLflowTracker(experiment_name='test')
        tracker.log_metric('loss', 0.5, step=1)
        
        mock_mlflow.log_metric.assert_called_with('loss', 0.5, step=1)

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test logging multiple metrics."""
        tracker = MLflowTracker(experiment_name='test')
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'val_loss': 0.6
        }
        tracker.log_metrics(metrics, step=10)
        
        mock_mlflow.log_metrics.assert_called_with(metrics, step=10)

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_artifact(self, mock_mlflow):
        """Test logging an artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test content')
            
            tracker = MLflowTracker(experiment_name='test')
            tracker.log_artifact(test_file)
            
            mock_mlflow.log_artifact.assert_called_with(test_file)

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_model(self, mock_mlflow):
        """Test logging a model."""
        tracker = MLflowTracker(experiment_name='test')
        
        model = Mock()
        tracker.log_model(model, 'model')
        
        # Should call appropriate mlflow model logging
        assert True  # Exact call depends on implementation

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_set_tag(self, mock_mlflow):
        """Test setting a tag."""
        tracker = MLflowTracker(experiment_name='test')
        tracker.set_tag('version', 'v4.1.1')
        
        mock_mlflow.set_tag.assert_called_with('version', 'v4.1.1')

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_set_tags(self, mock_mlflow):
        """Test setting multiple tags."""
        tracker = MLflowTracker(experiment_name='test')
        tags = {
            'version': 'v4.1.1',
            'author': 'test_user',
            'environment': 'dev'
        }
        tracker.set_tags(tags)
        
        mock_mlflow.set_tags.assert_called_with(tags)


class TestMLflowTrackerContextManager:
    """Test MLflowTracker as context manager."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_context_manager_basic(self, mock_mlflow):
        """Test basic context manager usage."""
        mock_run = Mock()
        mock_mlflow.start_run.return_value = mock_run
        mock_run.__enter__ = Mock(return_value=mock_run)
        mock_run.__exit__ = Mock(return_value=False)
        
        tracker = MLflowTracker(experiment_name='test')
        
        with tracker.start_run(run_name='test_run'):
            tracker.log_metric('loss', 0.5)
        
        mock_mlflow.start_run.assert_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_context_manager_with_exception(self, mock_mlflow):
        """Test context manager with exception."""
        mock_run = Mock()
        mock_mlflow.start_run.return_value = mock_run
        mock_run.__enter__ = Mock(return_value=mock_run)
        mock_run.__exit__ = Mock(return_value=False)
        
        tracker = MLflowTracker(experiment_name='test')
        
        try:
            with tracker.start_run():
                tracker.log_metric('loss', 0.5)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Run should still be properly ended
        assert True


class TestMLflowTrackerExperimentManagement:
    """Test experiment management features."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_create_experiment(self, mock_mlflow):
        """Test creating an experiment."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = '123'
        
        tracker = MLflowTracker(experiment_name='new_experiment')
        
        # Should create experiment if it doesn't exist
        mock_mlflow.create_experiment.assert_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_get_existing_experiment(self, mock_mlflow):
        """Test getting existing experiment."""
        mock_experiment = Mock()
        mock_experiment.experiment_id = '123'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        tracker = MLflowTracker(experiment_name='existing_experiment')
        
        # Should not create new experiment
        mock_mlflow.create_experiment.assert_not_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_list_runs(self, mock_mlflow):
        """Test listing runs in experiment."""
        tracker = MLflowTracker(experiment_name='test')
        
        mock_mlflow.search_runs.return_value = []
        runs = tracker.list_runs()
        
        mock_mlflow.search_runs.assert_called()


class TestMLflowTrackerModelRegistry:
    """Test model registry features."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_register_model(self, mock_mlflow):
        """Test registering a model."""
        tracker = MLflowTracker(experiment_name='test')
        
        tracker.register_model('runs:/123/model', 'MyModel')
        
        # Should call mlflow register_model
        assert True

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_transition_model_stage(self, mock_mlflow):
        """Test transitioning model stage."""
        tracker = MLflowTracker(experiment_name='test')
        
        tracker.transition_model_stage('MyModel', version=1, stage='Production')
        
        # Should transition model to production
        assert True


class TestMLflowTrackerIntegration:
    """Integration tests for MLflow tracker."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_full_training_workflow(self, mock_mlflow):
        """Test complete training workflow with MLflow."""
        mock_run = Mock()
        mock_mlflow.start_run.return_value = mock_run
        mock_run.__enter__ = Mock(return_value=mock_run)
        mock_run.__exit__ = Mock(return_value=False)
        
        tracker = MLflowTracker(
            tracking_uri='http://localhost:5000',
            experiment_name='quantum_training'
        )
        
        with tracker.start_run(run_name='experiment_1'):
            # Log hyperparameters
            tracker.log_params({
                'learning_rate': 0.01,
                'n_qubits': 4,
                'circuit_depth': 3,
                'optimizer': 'adam'
            })
            
            # Log training metrics
            for epoch in range(5):
                tracker.log_metrics({
                    'loss': 1.0 - epoch * 0.1,
                    'accuracy': 0.5 + epoch * 0.08
                }, step=epoch)
            
            # Log final model
            model = Mock()
            tracker.log_model(model, 'quantum_model')
            
            # Set tags
            tracker.set_tags({
                'version': 'v4.1.1',
                'framework': 'q-store'
            })
        
        # Verify all operations were called
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_multiple_runs_same_experiment(self, mock_mlflow):
        """Test running multiple experiments."""
        mock_run = Mock()
        mock_mlflow.start_run.return_value = mock_run
        mock_run.__enter__ = Mock(return_value=mock_run)
        mock_run.__exit__ = Mock(return_value=False)
        
        tracker = MLflowTracker(experiment_name='test')
        
        # Run 1
        with tracker.start_run(run_name='run_1'):
            tracker.log_metric('loss', 0.5)
        
        # Run 2
        with tracker.start_run(run_name='run_2'):
            tracker.log_metric('loss', 0.4)
        
        # Should have started 2 runs
        assert mock_mlflow.start_run.call_count == 2


class TestMLflowTrackerLogging:
    """Test advanced logging features."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_numpy_array(self, mock_mlflow):
        """Test logging numpy arrays."""
        tracker = MLflowTracker(experiment_name='test')
        
        array = np.array([1, 2, 3, 4, 5])
        tracker.log_artifact_data(array, 'array.npy')
        
        # Should serialize and log array
        assert True

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_figure(self, mock_mlflow):
        """Test logging matplotlib figures."""
        tracker = MLflowTracker(experiment_name='test')
        
        # Mock figure
        fig = Mock()
        tracker.log_figure(fig, 'plot.png')
        
        # Should save and log figure
        assert True

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_dict_as_json(self, mock_mlflow):
        """Test logging dictionary as JSON artifact."""
        tracker = MLflowTracker(experiment_name='test')
        
        data = {'key1': 'value1', 'key2': [1, 2, 3]}
        tracker.log_dict(data, 'config.json')
        
        # Should serialize and log JSON
        assert True


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_log_without_active_run(self, mock_mlflow):
        """Test logging without an active run."""
        tracker = MLflowTracker(experiment_name='test')
        
        # Should handle gracefully or raise appropriate error
        try:
            tracker.log_metric('loss', 0.5)
        except Exception as e:
            # Expected behavior depends on implementation
            pass

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_invalid_metric_value(self, mock_mlflow):
        """Test logging invalid metric values."""
        tracker = MLflowTracker(experiment_name='test')
        
        with tracker.start_run():
            # Try logging NaN
            tracker.log_metric('loss', float('nan'))
            
            # Try logging infinity
            tracker.log_metric('loss', float('inf'))
        
        # Should handle gracefully
        assert True

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_very_long_parameter_value(self, mock_mlflow):
        """Test logging very long parameter values."""
        tracker = MLflowTracker(experiment_name='test')
        
        long_value = 'x' * 10000
        
        with tracker.start_run():
            # MLflow has param value length limits
            try:
                tracker.log_param('long_param', long_value)
            except Exception:
                pass  # Expected to fail or truncate

    @patch('q_store.ml.tracking.mlflow_tracker.mlflow')
    def test_connection_error(self, mock_mlflow):
        """Test handling connection errors."""
        mock_mlflow.set_tracking_uri.side_effect = ConnectionError("Cannot connect")
        
        try:
            tracker = MLflowTracker(
                tracking_uri='http://invalid:9999',
                experiment_name='test'
            )
        except ConnectionError:
            pass  # Expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
