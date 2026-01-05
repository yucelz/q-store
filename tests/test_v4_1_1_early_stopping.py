"""
Unit tests for Q-Store v4.1.1 Early Stopping and Convergence Detection.

Tests cover:
- EarlyStopping with various modes
- ConvergenceDetector
- Restore best weights functionality
- Baseline and delta thresholds
"""

import pytest
import numpy as np

from q_store.ml.early_stopping import (
    EarlyStopping,
    ConvergenceDetector,
    create_early_stopping,
)


class TestEarlyStopping:
    """Test EarlyStopping class."""

    def test_early_stopping_min_mode(self):
        """Test early stopping in minimization mode."""
        early_stop = EarlyStopping(
            patience=3,
            min_delta=0.01,
            mode='min'
        )
        
        # Improving metrics (decreasing loss)
        assert not early_stop.should_stop(0, 1.0)
        assert not early_stop.should_stop(1, 0.9)
        assert not early_stop.should_stop(2, 0.8)
        
        # Plateau (no improvement)
        assert not early_stop.should_stop(3, 0.8)  # patience 1
        assert not early_stop.should_stop(4, 0.81)  # patience 2
        assert not early_stop.should_stop(5, 0.82)  # patience 3
        assert early_stop.should_stop(6, 0.83)  # Should stop

    def test_early_stopping_max_mode(self):
        """Test early stopping in maximization mode."""
        early_stop = EarlyStopping(
            patience=2,
            min_delta=0.01,
            mode='max'
        )
        
        # Improving metrics (increasing accuracy)
        assert not early_stop.should_stop(0, 0.7)
        assert not early_stop.should_stop(1, 0.8)
        assert not early_stop.should_stop(2, 0.85)
        
        # Plateau
        assert not early_stop.should_stop(3, 0.85)  # patience 1
        assert not early_stop.should_stop(4, 0.84)  # patience 2
        assert early_stop.should_stop(5, 0.84)  # Should stop

    def test_early_stopping_with_baseline(self):
        """Test early stopping with baseline value."""
        early_stop = EarlyStopping(
            patience=5,
            mode='min',
            baseline=0.5
        )
        
        # Above baseline - should accumulate patience
        assert not early_stop.should_stop(0, 0.8)
        assert not early_stop.should_stop(1, 0.7)
        
        # Below baseline - resets patience
        assert not early_stop.should_stop(2, 0.4)
        assert not early_stop.should_stop(3, 0.45)

    def test_min_delta(self):
        """Test minimum delta threshold."""
        early_stop = EarlyStopping(
            patience=2,
            min_delta=0.1,
            mode='min'
        )
        
        # Small improvement (less than min_delta)
        assert not early_stop.should_stop(0, 1.0)
        assert not early_stop.should_stop(1, 0.95)  # Not enough improvement
        assert not early_stop.should_stop(2, 0.94)  # Still not enough
        assert early_stop.should_stop(3, 0.93)  # Should stop

    def test_get_best_value(self):
        """Test getting best value."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, 0.8)
        early_stop.should_stop(2, 0.9)
        
        assert early_stop.get_best_value() == 0.8

    def test_get_best_epoch(self):
        """Test getting best epoch."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, 0.8)
        early_stop.should_stop(2, 0.9)
        early_stop.should_stop(3, 1.0)
        
        assert early_stop.get_best_epoch() == 1

    def test_restore_best_weights(self):
        """Test restore best weights flag."""
        early_stop = EarlyStopping(
            patience=2,
            mode='min',
            restore_best_weights=True
        )
        
        # Simulate storing weights
        weights_epoch_1 = {'w1': 1.0, 'w2': 2.0}
        
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, 0.8)
        early_stop.best_weights = weights_epoch_1
        early_stop.should_stop(2, 0.9)
        
        assert early_stop.restore_best_weights
        assert early_stop.best_weights == weights_epoch_1

    def test_wait_counter(self):
        """Test patience wait counter."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        early_stop.should_stop(0, 1.0)
        assert early_stop.wait == 0
        
        early_stop.should_stop(1, 1.0)  # No improvement
        assert early_stop.wait == 1
        
        early_stop.should_stop(2, 0.9)  # Improvement
        assert early_stop.wait == 0  # Reset


class TestConvergenceDetector:
    """Test ConvergenceDetector class."""

    def test_convergence_by_threshold(self):
        """Test convergence detection by threshold."""
        detector = ConvergenceDetector(
            method='threshold',
            threshold=0.01,
            patience=3
        )
        
        # Large changes
        assert not detector.has_converged(1.0)
        assert not detector.has_converged(0.5)
        
        # Small changes (converged)
        assert not detector.has_converged(0.505)
        assert not detector.has_converged(0.506)
        assert not detector.has_converged(0.507)
        assert detector.has_converged(0.508)  # After patience

    def test_convergence_by_relative_change(self):
        """Test convergence by relative change."""
        detector = ConvergenceDetector(
            method='relative',
            threshold=0.05,  # 5% change
            patience=2
        )
        
        # Start with value
        assert not detector.has_converged(100.0)
        
        # Large relative change (10%)
        assert not detector.has_converged(90.0)
        
        # Small relative changes
        assert not detector.has_converged(89.0)  # ~1%
        assert not detector.has_converged(88.5)  # ~0.6%
        assert detector.has_converged(88.3)  # Converged after patience

    def test_convergence_by_variance(self):
        """Test convergence by variance in window."""
        detector = ConvergenceDetector(
            method='variance',
            threshold=0.001,
            window_size=5
        )
        
        # Add values with high variance
        for val in [1.0, 2.0, 3.0, 2.0, 1.0]:
            assert not detector.has_converged(val)
        
        # Add values with low variance
        for val in [2.0, 2.01, 2.02, 2.01, 2.0]:
            result = detector.has_converged(val)
        
        # Should converge when variance is low
        assert result

    def test_get_convergence_info(self):
        """Test getting convergence information."""
        detector = ConvergenceDetector(method='threshold', threshold=0.01)
        
        detector.has_converged(1.0)
        detector.has_converged(0.9)
        detector.has_converged(0.85)
        
        info = detector.get_convergence_info()
        
        assert 'converged' in info
        assert 'iterations' in info
        assert info['iterations'] == 3

    def test_reset_detector(self):
        """Test resetting detector."""
        detector = ConvergenceDetector(method='threshold', threshold=0.01, patience=2)
        
        detector.has_converged(1.0)
        detector.has_converged(1.0)
        
        detector.reset()
        
        assert len(detector.history) == 0
        assert detector.patience_counter == 0


class TestEarlyStoppingFactory:
    """Test create_early_stopping factory function."""

    def test_create_basic_early_stopping(self):
        """Test creating basic early stopping."""
        early_stop = create_early_stopping(
            monitor='loss',
            patience=5,
            mode='min'
        )
        
        assert isinstance(early_stop, EarlyStopping)
        assert early_stop.patience == 5
        assert early_stop.mode == 'min'

    def test_create_with_convergence_detector(self):
        """Test creating early stopping with convergence detector."""
        early_stop = create_early_stopping(
            monitor='loss',
            patience=3,
            mode='min',
            convergence_method='threshold',
            convergence_threshold=0.001
        )
        
        assert isinstance(early_stop, EarlyStopping)


class TestEarlyStoppingIntegration:
    """Integration tests for early stopping."""

    def test_training_loop_simulation(self):
        """Test early stopping in simulated training loop."""
        early_stop = EarlyStopping(
            patience=5,
            min_delta=0.001,
            mode='min',
            verbose=False
        )
        
        # Simulate training with early stopping
        losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44]
        
        stopped_at = None
        for epoch, loss in enumerate(losses):
            if early_stop.should_stop(epoch, loss):
                stopped_at = epoch
                break
        
        # Should stop before reaching all epochs
        assert stopped_at is not None
        assert stopped_at < len(losses)

    def test_never_stops_with_improvement(self):
        """Test that early stopping doesn't trigger with continuous improvement."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        # Continuously improving
        for epoch in range(20):
            loss = 1.0 - epoch * 0.04
            should_stop = early_stop.should_stop(epoch, loss)
            assert not should_stop

    def test_stops_at_exact_patience(self):
        """Test stopping at exactly patience epochs."""
        patience = 5
        early_stop = EarlyStopping(patience=patience, mode='min')
        
        # Good performance then plateau
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, 0.5)
        
        # Plateau for patience epochs
        for i in range(patience):
            should_stop = early_stop.should_stop(2 + i, 0.5)
            if i < patience - 1:
                assert not should_stop
            else:
                assert should_stop


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value(self):
        """Test with single value."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        result = early_stop.should_stop(0, 1.0)
        assert not result

    def test_nan_values(self):
        """Test handling of NaN values."""
        early_stop = EarlyStopping(patience=2, mode='min')
        
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, np.nan)
        
        # Should handle NaN gracefully (treat as no improvement)
        assert early_stop.wait > 0

    def test_inf_values(self):
        """Test handling of infinite values."""
        early_stop = EarlyStopping(patience=2, mode='min')
        
        early_stop.should_stop(0, 1.0)
        early_stop.should_stop(1, np.inf)
        
        # Should handle inf gracefully
        assert early_stop.wait > 0

    def test_zero_patience(self):
        """Test with zero patience."""
        early_stop = EarlyStopping(patience=0, mode='min')
        
        early_stop.should_stop(0, 1.0)
        # With patience=0, should stop on first non-improvement
        result = early_stop.should_stop(1, 1.0)
        assert result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
