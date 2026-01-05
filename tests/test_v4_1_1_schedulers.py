"""
Unit tests for Q-Store v4.1.1 Learning Rate Schedulers.

Tests cover:
- StepLR
- ExponentialLR
- CosineAnnealingLR
- CyclicLR
- OneCycleLR
- ReduceLROnPlateau
- WarmupScheduler
- Scheduler factory function
"""

import pytest
import numpy as np
import math

from q_store.ml.schedulers import (
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    WarmupScheduler,
    create_scheduler,
)


class TestStepLR:
    """Test StepLR scheduler."""

    def test_step_lr_basic(self):
        """Test basic step decay."""
        scheduler = StepLR(
            initial_lr=0.1,
            step_size=10,
            gamma=0.1
        )
        
        # Initial LR
        lr = scheduler.step(0)
        assert lr == 0.1
        
        # Before step
        lr = scheduler.step(9)
        assert lr == 0.1
        
        # After first step
        lr = scheduler.step(10)
        assert lr == pytest.approx(0.01)
        
        # After second step
        lr = scheduler.step(20)
        assert lr == pytest.approx(0.001)

    def test_step_lr_current_epoch(self):
        """Test getting current epoch."""
        scheduler = StepLR(initial_lr=0.1, step_size=10, gamma=0.1)
        
        scheduler.step(5)
        assert scheduler.get_current_epoch() == 5


class TestExponentialLR:
    """Test ExponentialLR scheduler."""

    def test_exponential_decay(self):
        """Test exponential decay."""
        scheduler = ExponentialLR(
            initial_lr=0.1,
            gamma=0.95
        )
        
        lr0 = scheduler.step(0)
        assert lr0 == 0.1
        
        lr1 = scheduler.step(1)
        assert lr1 == pytest.approx(0.1 * 0.95)
        
        lr10 = scheduler.step(10)
        assert lr10 == pytest.approx(0.1 * (0.95 ** 10))


class TestCosineAnnealingLR:
    """Test CosineAnnealingLR scheduler."""

    def test_cosine_annealing(self):
        """Test cosine annealing schedule."""
        scheduler = CosineAnnealingLR(
            initial_lr=0.1,
            T_max=100,
            eta_min=0.0
        )
        
        # At start
        lr0 = scheduler.step(0)
        assert lr0 == 0.1
        
        # At middle (should be at minimum)
        lr50 = scheduler.step(50)
        assert lr50 < 0.01
        
        # At end (should be back at minimum)
        lr100 = scheduler.step(100)
        assert lr100 == pytest.approx(0.0)

    def test_cosine_annealing_with_restarts(self):
        """Test cosine annealing with warm restarts."""
        scheduler = CosineAnnealingLR(
            initial_lr=0.1,
            T_max=50,
            eta_min=0.001,
            T_mult=2
        )
        
        # First cycle
        lr0 = scheduler.step(0)
        assert lr0 == 0.1
        
        # After first cycle (should restart)
        lr50 = scheduler.step(50)
        assert lr50 == pytest.approx(0.1)


class TestCyclicLR:
    """Test CyclicLR scheduler."""

    def test_cyclic_lr_triangular(self):
        """Test triangular cyclic LR."""
        scheduler = CyclicLR(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode='triangular'
        )
        
        lrs = [scheduler.step(i) for i in range(40)]
        
        # Should cycle between base_lr and max_lr
        assert min(lrs) >= 0.001
        assert max(lrs) <= 0.01

    def test_cyclic_lr_triangular2(self):
        """Test triangular2 cyclic LR."""
        scheduler = CyclicLR(
            base_lr=0.001,
            max_lr=0.01,
            step_size=10,
            mode='triangular2'
        )
        
        lrs = [scheduler.step(i) for i in range(60)]
        
        # Max LR should decrease over cycles
        first_cycle_max = max(lrs[:20])
        second_cycle_max = max(lrs[20:40])
        assert second_cycle_max < first_cycle_max


class TestOneCycleLR:
    """Test OneCycleLR scheduler."""

    def test_one_cycle_lr(self):
        """Test one cycle LR policy."""
        scheduler = OneCycleLR(
            max_lr=0.1,
            total_steps=100,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        lrs = [scheduler.step(i) for i in range(100)]
        
        # LR should increase then decrease
        max_idx = np.argmax(lrs)
        assert 20 < max_idx < 40  # Peak around 30% of cycle
        
        # Should reach max_lr
        assert max(lrs) == pytest.approx(0.1, rel=0.1)
        
        # Should end near zero
        assert lrs[-1] < 0.01


class TestReduceLROnPlateau:
    """Test ReduceLROnPlateau scheduler."""

    def test_reduce_on_plateau_min_mode(self):
        """Test reduce on plateau for minimization."""
        scheduler = ReduceLROnPlateau(
            initial_lr=0.1,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        # Improving metrics (decreasing)
        for i in range(5):
            lr = scheduler.step(i, metric=10.0 - i)
            assert lr == 0.1  # Should not reduce yet
        
        # Plateau (no improvement)
        for i in range(6):
            lr = scheduler.step(5 + i, metric=5.0)
        
        # After patience, LR should be reduced
        lr = scheduler.get_last_lr()
        assert lr < 0.1

    def test_reduce_on_plateau_max_mode(self):
        """Test reduce on plateau for maximization."""
        scheduler = ReduceLROnPlateau(
            initial_lr=0.1,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # Improving metrics (increasing)
        for i in range(5):
            scheduler.step(i, metric=float(i))
        
        # Plateau
        for i in range(5):
            scheduler.step(5 + i, metric=5.0)
        
        # Should reduce after patience
        lr = scheduler.get_last_lr()
        assert lr < 0.1

    def test_reduce_on_plateau_cooldown(self):
        """Test cooldown period."""
        scheduler = ReduceLROnPlateau(
            initial_lr=0.1,
            factor=0.1,
            patience=2,
            cooldown=2
        )
        
        # Trigger reduction
        for i in range(5):
            scheduler.step(i, metric=1.0)
        
        # During cooldown, shouldn't reduce again immediately
        initial_lr = scheduler.get_last_lr()
        for i in range(3):
            scheduler.step(5 + i, metric=1.0)
        
        # Should still be in cooldown
        assert scheduler.get_last_lr() == initial_lr


class TestWarmupScheduler:
    """Test WarmupScheduler."""

    def test_linear_warmup(self):
        """Test linear warmup."""
        base_scheduler = StepLR(initial_lr=0.1, step_size=100, gamma=0.1)
        scheduler = WarmupScheduler(
            base_scheduler=base_scheduler,
            warmup_epochs=10,
            warmup_start_lr=0.0,
            warmup_mode='linear'
        )
        
        # During warmup
        lr0 = scheduler.step(0)
        assert lr0 == 0.0
        
        lr5 = scheduler.step(5)
        assert 0.0 < lr5 < 0.1
        
        # After warmup
        lr10 = scheduler.step(10)
        assert lr10 == 0.1

    def test_exponential_warmup(self):
        """Test exponential warmup."""
        base_scheduler = StepLR(initial_lr=0.1, step_size=100, gamma=0.1)
        scheduler = WarmupScheduler(
            base_scheduler=base_scheduler,
            warmup_epochs=10,
            warmup_start_lr=0.001,
            warmup_mode='exp'
        )
        
        lrs = [scheduler.step(i) for i in range(15)]
        
        # Should increase during warmup
        assert lrs[0] < lrs[5] < lrs[9]
        
        # After warmup, should follow base scheduler
        assert lrs[10] == 0.1


class TestSchedulerFactory:
    """Test create_scheduler factory function."""

    def test_create_step_scheduler(self):
        """Test creating StepLR via factory."""
        scheduler = create_scheduler(
            'step',
            initial_lr=0.1,
            step_size=10,
            gamma=0.1
        )
        
        assert isinstance(scheduler, StepLR)
        assert scheduler.step(0) == 0.1

    def test_create_cosine_scheduler(self):
        """Test creating CosineAnnealingLR via factory."""
        scheduler = create_scheduler(
            'cosine',
            initial_lr=0.1,
            T_max=100
        )
        
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_create_cyclic_scheduler(self):
        """Test creating CyclicLR via factory."""
        scheduler = create_scheduler(
            'cyclic',
            base_lr=0.001,
            max_lr=0.01,
            step_size=10
        )
        
        assert isinstance(scheduler, CyclicLR)

    def test_create_onecycle_scheduler(self):
        """Test creating OneCycleLR via factory."""
        scheduler = create_scheduler(
            'onecycle',
            max_lr=0.1,
            total_steps=100
        )
        
        assert isinstance(scheduler, OneCycleLR)

    def test_create_plateau_scheduler(self):
        """Test creating ReduceLROnPlateau via factory."""
        scheduler = create_scheduler(
            'plateau',
            initial_lr=0.1,
            mode='min',
            patience=5
        )
        
        assert isinstance(scheduler, ReduceLROnPlateau)

    def test_create_unknown_scheduler(self):
        """Test creating unknown scheduler raises error."""
        with pytest.raises((ValueError, KeyError)):
            create_scheduler('unknown', initial_lr=0.1)


class TestSchedulerIntegration:
    """Integration tests for schedulers."""

    def test_scheduler_with_training_loop(self):
        """Test scheduler in simulated training loop."""
        scheduler = CosineAnnealingLR(
            initial_lr=0.1,
            T_max=50,
            eta_min=0.001
        )
        
        lrs = []
        for epoch in range(100):
            lr = scheduler.step(epoch)
            lrs.append(lr)
        
        # Check LR decreases then increases (with restart)
        assert lrs[0] > lrs[25]
        assert len(lrs) == 100

    def test_multiple_schedulers_chained(self):
        """Test chaining schedulers with warmup."""
        base = CosineAnnealingLR(initial_lr=0.1, T_max=90, eta_min=0.0)
        scheduler = WarmupScheduler(
            base_scheduler=base,
            warmup_epochs=10,
            warmup_start_lr=0.0
        )
        
        lrs = [scheduler.step(i) for i in range(100)]
        
        # Warmup phase
        assert lrs[0] == 0.0
        assert lrs[5] < lrs[9]
        
        # Cosine annealing phase
        assert lrs[10] == 0.1
        assert lrs[50] < lrs[10]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
