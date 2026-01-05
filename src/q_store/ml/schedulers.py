"""
Learning Rate Schedulers for Quantum ML Training.

This module provides advanced learning rate scheduling strategies optimized
for quantum machine learning, including adaptive, cyclic, and warmup schedulers.

Key Components:
    - LRScheduler: Base scheduler class
    - StepLR: Step decay scheduler
    - ExponentialLR: Exponential decay
    - CosineAnnealingLR: Cosine annealing
    - CyclicLR: Cyclic learning rate
    - OneCycleLR: One-cycle policy
    - ReduceLROnPlateau: Reduce on metric plateau
    - WarmupScheduler: Learning rate warmup

Example:
    >>> from q_store.ml.schedulers import CosineAnnealingLR
    >>>
    >>> scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100, eta_min=1e-6)
    >>> for epoch in range(100):
    ...     lr = scheduler.step(epoch)
    ...     optimizer.lr = lr
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import numpy as np

logger = logging.getLogger(__name__)


class LRScheduler(ABC):
    """
    Base learning rate scheduler.

    All schedulers inherit from this base class and implement
    the step() method to compute the learning rate for each epoch/step.

    Args:
        initial_lr: Initial learning rate
        verbose: Whether to log LR changes (default: False)
    """

    def __init__(self, initial_lr: float, verbose: bool = False):
        """
        Initialize learning rate scheduler.

        Args:
            initial_lr: Initial learning rate
            verbose: Whether to log changes
        """
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.current_lr = initial_lr
        self.step_count = 0

    @abstractmethod
    def step(self, epoch: Optional[int] = None) -> float:
        """
        Compute learning rate for current epoch/step.

        Args:
            epoch: Current epoch (if None, uses internal counter)

        Returns:
            Learning rate for this epoch
        """
        pass

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.step_count = 0

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class StepLR(LRScheduler):
    """
    Step decay learning rate scheduler.

    Decays the learning rate by gamma every step_size epochs.

    Args:
        initial_lr: Initial learning rate
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
        verbose: Whether to log changes

    Example:
        >>> scheduler = StepLR(initial_lr=0.1, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     lr = scheduler.step(epoch)
        ...     print(f"Epoch {epoch}: LR = {lr}")
    """

    def __init__(
        self,
        initial_lr: float,
        step_size: int,
        gamma: float = 0.1,
        verbose: bool = False
    ):
        """Initialize step decay scheduler."""
        super().__init__(initial_lr, verbose)
        self.step_size = step_size
        self.gamma = gamma

        logger.info(
            f"Initialized StepLR: initial_lr={initial_lr}, "
            f"step_size={step_size}, gamma={gamma}"
        )

    def step(self, epoch: Optional[int] = None) -> float:
        """Compute learning rate with step decay."""
        if epoch is None:
            epoch = self.step_count
            self.step_count += 1

        # Calculate number of decay steps
        decay_steps = epoch // self.step_size

        # Compute new learning rate
        new_lr = self.initial_lr * (self.gamma ** decay_steps)

        if new_lr != self.current_lr and self.verbose:
            logger.info(f"Epoch {epoch}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


class ExponentialLR(LRScheduler):
    """
    Exponential decay learning rate scheduler.

    Decays the learning rate exponentially: lr = initial_lr * gamma^epoch

    Args:
        initial_lr: Initial learning rate
        gamma: Multiplicative factor (e.g., 0.95)
        verbose: Whether to log changes

    Example:
        >>> scheduler = ExponentialLR(initial_lr=0.1, gamma=0.95)
        >>> for epoch in range(100):
        ...     lr = scheduler.step(epoch)
    """

    def __init__(
        self,
        initial_lr: float,
        gamma: float = 0.95,
        verbose: bool = False
    ):
        """Initialize exponential decay scheduler."""
        super().__init__(initial_lr, verbose)
        self.gamma = gamma

        logger.info(f"Initialized ExponentialLR: initial_lr={initial_lr}, gamma={gamma}")

    def step(self, epoch: Optional[int] = None) -> float:
        """Compute learning rate with exponential decay."""
        if epoch is None:
            epoch = self.step_count
            self.step_count += 1

        new_lr = self.initial_lr * (self.gamma ** epoch)

        if new_lr != self.current_lr and self.verbose:
            logger.info(f"Epoch {epoch}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.

    Anneals the learning rate using a cosine function from initial_lr to eta_min
    over T_max epochs.

    Args:
        initial_lr: Initial learning rate
        T_max: Maximum number of iterations (half period)
        eta_min: Minimum learning rate (default: 0)
        verbose: Whether to log changes

    Example:
        >>> scheduler = CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=1e-6)
        >>> for epoch in range(100):
        ...     lr = scheduler.step(epoch)
    """

    def __init__(
        self,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0,
        verbose: bool = False
    ):
        """Initialize cosine annealing scheduler."""
        super().__init__(initial_lr, verbose)
        self.T_max = T_max
        self.eta_min = eta_min

        logger.info(
            f"Initialized CosineAnnealingLR: initial_lr={initial_lr}, "
            f"T_max={T_max}, eta_min={eta_min}"
        )

    def step(self, epoch: Optional[int] = None) -> float:
        """Compute learning rate with cosine annealing."""
        if epoch is None:
            epoch = self.step_count
            self.step_count += 1

        # Cosine annealing formula
        new_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * epoch / self.T_max)) / 2

        if abs(new_lr - self.current_lr) > 1e-8 and self.verbose:
            logger.info(f"Epoch {epoch}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


class CyclicLR(LRScheduler):
    """
    Cyclic learning rate scheduler.

    Cycles the learning rate between base_lr and max_lr with a given step_size.

    Args:
        base_lr: Lower learning rate bound
        max_lr: Upper learning rate bound
        step_size: Number of iterations in half a cycle
        mode: Cycling mode ('triangular', 'triangular2', 'exp_range')
        gamma: Decay constant for exp_range mode (default: 1.0)
        verbose: Whether to log changes

    Example:
        >>> scheduler = CyclicLR(base_lr=0.001, max_lr=0.1, step_size=500)
        >>> for step in range(5000):
        ...     lr = scheduler.step(step)
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = 'triangular',
        gamma: float = 1.0,
        verbose: bool = False
    ):
        """Initialize cyclic learning rate scheduler."""
        super().__init__(base_lr, verbose)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        logger.info(
            f"Initialized CyclicLR: base_lr={base_lr}, max_lr={max_lr}, "
            f"step_size={step_size}, mode={mode}"
        )

    def step(self, iteration: Optional[int] = None) -> float:
        """Compute learning rate for current iteration."""
        if iteration is None:
            iteration = self.step_count
            self.step_count += 1

        cycle = math.floor(1 + iteration / (2 * self.step_size))
        x = abs(iteration / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** iteration
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        new_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor

        if abs(new_lr - self.current_lr) > 1e-8 and self.verbose:
            logger.debug(f"Iteration {iteration}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


class OneCycleLR(LRScheduler):
    """
    One-cycle learning rate policy.

    Implements the one-cycle policy: warm up from max_lr/div_factor to max_lr,
    then anneal down to max_lr*final_div_factor.

    Args:
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of cycle spent increasing LR (default: 0.3)
        div_factor: Initial LR = max_lr/div_factor (default: 25)
        final_div_factor: Final LR = max_lr/final_div_factor (default: 10000)
        verbose: Whether to log changes

    Example:
        >>> scheduler = OneCycleLR(max_lr=0.1, total_steps=10000)
        >>> for step in range(10000):
        ...     lr = scheduler.step(step)
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25,
        final_div_factor: float = 10000,
        verbose: bool = False
    ):
        """Initialize one-cycle LR scheduler."""
        initial_lr = max_lr / div_factor
        super().__init__(initial_lr, verbose)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up

        logger.info(
            f"Initialized OneCycleLR: max_lr={max_lr}, total_steps={total_steps}, "
            f"pct_start={pct_start}"
        )

    def step(self, iteration: Optional[int] = None) -> float:
        """Compute learning rate for current iteration."""
        if iteration is None:
            iteration = self.step_count
            self.step_count += 1

        if iteration < self.step_size_up:
            # Warmup phase
            pct = iteration / self.step_size_up
            new_lr = self.initial_lr + pct * (self.max_lr - self.initial_lr)
        else:
            # Annealing phase
            pct = (iteration - self.step_size_up) / self.step_size_down
            final_lr = self.max_lr / self.final_div_factor
            new_lr = self.max_lr - pct * (self.max_lr - final_lr)

        if abs(new_lr - self.current_lr) > 1e-8 and self.verbose:
            logger.debug(f"Iteration {iteration}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when a metric has stopped improving.

    Monitors a metric and reduces LR by factor when the metric plateaus.

    Args:
        initial_lr: Initial learning rate
        mode: 'min' for loss, 'max' for accuracy (default: 'min')
        factor: Factor to reduce LR (default: 0.1)
        patience: Number of epochs with no improvement before reducing (default: 10)
        threshold: Threshold for measuring improvement (default: 1e-4)
        min_lr: Minimum learning rate (default: 1e-7)
        verbose: Whether to log changes

    Example:
        >>> scheduler = ReduceLROnPlateau(initial_lr=0.01, patience=10)
        >>> for epoch in range(100):
        ...     loss = train_epoch()
        ...     lr = scheduler.step(loss)
    """

    def __init__(
        self,
        initial_lr: float,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-7,
        verbose: bool = False
    ):
        """Initialize ReduceLROnPlateau scheduler."""
        super().__init__(initial_lr, verbose)

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        self.best_metric = None
        self.num_bad_epochs = 0

        logger.info(
            f"Initialized ReduceLROnPlateau: initial_lr={initial_lr}, "
            f"mode={mode}, factor={factor}, patience={patience}"
        )

    def step(self, metric: float) -> float:
        """
        Update learning rate based on metric.

        Args:
            metric: Current metric value (loss or accuracy)

        Returns:
            Current learning rate
        """
        if self.best_metric is None:
            self.best_metric = metric
            return self.current_lr

        # Check if metric improved
        if self.mode == 'min':
            improved = metric < self.best_metric - self.threshold
        else:  # mode == 'max'
            improved = metric > self.best_metric + self.threshold

        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if plateau detected
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)

            if new_lr != self.current_lr:
                if self.verbose:
                    logger.info(
                        f"Metric plateau detected. Reducing LR: "
                        f"{self.current_lr:.6f} -> {new_lr:.6f}"
                    )
                self.current_lr = new_lr
                self.num_bad_epochs = 0

        return self.current_lr


class WarmupScheduler(LRScheduler):
    """
    Learning rate warmup scheduler.

    Linearly increases learning rate from 0 to target_lr over warmup_steps,
    then optionally applies a base scheduler.

    Args:
        target_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        base_scheduler: Optional scheduler to apply after warmup
        verbose: Whether to log changes

    Example:
        >>> base = CosineAnnealingLR(initial_lr=0.1, T_max=100)
        >>> scheduler = WarmupScheduler(target_lr=0.1, warmup_steps=10, base_scheduler=base)
        >>> for step in range(110):
        ...     lr = scheduler.step(step)
    """

    def __init__(
        self,
        target_lr: float,
        warmup_steps: int,
        base_scheduler: Optional[LRScheduler] = None,
        verbose: bool = False
    ):
        """Initialize warmup scheduler."""
        super().__init__(0.0, verbose)
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler

        logger.info(
            f"Initialized WarmupScheduler: target_lr={target_lr}, "
            f"warmup_steps={warmup_steps}"
        )

    def step(self, step: Optional[int] = None) -> float:
        """Compute learning rate with warmup."""
        if step is None:
            step = self.step_count
            self.step_count += 1

        if step < self.warmup_steps:
            # Warmup phase
            new_lr = self.target_lr * (step + 1) / self.warmup_steps
        else:
            # After warmup
            if self.base_scheduler:
                new_lr = self.base_scheduler.step(step - self.warmup_steps)
            else:
                new_lr = self.target_lr

        if abs(new_lr - self.current_lr) > 1e-8 and self.verbose:
            logger.debug(f"Step {step}: LR {self.current_lr:.6f} -> {new_lr:.6f}")

        self.current_lr = new_lr
        return self.current_lr


def create_scheduler(
    scheduler_type: str,
    initial_lr: float,
    **kwargs
) -> LRScheduler:
    """
    Convenience function to create a learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler ('step', 'exponential', 'cosine', etc.)
        initial_lr: Initial learning rate
        **kwargs: Scheduler-specific parameters

    Returns:
        LRScheduler instance

    Example:
        >>> scheduler = create_scheduler('cosine', initial_lr=0.01, T_max=100)
        >>> scheduler = create_scheduler('step', initial_lr=0.1, step_size=30)
    """
    schedulers = {
        'step': StepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'cyclic': CyclicLR,
        'onecycle': OneCycleLR,
        'plateau': ReduceLROnPlateau,
        'warmup': WarmupScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. "
            f"Supported: {list(schedulers.keys())}"
        )

    scheduler_cls = schedulers[scheduler_type]

    # Handle special cases
    if scheduler_type == 'cyclic':
        # CyclicLR uses base_lr instead of initial_lr
        kwargs['base_lr'] = initial_lr
        return scheduler_cls(**kwargs)
    elif scheduler_type == 'onecycle':
        # OneCycleLR uses max_lr
        kwargs['max_lr'] = initial_lr
        return scheduler_cls(**kwargs)
    elif scheduler_type == 'warmup':
        # WarmupScheduler uses target_lr
        kwargs['target_lr'] = initial_lr
        return scheduler_cls(**kwargs)
    else:
        return scheduler_cls(initial_lr=initial_lr, **kwargs)
