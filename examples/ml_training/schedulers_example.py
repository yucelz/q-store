"""
Learning Rate Schedulers Example - Demonstrates various LR scheduling strategies.

This example shows:
- Step decay, exponential decay, cosine annealing
- Cyclic and one-cycle learning rates
- Adaptive scheduling (ReduceLROnPlateau)
- Warmup strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from q_store.ml.schedulers import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    WarmupScheduler
)


def example_step_lr():
    """Step decay scheduler."""
    print("\n" + "="*70)
    print("Example 1: Step LR Scheduler")
    print("="*70)

    scheduler = StepLR(
        initial_lr=0.1,
        step_size=30,
        gamma=0.1,
        verbose=True
    )

    # Simulate training
    lrs = []
    for epoch in range(100):
        lr = scheduler.step(epoch)
        lrs.append(lr)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch}: LR = {lr:.6f}")

    print("✓ Step LR scheduler successful")

    return lrs


def example_exponential_lr():
    """Exponential decay scheduler."""
    print("\n" + "="*70)
    print("Example 2: Exponential LR Scheduler")
    print("="*70)

    scheduler = ExponentialLR(
        initial_lr=0.1,
        gamma=0.95,
        verbose=False
    )

    lrs = []
    for epoch in range(100):
        lr = scheduler.step(epoch)
        lrs.append(lr)

    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  LR at epoch 50: {lrs[50]:.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print(f"  Decay rate: {scheduler.gamma}")
    print("✓ Exponential LR scheduler successful")

    return lrs


def example_cosine_annealing():
    """Cosine annealing scheduler."""
    print("\n" + "="*70)
    print("Example 3: Cosine Annealing LR Scheduler")
    print("="*70)

    scheduler = CosineAnnealingLR(
        initial_lr=0.1,
        T_max=100,
        eta_min=1e-6,
        verbose=False
    )

    lrs = []
    for epoch in range(100):
        lr = scheduler.step(epoch)
        lrs.append(lr)

    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Min LR (middle): {min(lrs):.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print(f"  T_max: {scheduler.T_max}")
    print("✓ Cosine annealing scheduler successful")

    return lrs


def example_cyclic_lr():
    """Cyclic learning rate scheduler."""
    print("\n" + "="*70)
    print("Example 4: Cyclic LR Scheduler")
    print("="*70)

    scheduler = CyclicLR(
        base_lr=0.001,
        max_lr=0.1,
        step_size=20,
        mode='triangular',
        verbose=False
    )

    lrs = []
    for step in range(200):
        lr = scheduler.step(step)
        lrs.append(lr)

    print(f"  Base LR: {scheduler.base_lr:.6f}")
    print(f"  Max LR: {scheduler.max_lr:.6f}")
    print(f"  Step size: {scheduler.step_size}")
    print(f"  Mode: {scheduler.mode}")
    print(f"  LR range: [{min(lrs):.6f}, {max(lrs):.6f}]")
    print("✓ Cyclic LR scheduler successful")

    return lrs


def example_one_cycle_lr():
    """One-cycle learning rate policy."""
    print("\n" + "="*70)
    print("Example 5: One-Cycle LR Scheduler")
    print("="*70)

    scheduler = OneCycleLR(
        max_lr=0.1,
        total_steps=100,
        pct_start=0.3,
        verbose=False
    )

    lrs = []
    for step in range(100):
        lr = scheduler.step(step)
        lrs.append(lr)

    print(f"  Max LR: {scheduler.max_lr:.6f}")
    print(f"  Total steps: {scheduler.total_steps}")
    print(f"  Peak at: ~{int(scheduler.total_steps * scheduler.pct_start)} steps")
    print(f"  LR at peak: {max(lrs):.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print("✓ One-cycle LR scheduler successful")

    return lrs


def example_reduce_on_plateau():
    """Reduce LR on plateau scheduler."""
    print("\n" + "="*70)
    print("Example 6: ReduceLROnPlateau Scheduler")
    print("="*70)

    scheduler = ReduceLROnPlateau(
        initial_lr=0.1,
        factor=0.5,
        patience=10,
        mode='min',
        verbose=True
    )

    # Simulate training with plateaus
    lrs = []
    losses = []

    for epoch in range(100):
        # Simulate decreasing loss with plateaus
        if epoch < 30:
            loss = 1.0 - epoch * 0.02
        elif epoch < 60:
            loss = 0.4 + np.random.randn() * 0.01  # Plateau
        else:
            loss = 0.4 - (epoch - 60) * 0.005

        losses.append(loss)
        lr = scheduler.step(loss)
        lrs.append(lr)

    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print(f"  Factor: {scheduler.factor}")
    print(f"  Patience: {scheduler.patience}")
    print("✓ ReduceLROnPlateau scheduler successful")

    return lrs, losses


def example_warmup_scheduler():
    """Warmup scheduler."""
    print("\n" + "="*70)
    print("Example 7: Warmup Scheduler")
    print("="*70)

    warmup = WarmupScheduler(
        target_lr=0.1,
        warmup_steps=20,
        verbose=False
    )
    lrs = []
    for step in range(50):
        lr = warmup.step(step)
        lrs.append(lr)

    print(f"  Initial LR: {lrs[0]:.6f}")
    print(f"  Target LR: {warmup.target_lr:.6f}")
    print(f"  Warmup steps: {warmup.warmup_steps}")
    print(f"  LR at warmup end: {lrs[20]:.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    print("✓ Warmup scheduler successful")

    return lrs


def example_combined_warmup_cosine():
    """Combine warmup with cosine annealing."""
    print("\n" + "="*70)
    print("Example 8: Combined Warmup + Cosine Annealing")
    print("="*70)

    warmup = WarmupScheduler(
        target_lr=0.1,
        warmup_steps=20
    )

    cosine = CosineAnnealingLR(
        initial_lr=0.1,
        T_max=80,
        eta_min=1e-6
    )

    lrs = []
    for step in range(100):
        if step < 20:
            lr = warmup.step(step)
        else:
            lr = cosine.step(step - 20)
        lrs.append(lr)

    print(f"  Warmup phase (0-20): {lrs[0]:.6f} → {lrs[19]:.6f}")
    print(f"  Annealing phase (20-100): {lrs[20]:.6f} → {lrs[-1]:.6f}")
    print("✓ Combined scheduler successful")

    return lrs


def visualize_schedulers():
    """Visualize all schedulers."""
    print("\n" + "="*70)
    print("Example 9: Visualize All Schedulers")
    print("="*70)

    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Learning Rate Schedulers', fontsize=16)

        # Step LR
        scheduler = StepLR(0.1, step_size=30, gamma=0.1)
        lrs = [scheduler.step(i) for i in range(100)]
        axes[0, 0].plot(lrs)
        axes[0, 0].set_title('Step LR')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].grid(True, alpha=0.3)

        # Exponential LR
        scheduler = ExponentialLR(0.1, gamma=0.95)
        lrs = [scheduler.step(i) for i in range(100)]
        axes[0, 1].plot(lrs)
        axes[0, 1].set_title('Exponential LR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)

        # Cosine Annealing
        scheduler = CosineAnnealingLR(0.1, T_max=100, eta_min=1e-6)
        lrs = [scheduler.step(i) for i in range(100)]
        axes[0, 2].plot(lrs)
        axes[0, 2].set_title('Cosine Annealing')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True, alpha=0.3)

        # Cyclic LR (triangular)
        scheduler = CyclicLR(0.001, 0.1, step_size=20, mode='triangular')
        lrs = [scheduler.step(i) for i in range(200)]
        axes[1, 0].plot(lrs)
        axes[1, 0].set_title('Cyclic LR (Triangular)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # One-Cycle LR
        scheduler = OneCycleLR(max_lr=0.1, total_steps=100, pct_start=0.3)
        lrs = [scheduler.step(i) for i in range(100)]
        axes[1, 1].plot(lrs)
        axes[1, 1].set_title('One-Cycle LR')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)

        # Warmup
        scheduler = WarmupScheduler(target_lr=0.1, warmup_steps=20)
        lrs = [scheduler.step(i) for i in range(50)]
        axes[1, 2].plot(lrs)
        axes[1, 2].set_title('Warmup')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True, alpha=0.3)

        # Combined Warmup + Cosine
        warmup = WarmupScheduler(target_lr=0.1, warmup_steps=20)
        cosine = CosineAnnealingLR(initial_lr=0.1, T_max=80, eta_min=1e-6)
        lrs = []
        for i in range(100):
            if i < 20:
                lrs.append(warmup.step(i))
            else:
                lrs.append(cosine.step(i - 20))
        axes[2, 0].plot(lrs)
        axes[2, 0].set_title('Warmup + Cosine')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].grid(True, alpha=0.3)

        # Cyclic (exp_range)
        scheduler = CyclicLR(0.001, 0.1, step_size=20, mode='exp_range', gamma=0.99)
        lrs = [scheduler.step(i) for i in range(200)]
        axes[2, 1].plot(lrs)
        axes[2, 1].set_title('Cyclic LR (Exp Range)')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].grid(True, alpha=0.3)

        # ReduceLROnPlateau (simulated)
        scheduler = ReduceLROnPlateau(initial_lr=0.1, factor=0.5, patience=10)
        lrs = []
        for epoch in range(100):
            if epoch < 30:
                loss = 1.0 - epoch * 0.02
            elif epoch < 60:
                loss = 0.4
            else:
                loss = 0.4 - (epoch - 60) * 0.005
            lrs.append(scheduler.step(loss))
        axes[2, 2].plot(lrs)
        axes[2, 2].set_title('ReduceLROnPlateau')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Learning Rate')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lr_schedulers_visualization.png', dpi=300, bbox_inches='tight')
        print("  Saved visualization to: lr_schedulers_visualization.png")
        print("✓ Visualization successful")

    except Exception as e:
        print(f"⚠ Visualization failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Learning Rate Schedulers Examples")
    print("="*70)

    # Example 1: Step LR
    try:
        example_step_lr()
    except Exception as e:
        print(f"⚠ Step LR failed: {e}")

    # Example 2: Exponential LR
    try:
        example_exponential_lr()
    except Exception as e:
        print(f"⚠ Exponential LR failed: {e}")

    # Example 3: Cosine Annealing
    try:
        example_cosine_annealing()
    except Exception as e:
        print(f"⚠ Cosine annealing failed: {e}")

    # Example 4: Cyclic LR
    try:
        example_cyclic_lr()
    except Exception as e:
        print(f"⚠ Cyclic LR failed: {e}")

    # Example 5: One-Cycle LR
    try:
        example_one_cycle_lr()
    except Exception as e:
        print(f"⚠ One-cycle LR failed: {e}")

    # Example 6: ReduceLROnPlateau
    try:
        example_reduce_on_plateau()
    except Exception as e:
        print(f"⚠ ReduceLROnPlateau failed: {e}")

    # Example 7: Warmup
    try:
        example_warmup_scheduler()
    except Exception as e:
        print(f"⚠ Warmup scheduler failed: {e}")

    # Example 8: Combined
    try:
        example_combined_warmup_cosine()
    except Exception as e:
        print(f"⚠ Combined scheduler failed: {e}")

    # Example 9: Visualize
    try:
        visualize_schedulers()
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
