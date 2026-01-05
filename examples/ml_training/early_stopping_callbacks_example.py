"""
Early Stopping and Callbacks Example - Demonstrates training control mechanisms.

This example shows:
- Early stopping to prevent overfitting
- Convergence detection
- Model checkpointing
- CSV logging
- Progress tracking
"""

import numpy as np
import time
from q_store.ml.early_stopping import (
    EarlyStopping,
    ConvergenceDetector
)
from q_store.ml.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    ProgressCallback,
    LearningRateLogger,
    CallbackList
)


def example_early_stopping_basic():
    """Basic early stopping example."""
    print("\n" + "="*70)
    print("Example 1: Basic Early Stopping")
    print("="*70)

    early_stop = EarlyStopping(
        patience=10,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True,
        verbose=True
    )

    # Simulate training
    print("\nSimulating training with overfitting:")
    for epoch in range(100):
        # Simulate loss: improves then plateaus/increases
        if epoch < 30:
            loss = 1.0 - epoch * 0.02
        else:
            loss = 0.4 + (epoch - 30) * 0.005  # Starts increasing

        should_stop = early_stop.should_stop(epoch, loss)

        if epoch % 10 == 0 or should_stop:
            print(f"  Epoch {epoch}: loss={loss:.4f}, best={early_stop.best_value:.4f}")

        if should_stop:
            print(f"\n✓ Early stopping at epoch {epoch}")
            print(f"  Best epoch: {early_stop.best_epoch}")
            print(f"  Best loss: {early_stop.best_value:.4f}")
            break

    return early_stop


def example_early_stopping_accuracy():
    """Early stopping for accuracy (maximize)."""
    print("\n" + "="*70)
    print("Example 2: Early Stopping for Accuracy")
    print("="*70)

    early_stop = EarlyStopping(
        patience=15,
        min_delta=0.001,
        mode='max',  # Maximize accuracy
        verbose=True
    )

    print("\nSimulating training:")
    for epoch in range(100):
        # Simulate accuracy: improves then plateaus
        if epoch < 40:
            acc = 0.5 + epoch * 0.01
        else:
            acc = 0.9 + np.random.randn() * 0.001  # Plateau

        should_stop = early_stop.should_stop(epoch, acc)

        if epoch % 10 == 0 or should_stop:
            print(f"  Epoch {epoch}: acc={acc:.4f}, best={early_stop.best_value:.4f}")

        if should_stop:
            print(f"\n✓ Early stopping at epoch {epoch}")
            print(f"  Best epoch: {early_stop.best_epoch}")
            print(f"  Best accuracy: {early_stop.best_value:.4f}")
            break

    return early_stop


def example_convergence_detector():
    """Detect training convergence."""
    print("\n" + "="*70)
    print("Example 3: Convergence Detection")
    print("="*70)

    detector = ConvergenceDetector(
        window_size=10,
        threshold=0.0001,
        patience=5,
        verbose=True
    )

    print("\nSimulating training:")
    for epoch in range(100):
        # Simulate converging loss
        loss = 0.1 + 0.9 / (epoch + 1) + np.random.randn() * 0.001

        has_converged = detector.check_convergence(epoch, loss)

        if epoch % 10 == 0 or has_converged:
            print(f"  Epoch {epoch}: loss={loss:.4f}, std={detector.std:.6f}")

        if has_converged:
            print(f"\n✓ Training converged at epoch {epoch}")
            print(f"  Final loss: {loss:.4f}")
            print(f"  Standard deviation: {detector.std:.6f}")
            break

    return detector


def example_model_checkpoint():
    """Save best models during training."""
    print("\n" + "="*70)
    print("Example 4: Model Checkpointing")
    print("="*70)

    checkpoint = ModelCheckpoint(
        filepath='best_model_{epoch:03d}_{val_loss:.4f}.pkl',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=True
    )

    # Simulate training
    checkpoint.on_train_begin()

    print("\nSimulating training:")
    for epoch in range(20):
        # Training
        train_loss = 1.0 - epoch * 0.03

        # Validation
        val_loss = 1.0 - epoch * 0.025 + np.random.randn() * 0.02

        logs = {
            'loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch
        }

        checkpoint.on_epoch_begin(epoch)
        checkpoint.on_epoch_end(epoch, logs)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    checkpoint.on_train_end()

    print(f"\n✓ Best model saved at epoch {checkpoint.best_epoch}")
    print(f"  Best val_loss: {checkpoint.best_value:.4f}")

    return checkpoint


def example_csv_logger():
    """Log metrics to CSV."""
    print("\n" + "="*70)
    print("Example 5: CSV Logger")
    print("="*70)

    logger = CSVLogger(
        filename='training_log.csv',
        separator=',',
        append=False
    )

    logger.on_train_begin()

    print("\nLogging training metrics:")
    for epoch in range(10):
        logs = {
            'epoch': epoch,
            'loss': 1.0 - epoch * 0.05,
            'accuracy': 0.5 + epoch * 0.04,
            'val_loss': 1.0 - epoch * 0.04,
            'val_accuracy': 0.5 + epoch * 0.035,
            'lr': 0.01 * (0.95 ** epoch)
        }

        logger.on_epoch_end(epoch, logs)

        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}")

    logger.on_train_end()

    print("\n✓ Metrics saved to training_log.csv")

    return logger


def example_progress_callback():
    """Track training progress."""
    print("\n" + "="*70)
    print("Example 6: Progress Tracking")
    print("="*70)

    progress = ProgressCallback(
        total_epochs=50,
        print_every=10,
        show_eta=True
    )

    progress.on_train_begin()

    print("\nSimulating training:")
    for epoch in range(50):
        progress.on_epoch_begin(epoch)

        # Simulate epoch training
        time.sleep(0.01)

        logs = {
            'loss': 1.0 - epoch * 0.015,
            'val_loss': 1.0 - epoch * 0.012
        }

        progress.on_epoch_end(epoch, logs)

    progress.on_train_end()

    print("\n✓ Training progress tracked")

    return progress


def example_lr_logger():
    """Log learning rate changes."""
    print("\n" + "="*70)
    print("Example 7: Learning Rate Logger")
    print("="*70)

    lr_logger = LearningRateLogger()

    lr_logger.on_train_begin()

    print("\nLogging LR changes:")
    for epoch in range(20):
        # Simulate LR schedule
        if epoch < 10:
            lr = 0.01
        elif epoch < 15:
            lr = 0.001
        else:
            lr = 0.0001

        logs = {'lr': lr}
        lr_logger.on_epoch_end(epoch, logs)

        if epoch in [0, 9, 10, 14, 15, 19]:
            print(f"  Epoch {epoch}: LR = {lr:.6f}")

    lr_logger.on_train_end()

    print("\n✓ LR logging successful")
    print(f"  LR history: {len(lr_logger.lr_history)} epochs")

    return lr_logger


def example_callback_list():
    """Use multiple callbacks together."""
    print("\n" + "="*70)
    print("Example 8: Callback List")
    print("="*70)

    # Create multiple callbacks
    callbacks = CallbackList([
        EarlyStopping(patience=10, verbose=False),
        ModelCheckpoint('model.pkl', monitor='val_loss', save_best_only=True, verbose=False),
        CSVLogger('metrics.csv'),
        LearningRateLogger()
    ])

    callbacks.on_train_begin()

    print("\nSimulating training with multiple callbacks:")
    for epoch in range(30):
        callbacks.on_epoch_begin(epoch)

        # Simulate training
        logs = {
            'epoch': epoch,
            'loss': 1.0 - epoch * 0.02,
            'val_loss': 1.0 - epoch * 0.015 + np.random.randn() * 0.01,
            'accuracy': 0.5 + epoch * 0.015,
            'lr': 0.01 * (0.95 ** epoch)
        }

        callbacks.on_epoch_end(epoch, logs)

        # Check for early stopping
        if callbacks.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")

    callbacks.on_train_end()

    print("\n✓ Multi-callback training successful")
    print("  Files created: model.pkl, metrics.csv")

    return callbacks


def example_complete_training_loop():
    """Complete training loop with all components."""
    print("\n" + "="*70)
    print("Example 9: Complete Training Loop")
    print("="*70)

    # Setup callbacks
    callbacks = CallbackList([
        EarlyStopping(
            patience=15,
            min_delta=0.001,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            filepath='best_quantum_model.pkl',
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        ),
        CSVLogger('complete_training_log.csv'),
        ProgressCallback(total_epochs=100, print_every=10),
        LearningRateLogger()
    ])

    # Setup convergence detector
    convergence = ConvergenceDetector(
        window_size=10,
        threshold=0.0001,
        patience=5
    )

    # Training loop
    callbacks.on_train_begin()

    print("\nStarting complete training:")
    for epoch in range(100):
        callbacks.on_epoch_begin(epoch)

        # Simulate training
        train_loss = 1.0 / (epoch + 1) + np.random.randn() * 0.01
        val_loss = 1.0 / (epoch + 1) + np.random.randn() * 0.015
        accuracy = 1.0 - 1.0 / (epoch + 2)
        lr = 0.01 * (0.95 ** epoch)

        logs = {
            'epoch': epoch,
            'loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'lr': lr
        }

        callbacks.on_epoch_end(epoch, logs)

        # Check convergence
        if convergence.check_convergence(epoch, val_loss):
            print(f"\n  Training converged at epoch {epoch}")
            break

        # Check early stopping
        if callbacks.should_stop:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    callbacks.on_train_end()

    print("\n✓ Complete training loop successful")
    print("  Training completed with early stopping and convergence detection")

    return callbacks


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Early Stopping & Callbacks Examples")
    print("="*70)

    # Early stopping examples
    try:
        example_early_stopping_basic()
    except Exception as e:
        print(f"⚠ Early stopping (basic) failed: {e}")

    try:
        example_early_stopping_accuracy()
    except Exception as e:
        print(f"⚠ Early stopping (accuracy) failed: {e}")

    try:
        example_convergence_detector()
    except Exception as e:
        print(f"⚠ Convergence detector failed: {e}")

    # Callback examples
    try:
        example_model_checkpoint()
    except Exception as e:
        print(f"⚠ Model checkpoint failed: {e}")

    try:
        example_csv_logger()
    except Exception as e:
        print(f"⚠ CSV logger failed: {e}")

    try:
        example_progress_callback()
    except Exception as e:
        print(f"⚠ Progress callback failed: {e}")

    try:
        example_lr_logger()
    except Exception as e:
        print(f"⚠ LR logger failed: {e}")

    try:
        example_callback_list()
    except Exception as e:
        print(f"⚠ Callback list failed: {e}")

    # Complete example
    try:
        example_complete_training_loop()
    except Exception as e:
        print(f"⚠ Complete training loop failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
