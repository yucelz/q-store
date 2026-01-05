"""
MLflow Experiment Tracking with Q-Store v4.1.1

Demonstrates comprehensive experiment tracking:
- MLflow integration
- Structured logging
- Metrics tracking and analysis
- Model versioning

Prerequisites:
    pip install q-store[ml,tracking]

Optional: Start MLflow UI
    mlflow ui --backend-store-uri ./mlruns
"""

import asyncio
import numpy as np
from pathlib import Path

from q_store.backends import BackendManager
from q_store.ml import (
    QuantumModel,
    QuantumTrainer,
    TrainingConfig,
)

# v4.1.1: Tracking imports
from q_store.ml import (
    MLflowTracker,
    MLflowConfig,
    QuantumMLLogger,
    LogLevel,
    MetricsTracker,
    MetricsAnalyzer,
    MLflowCallback,
)

from q_store.data import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
    QuantumPreprocessor,
    QuantumDataGenerator,
)


async def experiment_1_baseline():
    """Baseline experiment with default parameters."""
    print("\n" + "=" * 80)
    print("Experiment 1: Baseline Configuration")
    print("=" * 80)

    # Configure MLflow
    mlflow_config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_mnist_experiments',
        run_name_prefix='baseline'
    )

    tracker = MLflowTracker(mlflow_config)

    if not tracker.mlflow_available:
        print("⚠ MLflow not available. Install with: pip install mlflow")
        return

    # Start MLflow run
    tracker.start_run(
        run_name='baseline_run_1',
        tags={
            'model_type': 'quantum_neural_network',
            'experiment_type': 'baseline',
            'framework': 'q-store'
        }
    )

    # Log hyperparameters
    params = {
        'n_qubits': 6,
        'circuit_depth': 3,
        'learning_rate': 0.01,
        'batch_size': 32,
        'optimizer': 'adam',
        'gradient_method': 'spsa_subsampled'
    }
    tracker.log_params(params)

    # Simulate training
    print("\nSimulating training...")
    for epoch in range(20):
        # Simulate metrics
        train_loss = 0.5 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.01)
        val_loss = 0.6 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.02)
        accuracy = min(0.95, 0.5 + epoch * 0.02 + np.random.normal(0, 0.01))

        # Log metrics
        tracker.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': params['learning_rate']
        }, step=epoch)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={accuracy:.4f}")

    # Log final results
    tracker.log_metric('final_accuracy', accuracy)
    tracker.set_tag('status', 'completed')

    # End run
    tracker.end_run()

    print("\n✓ Experiment 1 complete")
    print(f"  Run ID: {tracker.get_run_id()}")
    print(f"  Final accuracy: {accuracy:.4f}")


async def experiment_2_with_lr_scheduling():
    """Experiment with learning rate scheduling."""
    print("\n" + "=" * 80)
    print("Experiment 2: With Learning Rate Scheduling")
    print("=" * 80)

    mlflow_config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_mnist_experiments',
    )

    tracker = MLflowTracker(mlflow_config)

    if not tracker.mlflow_available:
        return

    tracker.start_run(
        run_name='lr_scheduling_run_1',
        tags={
            'model_type': 'quantum_neural_network',
            'experiment_type': 'lr_scheduling',
            'scheduler': 'cosine_annealing'
        }
    )

    # Log hyperparameters
    params = {
        'n_qubits': 8,
        'circuit_depth': 4,
        'learning_rate': 0.05,
        'lr_scheduler': 'cosine',
        'batch_size': 32,
        'optimizer': 'adam'
    }
    tracker.log_params(params)

    # Simulate training with LR decay
    print("\nSimulating training with LR scheduling...")
    for epoch in range(30):
        # Cosine annealing LR
        lr = 0.001 + (0.05 - 0.001) * (1 + np.cos(np.pi * epoch / 30)) / 2

        # Simulate metrics (better convergence with scheduling)
        train_loss = 0.4 * np.exp(-epoch * 0.12) + np.random.normal(0, 0.008)
        val_loss = 0.5 * np.exp(-epoch * 0.10) + np.random.normal(0, 0.015)
        accuracy = min(0.97, 0.6 + epoch * 0.015 + np.random.normal(0, 0.008))

        tracker.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': lr
        }, step=epoch)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f}, lr={lr:.6f}, acc={accuracy:.4f}")

    tracker.log_metric('final_accuracy', accuracy)
    tracker.set_tag('status', 'completed')
    tracker.end_run()

    print("\n✓ Experiment 2 complete")
    print(f"  Final accuracy: {accuracy:.4f} (improved with LR scheduling)")


async def experiment_3_comprehensive_tracking():
    """Comprehensive tracking with all v4.1.1 features."""
    print("\n" + "=" * 80)
    print("Experiment 3: Comprehensive Tracking")
    print("=" * 80)

    # Initialize all trackers
    mlflow_tracker = MLflowTracker(
        MLflowConfig(
            tracking_uri='./mlruns',
            experiment_name='quantum_mnist_experiments',
        )
    )

    if not mlflow_tracker.mlflow_available:
        return

    # Initialize structured logger
    logger = QuantumMLLogger(
        name='experiment_3',
        log_level=LogLevel.INFO,
        log_file='experiment_3.log',
        structured=True
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(auto_compute_stats=True)
    analyzer = MetricsAnalyzer(metrics_tracker)

    # Start MLflow run
    mlflow_tracker.start_run(
        run_name='comprehensive_tracking',
        tags={
            'experiment_type': 'comprehensive',
            'features': 'mlflow+logger+metrics'
        }
    )

    # Log experiment info
    logger.experiment_info({
        'experiment_id': 'exp_003',
        'description': 'Comprehensive tracking demonstration',
        'features': ['mlflow', 'structured_logging', 'metrics_analysis']
    })

    # Log hyperparameters
    params = {
        'n_qubits': 10,
        'circuit_depth': 5,
        'learning_rate': 0.02,
        'batch_size': 64,
        'enable_early_stopping': True,
        'lr_scheduler': 'onecycle'
    }

    mlflow_tracker.log_params(params)
    logger.params(params)

    # Training simulation
    logger.training_start(epochs=40, batch_size=64)
    print("\nRunning comprehensive tracked training...")

    for epoch in range(40):
        logger.epoch_start(epoch)

        # Simulate training
        train_loss = 0.3 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.005)
        val_loss = 0.35 * np.exp(-epoch * 0.12) + np.random.normal(0, 0.01)
        accuracy = min(0.98, 0.65 + epoch * 0.012 + np.random.normal(0, 0.005))
        grad_norm = 0.5 * np.exp(-epoch * 0.08)

        # Track with all systems
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'gradient_norm': grad_norm
        }

        mlflow_tracker.log_metrics(metrics, step=epoch)
        metrics_tracker.add_metrics(metrics, step=epoch)
        logger.epoch_end(epoch, metrics)

        if epoch % 10 == 0:
            # Get metrics statistics
            train_loss_stats = metrics_tracker.get_statistics('train_loss')
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f} (mean={train_loss_stats['mean']:.4f})")

        # Check for overfitting
        if epoch > 10:
            is_overfitting = analyzer.detect_overfitting('train_loss', 'val_loss', threshold=0.05)
            if is_overfitting:
                logger.warning(f"Potential overfitting detected at epoch {epoch}")
                mlflow_tracker.set_tag('overfitting_detected', 'true')

    # Training complete
    logger.training_end(final_loss=train_loss, final_accuracy=accuracy)

    # Analyze metrics
    print("\n" + "-" * 80)
    print("Metrics Analysis:")
    print("-" * 80)

    summary = metrics_tracker.get_metrics_summary()
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Min:  {stats['min']:.6f}")
        print(f"  Max:  {stats['max']:.6f}")

        trend = analyzer.detect_trend(metric, window=10)
        print(f"  Trend: {trend}")

    # Check convergence
    convergence = analyzer.get_convergence_status('train_loss', window=10)
    print(f"\nConvergence Status:")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Variance: {convergence['variance']:.6f}")
    print(f"  Trend: {convergence['trend']}")

    # Export metrics
    metrics_data = metrics_tracker.export_to_dict()
    mlflow_tracker.log_dict(metrics_data, 'metrics_history.json')

    # Log final results
    mlflow_tracker.log_metric('final_accuracy', accuracy)
    mlflow_tracker.log_metric('convergence_variance', convergence['variance'])
    mlflow_tracker.set_tag('converged', str(convergence['converged']))
    mlflow_tracker.set_tag('status', 'completed')

    mlflow_tracker.end_run()

    print("\n✓ Experiment 3 complete")
    print(f"  Final accuracy: {accuracy:.4f}")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Log file: experiment_3.log")


async def main():
    """Run all experiments."""

    print("=" * 80)
    print("Q-Store v4.1.1: MLflow Experiment Tracking Examples")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Basic MLflow tracking")
    print("  2. Tracking with LR scheduling")
    print("  3. Comprehensive tracking (MLflow + Logger + Metrics)")
    print("\nNote: Start MLflow UI with: mlflow ui --backend-store-uri ./mlruns")
    print("=" * 80)

    # Run experiments
    await experiment_1_baseline()
    await experiment_2_with_lr_scheduling()
    await experiment_3_comprehensive_tracking()

    # Summary
    print("\n\n" + "=" * 80)
    print("All Experiments Complete!")
    print("=" * 80)
    print("""
Next Steps:

1. View experiments in MLflow UI:
   mlflow ui --backend-store-uri ./mlruns
   Then open: http://localhost:5000

2. Compare experiments:
   - Select multiple runs
   - Click "Compare" button
   - View metrics charts and parameter comparison

3. Access logged artifacts:
   - Check ./mlruns/[experiment_id]/[run_id]/artifacts/

4. Query programmatically:
   from q_store.ml import MLflowTracker
   tracker = MLflowTracker(...)
   # Use tracker.get_*() methods

Key Features Demonstrated:
✓ Multiple experiment tracking
✓ Hyperparameter logging
✓ Metrics tracking over time
✓ Structured logging
✓ Metrics analysis and convergence detection
✓ Overfitting detection
✓ Artifact logging
✓ Tags and metadata
    """)
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
