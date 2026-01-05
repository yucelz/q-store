"""
MLflow Tracking Example - Demonstrates MLflow integration for experiment tracking.

This example shows:
- MLflow configuration and connection
- Parameter and metric logging
- Model tracking and versioning
- Artifact management
- Run comparison
"""

import numpy as np
import time
from q_store.ml.tracking import MLflowTracker, MLflowConfig


def example_basic_mlflow_tracking():
    """Basic MLflow tracking example."""
    print("\n" + "="*70)
    print("Example 1: Basic MLflow Tracking")
    print("="*70)

    # Configure MLflow
    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_basics',
        run_name_prefix='basic_run'
    )

    try:
        tracker = MLflowTracker(config)

        # Start run
        tracker.start_run(run_name='basic_experiment_001')

        # Log parameters
        tracker.log_params({
            'n_qubits': 4,
            'circuit_depth': 3,
            'learning_rate': 0.01,
            'batch_size': 32,
            'optimizer': 'adam'
        })

        # Simulate training
        print("\nSimulating training:")
        for epoch in range(10):
            loss = 1.0 - epoch * 0.08
            accuracy = 0.5 + epoch * 0.04

            tracker.log_metric('train_loss', loss, step=epoch)
            tracker.log_metric('train_accuracy', accuracy, step=epoch)

            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}, acc={accuracy:.4f}")

        # Log final metrics
        tracker.log_metrics({
            'final_loss': loss,
            'final_accuracy': accuracy
        })

        # End run
        tracker.end_run()

        print("\n✓ Basic MLflow tracking successful")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Tracking URI: {config.tracking_uri}")

        return tracker

    except Exception as e:
        print(f"⚠ MLflow not available: {e}")
        print("Install with: pip install mlflow")
        return None


def example_multiple_runs():
    """Track multiple training runs."""
    print("\n" + "="*70)
    print("Example 2: Multiple Training Runs")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_hyperparameter_search'
    )

    try:
        tracker = MLflowTracker(config)

        # Run multiple experiments with different hyperparameters
        learning_rates = [0.001, 0.01, 0.1]

        print("\nRunning experiments with different learning rates:")
        for i, lr in enumerate(learning_rates):
            tracker.start_run(run_name=f'lr_{lr}')

            # Log hyperparameters
            tracker.log_params({
                'n_qubits': 4,
                'learning_rate': lr,
                'batch_size': 32
            })

            # Simulate training
            for epoch in range(20):
                # Different LR produces different results
                loss = 1.0 / (1 + epoch * lr)
                accuracy = 1.0 - 1.0 / (1 + epoch * lr * 0.5)

                tracker.log_metric('loss', loss, step=epoch)
                tracker.log_metric('accuracy', accuracy, step=epoch)

            # Log final results
            tracker.log_metrics({
                'final_loss': loss,
                'final_accuracy': accuracy
            })

            tracker.end_run()

            print(f"  Run {i+1}: LR={lr}, Final loss={loss:.4f}, Final acc={accuracy:.4f}")

        print("\n✓ Multiple runs tracked successfully")
        print("  View results: mlflow ui --backend-store-uri ./mlruns")

        return tracker

    except Exception as e:
        print(f"⚠ MLflow tracking failed: {e}")
        return None


def example_model_logging():
    """Log models to MLflow."""
    print("\n" + "="*70)
    print("Example 3: Model Logging")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_models',
        auto_log_models=True
    )

    try:
        tracker = MLflowTracker(config)
        tracker.start_run(run_name='model_logging_test')

        # Log parameters
        tracker.log_params({
            'model_type': 'quantum_neural_network',
            'n_qubits': 8,
            'n_layers': 4
        })

        # Simulate model (simple dict for demo)
        model = {
            'weights': np.random.randn(32),
            'biases': np.random.randn(4),
            'architecture': 'quantum_cnn'
        }

        # Log model
        tracker.log_model(
            model,
            artifact_path='model',
            model_name='quantum_model_v1'
        )

        print("  Model logged to MLflow")

        # Log model metadata
        tracker.log_dict({
            'model_architecture': {
                'type': 'quantum_cnn',
                'layers': [
                    {'type': 'quantum_conv', 'qubits': 8},
                    {'type': 'entanglement', 'pattern': 'linear'},
                    {'type': 'measurement'}
                ]
            }
        }, 'model_architecture.json')

        tracker.end_run()

        print("✓ Model logging successful")

        return tracker

    except Exception as e:
        print(f"⚠ Model logging failed: {e}")
        return None


def example_artifact_logging():
    """Log artifacts (plots, data, configs)."""
    print("\n" + "="*70)
    print("Example 4: Artifact Logging")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_artifacts'
    )

    try:
        tracker = MLflowTracker(config)
        tracker.start_run(run_name='artifact_test')

        # Log configuration as artifact
        config_dict = {
            'model': {
                'n_qubits': 8,
                'depth': 4,
                'entanglement': 'linear'
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'lr': 0.01
            }
        }
        tracker.log_dict(config_dict, 'config.json')
        print("  Logged config.json")

        # Log training data statistics
        data_stats = {
            'train_samples': 5000,
            'test_samples': 1000,
            'num_features': 784,
            'num_classes': 10
        }
        tracker.log_dict(data_stats, 'data_stats.json')
        print("  Logged data_stats.json")

        # Log numpy arrays
        weights = np.random.randn(100)
        tracker.log_numpy(weights, 'weights.npy')
        print("  Logged weights.npy")

        tracker.end_run()

        print("✓ Artifact logging successful")

        return tracker

    except Exception as e:
        print(f"⚠ Artifact logging failed: {e}")
        return None


def example_tags_and_notes():
    """Add tags and notes to runs."""
    print("\n" + "="*70)
    print("Example 5: Tags and Notes")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_tagged',
        tags={
            'project': 'quantum_classification',
            'team': 'research'
        }
    )

    try:
        tracker = MLflowTracker(config)
        tracker.start_run(run_name='tagged_run')

        # Set tags
        tracker.set_tags({
            'model_version': 'v1.0',
            'framework': 'q-store',
            'dataset': 'fashion_mnist',
            'status': 'experimental'
        })

        # Log parameters
        tracker.log_params({
            'n_qubits': 4,
            'learning_rate': 0.01
        })

        # Log metrics
        for epoch in range(5):
            tracker.log_metric('loss', 1.0 - epoch * 0.1, step=epoch)

        # Add notes
        notes = """
        Experimental run with 4 qubits.
        Testing new entanglement pattern.
        Results look promising.
        """
        tracker.log_text(notes, 'notes.txt')

        tracker.end_run()

        print("✓ Tags and notes added successfully")
        print("  Tags: model_version, framework, dataset, status")

        return tracker

    except Exception as e:
        print(f"⚠ Tags and notes failed: {e}")
        return None


def example_nested_runs():
    """Create nested runs for hyperparameter tuning."""
    print("\n" + "="*70)
    print("Example 6: Nested Runs (Hyperparameter Tuning)")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_tuning'
    )

    try:
        tracker = MLflowTracker(config)

        # Parent run
        tracker.start_run(run_name='hyperparameter_tuning_session')

        tracker.log_params({
            'tuning_method': 'grid_search',
            'param_space': 'lr=[0.001, 0.01, 0.1]'
        })

        best_accuracy = 0
        best_lr = 0

        print("\nRunning nested experiments:")
        for lr in [0.001, 0.01, 0.1]:
            # Child run
            with tracker.start_nested_run(run_name=f'lr_{lr}'):
                tracker.log_param('learning_rate', lr)

                # Simulate training
                accuracy = 0.5 + lr * 2
                tracker.log_metric('final_accuracy', accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_lr = lr

                print(f"  LR={lr}: accuracy={accuracy:.4f}")

        # Log best results to parent
        tracker.log_metrics({
            'best_accuracy': best_accuracy,
            'best_learning_rate': best_lr
        })

        tracker.end_run()

        print(f"\n✓ Nested runs successful")
        print(f"  Best LR: {best_lr}, Best accuracy: {best_accuracy:.4f}")

        return tracker

    except Exception as e:
        print(f"⚠ Nested runs failed: {e}")
        return None


def example_complete_mlflow_workflow():
    """Complete MLflow workflow with all features."""
    print("\n" + "="*70)
    print("Example 7: Complete MLflow Workflow")
    print("="*70)

    config = MLflowConfig(
        tracking_uri='./mlruns',
        experiment_name='quantum_ml_complete',
        run_name_prefix='complete',
        tags={'project': 'demo', 'version': 'v1'}
    )

    try:
        tracker = MLflowTracker(config)

        # Step 1: Start run
        print("\nStep 1: Start MLflow run")
        tracker.start_run(run_name='complete_workflow')

        # Step 2: Log configuration
        print("Step 2: Log configuration")
        tracker.log_params({
            'n_qubits': 8,
            'circuit_depth': 4,
            'learning_rate': 0.01,
            'batch_size': 32,
            'optimizer': 'adam',
            'epochs': 50
        })

        # Step 3: Set tags
        print("Step 3: Set tags")
        tracker.set_tags({
            'model': 'quantum_cnn',
            'dataset': 'fashion_mnist',
            'status': 'completed'
        })

        # Step 4: Training loop
        print("Step 4: Simulate training")
        for epoch in range(10):
            train_loss = 1.0 - epoch * 0.08
            val_loss = 1.0 - epoch * 0.07
            accuracy = 0.5 + epoch * 0.04

            tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy
            }, step=epoch)

            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Step 5: Log artifacts
        print("Step 5: Log artifacts")
        tracker.log_dict({
            'training_summary': {
                'final_loss': train_loss,
                'final_accuracy': accuracy,
                'total_epochs': 10
            }
        }, 'summary.json')

        # Step 6: Log model
        print("Step 6: Log model")
        model = {'weights': np.random.randn(64)}
        tracker.log_model(model, 'model', 'quantum_model')

        # Step 7: End run
        print("Step 7: End run")
        tracker.end_run()

        print("\n✓ Complete MLflow workflow successful")
        print("\nTo view results, run:")
        print("  mlflow ui --backend-store-uri ./mlruns")
        print("  Then open: http://localhost:5000")

        return tracker

    except Exception as e:
        print(f"⚠ Complete workflow failed: {e}")
        return None


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store MLflow Tracking Examples")
    print("="*70)
    print("\nNOTE: These examples require MLflow to be installed.")
    print("Install with: pip install mlflow")
    print("="*70)

    # Example 1: Basic tracking
    example_basic_mlflow_tracking()

    # Example 2: Multiple runs
    example_multiple_runs()

    # Example 3: Model logging
    example_model_logging()

    # Example 4: Artifact logging
    example_artifact_logging()

    # Example 5: Tags and notes
    example_tags_and_notes()

    # Example 6: Nested runs
    example_nested_runs()

    # Example 7: Complete workflow
    example_complete_mlflow_workflow()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Start MLflow UI: mlflow ui --backend-store-uri ./mlruns")
    print("2. Open browser: http://localhost:5000")
    print("3. Explore your experiments, compare runs, and view metrics")


if __name__ == '__main__':
    main()
