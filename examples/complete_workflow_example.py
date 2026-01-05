"""
Complete End-to-End Workflow Example

This example demonstrates a complete quantum ML pipeline using all new features:
1. Data loading and preprocessing
2. Quantum adaptation
3. Training with callbacks and schedulers
4. MLflow tracking
5. Hyperparameter tuning

This is a comprehensive example showing how all components work together.
"""

import numpy as np
import time
from pathlib import Path

# Data management imports
from q_store.data.loaders import DatasetLoader, DatasetConfig, DatasetSource
from q_store.data.preprocessing import QuantumPreprocessor, DataSplitter
from q_store.data.adapters import DimensionReducer, QuantumDataAdapter, EncodingType
from q_store.data.generators import QuantumDataGenerator
from q_store.data.validation import DataValidator, DataProfiler
from q_store.data.augmentation import QuantumAugmentation

# ML training imports
from q_store.ml.schedulers import CosineAnnealingLR, WarmupScheduler
from q_store.ml.early_stopping import EarlyStopping, ConvergenceDetector
from q_store.ml.callbacks import (
    CallbackList, ModelCheckpoint, CSVLogger,
    ProgressCallback, LearningRateLogger
)

# Optional: MLflow tracking
try:
    from q_store.ml.tracking import MLflowTracker, MLflowConfig
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠ MLflow not available. Install with: pip install mlflow")

# Optional: Hyperparameter tuning
try:
    from q_store.ml.tuning import RandomSearch
    TUNING_AVAILABLE = True
except ImportError:
    TUNING_AVAILABLE = False


def simulate_quantum_model_training(x_batch, y_batch, lr):
    """
    Simulate quantum model training for demonstration.
    In real use, replace with actual quantum circuit training.
    """
    # Simulate loss based on data and learning rate
    loss = 0.5 + np.random.randn() * 0.1 + (1.0 - lr) * 0.1
    accuracy = 0.7 + np.random.randn() * 0.05

    # Simulate training time
    time.sleep(0.01)

    return loss, accuracy


def load_and_prepare_data():
    """Step 1: Load and prepare dataset."""
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*70)

    # Load Fashion MNIST
    print("\n1.1 Loading Fashion MNIST dataset...")
    config = DatasetConfig(
        name='fashion_mnist',
        source=DatasetSource.KERAS,
        source_params={
            'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
        }
    )

    dataset = DatasetLoader.load(config)
    print(f"✓ Loaded {dataset.num_samples} samples")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Shape: {dataset.x_train.shape}")

    # Validate data
    print("\n1.2 Validating data quality...")
    validator = DataValidator()
    is_valid, message = validator.validate_shapes(
        dataset.x_train, dataset.y_train
    )
    print(f"✓ Data validation: {message}")

    # Profile data
    print("\n1.3 Profiling dataset...")
    profiler = DataProfiler()
    stats = profiler.profile(dataset.x_train[:1000], dataset.y_train[:1000])
    print(f"✓ Dataset profile:")
    print(f"  Mean: {stats['feature_mean']:.3f}")
    print(f"  Std: {stats['feature_std']:.3f}")
    print(f"  Balanced: {stats['is_balanced']}")

    return dataset


def preprocess_data(dataset):
    """Step 2: Preprocess and adapt data for quantum."""
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING AND QUANTUM ADAPTATION")
    print("="*70)

    # Flatten images
    print("\n2.1 Flattening images...")
    x_train = dataset.x_train.reshape(dataset.x_train.shape[0], -1)
    x_test = dataset.x_test.reshape(dataset.x_test.shape[0], -1)
    print(f"✓ Flattened to shape: {x_train.shape}")

    # Use subset for faster demo
    x_train = x_train[:5000]
    y_train = dataset.y_train[:5000]
    x_test = x_test[:1000]
    y_test = dataset.y_test[:1000]

    # Normalize
    print("\n2.2 Normalizing data...")
    preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
    x_train_norm = preprocessor.fit_transform(x_train)
    x_test_norm = preprocessor.transform(x_test)
    print(f"✓ Normalized to range [{x_train_norm.min():.3f}, {x_train_norm.max():.3f}]")

    # Split into train/val
    print("\n2.3 Creating validation split...")
    splitter = DataSplitter(
        split_ratio={'train': 0.8, 'val': 0.2},
        stratify=True,
        random_seed=42
    )
    splits = splitter.split(x_train_norm, y_train)
    print(f"✓ Train: {len(splits['x_train'])}, Val: {len(splits['x_val'])}")

    # Reduce dimensions for quantum
    print("\n2.4 Reducing dimensions with PCA...")
    reducer = DimensionReducer(method='pca', target_dim=16)
    x_train_reduced = reducer.fit_transform(splits['x_train'])
    x_val_reduced = reducer.transform(splits['x_val'])
    x_test_reduced = reducer.transform(x_test_norm)
    print(f"✓ Reduced to {x_train_reduced.shape[1]} dimensions")
    print(f"  Variance retained: {reducer.explained_variance_ratio_.sum():.3f}")

    # Prepare for quantum encoding
    print("\n2.5 Preparing for quantum encoding...")
    adapter = QuantumDataAdapter(n_qubits=4, encoding=EncodingType.AMPLITUDE)
    x_train_quantum = adapter.prepare(x_train_reduced)
    x_val_quantum = adapter.prepare(x_val_reduced)
    x_test_quantum = adapter.prepare(x_test_reduced)
    print(f"✓ Prepared for {adapter.n_qubits}-qubit {adapter.encoding.value} encoding")

    # Optional augmentation
    print("\n2.6 Applying quantum augmentation...")
    augmentation = QuantumAugmentation(
        phase_shift_range=0.05,
        amplitude_noise=0.01,
        probability=0.3
    )
    x_train_aug = augmentation.apply(x_train_quantum)
    print(f"✓ Augmentation applied")

    return {
        'x_train': x_train_aug,
        'y_train': splits['y_train'],
        'x_val': x_val_quantum,
        'y_val': splits['y_val'],
        'x_test': x_test_quantum,
        'y_test': y_test
    }


def setup_training_components(total_epochs):
    """Step 3: Setup training components."""
    print("\n" + "="*70)
    print("STEP 3: SETTING UP TRAINING COMPONENTS")
    print("="*70)

    # Learning rate scheduler with warmup
    print("\n3.1 Configuring learning rate scheduler...")
    warmup = WarmupScheduler(initial_lr=1e-6, target_lr=0.01, warmup_steps=10)
    scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=total_epochs-10, eta_min=1e-6)
    print("✓ Warmup (10 steps) + Cosine Annealing")

    # Early stopping
    print("\n3.2 Configuring early stopping...")
    early_stop = EarlyStopping(
        patience=15,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True,
        verbose=True
    )
    print("✓ Early stopping (patience=15)")

    # Convergence detector
    print("\n3.3 Configuring convergence detection...")
    convergence = ConvergenceDetector(
        window_size=10,
        threshold=0.0001,
        patience=5
    )
    print("✓ Convergence detector")

    # Callbacks
    print("\n3.4 Setting up callbacks...")
    callbacks = CallbackList([
        ModelCheckpoint(
            filepath='checkpoints/best_model_epoch_{epoch:03d}.pkl',
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        ),
        CSVLogger('training_log.csv'),
        ProgressCallback(total_epochs=total_epochs, print_every=5),
        LearningRateLogger()
    ])
    print("✓ Callbacks: Checkpoint, CSV Logger, Progress, LR Logger")

    # MLflow tracking
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        print("\n3.5 Setting up MLflow tracking...")
        config = MLflowConfig(
            tracking_uri='./mlruns',
            experiment_name='fashion_mnist_quantum_complete',
            run_name_prefix='complete_run'
        )
        mlflow_tracker = MLflowTracker(config)
        print("✓ MLflow tracker configured")

    return {
        'warmup': warmup,
        'scheduler': scheduler,
        'early_stop': early_stop,
        'convergence': convergence,
        'callbacks': callbacks,
        'mlflow': mlflow_tracker
    }


def train_model(data, components, hyperparams):
    """Step 4: Train the model."""
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)

    # Create data generators
    print("\n4.1 Creating data generators...")
    train_gen = QuantumDataGenerator(
        data['x_train'], data['y_train'],
        batch_size=hyperparams['batch_size'],
        shuffle=True
    )
    val_gen = QuantumDataGenerator(
        data['x_val'], data['y_val'],
        batch_size=hyperparams['batch_size'],
        shuffle=False
    )
    print(f"✓ Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")

    # Start MLflow run
    if components['mlflow']:
        print("\n4.2 Starting MLflow run...")
        components['mlflow'].start_run(run_name='complete_training')
        components['mlflow'].log_params(hyperparams)
        print("✓ MLflow run started")

    # Initialize callbacks
    components['callbacks'].on_train_begin()

    # Training loop
    print("\n4.3 Training loop:")
    print("-" * 70)

    best_val_loss = float('inf')

    for epoch in range(hyperparams['max_epochs']):
        components['callbacks'].on_epoch_begin(epoch)

        # Get learning rate
        if epoch < 10:
            lr = components['warmup'].step(epoch)
        else:
            lr = components['scheduler'].step(epoch - 10)

        # Training phase
        train_losses = []
        train_accs = []
        for x_batch, y_batch in train_gen:
            loss, acc = simulate_quantum_model_training(x_batch, y_batch, lr)
            train_losses.append(loss)
            train_accs.append(acc)

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)

        # Validation phase
        val_losses = []
        val_accs = []
        for x_batch, y_batch in val_gen:
            loss, acc = simulate_quantum_model_training(x_batch, y_batch, lr)
            val_losses.append(loss)
            val_accs.append(acc)

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        # Log metrics
        logs = {
            'epoch': epoch,
            'lr': lr,
            'loss': train_loss,
            'accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }

        # Update callbacks
        components['callbacks'].on_epoch_end(epoch, logs)

        # Log to MLflow
        if components['mlflow']:
            components['mlflow'].log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': lr
            }, step=epoch)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check early stopping
        if components['early_stop'].should_stop(epoch, val_loss):
            print(f"\n✓ Early stopping triggered at epoch {epoch}")
            break

        # Check convergence
        if components['convergence'].check_convergence(epoch, val_loss):
            print(f"\n✓ Training converged at epoch {epoch}")
            break

        # Check callback stopping
        if components['callbacks'].should_stop:
            print(f"\n✓ Training stopped by callback at epoch {epoch}")
            break

    # Training completed
    components['callbacks'].on_train_end()

    if components['mlflow']:
        components['mlflow'].log_metrics({
            'best_val_loss': best_val_loss,
            'final_epoch': epoch
        })
        components['mlflow'].end_run()

    print("\n" + "-" * 70)
    print(f"✓ Training completed at epoch {epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    return best_val_loss


def evaluate_model(data):
    """Step 5: Evaluate on test set."""
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("="*70)

    print("\n5.1 Evaluating on test set...")
    test_gen = QuantumDataGenerator(
        data['x_test'], data['y_test'],
        batch_size=32,
        shuffle=False
    )

    test_losses = []
    test_accs = []

    for x_batch, y_batch in test_gen:
        loss, acc = simulate_quantum_model_training(x_batch, y_batch, 0.01)
        test_losses.append(loss)
        test_accs.append(acc)

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)

    print(f"✓ Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


def hyperparameter_tuning(data):
    """Step 6: Optional hyperparameter tuning."""
    print("\n" + "="*70)
    print("STEP 6: HYPERPARAMETER TUNING (OPTIONAL)")
    print("="*70)

    if not TUNING_AVAILABLE:
        print("⚠ Tuning not available. Skipping...")
        return None

    print("\n6.1 Running random search for hyperparameters...")

    param_distributions = {
        'learning_rate': ('log_uniform', 1e-4, 1e-1),
        'batch_size': ('choice', [16, 32, 64]),
        'warmup_steps': ('int_uniform', 5, 15)
    }

    def objective(params):
        # Quick training with these params
        hyperparams = {
            'n_qubits': 4,
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'warmup_steps': params['warmup_steps'],
            'max_epochs': 20
        }

        # Simulate quick training
        val_loss = np.random.rand() + params['learning_rate'] * 2
        return val_loss

    tuner = RandomSearch(
        param_distributions,
        n_trials=10,
        scoring='min',
        verbose=True
    )

    best_params, best_score = tuner.search(objective)

    print(f"\n✓ Hyperparameter tuning completed")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_score:.4f}")

    return best_params


def main():
    """Run complete end-to-end workflow."""
    print("\n" + "="*70)
    print("Q-STORE COMPLETE END-TO-END WORKFLOW")
    print("Fashion MNIST Classification with Quantum Neural Network")
    print("="*70)

    # Configuration
    hyperparams = {
        'n_qubits': 4,
        'circuit_depth': 3,
        'learning_rate': 0.01,
        'batch_size': 32,
        'max_epochs': 50,
        'encoding': 'amplitude'
    }

    print("\nConfiguration:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    # Step 1: Load data
    dataset = load_and_prepare_data()

    # Step 2: Preprocess and adapt
    data = preprocess_data(dataset)

    # Step 3: Setup training
    components = setup_training_components(hyperparams['max_epochs'])

    # Step 4: Train
    best_val_loss = train_model(data, components, hyperparams)

    # Step 5: Evaluate
    test_loss, test_acc = evaluate_model(data)

    # Step 6: Optional tuning
    # best_params = hyperparameter_tuning(data)

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nSummary:")
    print(f"  Dataset: Fashion MNIST (5000 train, 1000 test)")
    print(f"  Quantum encoding: {hyperparams['encoding']} ({hyperparams['n_qubits']} qubits)")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print("\nGenerated artifacts:")
    print("  - checkpoints/best_model_*.pkl")
    print("  - training_log.csv")
    print("  - mlruns/ (MLflow data)")

    if MLFLOW_AVAILABLE:
        print("\nView results:")
        print("  mlflow ui --backend-store-uri ./mlruns")
        print("  Open: http://localhost:5000")

    print("\n" + "="*70)
    print("All steps completed successfully! 🎉")
    print("="*70)


if __name__ == '__main__':
    # Create output directories
    Path('checkpoints').mkdir(exist_ok=True)

    # Run complete workflow
    main()
