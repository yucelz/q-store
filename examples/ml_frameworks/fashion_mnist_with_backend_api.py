"""
Fashion MNIST Training with Q-Store Backend API (v4.1.1)

This example demonstrates the complete v4.1.1 workflow:
- Loading data from Q-Store Backend API
- Using quantum data preprocessing
- Training with enhanced QuantumTrainer
- Learning rate scheduling
- Early stopping
- Callbacks for tracking

Prerequisites:
    pip install q-store[ml,datasets,tracking]
"""

import asyncio
import numpy as np
from q_store.backends import BackendManager
from q_store.ml import (
    QuantumModel,
    QuantumTrainer,
    TrainingConfig,
)

# v4.1.1: New imports
from q_store.data import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
    QuantumPreprocessor,
    QuantumDataGenerator,
)
from q_store.ml import (
    ModelCheckpoint,
    CSVLogger,
    ProgressCallback,
    create_scheduler,
    EarlyStopping,
)


async def main():
    """Main training function."""

    print("=" * 80)
    print("Q-Store v4.1.1: Fashion MNIST with Backend API Example")
    print("=" * 80)

    # =========================================================================
    # 1. Load Dataset from Backend API (or fallback to Keras)
    # =========================================================================
    print("\n[1/6] Loading Fashion MNIST dataset...")

    # Try Backend API first, fallback to Keras
    try:
        dataset_config = DatasetConfig(
            name='fashion_mnist',
            source=DatasetSource.BACKEND_API,
            source_params={
                'api_url': 'http://localhost:8000',  # Update with your backend URL
                'dataset_id': 'fashion_mnist'
            }
        )
        dataset = DatasetLoader.load(dataset_config)
        print("✓ Loaded from Backend API")
    except Exception as e:
        print(f"Backend API not available ({e}), using Keras fallback...")
        dataset_config = DatasetConfig(
            name='fashion_mnist',
            source=DatasetSource.KERAS,
            source_params={
                'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
            }
        )
        dataset = DatasetLoader.load(dataset_config)
        print("✓ Loaded from Keras")

    print(f"  Training samples: {len(dataset.x_train)}")
    print(f"  Test samples: {len(dataset.x_test)}")
    print(f"  Input shape: {dataset.x_train.shape}")

    # =========================================================================
    # 2. Preprocess Data for Quantum ML
    # =========================================================================
    print("\n[2/6] Preprocessing data for quantum ML...")

    # Flatten images
    x_train = dataset.x_train.reshape(len(dataset.x_train), -1)
    x_test = dataset.x_test.reshape(len(dataset.x_test), -1)

    # Normalize to [0, 1]
    preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    # For demo, use subset
    n_train = 1000
    n_test = 200
    x_train = x_train[:n_train]
    y_train = dataset.y_train[:n_train]
    x_test = x_test[:n_test]
    y_test = dataset.y_test[:n_test]

    print(f"✓ Preprocessed {n_train} training samples")
    print(f"  Feature range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    # =========================================================================
    # 3. Create Data Generators
    # =========================================================================
    print("\n[3/6] Creating data generators...")

    train_generator = QuantumDataGenerator(
        x_train, y_train,
        batch_size=32,
        shuffle=True
    )

    val_generator = QuantumDataGenerator(
        x_test, y_test,
        batch_size=32,
        shuffle=False
    )

    print(f"✓ Created generators: batch_size=32")

    # =========================================================================
    # 4. Configure Training with v4.1.1 Features
    # =========================================================================
    print("\n[4/6] Configuring training with v4.1.1 features...")

    # Create callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath='models/best_model.pkl',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        # Log to CSV
        CSVLogger('training_log.csv'),
        # Show progress
        ProgressCallback(print_freq=5),
    ]

    # Training configuration
    config = TrainingConfig(
        # Backend
        pinecone_api_key='mock-key',
        quantum_sdk='mock',

        # Model architecture
        n_qubits=8,
        circuit_depth=3,

        # Training hyperparameters
        learning_rate=0.01,
        batch_size=32,
        epochs=50,

        # Optimizer
        optimizer='adam',
        gradient_method='spsa_subsampled',

        # v4.1.1: Learning rate scheduling
        lr_scheduler='cosine',
        lr_scheduler_params={
            'T_max': 50,
            'eta_min': 0.001
        },

        # v4.1.1: Early stopping
        enable_early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_monitor='val_loss',

        # v4.1.1: Callbacks
        callbacks=callbacks,

        # Monitoring
        log_interval=5,
        checkpoint_interval=10,
    )

    print("✓ Configuration:")
    print(f"  - n_qubits: {config.n_qubits}")
    print(f"  - circuit_depth: {config.circuit_depth}")
    print(f"  - learning_rate: {config.learning_rate}")
    print(f"  - lr_scheduler: {config.lr_scheduler}")
    print(f"  - early_stopping: enabled (patience={config.early_stopping_patience})")
    print(f"  - callbacks: {len(callbacks)}")

    # =========================================================================
    # 5. Initialize Model and Trainer
    # =========================================================================
    print("\n[5/6] Initializing model and trainer...")

    # Backend manager
    backend_manager = BackendManager(
        sdk_type='mock',
        api_key=None
    )

    # Create model
    model = QuantumModel(
        input_dim=x_train.shape[1],
        n_qubits=config.n_qubits,
        output_dim=10,  # 10 classes
        backend=backend_manager.get_backend(),
        depth=config.circuit_depth
    )

    # Create trainer
    trainer = QuantumTrainer(config, backend_manager)

    print("✓ Model initialized:")
    print(f"  - Input dim: {x_train.shape[1]}")
    print(f"  - Output dim: 10")
    print(f"  - Parameters: ~{model.n_qubits * model.depth * 3}")

    # =========================================================================
    # 6. Train Model
    # =========================================================================
    print("\n[6/6] Training model...")
    print("-" * 80)

    try:
        await trainer.train(
            model=model,
            train_loader=train_generator,
            val_loader=val_generator,
            epochs=config.epochs
        )

        print("-" * 80)
        print("\n✓ Training completed successfully!")

        # Show training summary
        if trainer.training_history:
            final_metrics = trainer.training_history[-1]
            print("\nFinal Metrics:")
            print(f"  - Loss: {final_metrics.loss:.4f}")
            print(f"  - Gradient norm: {final_metrics.gradient_norm:.4f}")
            print(f"  - Learning rate: {final_metrics.learning_rate:.6f}")
            print(f"  - Epoch time: {final_metrics.epoch_time_ms/1000:.2f}s")

        # Early stopping info
        if trainer.early_stopping and trainer.early_stopping.stopped_epoch > 0:
            print(f"\nEarly stopping triggered at epoch {trainer.early_stopping.stopped_epoch}")
            print(f"Best epoch: {trainer.early_stopping.best_epoch}")
            print(f"Best value: {trainer.early_stopping.best_value:.4f}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        raise

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total epochs: {len(trainer.training_history)}")
    print(f"Checkpoint saved: models/best_model.pkl")
    print(f"Training log: training_log.csv")
    print("\nv4.1.1 Features Used:")
    print("  ✓ Backend API dataset loading")
    print("  ✓ Quantum preprocessing")
    print("  ✓ Data generators")
    print("  ✓ Learning rate scheduling (Cosine)")
    print("  ✓ Early stopping")
    print("  ✓ Training callbacks (3)")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
