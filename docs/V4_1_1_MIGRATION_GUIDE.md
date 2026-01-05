# Q-Store v4.1.1 Migration Guide

## Overview

Q-Store v4.1.1 introduces comprehensive data management and training enhancement features while maintaining backward compatibility with v4.1.0. This guide helps you migrate existing code and leverage new features.

**Release Date**: January 2026
**Previous Version**: 4.1.0
**Breaking Changes**: None (fully backward compatible)

---

## Table of Contents

1. [What's New in v4.1.1](#whats-new-in-v411)
2. [Installation](#installation)
3. [Quick Migration Steps](#quick-migration-steps)
4. [Feature-by-Feature Migration](#feature-by-feature-migration)
5. [Code Examples](#code-examples)
6. [Common Migration Patterns](#common-migration-patterns)
7. [Troubleshooting](#troubleshooting)

---

## What's New in v4.1.1

### 🎯 Data Management Layer (Phase 1)
- **Dataset Loading**: Load from Keras, HuggingFace, Backend API, and local files
- **Data Preprocessing**: Quantum-specific normalization and transformations
- **Data Validation**: Automated checks for quantum compatibility
- **Data Augmentation**: Quantum, classical, and hybrid augmentation
- **Data Generators**: Efficient batch generation with augmentation support

### 🚀 Training Enhancements (Phase 2)
- **Learning Rate Schedulers**: 8 schedulers (Step, Cosine, Cyclic, OneCycle, etc.)
- **Early Stopping**: Prevent overfitting with convergence detection
- **Training Callbacks**: 7 callbacks (ModelCheckpoint, CSV, MLflow, TensorBoard, etc.)
- **Enhanced QuantumTrainer**: Integrated support for all new features

### 📊 Experiment Tracking (Phase 3)
- **MLflow Integration**: Complete experiment tracking and model registry
- **Structured Logging**: QuantumMLLogger with JSON support
- **Metrics Tracking**: Advanced metrics analysis and visualization

### 🔍 Hyperparameter Tuning (Phase 4)
- **Grid Search**: Exhaustive parameter exploration
- **Random Search**: Efficient random sampling
- **Bayesian Optimization**: Smart Gaussian process-based search
- **Optuna Integration**: State-of-the-art optimization

---

## Installation

### Upgrade from v4.1.0

```bash
# Basic upgrade
pip install --upgrade q-store

# With all new features
pip install --upgrade "q-store[all]"
```

### Feature-Specific Installation

```bash
# Data management only
pip install "q-store[datasets,augmentation]"

# Experiment tracking
pip install "q-store[tracking]"

# Hyperparameter tuning
pip install "q-store[tuning]"

# Everything
pip install "q-store[all]"
```

### Verify Installation

```python
import q_store
print(q_store.__version__)  # Should be 4.1.1

# Check available features
from q_store.data import DatasetLoader
from q_store.ml import (
    create_scheduler,
    EarlyStopping,
    MLflowTracker,
    OptunaTuner
)
print("✓ All v4.1.1 features available")
```

---

## Quick Migration Steps

### Step 1: Update Dependencies

**Before (v4.1.0)**:
```toml
# requirements.txt or pyproject.toml
q-store>=4.1.0
```

**After (v4.1.1)**:
```toml
q-store>=4.1.1
# Optional dependencies
datasets>=2.14.0  # For HuggingFace support
mlflow>=2.7.0     # For experiment tracking
optuna>=3.3.0     # For advanced tuning
```

### Step 2: Update Imports (Optional)

Your existing v4.1.0 code continues to work without changes. To use new features, add:

```python
# NEW: Data management
from q_store.data import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
    QuantumPreprocessor,
    QuantumDataGenerator,
)

# NEW: Training enhancements
from q_store.ml import (
    create_scheduler,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)

# NEW: Experiment tracking
from q_store.ml import (
    MLflowTracker,
    QuantumMLLogger,
    MetricsTracker,
)

# NEW: Hyperparameter tuning
from q_store.ml import (
    GridSearch,
    RandomSearch,
    OptunaTuner,
)
```

### Step 3: Enhance Existing Training (Optional)

**Before (v4.1.0)**:
```python
config = TrainingConfig(
    learning_rate=0.01,
    epochs=100,
    # ... other params
)

trainer = QuantumTrainer(config, backend_manager)
await trainer.train(model, train_loader, val_loader)
```

**After (v4.1.1)** - Enhanced with new features:
```python
# Add learning rate scheduling
config = TrainingConfig(
    learning_rate=0.01,
    epochs=100,
    # NEW: LR scheduling
    lr_scheduler='cosine',
    lr_scheduler_params={'T_max': 100, 'eta_min': 0.001},
    # NEW: Early stopping
    enable_early_stopping=True,
    early_stopping_patience=10,
    # NEW: Callbacks
    callbacks=[
        ModelCheckpoint('best_model.pkl', monitor='val_loss'),
        CSVLogger('training.csv'),
    ],
)

trainer = QuantumTrainer(config, backend_manager)
await trainer.train(model, train_loader, val_loader)
```

---

## Feature-by-Feature Migration

### 1. Dataset Loading

#### Before: Manual Data Loading
```python
# v4.1.0: Manual loading
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Manual preprocessing
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
```

#### After: Unified Dataset Loading
```python
# v4.1.1: Unified loading
from q_store.data import DatasetLoader, DatasetConfig, DatasetSource

# Load from multiple sources
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
)
dataset = DatasetLoader.load(config)

# Access data
x_train, y_train = dataset.x_train, dataset.y_train
x_test, y_test = dataset.x_test, dataset.y_test

# Automatic preprocessing
from q_store.data import QuantumPreprocessor
preprocessor = QuantumPreprocessor(method='minmax')
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
```

### 2. Data Generators

#### Before: Manual Batching
```python
# v4.1.0: Manual iteration
async def manual_data_loader(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        yield batch_x, batch_y

train_loader = manual_data_loader(x_train, y_train, 32)
```

#### After: Efficient Generators
```python
# v4.1.1: Built-in generators
from q_store.data import QuantumDataGenerator

train_generator = QuantumDataGenerator(
    x_train, y_train,
    batch_size=32,
    shuffle=True,
    augmentation=None,  # Optional augmentation
    preprocessing=None   # Optional preprocessing
)

# Use directly in training
await trainer.train(model, train_generator, val_generator)
```

### 3. Learning Rate Scheduling

#### Before: Manual LR Updates
```python
# v4.1.0: Manual LR adjustment
config = TrainingConfig(learning_rate=0.01, epochs=100)
trainer = QuantumTrainer(config, backend_manager)

# Manual LR decay in training loop
for epoch in range(100):
    if epoch > 50:
        config.learning_rate *= 0.95
    # ... training
```

#### After: Automatic Scheduling
```python
# v4.1.1: Automatic scheduling
config = TrainingConfig(
    learning_rate=0.01,
    epochs=100,
    # Add scheduler
    lr_scheduler='cosine',
    lr_scheduler_params={
        'T_max': 100,
        'eta_min': 0.001
    }
)

trainer = QuantumTrainer(config, backend_manager)
# LR automatically adjusted during training
await trainer.train(model, train_loader, val_loader)
```

### 4. Early Stopping

#### Before: Manual Convergence Checks
```python
# v4.1.0: Manual early stopping
best_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(100):
    metrics = await trainer.train_epoch(model, train_loader, epoch)
    val_metrics = await trainer.validate(model, val_loader)

    if val_metrics['loss'] < best_loss:
        best_loss = val_metrics['loss']
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

#### After: Automatic Early Stopping
```python
# v4.1.1: Automatic early stopping
config = TrainingConfig(
    epochs=100,
    enable_early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_monitor='val_loss'
)

trainer = QuantumTrainer(config, backend_manager)
# Automatically stops when converged
await trainer.train(model, train_loader, val_loader)

# Access early stopping info
if trainer.early_stopping.stopped_epoch > 0:
    print(f"Stopped at epoch {trainer.early_stopping.stopped_epoch}")
    print(f"Best epoch: {trainer.early_stopping.best_epoch}")
```

### 5. Experiment Tracking

#### Before: Manual Logging
```python
# v4.1.0: Manual logging
import logging
logger = logging.getLogger(__name__)

for epoch in range(100):
    metrics = await trainer.train_epoch(model, train_loader, epoch)
    logger.info(f"Epoch {epoch}: loss={metrics.loss}")

    # Manual metric storage
    with open('metrics.txt', 'a') as f:
        f.write(f"{epoch},{metrics.loss}\n")
```

#### After: MLflow Integration
```python
# v4.1.1: MLflow tracking
from q_store.ml import MLflowTracker, MLflowConfig

config = MLflowConfig(
    experiment_name='my_experiment',
    tracking_uri='./mlruns'
)
tracker = MLflowTracker(config)

tracker.start_run(run_name='experiment_1')

# Log parameters
tracker.log_params({
    'n_qubits': 8,
    'learning_rate': 0.01,
    'epochs': 100
})

# Training with automatic callback
from q_store.ml import MLflowCallback
config = TrainingConfig(
    callbacks=[MLflowCallback(tracker=tracker)]
)

await trainer.train(model, train_loader, val_loader)

tracker.end_run()

# View in UI: mlflow ui --backend-store-uri ./mlruns
```

### 6. Hyperparameter Tuning

#### Before: Manual Grid Search
```python
# v4.1.0: Manual parameter search
best_score = float('inf')
best_params = None

for lr in [0.001, 0.01, 0.1]:
    for n_qubits in [4, 6, 8]:
        for depth in [2, 3, 4]:
            config = TrainingConfig(
                learning_rate=lr,
                n_qubits=n_qubits,
                circuit_depth=depth
            )
            # Train and evaluate
            score = await train_and_evaluate(config)
            if score < best_score:
                best_score = score
                best_params = {'lr': lr, 'n_qubits': n_qubits, 'depth': depth}
```

#### After: Automatic Tuning
```python
# v4.1.1: Automatic tuning with Optuna
from q_store.ml import OptunaTuner, OptunaConfig

config = OptunaConfig(
    study_name='parameter_search',
    direction='minimize',
    n_trials=50
)

tuner = OptunaTuner(config)

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    n_qubits = trial.suggest_int('n_qubits', 4, 12)
    depth = trial.suggest_int('depth', 2, 6)

    # Train and return score
    return train_and_evaluate(lr, n_qubits, depth)

best_params = tuner.optimize(objective)
print(f"Best params: {best_params}")
```

---

## Code Examples

### Complete Migration Example

**v4.1.0 Code**:
```python
import asyncio
import tensorflow as tf
from q_store.backends import BackendManager
from q_store.ml import QuantumModel, QuantumTrainer, TrainingConfig

async def main():
    # Load data manually
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    # Basic configuration
    config = TrainingConfig(
        pinecone_api_key='mock-key',
        quantum_sdk='mock',
        n_qubits=8,
        learning_rate=0.01,
        epochs=50
    )

    # Initialize and train
    backend_manager = BackendManager('mock', None)
    model = QuantumModel(784, 8, 10, backend_manager.get_backend(), depth=3)
    trainer = QuantumTrainer(config, backend_manager)

    # Manual data loader
    async def data_loader(x, y, batch_size=32):
        for i in range(0, len(x), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]

    train_loader = data_loader(x_train, y_train)
    val_loader = data_loader(x_test, y_test)

    await trainer.train(model, train_loader, val_loader)

asyncio.run(main())
```

**v4.1.1 Enhanced Code**:
```python
import asyncio
from q_store.backends import BackendManager
from q_store.ml import (
    QuantumModel, QuantumTrainer, TrainingConfig,
    ModelCheckpoint, CSVLogger, MLflowTracker, MLflowConfig
)
from q_store.data import (
    DatasetLoader, DatasetConfig, DatasetSource,
    QuantumPreprocessor, QuantumDataGenerator
)

async def main():
    # 1. Load data with unified loader
    config_data = DatasetConfig(
        name='fashion_mnist',
        source=DatasetSource.KERAS,
        source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
    )
    dataset = DatasetLoader.load(config_data)

    # 2. Preprocess for quantum
    preprocessor = QuantumPreprocessor(method='minmax')
    x_train = preprocessor.fit_transform(dataset.x_train.reshape(-1, 784))
    x_test = preprocessor.transform(dataset.x_test.reshape(-1, 784))

    # 3. Create efficient generators
    train_gen = QuantumDataGenerator(x_train, dataset.y_train, batch_size=32)
    val_gen = QuantumDataGenerator(x_test, dataset.y_test, batch_size=32)

    # 4. Setup experiment tracking
    mlflow_tracker = MLflowTracker(MLflowConfig(
        experiment_name='fashion_mnist',
        tracking_uri='./mlruns'
    ))
    mlflow_tracker.start_run(run_name='enhanced_v4.1.1')

    # 5. Enhanced training configuration
    config = TrainingConfig(
        pinecone_api_key='mock-key',
        quantum_sdk='mock',
        n_qubits=8,
        learning_rate=0.02,
        epochs=50,
        # NEW: LR scheduling
        lr_scheduler='cosine',
        lr_scheduler_params={'T_max': 50, 'eta_min': 0.001},
        # NEW: Early stopping
        enable_early_stopping=True,
        early_stopping_patience=10,
        # NEW: Callbacks
        callbacks=[
            ModelCheckpoint('best_model.pkl', monitor='val_loss'),
            CSVLogger('training.csv'),
        ]
    )

    # Log params to MLflow
    mlflow_tracker.log_params({
        'n_qubits': config.n_qubits,
        'learning_rate': config.learning_rate,
        'lr_scheduler': config.lr_scheduler
    })

    # 6. Train with all enhancements
    backend_manager = BackendManager('mock', None)
    model = QuantumModel(784, 8, 10, backend_manager.get_backend(), depth=3)
    trainer = QuantumTrainer(config, backend_manager)

    await trainer.train(model, train_gen, val_gen)

    # 7. Log final results
    if trainer.training_history:
        final = trainer.training_history[-1]
        mlflow_tracker.log_metrics({
            'final_loss': final.loss,
            'final_lr': final.learning_rate
        })

    mlflow_tracker.end_run()
    print("✓ Training complete! View in MLflow UI: mlflow ui")

asyncio.run(main())
```

---

## Common Migration Patterns

### Pattern 1: Add MLflow Tracking to Existing Code

```python
# Minimal changes to existing code:
from q_store.ml import MLflowCallback, MLflowTracker, MLflowConfig

# 1. Create tracker
tracker = MLflowTracker(MLflowConfig(experiment_name='my_exp'))
tracker.start_run()

# 2. Add callback to existing config
config.callbacks = [MLflowCallback(tracker=tracker)]

# 3. Your existing training code works as-is
await trainer.train(model, train_loader, val_loader)

tracker.end_run()
```

### Pattern 2: Add Early Stopping Without Changing Logic

```python
# Just add to config - no other changes needed
config = TrainingConfig(
    # ... your existing params ...
    enable_early_stopping=True,
    early_stopping_patience=10,
    early_stopping_monitor='val_loss'
)

# Training automatically stops when converged
await trainer.train(model, train_loader, val_loader)
```

### Pattern 3: Migrate Custom Data Loading to Generators

**Before**:
```python
async def my_custom_loader(data, labels, batch_size):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    for start in range(0, len(data), batch_size):
        batch_indices = indices[start:start+batch_size]
        yield data[batch_indices], labels[batch_indices]
```

**After**:
```python
# Use built-in generator instead
from q_store.data import QuantumDataGenerator

generator = QuantumDataGenerator(
    data, labels,
    batch_size=batch_size,
    shuffle=True  # Built-in shuffling
)

# Works exactly like your custom loader
await trainer.train(model, generator, val_generator)
```

---

## Troubleshooting

### Issue 1: Import Errors

**Error**: `ImportError: cannot import name 'DatasetLoader'`

**Solution**:
```bash
# Install data management dependencies
pip install "q-store[datasets]"

# Or install all features
pip install "q-store[all]"
```

### Issue 2: MLflow Not Found

**Error**: `MLflow not available. Install with: pip install mlflow`

**Solution**:
```bash
# Install tracking dependencies
pip install "q-store[tracking]"

# Verify
python -c "import mlflow; print(mlflow.__version__)"
```

### Issue 3: Scheduler Not Working

**Problem**: Learning rate doesn't change during training

**Check**:
```python
# 1. Verify scheduler is configured
assert config.lr_scheduler is not None

# 2. Check if scheduler initialized
assert trainer.lr_scheduler is not None

# 3. Monitor LR changes
from q_store.ml import LearningRateLogger
config.callbacks.append(LearningRateLogger(verbose=True))
```

### Issue 4: Early Stopping Not Triggering

**Problem**: Training runs for all epochs despite convergence

**Debug**:
```python
# 1. Check monitor metric exists
# Make sure 'val_loss' (or your metric) is in logs

# 2. Verify patience is reasonable
config.early_stopping_patience = 5  # Try smaller value

# 3. Check min_delta
config.early_stopping_min_delta = 0.0  # Try removing threshold

# 4. Print early stopping state after training
if trainer.early_stopping:
    print(f"Best epoch: {trainer.early_stopping.best_epoch}")
    print(f"Best value: {trainer.early_stopping.best_value}")
    print(f"Wait count: {trainer.early_stopping.wait}")
```

### Issue 5: Callback Errors

**Error**: `TypeError: 'NoneType' object is not iterable` in callbacks

**Solution**:
```python
# Make sure callbacks is a list, not None
config = TrainingConfig(
    callbacks=[  # Use list, even if empty
        ModelCheckpoint('model.pkl'),
    ]
    # NOT: callbacks=None  # This can cause issues
)

# Or use empty list
config.callbacks = config.callbacks or []
```

---

## Next Steps

1. **Read the [Data Management Guide](DATA_MANAGEMENT_GUIDE.md)** for detailed dataset loading examples

2. **Check [API Reference](API_REFERENCE_V4_1_1.md)** for complete API documentation

3. **Run the Examples**:
   ```bash
   # Fashion MNIST with new features
   python examples/ml_frameworks/fashion_mnist_with_backend_api.py

   # Hyperparameter tuning
   python examples/ml_frameworks/hyperparameter_tuning_example.py

   # MLflow tracking
   python examples/ml_frameworks/mlflow_tracking_example.py

   # Complete end-to-end workflow
   python examples/ml_frameworks/end_to_end_workflow.py
   ```

4. **Join the Community**:
   - GitHub: https://github.com/yucelz/q-store
   - Report Issues: https://github.com/yucelz/q-store/issues
   - Documentation: https://github.com/yucelz/q-store/tree/main/docs

---

## Summary

✅ **Backward Compatible**: All v4.1.0 code works without changes
✅ **Optional Features**: Use only what you need
✅ **Easy Migration**: Add new features incrementally
✅ **Better Performance**: Enhanced training with minimal code changes
✅ **Complete Tracking**: Built-in experiment management

**Welcome to Q-Store v4.1.1! 🎉**
