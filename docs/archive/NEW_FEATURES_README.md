# Q-Store New Features Examples

This directory contains comprehensive examples demonstrating the new implementations in Q-Store v4.1.1.

## Directory Structure

```
examples/
├── data_management/          # Data loading, preprocessing, and augmentation
│   ├── data_loaders_example.py
│   ├── data_adapters_example.py
│   ├── data_preprocessing_example.py
│   ├── backend_client_example.py
│   └── generators_validation_augmentation_example.py
│
├── ml_training/              # ML training features
│   ├── schedulers_example.py
│   ├── early_stopping_callbacks_example.py
│   ├── mlflow_tracking_example.py
│   └── hyperparameter_tuning_example.py
│
└── NEW_FEATURES_README.md    # This file
```

## Data Management Examples

### 1. Data Loaders (`data_loaders_example.py`)

Demonstrates the unified dataset loading system with support for multiple sources.

**Features:**
- Load from Keras datasets (Fashion MNIST, MNIST, CIFAR-10)
- Load from HuggingFace datasets
- Load from Q-Store Backend API
- Load from local files (NumPy, CSV)
- Automatic preprocessing and validation splitting
- Dataset registry for common configurations

**Run:**
```bash
python examples/data_management/data_loaders_example.py
```

**Key Classes:**
- `DatasetLoader`: Main loader with adapter pattern
- `DatasetConfig`: Configuration for dataset loading
- `DatasetSource`: Enum for data sources (KERAS, HUGGINGFACE, BACKEND_API, LOCAL_FILES)

### 2. Data Adapters (`data_adapters_example.py`)

Shows how to prepare classical data for quantum processing.

**Features:**
- Dimension reduction (PCA, SVD, feature selection)
- Quantum encoding preparation (amplitude, angle, basis)
- Image adaptation for quantum circuits
- Custom encoding functions
- Data validation for quantum compatibility

**Run:**
```bash
python examples/data_management/data_adapters_example.py
```

**Key Classes:**
- `QuantumDataAdapter`: Prepare data for quantum encoding
- `DimensionReducer`: Reduce data dimensions
- `QuantumImageAdapter`: Adapt images for quantum processing
- `EncodingType`: Quantum encoding schemes

### 3. Data Preprocessing (`data_preprocessing_example.py`)

Comprehensive preprocessing for quantum ML.

**Features:**
- Multiple normalization methods (minmax, zscore, L1, L2, robust)
- Train/validation/test splitting
- Stratified splitting for imbalanced datasets
- Feature scaling and clipping
- Inverse transformations

**Run:**
```bash
python examples/data_management/data_preprocessing_example.py
```

**Key Classes:**
- `QuantumPreprocessor`: Normalization and standardization
- `DataSplitter`: Train/val/test splitting
- `NormalizationMethod`: Enum for normalization methods

### 4. Backend API Client (`backend_client_example.py`)

Integration with Q-Store Backend API.

**Features:**
- Connect to Backend API
- List and retrieve datasets
- Download dataset data
- Import from HuggingFace
- Label Studio integration
- Apply augmentation via API

**Run:**
```bash
# Start backend first
python -m q_store.backend.main

# Then run example
python examples/data_management/backend_client_example.py
```

**Key Classes:**
- `BackendAPIClient`: REST client for Backend API
- `BackendConfig`: Configuration for API connection

### 5. Generators, Validation & Augmentation (`generators_validation_augmentation_example.py`)

Efficient data generation and quality control.

**Features:**
- Batch generators with shuffling
- Streaming generators for large datasets
- Data validation and profiling
- Outlier detection
- Quantum and classical augmentation
- Hybrid augmentation strategies

**Run:**
```bash
python examples/data_management/generators_validation_augmentation_example.py
```

**Key Classes:**
- `QuantumDataGenerator`: Batch generator
- `DataValidator`: Data validation
- `DataProfiler`: Dataset profiling
- `QuantumAugmentation`: Quantum-specific augmentation
- `HybridAugmentation`: Combined classical + quantum

## ML Training Examples

### 6. Learning Rate Schedulers (`schedulers_example.py`)

Advanced learning rate scheduling strategies.

**Features:**
- Step decay and exponential decay
- Cosine annealing
- Cyclic learning rates
- One-cycle policy
- ReduceLROnPlateau (adaptive)
- Warmup strategies
- Combined schedulers
- Visualization of all schedulers

**Run:**
```bash
python examples/ml_training/schedulers_example.py
```

**Key Classes:**
- `StepLR`, `ExponentialLR`, `CosineAnnealingLR`
- `CyclicLR`, `OneCycleLR`
- `ReduceLROnPlateau`
- `WarmupScheduler`

### 7. Early Stopping & Callbacks (`early_stopping_callbacks_example.py`)

Training control mechanisms and callbacks.

**Features:**
- Early stopping to prevent overfitting
- Convergence detection
- Model checkpointing (save best models)
- CSV logging
- Progress tracking
- Learning rate logging
- Multiple callbacks with CallbackList

**Run:**
```bash
python examples/ml_training/early_stopping_callbacks_example.py
```

**Key Classes:**
- `EarlyStopping`: Stop when metric stops improving
- `ConvergenceDetector`: Detect training convergence
- `ModelCheckpoint`: Save best models
- `CSVLogger`: Log metrics to CSV
- `ProgressCallback`: Track training progress
- `CallbackList`: Manage multiple callbacks

### 8. MLflow Tracking (`mlflow_tracking_example.py`)

Experiment tracking with MLflow integration.

**Features:**
- MLflow configuration and connection
- Parameter and metric logging
- Model tracking and versioning
- Artifact management (configs, plots, data)
- Tags and notes
- Nested runs for hyperparameter tuning
- Complete workflow example

**Run:**
```bash
# Install MLflow first
pip install mlflow

# Run example
python examples/ml_training/mlflow_tracking_example.py

# View results
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

**Key Classes:**
- `MLflowTracker`: Main interface for MLflow
- `MLflowConfig`: Configuration for MLflow

### 9. Hyperparameter Tuning (`hyperparameter_tuning_example.py`)

Various hyperparameter optimization strategies.

**Features:**
- Grid search (exhaustive)
- Random search
- Bayesian optimization
- Optuna integration (TPE, pruning)
- Parallel optimization
- Visualization of results
- Complete tuning workflow
- Method comparison

**Run:**
```bash
# Install optional dependencies
pip install bayesian-optimization optuna

# Run example
python examples/ml_training/hyperparameter_tuning_example.py
```

**Key Classes:**
- `GridSearch`: Exhaustive grid search
- `RandomSearch`: Random parameter sampling
- `BayesianOptimizer`: Gaussian process optimization
- `OptunaTuner`: Optuna integration
- `OptunaConfig`: Configuration for Optuna

## Installation Requirements

### Core Requirements
```bash
pip install q-store
```

### Optional Dependencies

For full functionality:
```bash
# MLflow tracking
pip install mlflow

# Bayesian optimization
pip install bayesian-optimization

# Advanced hyperparameter tuning
pip install optuna

# Data augmentation
pip install albumentations

# HuggingFace datasets
pip install datasets

# Visualization
pip install matplotlib
```

Or install all at once:
```bash
pip install q-store[ml,data,viz]
```

## Quick Start

### 1. Data Management Workflow

```python
from q_store.data import (
    DatasetLoader, DatasetConfig, DatasetSource,
    QuantumPreprocessor, DimensionReducer,
    QuantumDataGenerator
)

# Load dataset
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    preprocessing={'normalize': True, 'flatten': True}
)
dataset = DatasetLoader.load(config)

# Preprocess
preprocessor = QuantumPreprocessor(method='minmax')
x_train = preprocessor.fit_transform(dataset.x_train)

# Reduce dimensions
reducer = DimensionReducer(method='pca', target_dim=16)
x_reduced = reducer.fit_transform(x_train)

# Create generator
generator = QuantumDataGenerator(
    x_reduced, dataset.y_train,
    batch_size=32,
    shuffle=True
)
```

### 2. ML Training Workflow

```python
from q_store.ml import (
    CosineAnnealingLR,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    MLflowTracker
)

# Setup callbacks
early_stop = EarlyStopping(patience=10, mode='min')
checkpoint = ModelCheckpoint('best_model.pkl', save_best_only=True)
csv_logger = CSVLogger('training_log.csv')

# Setup scheduler
scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100)

# Setup MLflow tracking
tracker = MLflowTracker()
tracker.start_run(run_name='experiment_001')

# Training loop
for epoch in range(100):
    lr = scheduler.step(epoch)
    
    # Train...
    train_loss = train_epoch(lr)
    val_loss = validate()
    
    # Log to MLflow
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': lr
    }, step=epoch)
    
    # Callbacks
    if early_stop.should_stop(epoch, val_loss):
        break
    
    checkpoint.on_epoch_end(epoch, {'val_loss': val_loss})
    csv_logger.on_epoch_end(epoch, {'loss': train_loss, 'val_loss': val_loss})

tracker.end_run()
```

### 3. Hyperparameter Tuning Workflow

```python
from q_store.ml.tuning import OptunaTuner, OptunaConfig

# Configure tuner
config = OptunaConfig(
    study_name='my_study',
    direction='minimize',
    n_trials=50
)
tuner = OptunaTuner(config)

# Define objective
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    n_qubits = trial.suggest_int('n_qubits', 4, 12)
    depth = trial.suggest_int('circuit_depth', 2, 6)
    
    # Train model with these params
    loss = train_model(lr=lr, n_qubits=n_qubits, depth=depth)
    
    return loss

# Optimize
best_params = tuner.optimize(objective)
print(f"Best parameters: {best_params}")
```

## Common Use Cases

### Use Case 1: Complete Pipeline
```bash
# 1. Load and preprocess data
python examples/data_management/data_loaders_example.py
python examples/data_management/data_preprocessing_example.py

# 2. Setup training with callbacks
python examples/ml_training/early_stopping_callbacks_example.py
python examples/ml_training/schedulers_example.py

# 3. Track experiments
python examples/ml_training/mlflow_tracking_example.py

# 4. Tune hyperparameters
python examples/ml_training/hyperparameter_tuning_example.py
```

### Use Case 2: Data Preparation Only
```bash
# Load data
python examples/data_management/data_loaders_example.py

# Adapt for quantum
python examples/data_management/data_adapters_example.py

# Validate and augment
python examples/data_management/generators_validation_augmentation_example.py
```

### Use Case 3: Training Optimization
```bash
# Setup learning rate schedule
python examples/ml_training/schedulers_example.py

# Add early stopping
python examples/ml_training/early_stopping_callbacks_example.py

# Find best hyperparameters
python examples/ml_training/hyperparameter_tuning_example.py
```

## Tips and Best Practices

### Data Management
1. **Always validate data** before quantum encoding
2. **Use appropriate normalization** for your encoding type (amplitude encoding needs L2 norm)
3. **Profile your data** to understand class distribution and feature statistics
4. **Use generators** for large datasets to save memory
5. **Apply augmentation** to improve model generalization

### Training
1. **Start with warmup** when using large learning rates
2. **Monitor multiple metrics** with callbacks
3. **Save checkpoints** regularly, not just the best model
4. **Use early stopping** to prevent overfitting
5. **Log everything** to MLflow for reproducibility

### Hyperparameter Tuning
1. **Start with coarse grid search** to find promising regions
2. **Refine with random search** around best parameters
3. **Fine-tune with Bayesian optimization** for efficiency
4. **Use Optuna pruning** to stop unpromising trials early
5. **Compare methods** to find what works best for your problem

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Install missing dependencies
pip install q-store[ml,data,viz]
```

**MLflow UI not starting:**
```bash
# Specify correct backend
mlflow ui --backend-store-uri ./mlruns --port 5000
```

**Backend API connection failed:**
```bash
# Start backend first
python -m q_store.backend.main
```

**Memory errors with large datasets:**
```python
# Use streaming generator
generator = StreamingDataGenerator(data_loader, batch_size=32)
```

## Additional Resources

- **Main Documentation**: See `/docs` directory
- **API Reference**: See docstrings in source code
- **Full Examples**: See `/examples` directory
- **Tests**: See `/tests` directory for more usage patterns

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/yucelz/q-store/issues
- Documentation: `/docs`
- Examples: This directory

## License

See LICENSE file in the root directory.
