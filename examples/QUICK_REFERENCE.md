# Q-Store v4.1.1 New Features Quick Reference

## Data Management

### Loading Data

```python
from q_store.data import DatasetLoader, DatasetConfig, DatasetSource

# From Keras
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
)
dataset = DatasetLoader.load(config)

# From HuggingFace
config = DatasetConfig(
    name='mnist',
    source=DatasetSource.HUGGINGFACE,
    source_params={'dataset_name': 'mnist'}
)
dataset = DatasetLoader.load(config)

# From local files
config = DatasetConfig(
    name='my_data',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'x_train_path': 'x_train.npy',
        'y_train_path': 'y_train.npy'
    }
)
dataset = DatasetLoader.load(config)
```

### Preprocessing

```python
from q_store.data import QuantumPreprocessor, DataSplitter

# Normalize
preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
x_normalized = preprocessor.fit_transform(x_data)

# Split data
splitter = DataSplitter(split_ratio={'train': 0.8, 'val': 0.1, 'test': 0.1})
splits = splitter.split(x_data, y_data)
```

### Quantum Adaptation

```python
from q_store.data import DimensionReducer, QuantumDataAdapter, EncodingType

# Reduce dimensions
reducer = DimensionReducer(method='pca', target_dim=16)
x_reduced = reducer.fit_transform(x_data)

# Prepare for quantum
adapter = QuantumDataAdapter(n_qubits=4, encoding=EncodingType.AMPLITUDE)
x_quantum = adapter.prepare(x_reduced)
```

### Data Generators

```python
from q_store.data import QuantumDataGenerator

generator = QuantumDataGenerator(
    x_data, y_data,
    batch_size=32,
    shuffle=True
)

for x_batch, y_batch in generator:
    # Train on batch
    pass
```

### Validation

```python
from q_store.data import DataValidator, DataProfiler

# Validate
validator = DataValidator()
is_valid, message = validator.validate_all(x_data, y_data, n_qubits=4)

# Profile
profiler = DataProfiler()
stats = profiler.profile(x_data, y_data)
```

### Augmentation

```python
from q_store.data import QuantumAugmentation, HybridAugmentation

# Quantum augmentation
aug = QuantumAugmentation(phase_shift_range=0.1, amplitude_noise=0.01)
x_augmented = aug.apply(x_data)

# Hybrid augmentation
hybrid = HybridAugmentation(
    classical_transforms=[{'type': 'horizontal_flip', 'p': 0.5}],
    quantum_augmentation=True
)
x_augmented = hybrid.apply(images)
```

## ML Training

### Learning Rate Schedulers

```python
from q_store.ml import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    CyclicLR, OneCycleLR, ReduceLROnPlateau, WarmupScheduler
)

# Cosine annealing
scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100, eta_min=1e-6)
lr = scheduler.step(epoch)

# One-cycle
scheduler = OneCycleLR(max_lr=0.1, total_steps=100)
lr = scheduler.step(step)

# Warmup
warmup = WarmupScheduler(initial_lr=1e-6, target_lr=0.01, warmup_steps=10)
lr = warmup.step(step)
```

### Early Stopping

```python
from q_store.ml import EarlyStopping, ConvergenceDetector

# Early stopping
early_stop = EarlyStopping(patience=10, min_delta=0.001, mode='min')
if early_stop.should_stop(epoch, loss):
    break

# Convergence detection
convergence = ConvergenceDetector(window_size=10, threshold=0.0001)
if convergence.check_convergence(epoch, loss):
    break
```

### Callbacks

```python
from q_store.ml import (
    ModelCheckpoint, CSVLogger, ProgressCallback,
    LearningRateLogger, CallbackList
)

callbacks = CallbackList([
    ModelCheckpoint('best_model.pkl', monitor='val_loss', save_best_only=True),
    CSVLogger('training_log.csv'),
    ProgressCallback(total_epochs=100),
    LearningRateLogger()
])

callbacks.on_train_begin()
for epoch in range(100):
    callbacks.on_epoch_begin(epoch)
    # ... training ...
    callbacks.on_epoch_end(epoch, logs={'loss': loss, 'val_loss': val_loss})
callbacks.on_train_end()
```

### MLflow Tracking

```python
from q_store.ml.tracking import MLflowTracker, MLflowConfig

# Configure
config = MLflowConfig(
    tracking_uri='./mlruns',
    experiment_name='my_experiment'
)

tracker = MLflowTracker(config)

# Track experiment
tracker.start_run(run_name='run_001')
tracker.log_params({'n_qubits': 4, 'lr': 0.01})
tracker.log_metric('loss', loss_value, step=epoch)
tracker.log_model(model, 'model')
tracker.end_run()
```

### Hyperparameter Tuning

```python
from q_store.ml.tuning import (
    GridSearch, RandomSearch, BayesianOptimizer,
    OptunaTuner, OptunaConfig
)

# Grid search
param_grid = {'lr': [0.001, 0.01, 0.1], 'n_qubits': [4, 6, 8]}
grid = GridSearch(param_grid, scoring='min')
best_params, best_score = grid.search(objective_fn)

# Random search
param_dist = {
    'lr': ('log_uniform', 1e-4, 1e-1),
    'n_qubits': ('int_uniform', 4, 12)
}
random = RandomSearch(param_dist, n_trials=50, scoring='min')
best_params, best_score = random.search(objective_fn)

# Optuna
config = OptunaConfig(study_name='my_study', direction='minimize', n_trials=50)
tuner = OptunaTuner(config)

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    n_qubits = trial.suggest_int('n_qubits', 4, 12)
    # Train and return metric
    return loss

best_params = tuner.optimize(objective)
```

## Complete Training Loop

```python
from q_store.data import DatasetLoader, QuantumPreprocessor, QuantumDataGenerator
from q_store.ml import (
    CosineAnnealingLR, EarlyStopping, ModelCheckpoint,
    CSVLogger, CallbackList
)

# 1. Load data
dataset = DatasetLoader.load(config)

# 2. Preprocess
preprocessor = QuantumPreprocessor(method='minmax')
x_train = preprocessor.fit_transform(dataset.x_train)

# 3. Create generator
generator = QuantumDataGenerator(x_train, dataset.y_train, batch_size=32)

# 4. Setup training
scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100)
early_stop = EarlyStopping(patience=10)
callbacks = CallbackList([
    ModelCheckpoint('best_model.pkl', save_best_only=True),
    CSVLogger('training_log.csv')
])

# 5. Train
callbacks.on_train_begin()
for epoch in range(100):
    lr = scheduler.step(epoch)
    callbacks.on_epoch_begin(epoch)
    
    # Training loop
    for x_batch, y_batch in generator:
        loss = train_step(x_batch, y_batch, lr)
    
    val_loss = validate()
    callbacks.on_epoch_end(epoch, {'loss': loss, 'val_loss': val_loss})
    
    if early_stop.should_stop(epoch, val_loss):
        break

callbacks.on_train_end()
```

## Backend API

```python
from q_store.data import BackendAPIClient, BackendConfig

# Configure
config = BackendConfig(
    base_url='http://localhost:8000',
    api_key='your_api_key'
)
client = BackendAPIClient(config)

# List datasets
datasets = client.list_datasets()

# Get dataset
dataset = client.get_dataset(dataset_id='uuid-123')

# Download data
data = client.download_dataset_data(dataset_id='uuid-123', split='train')

# Upload dataset
client.upload_dataset('my_dataset', x_train, y_train, x_test, y_test)

# Import from HuggingFace
result = client.import_from_huggingface('mnist')
```

## Common Patterns

### Pattern 1: Data Pipeline
```python
# Load → Preprocess → Adapt → Generate
dataset = DatasetLoader.load(config)
x_processed = QuantumPreprocessor(method='minmax').fit_transform(dataset.x_train)
x_quantum = QuantumDataAdapter(n_qubits=4).prepare(x_processed)
generator = QuantumDataGenerator(x_quantum, dataset.y_train, batch_size=32)
```

### Pattern 2: Training Setup
```python
# Scheduler + Early Stop + Callbacks
scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100)
early_stop = EarlyStopping(patience=10)
callbacks = CallbackList([ModelCheckpoint('best.pkl'), CSVLogger('log.csv')])
```

### Pattern 3: Experiment Tracking
```python
# Setup MLflow → Train → Log
tracker = MLflowTracker(config)
tracker.start_run()
tracker.log_params(hyperparams)
# ... training loop with tracker.log_metric() ...
tracker.end_run()
```

### Pattern 4: Hyperparameter Tuning
```python
# Define objective → Choose method → Optimize
def objective(params):
    # Train with params and return metric
    return loss

tuner = OptunaTuner(config)
best_params = tuner.optimize(objective)
```

## Installation Commands

```bash
# Core
pip install q-store

# With ML features
pip install q-store[ml]

# With data features
pip install q-store[data]

# With visualization
pip install q-store[viz]

# Everything
pip install q-store[all]

# Or individual packages
pip install mlflow optuna bayesian-optimization albumentations datasets
```

## File Locations

After running examples, you'll find:

```
./checkpoints/          # Model checkpoints
./training_log.csv      # Training metrics
./mlruns/               # MLflow tracking data
./optuna_*.png          # Optuna visualizations
./lr_schedulers_*.png   # Scheduler visualizations
```

## View Results

```bash
# MLflow UI
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000

# TensorBoard (if using TensorBoard callback)
tensorboard --logdir ./logs
# Open: http://localhost:6006
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install q-store[all]` |
| MLflow not starting | `mlflow ui --backend-store-uri ./mlruns` |
| Backend connection failed | Start backend: `python -m q_store.backend.main` |
| Memory errors | Use `StreamingDataGenerator` |
| Slow training | Reduce batch size or data size |

## Next Steps

1. **Run complete example**: `python examples/complete_workflow_example.py`
2. **Read full guide**: [NEW_FEATURES_README.md](NEW_FEATURES_README.md)
3. **Explore examples**: Browse `data_management/` and `ml_training/`
4. **Check documentation**: See `/docs` directory
5. **View tests**: See `/tests` for more patterns

---

For detailed examples and explanations, see [NEW_FEATURES_README.md](NEW_FEATURES_README.md)
