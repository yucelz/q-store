# Q-Store Data Management Guide (v4.1.1)

Complete guide to the v4.1.1 data management layer for quantum machine learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Loading](#dataset-loading)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Validation](#data-validation)
5. [Data Generators](#data-generators)
6. [Data Augmentation](#data-augmentation)
7. [Advanced Topics](#advanced-topics)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

---

## Overview

The Q-Store data management layer provides a unified interface for loading, preprocessing, validating, and augmenting data for quantum machine learning.

### Key Features

- **4 Data Sources**: Keras, HuggingFace, Backend API, Local Files
- **5 File Formats**: NumPy, CSV, Images, HDF5, Parquet
- **Quantum-Specific Processing**: Optimized for quantum ML workflows
- **Validation & Profiling**: Automated compatibility checks
- **Efficient Generators**: Memory-efficient batch processing
- **Augmentation**: Quantum, classical, and hybrid techniques

### Installation

```bash
# Basic data management
pip install "q-store[datasets]"

# With augmentation support
pip install "q-store[datasets,augmentation]"

# Everything
pip install "q-store[all]"
```

---

## Dataset Loading

### Quick Start

```python
from q_store.data import DatasetLoader, DatasetConfig, DatasetSource

# Load Fashion MNIST from Keras
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
)

dataset = DatasetLoader.load(config)
print(f"Training: {dataset.x_train.shape}, Test: {dataset.x_test.shape}")
```

### Source 1: Keras Datasets

Load built-in TensorFlow/Keras datasets.

```python
from q_store.data import DatasetConfig, DatasetSource, DatasetLoader

# Fashion MNIST
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={
        'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
    }
)
dataset = DatasetLoader.load(config)

# MNIST
config = DatasetConfig(
    name='mnist',
    source=DatasetSource.KERAS,
    source_params={
        'dataset_module': 'tensorflow.keras.datasets.mnist'
    }
)
dataset = DatasetLoader.load(config)

# CIFAR-10
config = DatasetConfig(
    name='cifar10',
    source=DatasetSource.KERAS,
    source_params={
        'dataset_module': 'tensorflow.keras.datasets.cifar10'
    }
)
dataset = DatasetLoader.load(config)
```

**Available Keras Datasets**:

- `mnist` - Handwritten digits (28x28 grayscale)
- `fashion_mnist` - Fashion items (28x28 grayscale)
- `cifar10` - Objects (32x32 RGB, 10 classes)
- `cifar100` - Objects (32x32 RGB, 100 classes)
- `imdb` - Movie reviews (text classification)
- `reuters` - News articles (text classification)

### Source 2: HuggingFace Datasets

Access 500K+ datasets from HuggingFace Hub.

```python
# Requires: pip install datasets

from q_store.data import DatasetConfig, DatasetSource, DatasetLoader

# Load from HuggingFace
config = DatasetConfig(
    name='mnist',
    source=DatasetSource.HUGGINGFACE,
    source_params={
        'dataset_name': 'mnist',
        'split_mapping': {
            'train': 'train',
            'test': 'test'
        },
        'feature_column': 'image',
        'label_column': 'label'
    }
)

dataset = DatasetLoader.load(config)
```

**Common HuggingFace Datasets**:

```python
# CIFAR-10
config = DatasetConfig(
    name='cifar10',
    source=DatasetSource.HUGGINGFACE,
    source_params={
        'dataset_name': 'cifar10',
        'feature_column': 'img',
        'label_column': 'label'
    }
)

# IMDB Reviews
config = DatasetConfig(
    name='imdb',
    source=DatasetSource.HUGGINGFACE,
    source_params={
        'dataset_name': 'imdb',
        'feature_column': 'text',
        'label_column': 'label'
    }
)

# Custom dataset
config = DatasetConfig(
    name='my_dataset',
    source=DatasetSource.HUGGINGFACE,
    source_params={
        'dataset_name': 'username/my-dataset',
        'split_mapping': {'train': 'train[:80%]', 'test': 'train[80%:]'}
    }
)
```

### Source 3: Q-Store Backend API

Load datasets from Q-Store backend server.

```python
from q_store.data import DatasetConfig, DatasetSource, DatasetLoader

config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.BACKEND_API,
    source_params={
        'api_url': 'http://localhost:8000',
        'dataset_id': 'fashion_mnist',
        'api_key': None  # Optional
    }
)

dataset = DatasetLoader.load(config)
```

**Backend API Configuration**:

```python
# With authentication
config = DatasetConfig(
    name='my_dataset',
    source=DatasetSource.BACKEND_API,
    source_params={
        'api_url': 'https://api.q-store.tech',
        'dataset_id': 'my_dataset_id',
        'api_key': 'your-api-key',
        'timeout': 30  # Request timeout in seconds
    }
)

# With custom headers
config = DatasetConfig(
    name='my_dataset',
    source=DatasetSource.BACKEND_API,
    source_params={
        'api_url': 'https://api.q-store.tech',
        'dataset_id': 'my_dataset_id',
        'headers': {
            'Authorization': 'Bearer token',
            'X-Custom-Header': 'value'
        }
    }
)
```

### Source 4: Local Files

Load data from local file system.

#### NumPy Files

```python
from q_store.data import DatasetConfig, DatasetSource, DatasetLoader

# Single .npy file
config = DatasetConfig(
    name='my_data',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'numpy',
        'file_path': 'data/my_data.npy',
        'labels_path': 'data/my_labels.npy'  # Optional
    }
)

# .npz archive
config = DatasetConfig(
    name='my_data',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'numpy',
        'file_path': 'data/dataset.npz',
        'data_key': 'x',        # Key for features in .npz
        'labels_key': 'y'       # Key for labels in .npz
    }
)

dataset = DatasetLoader.load(config)
```

#### CSV Files

```python
# CSV file
config = DatasetConfig(
    name='iris',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'csv',
        'file_path': 'data/iris.csv',
        'feature_columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'label_column': 'species',
        'test_split': 0.2  # 20% for test set
    }
)

dataset = DatasetLoader.load(config)
```

#### Image Directories

```python
# Image directory (organized by class)
# Structure: data/images/class1/*.jpg, data/images/class2/*.jpg, ...
config = DatasetConfig(
    name='my_images',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'images',
        'directory': 'data/images',
        'image_size': (128, 128),  # Resize to this size
        'test_split': 0.2,
        'color_mode': 'rgb'  # 'rgb' or 'grayscale'
    }
)

dataset = DatasetLoader.load(config)
```

#### HDF5 Files

```python
# HDF5 file
config = DatasetConfig(
    name='large_dataset',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'hdf5',
        'file_path': 'data/dataset.h5',
        'data_key': 'features',
        'labels_key': 'labels'
    }
)

dataset = DatasetLoader.load(config)
```

#### Parquet Files

```python
# Parquet file
config = DatasetConfig(
    name='large_tabular',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'parquet',
        'file_path': 'data/dataset.parquet',
        'feature_columns': None,  # None = all except label
        'label_column': 'target',
        'test_split': 0.2
    }
)

dataset = DatasetLoader.load(config)
```

---

## Data Preprocessing

### Quantum Preprocessing

Prepare classical data for quantum encoding.

```python
from q_store.data import QuantumPreprocessor

# Normalization methods
preprocessor = QuantumPreprocessor(
    method='minmax',        # Options: 'minmax', 'zscore', 'l1', 'l2', 'robust'
    feature_range=(0, 1)   # For minmax
)

# Fit on training data
x_train_normalized = preprocessor.fit_transform(x_train)

# Transform test data
x_test_normalized = preprocessor.transform(x_test)

# Access preprocessing parameters
print(f"Min: {preprocessor.min_val}")
print(f"Max: {preprocessor.max_val}")
```

**Normalization Methods**:

```python
# Min-Max Normalization (recommended for quantum)
preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))

# Z-Score Standardization
preprocessor = QuantumPreprocessor(method='zscore')

# L1 Normalization
preprocessor = QuantumPreprocessor(method='l1')

# L2 Normalization
preprocessor = QuantumPreprocessor(method='l2')

# Robust Scaling (resistant to outliers)
preprocessor = QuantumPreprocessor(method='robust')
```

### Data Splitting

Split data for training, validation, and testing.

```python
from q_store.data import DataSplitter

# Train/Val/Test Split
splits = DataSplitter.train_val_test_split(
    x_data, y_data,
    split_ratio=(0.7, 0.15, 0.15),  # 70% train, 15% val, 15% test
    shuffle=True,
    random_seed=42
)

x_train, y_train = splits['train']
x_val, y_val = splits['val']
x_test, y_test = splits['test']

# K-Fold Cross-Validation
folds = DataSplitter.k_fold_split(x_data, y_data, n_splits=5, shuffle=True)

for fold_idx, (train_data, val_data) in enumerate(folds):
    x_train, y_train = train_data
    x_val, y_val = val_data
    print(f"Fold {fold_idx}: Train={len(x_train)}, Val={len(x_val)}")

# Time Series Split (preserves order)
splits = DataSplitter.time_series_split(x_data, y_data, n_splits=5)

for split_idx, (train_data, test_data) in enumerate(splits):
    x_train, y_train = train_data
    x_test, y_test = test_data
    print(f"Split {split_idx}: Train={len(x_train)}, Test={len(x_test)}")
```

### Quantum Data Adapters

Adapt data for specific quantum encoding schemes.

```python
from q_store.data import QuantumDataAdapter, EncodingType

# Amplitude encoding adapter
adapter = QuantumDataAdapter(
    n_qubits=8,
    encoding=EncodingType.AMPLITUDE,
    normalize=True
)

# Prepare data for quantum circuit
quantum_data = adapter.prepare(classical_data)
print(f"Input shape: {classical_data.shape}")
print(f"Output shape: {quantum_data.shape}")

# Angle encoding
adapter = QuantumDataAdapter(
    n_qubits=10,
    encoding=EncodingType.ANGLE,
    normalize=True
)

# Basis encoding
adapter = QuantumDataAdapter(
    n_qubits=8,
    encoding=EncodingType.BASIS,
    normalize=False
)
```

### Dimension Reduction

Reduce high-dimensional data for quantum processing.

```python
from q_store.data import DimensionReducer

# PCA reduction
reducer = DimensionReducer(
    method='pca',
    n_components=64,  # Reduce to 64 features
    random_seed=42
)

x_reduced = reducer.fit_transform(x_train)
x_test_reduced = reducer.transform(x_test)

# Other methods
reducer = DimensionReducer(method='autoencoder', n_components=32)
reducer = DimensionReducer(method='pool', n_components=64, pool_size=2)
reducer = DimensionReducer(method='random', n_components=50)
```

### Image Preprocessing

Specialized preprocessing for image data.

```python
from q_store.data import QuantumImageAdapter

adapter = QuantumImageAdapter(
    target_size=(16, 16),  # Downscale images
    encoding='amplitude',
    flatten=True,
    normalize=True
)

# Transform images
quantum_images = adapter.transform(images)
print(f"Original: {images.shape} → Quantum: {quantum_images.shape}")
```

---

## Data Validation

### Automated Validation

Check data compatibility with quantum ML.

```python
from q_store.data import DataValidator

validator = DataValidator()

# Validate dataset
validation_results = validator.validate_all(
    x_data, y_data,
    n_qubits=8,
    encoding='amplitude'
)

# Check results
for check_name, result in validation_results.items():
    status = "✓" if result['valid'] else "✗"
    print(f"{status} {check_name}: {result['message']}")
```

**Validation Checks**:

- ✓ Shape compatibility
- ✓ Data type checks
- ✓ NaN/infinity detection
- ✓ Value range validation
- ✓ Quantum encoding compatibility
- ✓ Label format checks

### Data Profiling

Get statistical overview of your dataset.

```python
from q_store.data import DataProfiler

profiler = DataProfiler()

# Profile dataset
profile = profiler.profile(
    x_data, y_data,
    include_outliers=True,
    include_correlations=True
)

# Print statistics
print(f"Samples: {profile['n_samples']}")
print(f"Features: {profile['n_features']}")
print(f"Classes: {profile['n_classes']}")
print(f"Class balance: {profile['class_balance']}")
print(f"Missing values: {profile['missing_values']}")

# Feature statistics
for feature_stats in profile['feature_statistics']:
    print(f"\nFeature: {feature_stats['name']}")
    print(f"  Mean: {feature_stats['mean']:.4f}")
    print(f"  Std: {feature_stats['std']:.4f}")
    print(f"  Range: [{feature_stats['min']:.4f}, {feature_stats['max']:.4f}]")

# Outlier detection
if profile['outliers']:
    print(f"\nOutliers detected: {profile['outliers']['n_outliers']}")
    print(f"Outlier ratio: {profile['outliers']['outlier_ratio']:.2%}")
```

### Outlier Detection

Identify and handle outliers.

```python
from q_store.data import OutlierDetector

# IQR method
detector = OutlierDetector(method='iqr', threshold=1.5)
x_clean = detector.fit_transform(x_data)

# Z-score method
detector = OutlierDetector(method='zscore', threshold=3.0)
x_clean = detector.fit_transform(x_data)

# Isolation Forest
detector = OutlierDetector(method='isolation_forest', contamination=0.1)
x_clean = detector.fit_transform(x_data)

# Get outlier mask
outlier_mask = detector.get_outlier_mask(x_data)
print(f"Outliers: {outlier_mask.sum()} / {len(outlier_mask)}")
```

---

## Data Generators

### Basic Usage

```python
from q_store.data import QuantumDataGenerator

# Create generator
generator = QuantumDataGenerator(
    x_data, y_data,
    batch_size=32,
    shuffle=True,
    augmentation=None,
    preprocessing=None
)

# Iterate over batches
for batch_x, batch_y in generator:
    print(f"Batch shape: {batch_x.shape}")
    # Use batch for training
```

### Async Generators

```python
# For async training loops
async for batch_x, batch_y in generator:
    # Async training operations
    await model.train_step(batch_x, batch_y)
```

### Streaming Generator

For large datasets that don't fit in memory.

```python
from q_store.data import StreamingDataGenerator

generator = StreamingDataGenerator(
    data_path='large_dataset.h5',
    batch_size=32,
    chunk_size=1000,  # Load 1000 samples at a time
    shuffle_chunks=True
)

for batch_x, batch_y in generator:
    # Process batch
    pass
```

### Balanced Batch Generator

Ensure balanced class distribution in each batch.

```python
from q_store.data import BalancedBatchGenerator

generator = BalancedBatchGenerator(
    x_data, y_data,
    batch_size=32,
    samples_per_class=None  # Auto-balance
)

for batch_x, batch_y in generator:
    # Each batch has balanced classes
    unique, counts = np.unique(batch_y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
```

### Infinite Generator

Continuously generate batches for training.

```python
from q_store.data import InfiniteDataGenerator

generator = InfiniteDataGenerator(
    x_data, y_data,
    batch_size=32,
    shuffle=True
)

# Generate specific number of batches
for i, (batch_x, batch_y) in enumerate(generator):
    if i >= 100:  # Stop after 100 batches
        break
    # Process batch
```

---

## Data Augmentation

### Quantum Augmentation

Quantum-specific augmentation techniques.

```python
from q_store.data import QuantumAugmentation

augmentation = QuantumAugmentation(
    phase_shift_range=0.1,     # Phase perturbation
    amplitude_noise=0.01,      # Amplitude noise
    rotation_range=0.1,        # Random rotations
    probability=0.5            # Apply with 50% probability
)

# Apply augmentation
x_augmented = augmentation.apply(x_data)
```

### Classical Augmentation

Classical image augmentation (wraps albumentations).

```python
# Requires: pip install albumentations

from q_store.data import ClassicalAugmentation

augmentation = ClassicalAugmentation(
    transforms=['horizontal_flip', 'rotate', 'brightness'],
    probability=0.5
)

# Apply to images
images_augmented = augmentation.apply(images)
```

**Available Transforms**:

- `horizontal_flip` - Horizontal flip
- `vertical_flip` - Vertical flip
- `rotate` - Random rotation
- `brightness` - Brightness adjustment
- `contrast` - Contrast adjustment
- `gaussian_noise` - Add Gaussian noise
- `blur` - Gaussian blur

### Hybrid Augmentation

Combine classical and quantum augmentation.

```python
from q_store.data import HybridAugmentation

augmentation = HybridAugmentation(
    classical_transforms=['horizontal_flip', 'rotate'],
    quantum_config={
        'phase_shift_range': 0.1,
        'amplitude_noise': 0.01
    },
    apply_classical_first=True,  # Classical → Quantum
    classical_probability=0.5,
    quantum_probability=0.5
)

# Apply hybrid augmentation
x_augmented = augmentation.apply(x_data)
```

### Custom Augmentation Pipeline

```python
from q_store.data import AugmentationPipeline, QuantumAugmentation, ClassicalAugmentation

pipeline = AugmentationPipeline([
    ClassicalAugmentation(transforms=['horizontal_flip']),
    QuantumAugmentation(phase_shift_range=0.05),
    # Add more augmentations...
])

# Apply pipeline
x_augmented = pipeline.apply(x_data)
```

### Generator with Augmentation

```python
from q_store.data import QuantumDataGenerator, QuantumAugmentation

augmentation = QuantumAugmentation(phase_shift_range=0.1)

generator = QuantumDataGenerator(
    x_data, y_data,
    batch_size=32,
    augmentation=augmentation  # Apply during batch generation
)

for batch_x, batch_y in generator:
    # batch_x is augmented
    pass
```

---

## Advanced Topics

### Custom Data Sources

Create custom dataset loaders.

```python
from q_store.data import SourceAdapter, Dataset, DatasetConfig

class CustomSourceAdapter(SourceAdapter):
    """Custom data source adapter."""

    def load(self, config: DatasetConfig) -> Dataset:
        # Implement custom loading logic
        x_train, y_train, x_test, y_test = self._load_custom_data(config)

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            metadata={}
        )

    def _load_custom_data(self, config):
        # Your custom loading logic here
        pass

# Register and use
from q_store.data import DatasetLoader
DatasetLoader.register_adapter('custom', CustomSourceAdapter())

# Use custom source
config = DatasetConfig(name='my_data', source='custom', source_params={})
dataset = DatasetLoader.load(config)
```

### Batch Preprocessing

Apply preprocessing during batch generation.

```python
from q_store.data import QuantumDataGenerator, QuantumPreprocessor

preprocessor = QuantumPreprocessor(method='minmax')

generator = QuantumDataGenerator(
    x_data, y_data,
    batch_size=32,
    preprocessing=preprocessor  # Apply per batch
)
```

### Memory-Efficient Loading

For very large datasets.

```python
from q_store.data import StreamingDataGenerator

# Stream from HDF5 file
generator = StreamingDataGenerator(
    data_path='huge_dataset.h5',
    batch_size=32,
    chunk_size=1000,    # Load 1000 samples at a time
    prefetch=True,      # Prefetch next chunk
    num_workers=4       # Parallel loading
)
```

---

## Best Practices

### 1. Always Validate Data

```python
from q_store.data import DataValidator

validator = DataValidator()
results = validator.validate_all(x_train, y_train, n_qubits=8)

if not all(r['valid'] for r in results.values()):
    print("⚠ Data validation failed!")
    for check, result in results.items():
        if not result['valid']:
            print(f"  ✗ {check}: {result['message']}")
```

### 2. Use Quantum Preprocessing

```python
from q_store.data import QuantumPreprocessor

# Always normalize to [0, 1] for quantum encoding
preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
```

### 3. Profile Your Dataset

```python
from q_store.data import DataProfiler

profiler = DataProfiler()
profile = profiler.profile(x_train, y_train)

print(f"Dataset: {profile['n_samples']} samples, {profile['n_features']} features")
print(f"Class balance: {profile['class_balance']}")
```

### 4. Use Efficient Generators

```python
from q_store.data import QuantumDataGenerator

# Use generators instead of loading all data
generator = QuantumDataGenerator(
    x_train, y_train,
    batch_size=32,
    shuffle=True
)

# Memory efficient training
await trainer.train(model, generator, val_generator)
```

### 5. Apply Augmentation

```python
from q_store.data import QuantumAugmentation

# Increase training data diversity
augmentation = QuantumAugmentation(phase_shift_range=0.05)

generator = QuantumDataGenerator(
    x_train, y_train,
    batch_size=32,
    augmentation=augmentation
)
```

---

## API Reference

### Core Classes

**DatasetLoader**

- `load(config: DatasetConfig) -> Dataset`
- `register_adapter(name: str, adapter: SourceAdapter)`

**DatasetConfig**

- `name: str`
- `source: DatasetSource`
- `source_params: Dict[str, Any]`

**Dataset**

- `x_train: np.ndarray`
- `y_train: np.ndarray`
- `x_test: np.ndarray`
- `y_test: np.ndarray`
- `metadata: Dict[str, Any]`

**QuantumPreprocessor**

- `fit_transform(data: np.ndarray) -> np.ndarray`
- `transform(data: np.ndarray) -> np.ndarray`
- Methods: 'minmax', 'zscore', 'l1', 'l2', 'robust'

**QuantumDataGenerator**

- `__init__(x_data, y_data, batch_size, shuffle, augmentation, preprocessing)`
- `__iter__()` / `__next__()`

**DataValidator**

- `validate_all(x_data, y_data, n_qubits, encoding) -> Dict`
- `check_shape(x_data, y_data) -> Dict`
- `check_quantum_compatibility(data, n_qubits, encoding) -> Dict`

**DataProfiler**

- `profile(x_data, y_data, include_outliers, include_correlations) -> Dict`

**QuantumAugmentation**

- `apply(data: np.ndarray) -> np.ndarray`

**ClassicalAugmentation**

- `apply(images: np.ndarray) -> np.ndarray`

**HybridAugmentation**

- `apply(data: np.ndarray) -> np.ndarray`

---

## Examples

See the `examples/` directory for complete examples:

- `fashion_mnist_with_backend_api.py` - Complete workflow with all data features
- `end_to_end_workflow.py` - Full pipeline including data management

---

## Next Steps

1. **Try the Examples**: Run the example scripts to see features in action
2. **Read the Migration Guide**: [V4_1_1_MIGRATION_GUIDE.md](V4_1_1_MIGRATION_GUIDE.md)
3. **Check API Reference**: [API_REFERENCE_V4_1_1.md](API_REFERENCE_V4_1_1.md)
4. **Join the Community**: [GitHub](https://github.com/yucelz/q-store)

---

**Q-Store v4.1.1 - Unified Data Management for Quantum ML** 🎉
