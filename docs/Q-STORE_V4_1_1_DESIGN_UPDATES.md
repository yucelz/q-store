# Q-Store v4.1.1 Design Updates Summary

**Date**: January 4, 2026
**Status**: Awaiting Confirmation

---

## Key Design Change: Generic Dataset Loaders

### Previous Design (Rejected)

❌ **Specific loader classes for each dataset:**
- `FashionMNISTLoader`
- `MNISTLoader`
- `CIFAR10Loader`
- `CIFAR100Loader`
- `CustomDatasetLoader`

**Problems:**
- Not scalable (need new class for each dataset)
- Code duplication across loaders
- Hard to maintain and extend
- Inconsistent interfaces

---

### New Design (Approved Architecture)

✅ **Generic, plugin-based loader system:**

#### Core Components

1. **`DatasetLoader`** - Unified loader class with registry pattern
2. **`DatasetConfig`** - Configuration object for dataset loading
3. **`DatasetSource`** - Enum for source types (Keras, HuggingFace, Backend API, local files)
4. **`Dataset`** - Unified container for all datasets
5. **Source Adapters** - Plugin architecture for different sources:
   - `KerasSourceAdapter`
   - `HuggingFaceSourceAdapter`
   - `BackendAPISourceAdapter`
   - `LocalFilesSourceAdapter`

#### Key Advantages

✅ **Single Unified Interface**
```python
# Same interface for all datasets
dataset = DatasetLoader.load(config)
```

✅ **Configuration-Driven**
```python
# Just change config, not code
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
)
```

✅ **Easily Extensible**
```python
# Add new source by creating adapter
class MyCustomSourceAdapter(SourceAdapter):
    def load(self, config, cache_dir):
        # Custom loading logic
        pass

# Register and use
DatasetLoader.register_adapter(DatasetSource.CUSTOM, MyCustomSourceAdapter())
```

✅ **Source Agnostic**
```python
# Load Fashion MNIST from any source with same code
# From Keras
config1 = DatasetConfig(name='fashion_mnist', source=DatasetSource.KERAS, ...)

# From HuggingFace
config2 = DatasetConfig(name='fashion_mnist', source=DatasetSource.HUGGINGFACE, ...)

# From Backend API
config3 = DatasetConfig(name='fashion_mnist', source=DatasetSource.BACKEND_API, ...)

# All return same Dataset object!
dataset1 = DatasetLoader.load(config1)
dataset2 = DatasetLoader.load(config2)
dataset3 = DatasetLoader.load(config3)
```

---

## Supported Data Sources

| Source | Adapter Class | Supported Formats | Use Cases |
|--------|---------------|-------------------|-----------|
| **Keras** | `KerasSourceAdapter` | Built-in TensorFlow datasets | Quick prototyping, standard benchmarks |
| **HuggingFace** | `HuggingFaceSourceAdapter` | 500K+ datasets from HF Hub | Large-scale datasets, NLP, vision |
| **Backend API** | `BackendAPISourceAdapter` | Backend-managed datasets | Production, team collaboration |
| **Local Files** | `LocalFilesSourceAdapter` | NumPy, CSV, images, HDF5, Parquet | Custom datasets, private data |

---

## Implementation Details

### File Structure

```
q_store/data/
├── __init__.py
├── loaders.py              # Generic loader with adapters
│   ├── DatasetSource (enum)
│   ├── DatasetConfig (dataclass)
│   ├── DatasetLoader (main class)
│   ├── Dataset (container)
│   ├── SourceAdapter (ABC)
│   ├── KerasSourceAdapter
│   ├── HuggingFaceSourceAdapter
│   ├── BackendAPISourceAdapter
│   └── LocalFilesSourceAdapter
├── adapters.py             # Quantum data adapters
├── preprocessing.py        # Preprocessing utilities
├── augmentation.py         # Data augmentation
├── generators.py           # Data generators
├── validation.py           # Data validation
└── backend_client.py       # Backend API client
```

### Usage Examples

#### Example 1: Load from Keras

```python
from q_store.data.loaders import DatasetLoader, DatasetConfig, DatasetSource

config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'},
    split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
)
dataset = DatasetLoader.load(config)

# Access data
x_train, y_train = dataset.get_split('train')
print(f"Loaded {dataset.num_samples} samples with {dataset.num_classes} classes")
```

#### Example 2: Load from HuggingFace

```python
config = DatasetConfig(
    name='cifar10',
    source=DatasetSource.HUGGINGFACE,
    source_params={'dataset_name': 'cifar10'}
)
dataset = DatasetLoader.load(config)
```

#### Example 3: Load from Backend API

```python
from q_store.data.backend_client import BackendAPIClient

api_client = BackendAPIClient(base_url="http://localhost:8000", api_key="your_key")
config = DatasetConfig(
    name='my_dataset',
    source=DatasetSource.BACKEND_API,
    source_params={'dataset_id': 'uuid-123', 'api_client': api_client}
)
dataset = DatasetLoader.load(config)
```

#### Example 4: Load from Local Files

```python
config = DatasetConfig(
    name='custom_dataset',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'numpy',
        'train_data': '/path/to/x_train.npy',
        'train_labels': '/path/to/y_train.npy',
        'test_data': '/path/to/x_test.npy',
        'test_labels': '/path/to/y_test.npy'
    }
)
dataset = DatasetLoader.load(config)
```

---

## Updated Implementation Tasks

### Phase 1: Generic Dataset Loaders

1. **Core Classes**
   - [ ] `DatasetSource` enum with all source types
   - [ ] `DatasetConfig` dataclass with validation
   - [ ] `Dataset` container class with utilities
   - [ ] `DatasetLoader` main class with registry

2. **Source Adapters**
   - [ ] `SourceAdapter` abstract base class
   - [ ] `KerasSourceAdapter` with SUPPORTED_DATASETS registry
   - [ ] `HuggingFaceSourceAdapter` with HF Hub integration
   - [ ] `BackendAPISourceAdapter` with BackendAPIClient integration
   - [ ] `LocalFilesSourceAdapter` with format handlers:
     - [ ] NumPy loader (`.npy`, `.npz`)
     - [ ] CSV loader
     - [ ] Image directory loader
     - [ ] HDF5 loader
     - [ ] Parquet loader

3. **Auto-Registration System**
   - [ ] Adapter registration mechanism
   - [ ] Default adapters auto-registered on import
   - [ ] Custom adapter registration support

4. **Testing**
   - [ ] Unit tests for each adapter
   - [ ] Integration tests for each source type
   - [ ] End-to-end loading tests

---

## Benefits Summary

| Aspect | Benefit |
|--------|---------|
| **Maintainability** | Single loader class instead of N dataset classes |
| **Extensibility** | Add new sources by creating adapters (no core changes) |
| **Consistency** | Same `Dataset` object from all sources |
| **Flexibility** | Configuration-driven, easy to parameterize |
| **Testability** | Test adapters independently |
| **Documentation** | Single API to document and learn |
| **Code Reuse** | Shared logic in base classes |

---

## Next Steps

1. ✅ Review and approve this design
2. ⏳ Update todo list to reflect generic loader implementation
3. ⏳ Begin implementation of Phase 1 (Generic Dataset Loaders)
4. ⏳ Create example scripts demonstrating each source type
5. ⏳ Write comprehensive tests

---

## Confirmation Required

**Please confirm:**
- ✅ Approve generic loader design
- ✅ Proceed with adapter pattern architecture
- ✅ Update todo list accordingly

**Once confirmed, I will:**
1. Update the todo list with specific tasks for generic loaders
2. Provide implementation priorities
3. Create detailed implementation guide for each adapter

---

**Status**: ⏳ Awaiting User Confirmation
