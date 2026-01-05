# Q-Store v4.1.1 Architecture Design

**Version**: 4.1.1
**Date**: January 4, 2026
**Status**: Design Phase
**Target**: Backend API Integration & Data Management Enhancement

---

## Executive Summary

Q-Store v4.1.1 focuses on **Backend API Integration** and **Data Management Enhancement** to bridge the gap between the Q-Store Backend API (which has comprehensive dataset management) and the Q-Store Core library (which currently lacks data loading and management utilities).

### Key Objectives

1. **Backend-Core Integration**: Enable seamless data flow from Backend API datasets to Core library training
2. **Data Management Layer**: Add dataset loaders, adapters, and preprocessing utilities
3. **Experiment Tracking**: MLflow integration for experiment management
4. **Training Enhancements**: Advanced schedulers, early stopping, hyperparameter tuning
5. **Production Readiness**: Enhanced monitoring, logging, and error handling

---

## Current State Analysis

### Q-Store Backend API v4.1.0 (85% Complete)

**Implemented:**
- ✅ **Dataset Management** (15 API endpoints)
  - HuggingFace integration (500K+ datasets)
  - Label Studio integration (enterprise labeling)
  - Albumentations integration (70+ transforms)
- ✅ **Database Models**: Dataset, relationships to TrainingJob
- ✅ **Service Layer**: HuggingFaceDatasetService, LabelStudioService, AugmentationService
- ✅ **Training Integration**: TrainingJob accepts `dataset_id`
- ✅ **Dependencies**: datasets, label-studio-sdk, albumentations

**Pending:**
- ❌ Postman collection documentation
- ❌ QUICKSTART.md updates
- ⏳ Database migration execution
- ⏳ Environment configuration

### Q-Store Core v4.1.0 (Production-Ready, Missing Data Layer)

**Implemented (151 modules):**
- ✅ **Quantum ML Training** (20 modules)
  - QuantumTrainer with adaptive training
  - SPSA gradient estimation (48x faster)
  - Multi-backend orchestration (2-3x throughput)
  - Natural gradient descent (2-3x fewer iterations)
- ✅ **Quantum Layers** (9 modules)
  - Quantum-first architecture (70% quantum compute)
  - PyTorch & TensorFlow integration
- ✅ **Backend Abstraction** (14 modules)
  - IonQ, Cirq, Qiskit, qsim, Lightning
  - Health monitoring, connection pooling
- ✅ **Optimization** (8 modules)
  - Adaptive circuit optimization (30-40% faster)
  - Smart caching (3-4x faster prep)
  - Adaptive shot allocation (20-30% savings)
- ✅ **Error Mitigation** (4 modules)
  - ZNE, PEC, measurement error correction
- ✅ **Advanced Analysis** (13 modules)
  - Entanglement measures, tomography, verification

**Critical Gaps:**
- ❌ **No dataset loaders** (Fashion MNIST, MNIST, CIFAR, custom)
- ❌ **No data adapters** for quantum ML
- ❌ **No preprocessing utilities** compatible with backend API
- ❌ **No data pipeline** integration with backend datasets
- ❌ **No experiment tracking** (MLflow, W&B)
- ❌ **No hyperparameter tuning** framework
- ❌ **No early stopping** in QuantumTrainer
- ❌ **No advanced schedulers** (cosine annealing, cyclic LR)
- ❌ **No data augmentation** utilities

---

## v4.1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Q-Store Backend API v4.1.0                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Dataset Management (85% Complete)                                 │ │
│  │  ├─ HuggingFace Integration (500K+ datasets)                       │ │
│  │  ├─ Label Studio Integration (enterprise labeling)                 │ │
│  │  ├─ Albumentations Integration (70+ transforms)                    │ │
│  │  ├─ Database: Dataset model with relationships                     │ │
│  │  └─ API: 15 endpoints for dataset operations                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Training Integration                                              │ │
│  │  ├─ TrainingJob.dataset_id (Foreign Key)                           │ │
│  │  ├─ DataConfig schema for data loading                             │ │
│  │  └─ Training tasks load data from datasets                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  │ REST API / SDK
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Q-Store Core v4.1.1 (NEW/ENHANCED)                    │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  🆕 Data Management Layer (q_store/data/)                          │ │
│  │  ├─ loaders.py          Dataset loaders (Fashion MNIST, MNIST, etc)│ │
│  │  ├─ adapters.py         Quantum data adapters                      │ │
│  │  ├─ preprocessing.py    Preprocessing utilities                    │ │
│  │  ├─ augmentation.py     Data augmentation for quantum ML           │ │
│  │  ├─ generators.py       Data generators for training               │ │
│  │  ├─ validation.py       Data validation and quality checks         │ │
│  │  └─ backend_client.py   Backend API client for dataset access      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  🔧 ML Training Enhancements (q_store/ml/)                         │ │
│  │  ├─ schedulers.py       Learning rate schedulers (cosine, cyclic)  │ │
│  │  ├─ early_stopping.py   Early stopping with convergence detection  │ │
│  │  ├─ callbacks.py        Training callbacks (logging, checkpointing)│ │
│  │  └─ quantum_trainer.py  ENHANCED with new features                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  🆕 Experiment Tracking (q_store/tracking/)                        │ │
│  │  ├─ mlflow_tracker.py   MLflow integration                         │ │
│  │  ├─ logger.py           Structured logging                         │ │
│  │  └─ metrics_tracker.py  Enhanced metrics tracking                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  🆕 Hyperparameter Tuning (q_store/tuning/)                        │ │
│  │  ├─ bayesian_optimizer.py  Bayesian optimization                   │ │
│  │  ├─ grid_search.py         Grid search                             │ │
│  │  ├─ random_search.py       Random search                           │ │
│  │  └─ optuna_integration.py  Optuna integration                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Existing Quantum ML Framework (151 modules)                       │ │
│  │  ├─ ml/                  Training infrastructure (20 modules)      │ │
│  │  ├─ layers/              Quantum layers (9 modules)                │ │
│  │  ├─ backends/            Backend abstraction (14 modules)          │ │
│  │  ├─ optimization/        Circuit optimization (8 modules)          │ │
│  │  ├─ mitigation/          Error mitigation (4 modules)              │ │
│  │  └─ ...                  Complete quantum ML stack                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Management Layer (`q_store/data/`)

#### 1.1 Generic Dataset Loaders (`loaders.py`)

**Purpose**: Unified, extensible dataset loading system with plugin architecture

**Design Philosophy:**
- **Single unified interface** instead of per-dataset classes
- **Source adapters** for different data sources (Keras, HuggingFace, Backend API, local files)
- **Dataset registry** for easy extension
- **Configuration-driven** approach for flexibility

**Core Classes:**

```python
class DatasetSource(Enum):
    """Supported dataset sources."""
    KERAS = "keras"
    HUGGINGFACE = "huggingface"
    BACKEND_API = "backend_api"
    LOCAL_FILES = "local_files"
    NUMPY_FILES = "numpy"
    CSV_FILES = "csv"
    IMAGE_DIRECTORY = "images"

class DatasetConfig:
    """Configuration for dataset loading."""
    def __init__(
        self,
        name: str,
        source: DatasetSource,
        source_params: Optional[Dict[str, Any]] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        split_config: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            name: Dataset identifier (e.g., 'fashion_mnist', 'cifar10')
            source: Source type (Keras, HuggingFace, etc.)
            source_params: Source-specific parameters
                Examples:
                - Keras: {'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
                - HuggingFace: {'dataset_name': 'fashion_mnist', 'split': 'train'}
                - Backend API: {'dataset_id': 'uuid-123', 'api_client': client}
                - Local: {'train_images': 'path/to/images.npy', 'train_labels': ...}
            preprocessing: Preprocessing config (normalize, resize, etc.)
            split_config: Train/val/test split ratios
        """
        pass

class DatasetLoader:
    """
    Unified dataset loader with plugin architecture.

    Usage:
        # Load Fashion MNIST from Keras
        config = DatasetConfig(
            name='fashion_mnist',
            source=DatasetSource.KERAS,
            source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
        )
        data = DatasetLoader.load(config)

        # Load from HuggingFace
        config = DatasetConfig(
            name='fashion_mnist',
            source=DatasetSource.HUGGINGFACE,
            source_params={'dataset_name': 'fashion_mnist'}
        )
        data = DatasetLoader.load(config)

        # Load from Backend API
        config = DatasetConfig(
            name='my_dataset',
            source=DatasetSource.BACKEND_API,
            source_params={'dataset_id': 'uuid-123', 'api_client': client}
        )
        data = DatasetLoader.load(config)
    """

    # Registry of source adapters
    _adapters: Dict[DatasetSource, 'SourceAdapter'] = {}

    @classmethod
    def register_adapter(cls, source: DatasetSource, adapter: 'SourceAdapter'):
        """Register a source adapter."""
        cls._adapters[source] = adapter

    @classmethod
    def load(
        cls,
        config: DatasetConfig,
        cache_dir: Optional[str] = None
    ) -> 'Dataset':
        """
        Load dataset using configuration.

        Returns:
            Dataset object with train/val/test splits
        """
        if config.source not in cls._adapters:
            raise ValueError(f"No adapter registered for source: {config.source}")

        adapter = cls._adapters[config.source]
        return adapter.load(config, cache_dir)

    @classmethod
    def list_available_datasets(
        cls,
        source: Optional[DatasetSource] = None
    ) -> List[Dict[str, Any]]:
        """List available datasets from a source."""
        pass

class SourceAdapter(ABC):
    """Abstract base class for dataset source adapters."""

    @abstractmethod
    def load(self, config: DatasetConfig, cache_dir: Optional[str]) -> 'Dataset':
        """Load dataset from this source."""
        pass

    @abstractmethod
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets from this source."""
        pass

class KerasSourceAdapter(SourceAdapter):
    """Adapter for Keras datasets."""

    # Built-in Keras datasets
    SUPPORTED_DATASETS = {
        'mnist': 'tensorflow.keras.datasets.mnist',
        'fashion_mnist': 'tensorflow.keras.datasets.fashion_mnist',
        'cifar10': 'tensorflow.keras.datasets.cifar10',
        'cifar100': 'tensorflow.keras.datasets.cifar100',
        'imdb': 'tensorflow.keras.datasets.imdb',
        'reuters': 'tensorflow.keras.datasets.reuters',
        'boston_housing': 'tensorflow.keras.datasets.boston_housing'
    }

    def load(self, config: DatasetConfig, cache_dir: Optional[str]) -> 'Dataset':
        """Load from Keras datasets."""
        dataset_module = config.source_params.get('dataset_module')
        # Dynamic import and load
        pass

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List Keras datasets."""
        return [{'name': name, 'module': module}
                for name, module in self.SUPPORTED_DATASETS.items()]

class HuggingFaceSourceAdapter(SourceAdapter):
    """Adapter for HuggingFace datasets."""

    def load(self, config: DatasetConfig, cache_dir: Optional[str]) -> 'Dataset':
        """Load from HuggingFace Hub."""
        from datasets import load_dataset
        dataset_name = config.source_params.get('dataset_name')
        hf_config = config.source_params.get('config')
        revision = config.source_params.get('revision')
        # Load and convert to Dataset object
        pass

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List HuggingFace datasets (search by query)."""
        pass

class BackendAPISourceAdapter(SourceAdapter):
    """Adapter for Q-Store Backend API."""

    def load(self, config: DatasetConfig, cache_dir: Optional[str]) -> 'Dataset':
        """Load from Backend API."""
        api_client = config.source_params.get('api_client')
        dataset_id = config.source_params.get('dataset_id')
        # Load via API client
        pass

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List datasets from Backend API."""
        pass

class LocalFilesSourceAdapter(SourceAdapter):
    """Adapter for local file datasets."""

    SUPPORTED_FORMATS = ['numpy', 'csv', 'images', 'hdf5', 'parquet']

    def load(self, config: DatasetConfig, cache_dir: Optional[str]) -> 'Dataset':
        """Load from local files."""
        file_format = config.source_params.get('format', 'numpy')

        if file_format == 'numpy':
            return self._load_numpy(config)
        elif file_format == 'csv':
            return self._load_csv(config)
        elif file_format == 'images':
            return self._load_images(config)
        elif file_format == 'hdf5':
            return self._load_hdf5(config)
        elif file_format == 'parquet':
            return self._load_parquet(config)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def _load_numpy(self, config: DatasetConfig) -> 'Dataset':
        """Load from .npy or .npz files."""
        pass

    def _load_csv(self, config: DatasetConfig) -> 'Dataset':
        """Load from CSV files."""
        pass

    def _load_images(self, config: DatasetConfig) -> 'Dataset':
        """Load from image directory."""
        pass

class Dataset:
    """
    Unified dataset container.

    Provides consistent interface regardless of source.
    """
    def __init__(
        self,
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.metadata = metadata or {}

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self.x_train) + (len(self.x_val) if self.x_val is not None else 0) + \
               (len(self.x_test) if self.x_test is not None else 0)

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(np.unique(self.y_train))

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of a single input sample."""
        return self.x_train.shape[1:]

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get specific split (train/val/test)."""
        if split == 'train':
            return self.x_train, self.y_train
        elif split == 'val':
            if self.x_val is None:
                raise ValueError("Validation split not available")
            return self.x_val, self.y_val
        elif split == 'test':
            if self.x_test is None:
                raise ValueError("Test split not available")
            return self.x_test, self.y_test
        else:
            raise ValueError(f"Invalid split: {split}")

    def save(self, path: str, format: str = 'npz'):
        """Save dataset to file."""
        pass

    @classmethod
    def load(cls, path: str) -> 'Dataset':
        """Load dataset from file."""
        pass

# Initialize default adapters
DatasetLoader.register_adapter(DatasetSource.KERAS, KerasSourceAdapter())
DatasetLoader.register_adapter(DatasetSource.HUGGINGFACE, HuggingFaceSourceAdapter())
DatasetLoader.register_adapter(DatasetSource.BACKEND_API, BackendAPISourceAdapter())
DatasetLoader.register_adapter(DatasetSource.LOCAL_FILES, LocalFilesSourceAdapter())
```

**Features:**

- ✅ **Unified Interface**: Single `DatasetLoader.load(config)` for all sources
- ✅ **Extensible**: Easy to add new sources via adapter pattern
- ✅ **Configuration-Driven**: Dataset loading via `DatasetConfig` objects
- ✅ **Source Agnostic**: Same code works with Keras, HuggingFace, Backend API, local files
- ✅ **Auto-Registration**: Adapters auto-register on import
- ✅ **Consistent Output**: All loaders return `Dataset` objects
- ✅ **Metadata Support**: Preserve dataset metadata across sources
- ✅ **Caching**: Optional caching for downloaded datasets

**Usage Examples:**

```python
# Example 1: Load Fashion MNIST from Keras
config = DatasetConfig(
    name='fashion_mnist',
    source=DatasetSource.KERAS,
    source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'},
    split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
)
dataset = DatasetLoader.load(config)

# Example 2: Load from HuggingFace
config = DatasetConfig(
    name='cifar10',
    source=DatasetSource.HUGGINGFACE,
    source_params={'dataset_name': 'cifar10'}
)
dataset = DatasetLoader.load(config)

# Example 3: Load from Backend API
from q_store.data.backend_client import BackendAPIClient
api_client = BackendAPIClient(base_url="http://localhost:8000", api_key="your_key")
config = DatasetConfig(
    name='my_custom_dataset',
    source=DatasetSource.BACKEND_API,
    source_params={'dataset_id': 'uuid-123', 'api_client': api_client}
)
dataset = DatasetLoader.load(config)

# Example 4: Load from local numpy files
config = DatasetConfig(
    name='custom_dataset',
    source=DatasetSource.LOCAL_FILES,
    source_params={
        'format': 'numpy',
        'train_data': 'path/to/x_train.npy',
        'train_labels': 'path/to/y_train.npy',
        'test_data': 'path/to/x_test.npy',
        'test_labels': 'path/to/y_test.npy'
    }
)
dataset = DatasetLoader.load(config)

# Access data
x_train, y_train = dataset.get_split('train')
print(f"Dataset: {dataset.name}")
print(f"Total samples: {dataset.num_samples}")
print(f"Classes: {dataset.num_classes}")
print(f"Input shape: {dataset.input_shape}")
```

#### 1.2 Quantum Data Adapters (`adapters.py`)

**Purpose**: Convert classical data to quantum-compatible format

**Classes:**
```python
class QuantumDataAdapter:
    """Base adapter for quantum data preparation."""
    @staticmethod
    def prepare_for_quantum(
        data: np.ndarray,
        target_dim: int,
        encoding: str = 'amplitude'  # 'amplitude', 'angle', 'basis'
    ) -> np.ndarray

class DimensionReducer:
    """Reduce data dimensions for quantum encoding."""
    @staticmethod
    def pca(data: np.ndarray, n_components: int) -> np.ndarray
    @staticmethod
    def autoencoder(data: np.ndarray, encoding_dim: int) -> np.ndarray
    @staticmethod
    def pooling(data: np.ndarray, pool_size: Tuple[int, int]) -> np.ndarray

class QuantumImageAdapter:
    """Adapt images for quantum processing."""
    @staticmethod
    def flatten_and_normalize(images: np.ndarray) -> np.ndarray
    @staticmethod
    def resize_for_qubits(images: np.ndarray, n_qubits: int) -> np.ndarray
    @staticmethod
    def extract_features(images: np.ndarray, method: str) -> np.ndarray
```

#### 1.3 Preprocessing Utilities (`preprocessing.py`)

**Purpose**: Data preprocessing for quantum ML

**Classes:**
```python
class QuantumPreprocessor:
    """Preprocessing utilities for quantum ML."""

    @staticmethod
    def normalize(
        data: np.ndarray,
        method: str = 'minmax',  # 'minmax', 'zscore', 'l2'
        feature_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray

    @staticmethod
    def reduce_dimensions(
        data: np.ndarray,
        target_dim: int,
        method: str = 'pca'  # 'pca', 'autoencoder', 'pool'
    ) -> np.ndarray

    @staticmethod
    def validate_for_quantum(
        data: np.ndarray,
        n_qubits: int,
        encoding: str = 'amplitude'
    ) -> bool

class DataSplitter:
    """Split datasets for training."""

    @staticmethod
    def train_val_test_split(
        x: np.ndarray,
        y: np.ndarray,
        split_ratio: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15},
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]

    @staticmethod
    def k_fold_split(
        x: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]
```

#### 1.4 Data Augmentation (`augmentation.py`)

**Purpose**: Data augmentation for quantum ML

**Classes:**
```python
class QuantumAugmentation:
    """Quantum-specific data augmentation."""

    @staticmethod
    def phase_shift(data: np.ndarray, shift_range: float = 0.1) -> np.ndarray

    @staticmethod
    def amplitude_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray

    @staticmethod
    def random_rotation(data: np.ndarray, angle_range: float = 0.1) -> np.ndarray

class ClassicalAugmentation:
    """Classical data augmentation (wraps albumentations)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize from albumentations config."""
        pass

    def apply(self, images: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        pass

class HybridAugmentation:
    """Combined classical + quantum augmentation."""

    def __init__(
        self,
        classical_config: Optional[Dict] = None,
        quantum_config: Optional[Dict] = None
    ):
        pass

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply hybrid augmentation."""
        pass
```

#### 1.5 Data Generators (`generators.py`)

**Purpose**: Efficient data generation for training

**Classes:**
```python
class QuantumDataGenerator:
    """Data generator for quantum ML training."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augmentation: Optional[Any] = None,
        preprocessing: Optional[Callable] = None
    ):
        pass

    def __iter__(self):
        """Iterate over batches."""
        pass

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch."""
        pass

class StreamingDataGenerator:
    """Streaming generator for large datasets."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        chunk_size: int = 10000
    ):
        pass
```

#### 1.6 Data Validation (`validation.py`)

**Purpose**: Data quality checks and validation

**Classes:**
```python
class DataValidator:
    """Validate data for quantum ML."""

    @staticmethod
    def check_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool

    @staticmethod
    def check_range(data: np.ndarray, min_val: float, max_val: float) -> bool

    @staticmethod
    def check_nan(data: np.ndarray) -> bool

    @staticmethod
    def check_quantum_compatibility(
        data: np.ndarray,
        n_qubits: int,
        encoding: str
    ) -> Tuple[bool, str]

class DataProfiler:
    """Profile dataset characteristics."""

    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict[str, Any]:
        """Mean, std, min, max, quartiles."""
        pass

    @staticmethod
    def compute_class_distribution(labels: np.ndarray) -> Dict[int, int]:
        """Class distribution."""
        pass

    @staticmethod
    def detect_outliers(data: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Outlier detection."""
        pass
```

#### 1.7 Backend API Client (`backend_client.py`)

**Purpose**: Client for Q-Store Backend API dataset access

**Classes:**
```python
class BackendAPIClient:
    """Client for Q-Store Backend API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        pass

    def list_datasets(
        self,
        project_id: Optional[str] = None,
        dataset_source: Optional[str] = None,
        status: str = 'ready'
    ) -> List[Dict]:
        """List available datasets."""
        pass

    def get_dataset(self, dataset_id: str) -> Dict:
        """Get dataset metadata."""
        pass

    def load_dataset_data(
        self,
        dataset_id: str,
        split: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset data from backend."""
        pass

    def import_huggingface_dataset(
        self,
        dataset_name: str,
        project_id: Optional[str] = None
    ) -> str:
        """Import HuggingFace dataset via backend."""
        pass
```

---

### 2. ML Training Enhancements (`q_store/ml/`)

#### 2.1 Learning Rate Schedulers (`schedulers.py`)

**Purpose**: Advanced learning rate scheduling

**Classes:**
```python
class LRScheduler:
    """Base learning rate scheduler."""

    def step(self, epoch: int) -> float:
        """Get learning rate for epoch."""
        pass

class StepLR(LRScheduler):
    """Step decay scheduler."""
    pass

class ExponentialLR(LRScheduler):
    """Exponential decay scheduler."""
    pass

class CosineAnnealingLR(LRScheduler):
    """Cosine annealing scheduler."""

    def __init__(
        self,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0
    ):
        pass

class CyclicLR(LRScheduler):
    """Cyclic learning rate."""

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int
    ):
        pass

class OneCycleLR(LRScheduler):
    """One cycle learning rate."""
    pass

class ReduceLROnPlateau(LRScheduler):
    """Reduce LR when metric plateaus."""

    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7
    ):
        pass

    def step(self, metric: float) -> float:
        """Update based on metric."""
        pass
```

#### 2.2 Early Stopping (`early_stopping.py`)

**Purpose**: Stop training when convergence is reached

**Classes:**
```python
class EarlyStopping:
    """Early stopping with convergence detection."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',  # 'min' or 'max'
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def check(
        self,
        current_value: float,
        current_weights: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if should stop training.
        Returns True if should stop.
        """
        pass

    def get_best_weights(self) -> Optional[np.ndarray]:
        """Get best weights if restore_best_weights=True."""
        pass

class ConvergenceDetector:
    """Detect training convergence."""

    @staticmethod
    def detect_plateau(
        history: List[float],
        window_size: int = 10,
        threshold: float = 1e-4
    ) -> bool:
        """Detect if metric has plateaued."""
        pass

    @staticmethod
    def detect_divergence(
        history: List[float],
        threshold: float = 1.5
    ) -> bool:
        """Detect if training is diverging."""
        pass
```

#### 2.3 Training Callbacks (`callbacks.py`)

**Purpose**: Extensible callback system for training

**Classes:**
```python
class Callback:
    """Base callback class."""

    def on_train_begin(self, logs: Optional[Dict] = None):
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        pass

class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min'
    ):
        pass

class TensorBoardCallback(Callback):
    """Log to TensorBoard."""
    pass

class MLflowCallback(Callback):
    """Log to MLflow."""
    pass

class ProgressBarCallback(Callback):
    """Display training progress."""
    pass

class MetricsLoggerCallback(Callback):
    """Log metrics to file."""
    pass
```

#### 2.4 Enhanced QuantumTrainer (`quantum_trainer.py` - MODIFICATIONS)

**Enhancements:**
```python
class QuantumTrainer:
    """Enhanced quantum trainer with new features."""

    def __init__(
        self,
        # Existing parameters...

        # NEW: Data loading
        dataset_id: Optional[str] = None,
        backend_client: Optional[BackendAPIClient] = None,
        data_generator: Optional[QuantumDataGenerator] = None,

        # NEW: Training enhancements
        lr_scheduler: Optional[LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        callbacks: Optional[List[Callback]] = None,

        # NEW: Experiment tracking
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """Initialize enhanced quantum trainer."""
        pass

    def train(
        self,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Enhanced training with:
        - Data loading from backend if dataset_id provided
        - Learning rate scheduling
        - Early stopping
        - Callback execution
        - MLflow tracking
        """
        pass
```

---

### 3. Experiment Tracking (`q_store/tracking/`)

#### 3.1 MLflow Integration (`mlflow_tracker.py`)

**Purpose**: Track experiments with MLflow

**Classes:**
```python
class MLflowTracker:
    """MLflow experiment tracking."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "quantum-ml",
        run_name: Optional[str] = None
    ):
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        import mlflow
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model"):
        """Log model."""
        import mlflow
        mlflow.log_artifact(artifact_path)

    def end_run(self):
        """End tracking run."""
        import mlflow
        mlflow.end_run()

class WeightsAndBiasesTracker:
    """W&B experiment tracking (alternative to MLflow)."""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        import wandb
        wandb.init(project=project, entity=entity, name=name, config=config)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        import wandb
        wandb.log(metrics, step=step)

    def finish(self):
        import wandb
        wandb.finish()
```

#### 3.2 Structured Logging (`logger.py`)

**Purpose**: Enhanced logging for debugging and monitoring

**Classes:**
```python
class QuantumMLLogger:
    """Structured logger for quantum ML."""

    def __init__(
        self,
        name: str = "q_store",
        level: str = "INFO",
        log_file: Optional[str] = None
    ):
        pass

    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with config."""
        pass

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        pass

    def log_quantum_metrics(self, metrics: Dict[str, Any]):
        """Log quantum-specific metrics."""
        pass

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context."""
        pass
```

---

### 4. Hyperparameter Tuning (`q_store/tuning/`)

#### 4.1 Bayesian Optimization (`bayesian_optimizer.py`)

**Purpose**: Bayesian hyperparameter optimization

**Classes:**
```python
class BayesianOptimizer:
    """Bayesian hyperparameter optimization."""

    def __init__(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_iterations: int = 50
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            objective_function: Function to optimize (takes params dict)
            search_space: Parameter search space
                Example: {
                    'learning_rate': (0.001, 0.1, 'log-uniform'),
                    'n_qubits': (4, 12, 'int'),
                    'circuit_depth': (2, 8, 'int')
                }
            n_iterations: Number of optimization iterations
        """
        pass

    def optimize(self) -> Dict[str, Any]:
        """Run optimization and return best parameters."""
        pass

class GridSearch:
    """Grid search for hyperparameter tuning."""

    def __init__(
        self,
        objective_function: Callable,
        param_grid: Dict[str, List[Any]]
    ):
        """
        Args:
            param_grid: Parameter grid
                Example: {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'n_qubits': [4, 8, 12],
                    'circuit_depth': [2, 4, 6]
                }
        """
        pass

    def search(self) -> Dict[str, Any]:
        """Run grid search and return best parameters."""
        pass

class RandomSearch:
    """Random search for hyperparameter tuning."""
    pass
```

#### 4.2 Optuna Integration (`optuna_integration.py`)

**Purpose**: Integration with Optuna framework

**Classes:**
```python
class OptunaOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        objective_function: Callable,
        n_trials: int = 100,
        direction: str = 'minimize'  # or 'maximize'
    ):
        pass

    def optimize(
        self,
        param_suggestions: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Run Optuna optimization.

        Args:
            param_suggestions: Parameter suggestion callbacks
                Example: {
                    'learning_rate': lambda trial: trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                    'n_qubits': lambda trial: trial.suggest_int('n_qubits', 4, 12)
                }
        """
        pass
```

---

## Implementation Roadmap

### Phase 1: Data Management Layer (Weeks 1-2)

**Priority: Critical**

1. **Generic Dataset Loaders** (`loaders.py`)
   - Core classes: `DatasetLoader`, `DatasetConfig`, `Dataset`, `DatasetSource` enum
   - Source adapters: `KerasSourceAdapter`, `HuggingFaceSourceAdapter`, `BackendAPISourceAdapter`, `LocalFilesSourceAdapter`
   - Adapter pattern with plugin architecture
   - Support for Keras, HuggingFace, Backend API, local files (numpy, CSV, images, HDF5, parquet)
   - Auto-registration system for adapters
   - Automatic caching and metadata preservation

2. **Data Adapters** (`adapters.py`)
   - QuantumDataAdapter, DimensionReducer
   - QuantumImageAdapter for image datasets
   - Generic adapter interfaces

3. **Preprocessing** (`preprocessing.py`)
   - QuantumPreprocessor (normalize, standardize, reduce_dimensions)
   - DataSplitter (train/val/test, k-fold)

4. **Backend API Client** (`backend_client.py`)
   - REST client for dataset endpoints
   - Authentication handling
   - Data loading from backend
   - Integration with BackendAPISourceAdapter

5. **Data Generators** (`generators.py`)
   - QuantumDataGenerator for efficient batching
   - StreamingDataGenerator for large datasets

**Deliverables:**
- Complete `q_store/data/` module with generic loader architecture
- Unit tests for all loaders and source adapters
- Integration tests with backend API
- Documentation and usage examples for each source type

### Phase 2: Training Enhancements (Week 3)

**Priority: High**

1. **Learning Rate Schedulers** (`schedulers.py`)
   - StepLR, ExponentialLR
   - CosineAnnealingLR, CyclicLR, OneCycleLR
   - ReduceLROnPlateau

2. **Early Stopping** (`early_stopping.py`)
   - EarlyStopping with patience
   - ConvergenceDetector

3. **Callbacks** (`callbacks.py`)
   - ModelCheckpoint
   - TensorBoardCallback, MLflowCallback
   - ProgressBarCallback, MetricsLoggerCallback

4. **Enhanced QuantumTrainer** (modifications)
   - Integrate LR schedulers
   - Integrate early stopping
   - Integrate callbacks
   - Support dataset loading

**Deliverables:**
- Enhanced training infrastructure
- Backward compatibility maintained
- Unit tests and integration tests
- Updated QuantumTrainer documentation

### Phase 3: Experiment Tracking (Week 3-4)

**Priority: Medium**

1. **MLflow Integration** (`mlflow_tracker.py`)
   - MLflowTracker for experiment tracking
   - Parameter, metric, and model logging

2. **Weights & Biases** (`logger.py`)
   - W&B tracker (alternative to MLflow)

3. **Structured Logging** (`logger.py`)
   - QuantumMLLogger with structured logs
   - Training lifecycle logging

**Deliverables:**
- Complete experiment tracking system
- Examples with MLflow and W&B
- Documentation for setup and usage

### Phase 4: Hyperparameter Tuning (Week 4-5)

**Priority: Medium**

1. **Bayesian Optimization** (`bayesian_optimizer.py`)
   - BayesianOptimizer implementation

2. **Grid/Random Search** (`bayesian_optimizer.py`)
   - GridSearch, RandomSearch

3. **Optuna Integration** (`optuna_integration.py`)
   - OptunaOptimizer wrapper

**Deliverables:**
- Complete hyperparameter tuning framework
- Examples for each method
- Benchmarks comparing methods

### Phase 5: Data Augmentation & Validation (Week 5-6)

**Priority: Low**

1. **Augmentation** (`augmentation.py`)
   - QuantumAugmentation (phase shift, amplitude noise)
   - ClassicalAugmentation (albumentations wrapper)
   - HybridAugmentation

2. **Validation** (`validation.py`)
   - DataValidator (shape, range, NaN checks)
   - DataProfiler (statistics, outliers)

**Deliverables:**
- Data augmentation utilities
- Data validation framework
- Unit tests and examples

### Phase 6: Documentation & Examples (Week 6-7)

**Priority: Medium**

1. **Documentation Updates**
   - API reference for new modules
   - Architecture documentation
   - Migration guide from v4.1.0 to v4.1.1

2. **Example Scripts**
   - Fashion MNIST with backend API
   - Hyperparameter tuning example
   - MLflow tracking example
   - Complete end-to-end workflow

3. **Integration Tests**
   - Backend API + Core integration
   - End-to-end training pipeline
   - Dataset loading from all sources

**Deliverables:**
- Complete documentation
- 5+ example scripts
- Comprehensive test suite

---

## File Structure (v4.1.1)

```
q-store/
├── src/q_store/
│   ├── data/                        🆕 NEW
│   │   ├── __init__.py
│   │   ├── loaders.py              🆕 Dataset loaders
│   │   ├── adapters.py             🆕 Quantum data adapters
│   │   ├── preprocessing.py        🆕 Preprocessing utilities
│   │   ├── augmentation.py         🆕 Data augmentation
│   │   ├── generators.py           🆕 Data generators
│   │   ├── validation.py           🆕 Data validation
│   │   └── backend_client.py       🆕 Backend API client
│   │
│   ├── ml/                          🔧 ENHANCED
│   │   ├── quantum_trainer.py      🔧 Enhanced with new features
│   │   ├── schedulers.py           🆕 Learning rate schedulers
│   │   ├── early_stopping.py       🆕 Early stopping
│   │   ├── callbacks.py            🆕 Training callbacks
│   │   └── [existing 16 modules]   ✅ Unchanged
│   │
│   ├── tracking/                    🆕 NEW
│   │   ├── __init__.py
│   │   ├── mlflow_tracker.py       🆕 MLflow integration
│   │   ├── logger.py               🆕 Structured logging
│   │   └── metrics_tracker.py      🆕 Enhanced metrics
│   │
│   ├── tuning/                      🆕 NEW
│   │   ├── __init__.py
│   │   ├── bayesian_optimizer.py   🆕 Bayesian optimization
│   │   ├── grid_search.py          🆕 Grid search
│   │   ├── random_search.py        🆕 Random search
│   │   └── optuna_integration.py   🆕 Optuna integration
│   │
│   └── [existing modules]           ✅ Unchanged (151 files)
│
├── examples/
│   ├── ml_frameworks/
│   │   ├── fashion_mnist_plain.py                  ✅ Existing
│   │   ├── fashion_mnist_with_backend_api.py       🆕 NEW
│   │   ├── hyperparameter_tuning_example.py        🆕 NEW
│   │   ├── mlflow_tracking_example.py              🆕 NEW
│   │   └── end_to_end_workflow.py                  🆕 NEW
│
├── docs/
│   ├── Q-STORE_V4_1_1_ARCHITECTURE_DESIGN.md       🆕 THIS FILE
│   ├── V4_1_1_MIGRATION_GUIDE.md                   🆕 NEW
│   ├── DATA_MANAGEMENT_GUIDE.md                    🆕 NEW
│   └── API_REFERENCE_V4_1_1.md                     🆕 NEW
│
└── tests/
    ├── test_data/                   🆕 NEW
    │   ├── test_loaders.py
    │   ├── test_adapters.py
    │   ├── test_preprocessing.py
    │   └── test_backend_client.py
    ├── test_ml/                     🔧 ENHANCED
    │   ├── test_schedulers.py       🆕 NEW
    │   ├── test_early_stopping.py   🆕 NEW
    │   └── test_callbacks.py        🆕 NEW
    ├── test_tracking/               🆕 NEW
    └── test_tuning/                 🆕 NEW
```

---

## Dependencies (New in v4.1.1)

```python
# requirements.txt additions

# Data management
requests>=2.31.0          # Backend API client
pillow>=10.2.0            # Image processing (already in backend)

# Experiment tracking
mlflow>=2.9.0             # MLflow tracking
wandb>=0.16.0             # Weights & Biases (optional)

# Hyperparameter tuning
scikit-optimize>=0.9.0    # Bayesian optimization
optuna>=3.5.0             # Optuna framework

# Data validation
pandas>=2.1.0             # Data profiling
```

---

## Backward Compatibility

### Guaranteed Compatibility

1. **Existing Code**: All existing v4.1.0 code will continue to work
2. **QuantumTrainer**: New parameters are optional, defaults maintain v4.1.0 behavior
3. **Module Structure**: No breaking changes to existing modules
4. **API**: All existing APIs preserved

### Deprecation Policy

- No deprecations in v4.1.1
- New features are purely additive
- Migration guide provided for adopting new features

---

## Success Metrics

### Technical Metrics

1. **Data Loading**
   - ✅ Support 5+ dataset formats (Fashion MNIST, MNIST, CIFAR-10, CIFAR-100, custom)
   - ✅ Load from 3+ sources (Keras, HuggingFace, Backend API)
   - ✅ <5 lines of code to load and prepare data

2. **Training Enhancements**
   - ✅ 10+ learning rate schedulers
   - ✅ Early stopping reduces training time by 20-40%
   - ✅ Callbacks provide extensible training lifecycle hooks

3. **Experiment Tracking**
   - ✅ MLflow integration tracks all experiments
   - ✅ Automatic parameter and metric logging
   - ✅ <10 lines of code to add tracking to existing code

4. **Hyperparameter Tuning**
   - ✅ 3+ optimization methods (Bayesian, Grid, Random)
   - ✅ Optuna integration for advanced tuning
   - ✅ 50-70% improvement in finding optimal hyperparameters

### User Experience Metrics

1. **Ease of Use**
   - ✅ Fashion MNIST training in <20 lines of code (including data loading)
   - ✅ Backend API integration seamless
   - ✅ Clear error messages and validation

2. **Documentation**
   - ✅ Complete API reference
   - ✅ 5+ end-to-end examples
   - ✅ Migration guide from v4.1.0

3. **Testing**
   - ✅ 95%+ code coverage
   - ✅ Integration tests with backend API
   - ✅ Performance benchmarks

---

## Conclusion

Q-Store v4.1.1 bridges the critical gap between the Backend API (which manages datasets) and the Core library (which performs quantum ML training). By adding a comprehensive data management layer, training enhancements, experiment tracking, and hyperparameter tuning, v4.1.1 provides a **complete end-to-end quantum ML platform**.

**Key Achievements:**
- ✅ Seamless Backend API integration
- ✅ Production-ready data management
- ✅ Advanced training capabilities
- ✅ Enterprise experiment tracking
- ✅ Automated hyperparameter optimization
- ✅ Backward compatible with v4.1.0

**Next Steps:**
1. Review and approve this architecture design
2. Begin Phase 1 implementation (Data Management Layer)
3. Iterative development with continuous testing
4. Documentation and example creation
5. Release Q-Store v4.1.1

---

**Document Version**: 1.0
**Status**: Awaiting Review & Approval
**Target Release**: Q1 2026
