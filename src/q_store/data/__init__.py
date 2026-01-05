"""
Q-Store Data Management Layer (v4.1.1).

This module provides a comprehensive data management system for quantum machine learning:

**Dataset Loading:**
- Keras datasets (built-in TensorFlow datasets)
- HuggingFace datasets (500K+ datasets from HF Hub)
- Q-Store Backend API datasets
- Local files (NumPy, CSV, images, HDF5, Parquet)

**Data Processing:**
- Quantum data adapters (dimension reduction, encoding preparation)
- Preprocessing utilities (normalization, standardization, splitting)
- Data validation and profiling
- Augmentation (quantum-specific, classical, hybrid)
- Efficient batch generators

**Backend Integration:**
- REST API client for Q-Store Backend
- Dataset management and retrieval

Example:
    >>> from q_store.data import DatasetLoader, DatasetConfig, DatasetSource
    >>> from q_store.data import QuantumPreprocessor, QuantumDataGenerator
    >>>
    >>> # Load and preprocess data
    >>> config = DatasetConfig(
    ...     name='fashion_mnist',
    ...     source=DatasetSource.KERAS,
    ...     source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
    ... )
    >>> dataset = DatasetLoader.load(config)
    >>>
    >>> # Preprocess for quantum ML
    >>> preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
    >>> x_processed = preprocessor.fit_transform(dataset.x_train)
    >>>
    >>> # Create data generator
    >>> generator = QuantumDataGenerator(x_processed, dataset.y_train, batch_size=32)
"""

# Dataset loaders
from .loaders import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
    Dataset,
    SourceAdapter,
    KerasSourceAdapter,
    HuggingFaceSourceAdapter,
    BackendAPISourceAdapter,
    LocalFilesSourceAdapter,
)

# Backend client
from .backend_client import (
    BackendAPIClient,
    BackendDatasetInfo,
    create_backend_client,
)

# Quantum data adapters
from .adapters import (
    QuantumDataAdapter,
    DimensionReducer,
    QuantumImageAdapter,
    EncodingType,
)

# Preprocessing
from .preprocessing import (
    QuantumPreprocessor,
    DataSplitter,
    NormalizationMethod,
)

# Data generators
from .generators import (
    QuantumDataGenerator,
    StreamingDataGenerator,
    InfiniteDataGenerator,
    BalancedBatchGenerator,
    create_data_generator,
)

# Validation and profiling
from .validation import (
    DataValidator,
    DataProfiler,
    OutlierDetector,
)

# Augmentation
from .augmentation import (
    QuantumAugmentation,
    ClassicalAugmentation,
    HybridAugmentation,
    AugmentationPipeline,
    create_augmentation,
)

__all__ = [
    # Loaders
    'DatasetLoader',
    'DatasetConfig',
    'DatasetSource',
    'Dataset',
    'SourceAdapter',
    'KerasSourceAdapter',
    'HuggingFaceSourceAdapter',
    'BackendAPISourceAdapter',
    'LocalFilesSourceAdapter',
    # Backend client
    'BackendAPIClient',
    'BackendDatasetInfo',
    'create_backend_client',
    # Adapters
    'QuantumDataAdapter',
    'DimensionReducer',
    'QuantumImageAdapter',
    'EncodingType',
    # Preprocessing
    'QuantumPreprocessor',
    'DataSplitter',
    'NormalizationMethod',
    # Generators
    'QuantumDataGenerator',
    'StreamingDataGenerator',
    'InfiniteDataGenerator',
    'BalancedBatchGenerator',
    'create_data_generator',
    # Validation
    'DataValidator',
    'DataProfiler',
    'OutlierDetector',
    # Augmentation
    'QuantumAugmentation',
    'ClassicalAugmentation',
    'HybridAugmentation',
    'AugmentationPipeline',
    'create_augmentation',
]

__version__ = '4.1.1'
