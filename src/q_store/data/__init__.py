"""
Q-Store Data Management Layer.

This module provides a unified, extensible dataset loading system with support for:
- Keras datasets (built-in TensorFlow datasets)
- HuggingFace datasets (500K+ datasets from HF Hub)
- Q-Store Backend API datasets
- Local files (NumPy, CSV, images, HDF5, Parquet)

The data layer uses a plugin architecture with source adapters, allowing easy extension
to new data sources.

Key components:
- DatasetLoader: Main loader class with registry pattern
- DatasetConfig: Configuration object for dataset loading
- DatasetSource: Enum for supported source types
- Dataset: Unified container for all datasets
- Source adapters: KerasSourceAdapter, HuggingFaceSourceAdapter, BackendAPISourceAdapter, LocalFilesSourceAdapter

Example:
    >>> from q_store.data import DatasetLoader, DatasetConfig, DatasetSource
    >>>
    >>> # Load Fashion MNIST from Keras
    >>> config = DatasetConfig(
    ...     name='fashion_mnist',
    ...     source=DatasetSource.KERAS,
    ...     source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
    ... )
    >>> dataset = DatasetLoader.load(config)
    >>> x_train, y_train = dataset.get_split('train')
"""

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

__all__ = [
    'DatasetLoader',
    'DatasetConfig',
    'DatasetSource',
    'Dataset',
    'SourceAdapter',
    'KerasSourceAdapter',
    'HuggingFaceSourceAdapter',
    'BackendAPISourceAdapter',
    'LocalFilesSourceAdapter',
]

__version__ = '4.1.1'
