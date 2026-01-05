"""
Generic Dataset Loader with Plugin Architecture.

This module provides a unified, extensible dataset loading system with support for
multiple data sources through a plugin-based adapter pattern.

Key Components:
    - DatasetSource: Enum defining supported data sources
    - DatasetConfig: Configuration object for dataset loading
    - Dataset: Unified container for loaded datasets
    - DatasetLoader: Main loader class with adapter registry
    - SourceAdapter: Abstract base class for source adapters
    - Concrete adapters: Keras, HuggingFace, BackendAPI, LocalFiles

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
    >>> print(f"Loaded {dataset.num_samples} samples with {dataset.num_classes} classes")
"""

import os
import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# Optional imports (will be checked at runtime)
try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)


class DatasetSource(Enum):
    """Supported dataset sources."""
    KERAS = "keras"
    HUGGINGFACE = "huggingface"
    BACKEND_API = "backend_api"
    LOCAL_FILES = "local_files"


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading.

    Attributes:
        name: Dataset identifier (e.g., 'fashion_mnist', 'cifar10')
        source: Source type (Keras, HuggingFace, Backend API, local files)
        source_params: Source-specific parameters (dict)
        preprocessing: Preprocessing configuration (optional)
        split_config: Train/val/test split ratios (optional)
        cache_dir: Directory for caching downloaded datasets (optional)

    Example:
        >>> # Keras source
        >>> config = DatasetConfig(
        ...     name='fashion_mnist',
        ...     source=DatasetSource.KERAS,
        ...     source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
        ... )
        >>>
        >>> # HuggingFace source
        >>> config = DatasetConfig(
        ...     name='fashion_mnist',
        ...     source=DatasetSource.HUGGINGFACE,
        ...     source_params={'dataset_name': 'fashion_mnist'}
        ... )
        >>>
        >>> # Local files
        >>> config = DatasetConfig(
        ...     name='my_dataset',
        ...     source=DatasetSource.LOCAL_FILES,
        ...     source_params={
        ...         'format': 'numpy',
        ...         'train_data': 'x_train.npy',
        ...         'train_labels': 'y_train.npy'
        ...     }
        ... )
    """
    name: str
    source: DatasetSource
    source_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    preprocessing: Optional[Dict[str, Any]] = None
    split_config: Optional[Dict[str, float]] = None
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")

        if not isinstance(self.source, DatasetSource):
            raise ValueError(f"Invalid source type: {self.source}")

        if self.source_params is None:
            self.source_params = {}

        # Validate split_config if provided
        if self.split_config:
            total = sum(self.split_config.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class Dataset:
    """
    Unified dataset container.

    Provides consistent interface regardless of data source. Contains train/val/test
    splits and metadata.

    Attributes:
        name: Dataset name
        x_train: Training data
        y_train: Training labels
        x_val: Validation data (optional)
        y_val: Validation labels (optional)
        x_test: Test data (optional)
        y_test: Test labels (optional)
        metadata: Additional metadata (dict)

    Properties:
        num_samples: Total number of samples across all splits
        num_classes: Number of unique classes
        input_shape: Shape of a single input sample
        num_features: Number of features (flattened)
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
        """
        Initialize Dataset container.

        Args:
            name: Dataset name
            x_train: Training data array
            y_train: Training labels array
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            x_test: Test data (optional)
            y_test: Test labels (optional)
            metadata: Additional metadata
        """
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.metadata = metadata or {}

        # Validate data
        if len(x_train) != len(y_train):
            raise ValueError(
                f"Training data and labels must have same length: "
                f"{len(x_train)} vs {len(y_train)}"
            )

        if x_val is not None and y_val is not None:
            if len(x_val) != len(y_val):
                raise ValueError(
                    f"Validation data and labels must have same length: "
                    f"{len(x_val)} vs {len(y_val)}"
                )

        if x_test is not None and y_test is not None:
            if len(x_test) != len(y_test):
                raise ValueError(
                    f"Test data and labels must have same length: "
                    f"{len(x_test)} vs {len(y_test)}"
                )

    @property
    def num_samples(self) -> int:
        """Total number of samples across all splits."""
        total = len(self.x_train)
        if self.x_val is not None:
            total += len(self.x_val)
        if self.x_test is not None:
            total += len(self.x_test)
        return total

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        all_labels = self.y_train
        if self.y_val is not None:
            all_labels = np.concatenate([all_labels, self.y_val])
        if self.y_test is not None:
            all_labels = np.concatenate([all_labels, self.y_test])
        return len(np.unique(all_labels))

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of a single input sample."""
        return self.x_train.shape[1:]

    @property
    def num_features(self) -> int:
        """Number of features (flattened dimension)."""
        return int(np.prod(self.input_shape))

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get specific split (train/val/test).

        Args:
            split: Split name ('train', 'val', or 'test')

        Returns:
            Tuple of (data, labels) for the requested split

        Raises:
            ValueError: If split is invalid or not available
        """
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
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def has_split(self, split: str) -> bool:
        """Check if a split is available."""
        if split == 'train':
            return True
        elif split == 'val':
            return self.x_val is not None
        elif split == 'test':
            return self.x_test is not None
        return False

    def save(self, path: str, format: str = 'npz'):
        """
        Save dataset to file.

        Args:
            path: File path to save to
            format: File format ('npz' or 'hdf5')
        """
        if format == 'npz':
            save_dict = {
                'x_train': self.x_train,
                'y_train': self.y_train
            }
            if self.x_val is not None:
                save_dict['x_val'] = self.x_val
                save_dict['y_val'] = self.y_val
            if self.x_test is not None:
                save_dict['x_test'] = self.x_test
                save_dict['y_test'] = self.y_test

            np.savez_compressed(path, **save_dict)
            logger.info(f"Saved dataset to {path}")

        elif format == 'hdf5':
            if not H5PY_AVAILABLE:
                raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

            with h5py.File(path, 'w') as f:
                f.create_dataset('x_train', data=self.x_train, compression='gzip')
                f.create_dataset('y_train', data=self.y_train, compression='gzip')

                if self.x_val is not None:
                    f.create_dataset('x_val', data=self.x_val, compression='gzip')
                    f.create_dataset('y_val', data=self.y_val, compression='gzip')

                if self.x_test is not None:
                    f.create_dataset('x_test', data=self.x_test, compression='gzip')
                    f.create_dataset('y_test', data=self.y_test, compression='gzip')

                # Save metadata
                for key, value in self.metadata.items():
                    f.attrs[key] = value

            logger.info(f"Saved dataset to {path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'hdf5'")

    @classmethod
    def load(cls, path: str) -> 'Dataset':
        """
        Load dataset from file.

        Args:
            path: File path to load from

        Returns:
            Dataset object

        Raises:
            ValueError: If file format is not supported
        """
        path_obj = Path(path)

        if path_obj.suffix == '.npz':
            data = np.load(path)
            return cls(
                name=path_obj.stem,
                x_train=data['x_train'],
                y_train=data['y_train'],
                x_val=data.get('x_val'),
                y_val=data.get('y_val'),
                x_test=data.get('x_test'),
                y_test=data.get('y_test')
            )

        elif path_obj.suffix in ['.h5', '.hdf5']:
            if not H5PY_AVAILABLE:
                raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

            with h5py.File(path, 'r') as f:
                metadata = dict(f.attrs)
                return cls(
                    name=path_obj.stem,
                    x_train=f['x_train'][()],
                    y_train=f['y_train'][()],
                    x_val=f.get('x_val', lambda: None)[()],
                    y_val=f.get('y_val', lambda: None)[()],
                    x_test=f.get('x_test', lambda: None)[()],
                    y_test=f.get('y_test', lambda: None)[()],
                    metadata=metadata
                )
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    def __repr__(self) -> str:
        """String representation of dataset."""
        splits_info = f"train={len(self.x_train)}"
        if self.x_val is not None:
            splits_info += f", val={len(self.x_val)}"
        if self.x_test is not None:
            splits_info += f", test={len(self.x_test)}"

        return (
            f"Dataset(name='{self.name}', "
            f"samples={self.num_samples}, "
            f"classes={self.num_classes}, "
            f"input_shape={self.input_shape}, "
            f"splits=({splits_info}))"
        )


class SourceAdapter(ABC):
    """
    Abstract base class for dataset source adapters.

    Each source adapter implements loading logic for a specific data source
    (Keras, HuggingFace, Backend API, local files, etc.).
    """

    @abstractmethod
    def load(self, config: DatasetConfig, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load dataset from this source.

        Args:
            config: Dataset configuration
            cache_dir: Optional cache directory for downloaded datasets

        Returns:
            Loaded Dataset object

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        pass

    @abstractmethod
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available datasets from this source.

        Returns:
            List of dataset metadata dictionaries
        """
        pass

    def _apply_split_config(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        split_config: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply train/val/test split configuration.

        Args:
            x_data: Input data
            y_data: Labels
            split_config: Split ratios (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15})
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        if split_config is None:
            # No split, return as train only
            return x_data, y_data, None, None

        # Shuffle if requested
        if shuffle:
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(len(x_data))
            x_data = x_data[indices]
            y_data = y_data[indices]

        # Calculate split indices
        n_samples = len(x_data)
        train_ratio = split_config.get('train', 0.7)
        val_ratio = split_config.get('val', 0.0)

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # Split data
        x_train = x_data[:train_end]
        y_train = y_data[:train_end]

        x_val = x_data[train_end:val_end] if val_ratio > 0 else None
        y_val = y_data[train_end:val_end] if val_ratio > 0 else None

        x_test = x_data[val_end:] if val_end < n_samples else None
        y_test = y_data[val_end:] if val_end < n_samples else None

        return x_train, y_train, x_val, y_val, x_test, y_test


class DatasetLoader:
    """
    Main dataset loader with registry pattern.

    Manages a registry of source adapters and provides unified loading interface.

    Example:
        >>> config = DatasetConfig(
        ...     name='fashion_mnist',
        ...     source=DatasetSource.KERAS,
        ...     source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
        ... )
        >>> dataset = DatasetLoader.load(config)
    """

    # Registry of source adapters
    _adapters: Dict[DatasetSource, SourceAdapter] = {}

    @classmethod
    def register_adapter(cls, source: DatasetSource, adapter: SourceAdapter):
        """
        Register a source adapter.

        Args:
            source: Source type
            adapter: Adapter instance

        Example:
            >>> adapter = MyCustomSourceAdapter()
            >>> DatasetLoader.register_adapter(DatasetSource.CUSTOM, adapter)
        """
        cls._adapters[source] = adapter
        logger.info(f"Registered adapter for source: {source.value}")

    @classmethod
    def unregister_adapter(cls, source: DatasetSource):
        """Unregister a source adapter."""
        if source in cls._adapters:
            del cls._adapters[source]
            logger.info(f"Unregistered adapter for source: {source.value}")

    @classmethod
    def load(
        cls,
        config: DatasetConfig,
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset using configuration.

        Args:
            config: Dataset configuration
            cache_dir: Optional cache directory for downloaded datasets

        Returns:
            Loaded Dataset object

        Raises:
            ValueError: If no adapter is registered for the source
            RuntimeError: If loading fails
        """
        if config.source not in cls._adapters:
            raise ValueError(
                f"No adapter registered for source: {config.source}. "
                f"Available sources: {list(cls._adapters.keys())}"
            )

        adapter = cls._adapters[config.source]
        logger.info(f"Loading dataset '{config.name}' from source: {config.source.value}")

        try:
            dataset = adapter.load(config, cache_dir or config.cache_dir)
            logger.info(f"Successfully loaded dataset: {dataset}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Failed to load dataset from {config.source.value}: {e}") from e

    @classmethod
    def list_available_datasets(
        cls,
        source: Optional[DatasetSource] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available datasets from source(s).

        Args:
            source: Specific source to list from (optional, lists all if None)

        Returns:
            Dictionary mapping source names to lists of dataset metadata
        """
        if source is not None:
            if source not in cls._adapters:
                raise ValueError(f"No adapter registered for source: {source}")
            return {source.value: cls._adapters[source].list_datasets()}

        # List from all sources
        results = {}
        for src, adapter in cls._adapters.items():
            try:
                results[src.value] = adapter.list_datasets()
            except Exception as e:
                logger.warning(f"Failed to list datasets from {src.value}: {e}")
                results[src.value] = []

        return results

    @classmethod
    def get_registered_sources(cls) -> List[DatasetSource]:
        """Get list of registered sources."""
        return list(cls._adapters.keys())


# ============================================================================
# Keras Source Adapter
# ============================================================================

class KerasSourceAdapter(SourceAdapter):
    """
    Adapter for TensorFlow/Keras built-in datasets.

    Supports loading from tensorflow.keras.datasets including:
    - MNIST
    - Fashion MNIST
    - CIFAR-10
    - CIFAR-100
    - IMDB
    - Reuters
    - Boston Housing

    Example:
        >>> config = DatasetConfig(
        ...     name='fashion_mnist',
        ...     source=DatasetSource.KERAS,
        ...     source_params={'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'}
        ... )
        >>> dataset = DatasetLoader.load(config)
    """

    # Built-in Keras datasets registry
    SUPPORTED_DATASETS = {
        'mnist': {
            'module': 'tensorflow.keras.datasets.mnist',
            'description': 'MNIST handwritten digits (60k train, 10k test, 28x28 grayscale)',
            'num_classes': 10,
            'input_shape': (28, 28),
        },
        'fashion_mnist': {
            'module': 'tensorflow.keras.datasets.fashion_mnist',
            'description': 'Fashion MNIST clothing items (60k train, 10k test, 28x28 grayscale)',
            'num_classes': 10,
            'input_shape': (28, 28),
        },
        'cifar10': {
            'module': 'tensorflow.keras.datasets.cifar10',
            'description': 'CIFAR-10 images (50k train, 10k test, 32x32 RGB)',
            'num_classes': 10,
            'input_shape': (32, 32, 3),
        },
        'cifar100': {
            'module': 'tensorflow.keras.datasets.cifar100',
            'description': 'CIFAR-100 images (50k train, 10k test, 32x32 RGB, 100 classes)',
            'num_classes': 100,
            'input_shape': (32, 32, 3),
        },
    }

    def load(self, config: DatasetConfig, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load dataset from Keras.

        Args:
            config: Dataset configuration with 'dataset_module' in source_params
            cache_dir: Not used (Keras handles caching internally)

        Returns:
            Dataset object with train/test splits

        Raises:
            ImportError: If tensorflow is not installed
            ValueError: If dataset module is invalid
        """
        dataset_module = config.source_params.get('dataset_module')
        if not dataset_module:
            raise ValueError(
                "dataset_module required in source_params. "
                f"Example: 'tensorflow.keras.datasets.fashion_mnist'"
            )

        try:
            # Dynamic import of dataset module
            module = importlib.import_module(dataset_module)
            logger.info(f"Loaded Keras dataset module: {dataset_module}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import {dataset_module}. "
                f"Make sure TensorFlow is installed: pip install tensorflow"
            ) from e

        try:
            # Load data using Keras API
            (x_train, y_train), (x_test, y_test) = module.load_data()
            logger.info(
                f"Loaded Keras dataset: "
                f"train={x_train.shape}, test={x_test.shape}"
            )

            # Apply split configuration if provided
            if config.split_config:
                # Combine train and test for custom splitting
                x_all = np.concatenate([x_train, x_test], axis=0)
                y_all = np.concatenate([y_train, y_test], axis=0)

                x_train, y_train, x_val, y_val, x_test, y_test = \
                    self._apply_split_config(
                        x_all, y_all,
                        split_config=config.split_config,
                        shuffle=True
                    )
            else:
                # Use Keras default split (train/test, no val)
                x_val, y_val = None, None

            # Create dataset object
            dataset = Dataset(
                name=config.name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                metadata={
                    'source': 'keras',
                    'module': dataset_module,
                    'original_train_size': len(x_train) if x_val is None else len(x_all),
                }
            )

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load Keras dataset: {e}") from e

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available Keras datasets.

        Returns:
            List of dataset metadata dictionaries
        """
        return [
            {
                'name': name,
                'source': 'keras',
                **info
            }
            for name, info in self.SUPPORTED_DATASETS.items()
        ]


# ============================================================================
# HuggingFace Source Adapter
# ============================================================================

class HuggingFaceSourceAdapter(SourceAdapter):
    """
    Adapter for HuggingFace Datasets Hub.

    Provides access to 500,000+ datasets from the HuggingFace Hub including:
    - Vision: ImageNet, COCO, Fashion MNIST, CIFAR, etc.
    - NLP: GLUE, SQuAD, WikiText, etc.
    - Audio: LibriSpeech, Common Voice, etc.
    - Multimodal: Various cross-domain datasets

    Example:
        >>> config = DatasetConfig(
        ...     name='fashion_mnist',
        ...     source=DatasetSource.HUGGINGFACE,
        ...     source_params={'dataset_name': 'fashion_mnist'}
        ... )
        >>> dataset = DatasetLoader.load(config)
    """

    def load(self, config: DatasetConfig, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load dataset from HuggingFace Hub.

        Args:
            config: Dataset configuration with source_params:
                - dataset_name (required): HF dataset name (e.g., 'fashion_mnist')
                - config_name (optional): Dataset configuration name
                - split (optional): Specific split to load (default: loads all)
                - revision (optional): Dataset version/revision
                - trust_remote_code (optional): Allow remote code execution
            cache_dir: Cache directory for downloaded datasets

        Returns:
            Dataset object with loaded data

        Raises:
            ImportError: If datasets library is not installed
            ValueError: If dataset_name is missing or invalid
            RuntimeError: If loading fails
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install with: pip install datasets"
            )

        # Extract parameters
        dataset_name = config.source_params.get('dataset_name')
        if not dataset_name:
            raise ValueError(
                "dataset_name required in source_params. "
                "Example: {'dataset_name': 'fashion_mnist'}"
            )

        hf_config = config.source_params.get('config_name')
        revision = config.source_params.get('revision')
        trust_remote_code = config.source_params.get('trust_remote_code', False)

        try:
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")

            # Load dataset from HF Hub
            hf_dataset = hf_load_dataset(
                dataset_name,
                name=hf_config,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )

            logger.info(f"HuggingFace dataset loaded. Available splits: {list(hf_dataset.keys())}")

            # Convert HF dataset to numpy arrays
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._convert_hf_dataset_to_numpy(hf_dataset, config)

            # Create Dataset object
            dataset = Dataset(
                name=config.name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                metadata={
                    'source': 'huggingface',
                    'dataset_name': dataset_name,
                    'config_name': hf_config,
                    'revision': revision,
                    'splits': list(hf_dataset.keys())
                }
            )

            return dataset

        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset '{dataset_name}': {e}"
            ) from e

    def _convert_hf_dataset_to_numpy(
        self,
        hf_dataset,
        config: DatasetConfig
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert HuggingFace dataset to numpy arrays.

        Args:
            hf_dataset: HuggingFace DatasetDict
            config: Dataset configuration

        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        # Detect data and label columns
        data_col, label_col = self._detect_columns(hf_dataset)

        # Extract train split
        if 'train' in hf_dataset:
            train_split = hf_dataset['train']
            x_train_raw = train_split[data_col]
            y_train = np.array(train_split[label_col])

            # Convert data based on type
            x_train = self._convert_data_column(x_train_raw)
        else:
            raise ValueError("No 'train' split found in HuggingFace dataset")

        # Extract validation split (if exists)
        if 'validation' in hf_dataset:
            val_split = hf_dataset['validation']
            x_val_raw = val_split[data_col]
            y_val = np.array(val_split[label_col])
            x_val = self._convert_data_column(x_val_raw)
        else:
            x_val, y_val = None, None

        # Extract test split (if exists)
        if 'test' in hf_dataset:
            test_split = hf_dataset['test']
            x_test_raw = test_split[data_col]
            y_test = np.array(test_split[label_col])
            x_test = self._convert_data_column(x_test_raw)
        else:
            x_test, y_test = None, None

        # Apply custom split configuration if provided and no val/test exist
        if config.split_config and (x_val is None or x_test is None):
            # Combine all available data
            all_x = [x_train]
            all_y = [y_train]
            if x_val is not None:
                all_x.append(x_val)
                all_y.append(y_val)
            if x_test is not None:
                all_x.append(x_test)
                all_y.append(y_test)

            x_all = np.concatenate(all_x, axis=0)
            y_all = np.concatenate(all_y, axis=0)

            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(
                    x_all, y_all,
                    split_config=config.split_config,
                    shuffle=True
                )

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _detect_columns(self, hf_dataset) -> Tuple[str, str]:
        """
        Detect data and label column names in HuggingFace dataset.

        Args:
            hf_dataset: HuggingFace DatasetDict

        Returns:
            Tuple of (data_column, label_column)
        """
        # Get column names from first available split
        split_name = list(hf_dataset.keys())[0]
        columns = hf_dataset[split_name].column_names

        # Common data column names
        data_candidates = ['image', 'img', 'pixel_values', 'text', 'sentence', 'input']
        # Common label column names
        label_candidates = ['label', 'labels', 'target', 'class', 'category']

        # Find data column
        data_col = None
        for candidate in data_candidates:
            if candidate in columns:
                data_col = candidate
                break

        if data_col is None:
            # Use first non-label column
            for col in columns:
                if col not in label_candidates:
                    data_col = col
                    break

        # Find label column
        label_col = None
        for candidate in label_candidates:
            if candidate in columns:
                label_col = candidate
                break

        if label_col is None:
            # Use last column as fallback
            label_col = columns[-1]

        logger.info(f"Detected columns: data='{data_col}', label='{label_col}'")
        return data_col, label_col

    def _convert_data_column(self, data_col) -> np.ndarray:
        """
        Convert HuggingFace data column to numpy array.

        Args:
            data_col: Data column from HuggingFace dataset

        Returns:
            Numpy array
        """
        # If already numpy array
        if isinstance(data_col, np.ndarray):
            return data_col

        # If list of PIL images
        if PIL_AVAILABLE and len(data_col) > 0:
            first_item = data_col[0]
            if hasattr(first_item, 'mode'):  # PIL Image
                # Convert PIL images to numpy arrays
                images = [np.array(img) for img in data_col]
                return np.array(images)

        # If list of lists/arrays
        try:
            return np.array(data_col)
        except Exception as e:
            logger.warning(f"Failed to convert data column to numpy: {e}")
            # Return as-is and let caller handle
            return np.array(data_col, dtype=object)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List popular HuggingFace datasets.

        Note: HuggingFace has 500K+ datasets. This returns a curated list
        of popular datasets. Use the HF Hub website for full search.

        Returns:
            List of dataset metadata dictionaries
        """
        # Curated list of popular datasets
        popular_datasets = [
            {
                'name': 'fashion_mnist',
                'description': 'Fashion MNIST - 70K fashion product images',
                'tasks': ['image-classification'],
                'source': 'huggingface'
            },
            {
                'name': 'mnist',
                'description': 'MNIST - Handwritten digits dataset',
                'tasks': ['image-classification'],
                'source': 'huggingface'
            },
            {
                'name': 'cifar10',
                'description': 'CIFAR-10 - 60K 32x32 color images in 10 classes',
                'tasks': ['image-classification'],
                'source': 'huggingface'
            },
            {
                'name': 'cifar100',
                'description': 'CIFAR-100 - 60K 32x32 color images in 100 classes',
                'tasks': ['image-classification'],
                'source': 'huggingface'
            },
            {
                'name': 'imdb',
                'description': 'IMDB - Movie review sentiment dataset',
                'tasks': ['text-classification'],
                'source': 'huggingface'
            },
            {
                'name': 'glue',
                'description': 'GLUE - General Language Understanding Evaluation',
                'tasks': ['text-classification'],
                'source': 'huggingface'
            },
        ]

        return popular_datasets


class BackendAPISourceAdapter(SourceAdapter):
    """
    Adapter for Q-Store Backend API datasets.

    Enables loading datasets managed by the Q-Store Backend, including:
    - HuggingFace datasets imported via backend
    - Label Studio annotated datasets
    - Custom datasets uploaded to backend
    - Augmented datasets (via Albumentations)

    The adapter uses BackendAPIClient for REST API communication.

    Required source_params:
        - dataset_id (str): UUID of the dataset in backend
        - api_client (BackendAPIClient): Authenticated API client

    Optional source_params:
        - base_url (str): Backend API URL (if api_client not provided)
        - api_key (str): API key for authentication (if api_client not provided)
        - jwt_token (str): JWT token for authentication (if api_client not provided)

    Example:
        >>> from q_store.data.backend_client import BackendAPIClient
        >>>
        >>> # Option 1: Provide api_client
        >>> api_client = BackendAPIClient(
        ...     base_url="http://localhost:8000",
        ...     api_key="your_api_key"
        ... )
        >>> config = DatasetConfig(
        ...     name='my_dataset',
        ...     source=DatasetSource.BACKEND_API,
        ...     source_params={
        ...         'dataset_id': 'uuid-123',
        ...         'api_client': api_client
        ...     }
        ... )
        >>> dataset = DatasetLoader.load(config)
        >>>
        >>> # Option 2: Provide connection params (client created automatically)
        >>> config = DatasetConfig(
        ...     name='my_dataset',
        ...     source=DatasetSource.BACKEND_API,
        ...     source_params={
        ...         'dataset_id': 'uuid-123',
        ...         'base_url': 'http://localhost:8000',
        ...         'api_key': 'your_api_key'
        ...     }
        ... )
        >>> dataset = DatasetLoader.load(config)
    """

    def load(self, config: DatasetConfig, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load dataset from Q-Store Backend API.

        Args:
            config: Dataset configuration with backend parameters
            cache_dir: Optional cache directory (not used for backend source)

        Returns:
            Dataset object with data loaded from backend

        Raises:
            ValueError: If required parameters are missing
            ImportError: If backend_client module is not available
            RuntimeError: If backend API request fails
        """
        try:
            from .backend_client import BackendAPIClient
        except ImportError:
            raise ImportError(
                "backend_client module is required for BackendAPISourceAdapter. "
                "Ensure backend_client.py is available in q_store.data/"
            )

        # Get dataset_id (required)
        dataset_id = config.source_params.get('dataset_id')
        if not dataset_id:
            raise ValueError(
                "BackendAPISourceAdapter requires 'dataset_id' in source_params. "
                "Example: source_params={'dataset_id': 'uuid-123', ...}"
            )

        # Get or create API client
        api_client = config.source_params.get('api_client')
        client_created = False

        if not api_client:
            # Create client from provided parameters
            base_url = config.source_params.get('base_url', 'http://localhost:8000')
            api_key = config.source_params.get('api_key')
            jwt_token = config.source_params.get('jwt_token')

            api_client = BackendAPIClient(
                base_url=base_url,
                api_key=api_key,
                jwt_token=jwt_token
            )
            client_created = True
            logger.info(f"Created BackendAPIClient for {base_url}")

        try:
            # Get dataset metadata
            logger.info(f"Fetching dataset metadata for {dataset_id}")
            dataset_info = api_client.get_dataset(dataset_id)

            # Download data for each available split
            x_train, y_train = None, None
            x_val, y_val = None, None
            x_test, y_test = None, None

            available_splits = dataset_info.splits or ['train', 'test']

            for split in available_splits:
                try:
                    logger.info(f"Downloading {split} split for dataset {dataset_id}")
                    x_data, y_data = api_client.download_dataset_data(
                        dataset_id=dataset_id,
                        split=split,
                        format='numpy'
                    )

                    if split == 'train':
                        x_train, y_train = x_data, y_data
                    elif split == 'val' or split == 'validation':
                        x_val, y_val = x_data, y_data
                    elif split == 'test':
                        x_test, y_test = x_data, y_data

                except Exception as e:
                    logger.warning(f"Failed to download {split} split: {e}")

            # If no validation split but have train and test, optionally create validation
            if x_val is None and x_train is not None and config.split_config:
                logger.info("No validation split found, applying split_config to training data")
                # Use _apply_split_config to create train/val/test splits from train data
                x_train, y_train, x_val, y_val, x_test_new, y_test_new = self._apply_split_config(
                    x_train, y_train,
                    config.split_config,
                    dataset_name=config.name
                )
                # If we didn't have a test split from backend, use the new one
                if x_test is None:
                    x_test, y_test = x_test_new, y_test_new

            # Create metadata
            metadata = {
                'source': 'backend_api',
                'dataset_id': dataset_id,
                'backend_source': dataset_info.source,
                'description': dataset_info.description,
                'backend_metadata': dataset_info.metadata
            }

            # Create Dataset object
            dataset = Dataset(
                name=config.name or dataset_info.name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                metadata=metadata
            )

            logger.info(f"Successfully loaded dataset from backend: {dataset}")
            return dataset

        finally:
            # Clean up client if we created it
            if client_created:
                api_client.close()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available datasets from backend.

        Note: This requires backend connection parameters to be available.
        Returns empty list if no connection is configured.

        Returns:
            List of dataset metadata dictionaries
        """
        try:
            from .backend_client import BackendAPIClient

            # Try to create a client with default parameters
            # In production, this would need proper configuration
            api_client = BackendAPIClient(base_url='http://localhost:8000')

            try:
                datasets_info = api_client.list_datasets()
                datasets = []

                for ds_info in datasets_info:
                    datasets.append({
                        'name': ds_info.name,
                        'id': ds_info.id,
                        'source': ds_info.source,
                        'description': ds_info.description,
                        'num_samples': ds_info.num_samples,
                        'num_classes': ds_info.num_classes,
                        'splits': ds_info.splits
                    })

                logger.info(f"Listed {len(datasets)} datasets from backend")
                return datasets

            finally:
                api_client.close()

        except Exception as e:
            logger.warning(f"Failed to list datasets from backend: {e}")
            return []


class LocalFilesSourceAdapter(SourceAdapter):
    """
    Adapter for loading datasets from local files.

    Supports multiple file formats:
    - NumPy: .npy, .npz files
    - CSV: .csv files
    - Images: Directory of images
    - HDF5: .h5, .hdf5 files
    - Parquet: .parquet files

    Required source_params:
        - format (str): File format ('numpy', 'csv', 'images', 'hdf5', 'parquet')
        - Additional params depend on format (see format-specific methods)

    Example - NumPy:
        >>> config = DatasetConfig(
        ...     name='my_dataset',
        ...     source=DatasetSource.LOCAL_FILES,
        ...     source_params={
        ...         'format': 'numpy',
        ...         'train_data': 'path/to/x_train.npy',
        ...         'train_labels': 'path/to/y_train.npy',
        ...         'test_data': 'path/to/x_test.npy',
        ...         'test_labels': 'path/to/y_test.npy'
        ...     }
        ... )

    Example - Images:
        >>> config = DatasetConfig(
        ...     name='my_images',
        ...     source=DatasetSource.LOCAL_FILES,
        ...     source_params={
        ...         'format': 'images',
        ...         'image_dir': 'path/to/images',
        ...         'labels_file': 'path/to/labels.csv'
        ...     }
        ... )
    """

    SUPPORTED_FORMATS = ['numpy', 'csv', 'images', 'hdf5', 'parquet']

    def load(self, config: DatasetConfig, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load dataset from local files.

        Args:
            config: Dataset configuration with format and file paths
            cache_dir: Not used for local files

        Returns:
            Dataset object with loaded data

        Raises:
            ValueError: If format is invalid or required params missing
            FileNotFoundError: If specified files don't exist
            RuntimeError: If loading fails
        """
        file_format = config.source_params.get('format')
        if not file_format:
            raise ValueError(
                "LocalFilesSourceAdapter requires 'format' in source_params. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_format}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Loading dataset from local files (format: {file_format})")

        # Dispatch to format-specific loader
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
            raise ValueError(f"Format handler not implemented: {file_format}")

    def _load_numpy(self, config: DatasetConfig) -> Dataset:
        """
        Load dataset from NumPy files (.npy or .npz).

        Required source_params:
            - train_data: Path to training data file
            - train_labels: Path to training labels file

        Optional source_params:
            - val_data, val_labels: Validation split files
            - test_data, test_labels: Test split files
            - data_file: Single .npz file with all splits

        Args:
            config: Dataset configuration

        Returns:
            Dataset object

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If file format is invalid
        """
        params = config.source_params

        # Check if single .npz file provided
        data_file = params.get('data_file')
        if data_file:
            return self._load_numpy_archive(config, data_file)

        # Load from separate files
        train_data_path = params.get('train_data')
        train_labels_path = params.get('train_labels')

        if not train_data_path or not train_labels_path:
            raise ValueError(
                "NumPy format requires 'train_data' and 'train_labels' paths, "
                "or a single 'data_file' (.npz archive)"
            )

        # Load training data
        x_train = self._load_npy_file(train_data_path)
        y_train = self._load_npy_file(train_labels_path)

        # Load validation data (optional)
        val_data_path = params.get('val_data')
        val_labels_path = params.get('val_labels')
        if val_data_path and val_labels_path:
            x_val = self._load_npy_file(val_data_path)
            y_val = self._load_npy_file(val_labels_path)
        else:
            x_val, y_val = None, None

        # Load test data (optional)
        test_data_path = params.get('test_data')
        test_labels_path = params.get('test_labels')
        if test_data_path and test_labels_path:
            x_test = self._load_npy_file(test_data_path)
            y_test = self._load_npy_file(test_labels_path)
        else:
            x_test, y_test = None, None

        # Apply split config if no val/test and split_config provided
        if config.split_config and (x_val is None or x_test is None):
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(x_train, y_train, config.split_config)

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            metadata={
                'source': 'local_files',
                'format': 'numpy',
                'train_data': train_data_path,
                'train_labels': train_labels_path
            }
        )

    def _load_numpy_archive(self, config: DatasetConfig, archive_path: str) -> Dataset:
        """
        Load dataset from a single .npz archive.

        Expected keys in archive:
            - x_train, y_train (required)
            - x_val, y_val (optional)
            - x_test, y_test (optional)

        Args:
            config: Dataset configuration
            archive_path: Path to .npz file

        Returns:
            Dataset object
        """
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")

        logger.info(f"Loading NumPy archive: {archive_path}")
        data = np.load(archive_path)

        # Check required keys
        if 'x_train' not in data or 'y_train' not in data:
            raise ValueError(
                f"Archive must contain 'x_train' and 'y_train'. "
                f"Found keys: {list(data.keys())}"
            )

        return Dataset(
            name=config.name,
            x_train=data['x_train'],
            y_train=data['y_train'],
            x_val=data.get('x_val'),
            y_val=data.get('y_val'),
            x_test=data.get('x_test'),
            y_test=data.get('y_test'),
            metadata={
                'source': 'local_files',
                'format': 'numpy_archive',
                'archive_path': archive_path
            }
        )

    def _load_npy_file(self, file_path: str) -> np.ndarray:
        """Load a single .npy file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug(f"Loading: {file_path}")
        return np.load(file_path)

    def _load_csv(self, config: DatasetConfig) -> Dataset:
        """
        Load dataset from CSV files.

        Required source_params:
            - train_file: Path to training CSV file
            - label_column: Name of label column

        Optional source_params:
            - val_file: Validation CSV file
            - test_file: Test CSV file
            - feature_columns: List of feature column names (if None, uses all except label)
            - sep: CSV separator (default: ',')
            - header: Row number for header (default: 0)

        Args:
            config: Dataset configuration

        Returns:
            Dataset object
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for CSV loading. "
                "Install with: pip install pandas"
            )

        params = config.source_params
        train_file = params.get('train_file')
        label_column = params.get('label_column')

        if not train_file or not label_column:
            raise ValueError(
                "CSV format requires 'train_file' and 'label_column' in source_params"
            )

        sep = params.get('sep', ',')
        header = params.get('header', 0)
        feature_columns = params.get('feature_columns')

        # Load training data
        x_train, y_train = self._load_csv_file(
            train_file, label_column, feature_columns, sep, header
        )

        # Load validation data (optional)
        val_file = params.get('val_file')
        if val_file:
            x_val, y_val = self._load_csv_file(
                val_file, label_column, feature_columns, sep, header
            )
        else:
            x_val, y_val = None, None

        # Load test data (optional)
        test_file = params.get('test_file')
        if test_file:
            x_test, y_test = self._load_csv_file(
                test_file, label_column, feature_columns, sep, header
            )
        else:
            x_test, y_test = None, None

        # Apply split config if needed
        if config.split_config and (x_val is None or x_test is None):
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(x_train, y_train, config.split_config)

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            metadata={
                'source': 'local_files',
                'format': 'csv',
                'train_file': train_file,
                'label_column': label_column
            }
        )

    def _load_csv_file(
        self,
        file_path: str,
        label_column: str,
        feature_columns: Optional[List[str]] = None,
        sep: str = ',',
        header: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from a single CSV file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading CSV: {file_path}")
        df = pd.read_csv(file_path, sep=sep, header=header)

        # Extract labels
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        y_data = df[label_column].values

        # Extract features
        if feature_columns:
            # Use specified feature columns
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            x_data = df[feature_columns].values
        else:
            # Use all columns except label
            feature_cols = [col for col in df.columns if col != label_column]
            x_data = df[feature_cols].values

        return x_data, y_data

    def _load_images(self, config: DatasetConfig) -> Dataset:
        """
        Load dataset from image directory.

        Required source_params:
            - image_dir: Root directory containing images
            - labels_file: CSV file with image names and labels
                          (columns: 'filename', 'label')

        Optional source_params:
            - image_size: Tuple of (height, width) to resize images
            - color_mode: 'rgb' or 'grayscale' (default: 'rgb')
            - file_extension: Image extension (default: '.jpg')

        Args:
            config: Dataset configuration

        Returns:
            Dataset object
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for image loading. "
                "Install with: pip install pillow"
            )

        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for reading labels file. "
                "Install with: pip install pandas"
            )

        params = config.source_params
        image_dir = params.get('image_dir')
        labels_file = params.get('labels_file')

        if not image_dir or not labels_file:
            raise ValueError(
                "Images format requires 'image_dir' and 'labels_file' in source_params"
            )

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        # Load labels
        logger.info(f"Loading labels from: {labels_file}")
        labels_df = pd.read_csv(labels_file)

        if 'filename' not in labels_df.columns or 'label' not in labels_df.columns:
            raise ValueError(
                "Labels CSV must contain 'filename' and 'label' columns"
            )

        # Load images
        image_size = params.get('image_size')
        color_mode = params.get('color_mode', 'rgb')

        images = []
        labels = []

        logger.info(f"Loading images from: {image_dir}")
        for idx, row in labels_df.iterrows():
            filename = row['filename']
            label = row['label']

            image_path = os.path.join(image_dir, filename)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}, skipping")
                continue

            # Load and preprocess image
            img = Image.open(image_path)

            # Convert color mode
            if color_mode == 'grayscale':
                img = img.convert('L')
            elif color_mode == 'rgb':
                img = img.convert('RGB')

            # Resize if specified
            if image_size:
                img = img.resize(image_size)

            # Convert to numpy array
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)

        x_data = np.array(images)
        y_data = np.array(labels)

        logger.info(f"Loaded {len(x_data)} images")

        # Apply split configuration
        if config.split_config:
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(x_data, y_data, config.split_config)
        else:
            # Default: all data as training
            x_train, y_train = x_data, y_data
            x_val, y_val = None, None
            x_test, y_test = None, None

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            metadata={
                'source': 'local_files',
                'format': 'images',
                'image_dir': image_dir,
                'labels_file': labels_file,
                'image_size': image_size,
                'color_mode': color_mode
            }
        )

    def _load_hdf5(self, config: DatasetConfig) -> Dataset:
        """
        Load dataset from HDF5 file.

        Required source_params:
            - hdf5_file: Path to HDF5 file

        Optional source_params:
            - train_data_key, train_labels_key: Keys for training data (default: 'x_train', 'y_train')
            - val_data_key, val_labels_key: Keys for validation data
            - test_data_key, test_labels_key: Keys for test data

        Args:
            config: Dataset configuration

        Returns:
            Dataset object
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 loading. "
                "Install with: pip install h5py"
            )

        params = config.source_params
        hdf5_file = params.get('hdf5_file')

        if not hdf5_file:
            raise ValueError("HDF5 format requires 'hdf5_file' in source_params")

        if not os.path.exists(hdf5_file):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

        logger.info(f"Loading HDF5 file: {hdf5_file}")

        with h5py.File(hdf5_file, 'r') as f:
            # Get dataset keys
            train_data_key = params.get('train_data_key', 'x_train')
            train_labels_key = params.get('train_labels_key', 'y_train')

            if train_data_key not in f or train_labels_key not in f:
                raise ValueError(
                    f"Training data keys not found in HDF5 file. "
                    f"Expected: '{train_data_key}', '{train_labels_key}'. "
                    f"Available keys: {list(f.keys())}"
                )

            # Load training data
            x_train = f[train_data_key][()]
            y_train = f[train_labels_key][()]

            # Load validation data (optional)
            val_data_key = params.get('val_data_key', 'x_val')
            val_labels_key = params.get('val_labels_key', 'y_val')
            if val_data_key in f and val_labels_key in f:
                x_val = f[val_data_key][()]
                y_val = f[val_labels_key][()]
            else:
                x_val, y_val = None, None

            # Load test data (optional)
            test_data_key = params.get('test_data_key', 'x_test')
            test_labels_key = params.get('test_labels_key', 'y_test')
            if test_data_key in f and test_labels_key in f:
                x_test = f[test_data_key][()]
                y_test = f[test_labels_key][()]
            else:
                x_test, y_test = None, None

            # Load metadata if available
            metadata = dict(f.attrs) if f.attrs else {}

        # Apply split config if needed
        if config.split_config and (x_val is None or x_test is None):
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(x_train, y_train, config.split_config)

        metadata.update({
            'source': 'local_files',
            'format': 'hdf5',
            'hdf5_file': hdf5_file
        })

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            metadata=metadata
        )

    def _load_parquet(self, config: DatasetConfig) -> Dataset:
        """
        Load dataset from Parquet files.

        Required source_params:
            - train_file: Path to training Parquet file
            - label_column: Name of label column

        Optional source_params:
            - val_file: Validation Parquet file
            - test_file: Test Parquet file
            - feature_columns: List of feature column names

        Args:
            config: Dataset configuration

        Returns:
            Dataset object
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "pyarrow is required for Parquet loading. "
                "Install with: pip install pyarrow"
            )

        params = config.source_params
        train_file = params.get('train_file')
        label_column = params.get('label_column')

        if not train_file or not label_column:
            raise ValueError(
                "Parquet format requires 'train_file' and 'label_column' in source_params"
            )

        feature_columns = params.get('feature_columns')

        # Load training data
        x_train, y_train = self._load_parquet_file(
            train_file, label_column, feature_columns
        )

        # Load validation data (optional)
        val_file = params.get('val_file')
        if val_file:
            x_val, y_val = self._load_parquet_file(
                val_file, label_column, feature_columns
            )
        else:
            x_val, y_val = None, None

        # Load test data (optional)
        test_file = params.get('test_file')
        if test_file:
            x_test, y_test = self._load_parquet_file(
                test_file, label_column, feature_columns
            )
        else:
            x_test, y_test = None, None

        # Apply split config if needed
        if config.split_config and (x_val is None or x_test is None):
            x_train, y_train, x_val, y_val, x_test, y_test = \
                self._apply_split_config(x_train, y_train, config.split_config)

        return Dataset(
            name=config.name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            metadata={
                'source': 'local_files',
                'format': 'parquet',
                'train_file': train_file,
                'label_column': label_column
            }
        )

    def _load_parquet_file(
        self,
        file_path: str,
        label_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from a single Parquet file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logger.info(f"Loading Parquet: {file_path}")
        table = pq.read_table(file_path)
        df = table.to_pandas()

        # Extract labels
        if label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in Parquet file. "
                f"Available columns: {list(df.columns)}"
            )
        y_data = df[label_column].values

        # Extract features
        if feature_columns:
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            x_data = df[feature_columns].values
        else:
            feature_cols = [col for col in df.columns if col != label_column]
            x_data = df[feature_cols].values

        return x_data, y_data

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available datasets from local files.

        Note: Since local files can be anywhere, this returns an empty list.
        Users should provide explicit file paths in source_params.

        Returns:
            Empty list
        """
        return []


# Auto-register default adapters
def _register_default_adapters():
    """Register default source adapters."""
    DatasetLoader.register_adapter(DatasetSource.KERAS, KerasSourceAdapter())
    DatasetLoader.register_adapter(DatasetSource.HUGGINGFACE, HuggingFaceSourceAdapter())
    DatasetLoader.register_adapter(DatasetSource.BACKEND_API, BackendAPISourceAdapter())
    DatasetLoader.register_adapter(DatasetSource.LOCAL_FILES, LocalFilesSourceAdapter())


# Auto-register on module import
_register_default_adapters()
