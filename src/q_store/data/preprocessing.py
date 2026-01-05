"""
Data Preprocessing Utilities for Quantum ML.

This module provides preprocessing utilities optimized for quantum machine learning,
including normalization, standardization, dimension reduction, and data splitting.

Key Components:
    - QuantumPreprocessor: Comprehensive preprocessing for quantum ML
    - DataSplitter: Train/val/test splitting with various strategies
    - NormalizationMethod: Enum for normalization methods

Example:
    >>> from q_store.data.preprocessing import QuantumPreprocessor, DataSplitter
    >>>
    >>> # Preprocess data
    >>> preprocessor = QuantumPreprocessor(method='minmax', feature_range=(0, 1))
    >>> x_processed = preprocessor.fit_transform(x_data)
    >>>
    >>> # Split data
    >>> splitter = DataSplitter(split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15})
    >>> splits = splitter.split(x_data, y_data)
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Supported normalization methods."""
    MINMAX = "minmax"  # Min-max normalization to [min, max]
    ZSCORE = "zscore"  # Z-score standardization (mean=0, std=1)
    L1 = "l1"  # L1 normalization (sum of absolute values = 1)
    L2 = "l2"  # L2 normalization (Euclidean norm = 1)
    ROBUST = "robust"  # Robust scaling using median and IQR


class QuantumPreprocessor:
    """
    Comprehensive preprocessing for quantum machine learning.

    Provides normalization, standardization, and validation optimized
    for quantum computing requirements.

    Args:
        method: Normalization method ('minmax', 'zscore', 'l1', 'l2', 'robust')
        feature_range: Target range for minmax normalization (default: (0, 1))
        clip_outliers: Whether to clip outliers (default: False)
        outlier_std: Number of std devs for outlier clipping (default: 3.0)

    Example:
        >>> preprocessor = QuantumPreprocessor(method='minmax')
        >>> x_normalized = preprocessor.fit_transform(x_data)
        >>> print(f"Normalized to range [{x_normalized.min()}, {x_normalized.max()}]")
    """

    def __init__(
        self,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1),
        clip_outliers: bool = False,
        outlier_std: float = 3.0
    ):
        """
        Initialize quantum preprocessor.

        Args:
            method: Normalization method
            feature_range: Target range for minmax
            clip_outliers: Whether to clip outliers
            outlier_std: Number of std devs for clipping
        """
        self.method = NormalizationMethod(method) if isinstance(method, str) else method
        self.feature_range = feature_range
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std

        # Fitted parameters
        self._fitted = False
        self._params = {}

        logger.info(
            f"Initialized QuantumPreprocessor: {self.method.value}, "
            f"range={feature_range}, clip_outliers={clip_outliers}"
        )

    def fit(self, data: np.ndarray) -> 'QuantumPreprocessor':
        """
        Fit preprocessor to data.

        Args:
            data: Training data (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        if self.method == NormalizationMethod.MINMAX:
            self._params['min'] = data.min(axis=0)
            self._params['max'] = data.max(axis=0)
            self._params['range'] = self._params['max'] - self._params['min']
            # Avoid division by zero
            self._params['range'] = np.where(
                self._params['range'] == 0,
                1.0,
                self._params['range']
            )

        elif self.method == NormalizationMethod.ZSCORE:
            self._params['mean'] = data.mean(axis=0)
            self._params['std'] = data.std(axis=0)
            # Avoid division by zero
            self._params['std'] = np.where(
                self._params['std'] == 0,
                1.0,
                self._params['std']
            )

        elif self.method == NormalizationMethod.ROBUST:
            self._params['median'] = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            self._params['iqr'] = q75 - q25
            # Avoid division by zero
            self._params['iqr'] = np.where(
                self._params['iqr'] == 0,
                1.0,
                self._params['iqr']
            )

        # L1 and L2 don't need fitting (per-sample normalization)

        self._fitted = True
        logger.info(f"Fitted preprocessor on data with shape {data.shape}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.

        Args:
            data: Data to transform (n_samples, n_features)

        Returns:
            Transformed data
        """
        if not self._fitted and self.method in [
            NormalizationMethod.MINMAX,
            NormalizationMethod.ZSCORE,
            NormalizationMethod.ROBUST
        ]:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Clip outliers if requested
        if self.clip_outliers:
            data = self._clip_outliers(data)

        # Apply normalization
        if self.method == NormalizationMethod.MINMAX:
            return self._normalize_minmax(data)
        elif self.method == NormalizationMethod.ZSCORE:
            return self._normalize_zscore(data)
        elif self.method == NormalizationMethod.L1:
            return self._normalize_l1(data)
        elif self.method == NormalizationMethod.L2:
            return self._normalize_l2(data)
        elif self.method == NormalizationMethod.ROBUST:
            return self._normalize_robust(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalization to feature_range."""
        min_val, max_val = self.feature_range
        normalized = (data - self._params['min']) / self._params['range']
        return normalized * (max_val - min_val) + min_val

    def _normalize_zscore(self, data: np.ndarray) -> np.ndarray:
        """Z-score standardization (mean=0, std=1)."""
        return (data - self._params['mean']) / self._params['std']

    def _normalize_l1(self, data: np.ndarray) -> np.ndarray:
        """L1 normalization (sum of absolute values = 1)."""
        norms = np.abs(data).sum(axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return data / norms

    def _normalize_l2(self, data: np.ndarray) -> np.ndarray:
        """L2 normalization (Euclidean norm = 1)."""
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return data / norms

    def _normalize_robust(self, data: np.ndarray) -> np.ndarray:
        """Robust scaling using median and IQR."""
        return (data - self._params['median']) / self._params['iqr']

    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """Clip outliers beyond outlier_std standard deviations."""
        if self.method == NormalizationMethod.ZSCORE:
            # Clip based on std from mean
            mean = self._params.get('mean', data.mean(axis=0))
            std = self._params.get('std', data.std(axis=0))
            lower = mean - self.outlier_std * std
            upper = mean + self.outlier_std * std
        else:
            # Clip based on percentiles
            lower = np.percentile(data, 1, axis=0)
            upper = np.percentile(data, 99, axis=0)

        return np.clip(data, lower, upper)

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """
        Convenience method for z-score standardization.

        Args:
            data: Data to standardize

        Returns:
            Standardized data (mean=0, std=1)
        """
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std == 0, 1, std)
        return (data - mean) / std

    @staticmethod
    def normalize(
        data: np.ndarray,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Convenience method for normalization.

        Args:
            data: Data to normalize
            method: Normalization method
            feature_range: Target range for minmax

        Returns:
            Normalized data
        """
        preprocessor = QuantumPreprocessor(method=method, feature_range=feature_range)
        return preprocessor.fit_transform(data)

    @staticmethod
    def reduce_dimensions(
        data: np.ndarray,
        target_dim: int,
        method: str = 'pca'
    ) -> np.ndarray:
        """
        Convenience method for dimension reduction.

        Args:
            data: Data to reduce
            target_dim: Target number of dimensions
            method: Reduction method ('pca', 'random')

        Returns:
            Reduced data
        """
        from .adapters import DimensionReducer

        reducer = DimensionReducer(method=method, target_dim=target_dim)
        return reducer.fit_transform(data)

    @staticmethod
    def validate_for_quantum(
        data: np.ndarray,
        n_qubits: int,
        encoding: str = 'amplitude'
    ) -> Tuple[bool, str]:
        """
        Validate data for quantum compatibility.

        Args:
            data: Data to validate
            n_qubits: Number of qubits
            encoding: Encoding type

        Returns:
            Tuple of (is_valid, message)
        """
        from .adapters import QuantumDataAdapter

        adapter = QuantumDataAdapter(n_qubits=n_qubits, encoding=encoding)
        return adapter.validate(data)


class DataSplitter:
    """
    Split datasets for training with various strategies.

    Supports:
    - Train/val/test splits
    - K-fold cross-validation
    - Stratified splitting
    - Time-series splitting

    Args:
        split_ratio: Dictionary of split ratios (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15})
        shuffle: Whether to shuffle before splitting (default: True)
        random_seed: Random seed for reproducibility (default: 42)
        stratify: Whether to maintain class distribution (default: False)

    Example:
        >>> splitter = DataSplitter(split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15})
        >>> splits = splitter.split(x_data, y_data)
        >>> x_train, y_train = splits['train']
        >>> x_val, y_val = splits['val']
        >>> x_test, y_test = splits['test']
    """

    def __init__(
        self,
        split_ratio: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
        random_seed: int = 42,
        stratify: bool = False
    ):
        """
        Initialize data splitter.

        Args:
            split_ratio: Split ratios dictionary
            shuffle: Whether to shuffle
            random_seed: Random seed
            stratify: Whether to stratify
        """
        self.split_ratio = split_ratio or {'train': 0.7, 'val': 0.15, 'test': 0.15}
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.stratify = stratify

        # Validate split ratios
        total = sum(self.split_ratio.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        logger.info(
            f"Initialized DataSplitter: {self.split_ratio}, "
            f"shuffle={shuffle}, stratify={stratify}"
        )

    def split(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets.

        Args:
            x_data: Input data
            y_data: Labels

        Returns:
            Dictionary mapping split names to (x, y) tuples
        """
        n_samples = len(x_data)

        if len(y_data) != n_samples:
            raise ValueError(
                f"x_data and y_data must have same length: "
                f"{n_samples} vs {len(y_data)}"
            )

        # Create indices
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_seed)
            if self.stratify:
                # Stratified shuffle
                indices = self._stratified_shuffle(y_data, rng)
            else:
                rng.shuffle(indices)

        # Calculate split points
        splits = {}
        current_idx = 0

        for split_name in ['train', 'val', 'test']:
            if split_name not in self.split_ratio:
                continue

            ratio = self.split_ratio[split_name]
            split_size = int(n_samples * ratio)

            # Handle last split (take remaining samples)
            if split_name == list(self.split_ratio.keys())[-1]:
                split_size = n_samples - current_idx

            # Extract split
            split_indices = indices[current_idx:current_idx + split_size]
            splits[split_name] = (
                x_data[split_indices],
                y_data[split_indices]
            )

            current_idx += split_size

        logger.info(
            f"Split {n_samples} samples into: " +
            ", ".join([f"{k}={len(v[0])}" for k, v in splits.items()])
        )

        return splits

    def _stratified_shuffle(
        self,
        y_data: np.ndarray,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Shuffle indices while maintaining class distribution."""
        indices = []
        unique_classes = np.unique(y_data)

        for cls in unique_classes:
            cls_indices = np.where(y_data == cls)[0]
            rng.shuffle(cls_indices)
            indices.append(cls_indices)

        # Interleave class indices
        all_indices = np.concatenate(indices)
        return all_indices

    @staticmethod
    def train_val_test_split(
        x_data: np.ndarray,
        y_data: np.ndarray,
        split_ratio: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Convenience method for train/val/test split.

        Args:
            x_data: Input data
            y_data: Labels
            split_ratio: Split ratios
            shuffle: Whether to shuffle
            random_seed: Random seed

        Returns:
            Dictionary with train/val/test splits
        """
        splitter = DataSplitter(
            split_ratio=split_ratio,
            shuffle=shuffle,
            random_seed=random_seed
        )
        return splitter.split(x_data, y_data)

    @staticmethod
    def k_fold_split(
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_splits: int = 5,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        K-fold cross-validation split.

        Args:
            x_data: Input data
            y_data: Labels
            n_splits: Number of folds
            shuffle: Whether to shuffle
            random_seed: Random seed

        Returns:
            List of (train, val) tuples for each fold
        """
        n_samples = len(x_data)
        indices = np.arange(n_samples)

        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(indices)

        fold_size = n_samples // n_splits
        folds = []

        for i in range(n_splits):
            # Validation indices for this fold
            val_start = i * fold_size
            val_end = val_start + fold_size if i < n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]

            # Training indices (everything else)
            train_indices = np.concatenate([
                indices[:val_start],
                indices[val_end:]
            ])

            # Create fold
            fold = (
                (x_data[train_indices], y_data[train_indices]),  # train
                (x_data[val_indices], y_data[val_indices])  # val
            )
            folds.append(fold)

        logger.info(
            f"Created {n_splits}-fold split with ~{fold_size} samples per fold"
        )
        return folds

    @staticmethod
    def time_series_split(
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Time-series cross-validation split.

        Creates expanding training sets with fixed validation size.

        Args:
            x_data: Input data (ordered by time)
            y_data: Labels
            n_splits: Number of splits

        Returns:
            List of (train, val) tuples
        """
        n_samples = len(x_data)
        val_size = n_samples // (n_splits + 1)

        splits = []
        for i in range(1, n_splits + 1):
            train_end = val_size * i
            val_end = train_end + val_size

            if val_end > n_samples:
                break

            splits.append((
                (x_data[:train_end], y_data[:train_end]),  # train
                (x_data[train_end:val_end], y_data[train_end:val_end])  # val
            ))

        logger.info(
            f"Created {len(splits)} time-series splits with validation size {val_size}"
        )
        return splits
