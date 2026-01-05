"""
Quantum Data Adapters.

This module provides adapters for converting classical data to quantum-compatible formats,
including dimension reduction, feature extraction, and encoding transformations.

Key Components:
    - QuantumDataAdapter: Base adapter for quantum data preparation
    - DimensionReducer: Reduce data dimensions for quantum encoding
    - QuantumImageAdapter: Adapt images for quantum processing
    - EncodingType: Enum for different quantum encoding schemes

Example:
    >>> from q_store.data.adapters import QuantumDataAdapter, DimensionReducer
    >>>
    >>> # Reduce dimensions
    >>> reducer = DimensionReducer(method='pca', target_dim=8)
    >>> reduced_data = reducer.fit_transform(data)
    >>>
    >>> # Prepare for quantum encoding
    >>> adapter = QuantumDataAdapter(n_qubits=8, encoding='amplitude')
    >>> quantum_data = adapter.prepare(reduced_data)
"""

import logging
from enum import Enum
from typing import Optional, Tuple, Union, Callable
import numpy as np

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    """Supported quantum encoding types."""
    AMPLITUDE = "amplitude"  # Amplitude encoding
    ANGLE = "angle"  # Angle encoding (rotation gates)
    BASIS = "basis"  # Basis encoding (computational basis states)


class QuantumDataAdapter:
    """
    Base adapter for preparing classical data for quantum processing.

    Handles:
    - Dimension validation and normalization
    - Encoding-specific transformations
    - Data validation for quantum compatibility

    Args:
        n_qubits: Number of qubits available
        encoding: Encoding type ('amplitude', 'angle', 'basis')
        normalize: Whether to normalize data (default: True)

    Example:
        >>> adapter = QuantumDataAdapter(n_qubits=8, encoding='amplitude')
        >>> quantum_data = adapter.prepare(classical_data)
        >>> print(f"Prepared {len(quantum_data)} samples for {adapter.n_qubits} qubits")
    """

    def __init__(
        self,
        n_qubits: int,
        encoding: Union[str, EncodingType] = EncodingType.AMPLITUDE,
        normalize: bool = True
    ):
        """
        Initialize quantum data adapter.

        Args:
            n_qubits: Number of qubits
            encoding: Encoding type
            normalize: Whether to normalize data
        """
        self.n_qubits = n_qubits
        self.encoding = EncodingType(encoding) if isinstance(encoding, str) else encoding
        self.normalize = normalize

        # Calculate maximum dimensions for each encoding
        if self.encoding == EncodingType.AMPLITUDE:
            self.max_features = 2 ** n_qubits
        elif self.encoding == EncodingType.ANGLE:
            self.max_features = n_qubits
        elif self.encoding == EncodingType.BASIS:
            self.max_features = n_qubits
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        logger.info(
            f"Initialized QuantumDataAdapter: {n_qubits} qubits, "
            f"{self.encoding.value} encoding, max {self.max_features} features"
        )

    def prepare(
        self,
        data: np.ndarray,
        target_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Prepare classical data for quantum encoding.

        Args:
            data: Classical data array (n_samples, n_features)
            target_dim: Target dimension (if None, uses max_features)

        Returns:
            Prepared data ready for quantum encoding

        Raises:
            ValueError: If data shape is incompatible
        """
        if data.ndim != 2:
            raise ValueError(
                f"Data must be 2D (n_samples, n_features), got shape {data.shape}"
            )

        n_samples, n_features = data.shape
        target_dim = target_dim or self.max_features

        # Check if dimension reduction needed
        if n_features > target_dim:
            logger.warning(
                f"Data has {n_features} features but target is {target_dim}. "
                f"Consider using DimensionReducer first."
            )
            # Truncate to target dimension
            data = data[:, :target_dim]
        elif n_features < target_dim:
            # Pad with zeros
            padding = np.zeros((n_samples, target_dim - n_features))
            data = np.concatenate([data, padding], axis=1)
            logger.info(f"Padded data from {n_features} to {target_dim} features")

        # Normalize if requested
        if self.normalize:
            data = self._normalize(data)

        # Apply encoding-specific transformations
        data = self._apply_encoding_transform(data)

        logger.debug(f"Prepared {n_samples} samples for quantum encoding")
        return data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data based on encoding type."""
        if self.encoding == EncodingType.AMPLITUDE:
            # L2 normalization for amplitude encoding
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            return data / norms

        elif self.encoding == EncodingType.ANGLE:
            # Scale to [0, 2π] for angle encoding
            data_min = data.min(axis=1, keepdims=True)
            data_max = data.max(axis=1, keepdims=True)
            data_range = data_max - data_min
            data_range = np.where(data_range == 0, 1, data_range)
            return 2 * np.pi * (data - data_min) / data_range

        elif self.encoding == EncodingType.BASIS:
            # Min-max normalization to [0, 1] for basis encoding
            data_min = data.min(axis=1, keepdims=True)
            data_max = data.max(axis=1, keepdims=True)
            data_range = data_max - data_min
            data_range = np.where(data_range == 0, 1, data_range)
            return (data - data_min) / data_range

        return data

    def _apply_encoding_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply encoding-specific transformations."""
        # Currently, normalization handles most encoding needs
        # Can be extended for more complex transformations
        return data

    def validate(self, data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate data for quantum compatibility.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if data.ndim != 2:
            return False, f"Data must be 2D, got {data.ndim}D"

        n_samples, n_features = data.shape

        if n_features > self.max_features:
            return False, (
                f"Data has {n_features} features but encoding supports max "
                f"{self.max_features} for {self.n_qubits} qubits"
            )

        if np.isnan(data).any():
            return False, "Data contains NaN values"

        if np.isinf(data).any():
            return False, "Data contains infinite values"

        return True, "Data is valid for quantum encoding"


class DimensionReducer:
    """
    Reduce data dimensions for quantum encoding.

    Supports multiple dimension reduction methods:
    - PCA (Principal Component Analysis)
    - Autoencoder (neural network-based)
    - Pooling (for images)
    - Random projection

    Args:
        method: Reduction method ('pca', 'autoencoder', 'pool', 'random')
        target_dim: Target number of dimensions
        **kwargs: Method-specific parameters

    Example:
        >>> reducer = DimensionReducer(method='pca', target_dim=8)
        >>> reduced_data = reducer.fit_transform(data)
        >>> print(f"Reduced from {data.shape[1]} to {reduced_data.shape[1]} dimensions")
    """

    SUPPORTED_METHODS = ['pca', 'autoencoder', 'pool', 'random']

    def __init__(
        self,
        method: str = 'pca',
        target_dim: int = 8,
        **kwargs
    ):
        """
        Initialize dimension reducer.

        Args:
            method: Reduction method
            target_dim: Target dimensions
            **kwargs: Method-specific parameters
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        self.method = method
        self.target_dim = target_dim
        self.kwargs = kwargs
        self._fitted = False
        self._components = None

        logger.info(f"Initialized DimensionReducer: {method}, target_dim={target_dim}")

    def fit(self, data: np.ndarray) -> 'DimensionReducer':
        """
        Fit the dimension reducer to data.

        Args:
            data: Training data (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        n_samples, n_features = data.shape

        if n_features <= self.target_dim:
            logger.warning(
                f"Data already has {n_features} features, "
                f"less than target {self.target_dim}"
            )
            self._fitted = True
            return self

        if self.method == 'pca':
            self._fit_pca(data)
        elif self.method == 'autoencoder':
            self._fit_autoencoder(data)
        elif self.method == 'pool':
            # Pooling doesn't require fitting
            self._fitted = True
        elif self.method == 'random':
            self._fit_random_projection(data)

        self._fitted = True
        logger.info(f"Fitted {self.method} reducer on {n_samples} samples")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to reduced dimensions.

        Args:
            data: Data to transform (n_samples, n_features)

        Returns:
            Reduced data (n_samples, target_dim)
        """
        if not self._fitted:
            raise RuntimeError("Reducer must be fitted before transform")

        if self.method == 'pca':
            return self._transform_pca(data)
        elif self.method == 'autoencoder':
            return self._transform_autoencoder(data)
        elif self.method == 'pool':
            return self._transform_pool(data)
        elif self.method == 'random':
            return self._transform_random_projection(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def _fit_pca(self, data: np.ndarray):
        """Fit PCA dimension reduction."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for PCA. "
                "Install with: pip install scikit-learn"
            )

        pca = PCA(n_components=self.target_dim, **self.kwargs)
        pca.fit(data)
        self._components = pca

        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA: Reduced to {self.target_dim} components "
            f"(explained variance: {explained_var:.2%})"
        )

    def _transform_pca(self, data: np.ndarray) -> np.ndarray:
        """Transform using PCA."""
        return self._components.transform(data)

    def _fit_autoencoder(self, data: np.ndarray):
        """Fit autoencoder dimension reduction."""
        raise NotImplementedError(
            "Autoencoder reduction not yet implemented. "
            "Use 'pca' or 'random' method instead."
        )

    def _transform_autoencoder(self, data: np.ndarray) -> np.ndarray:
        """Transform using autoencoder."""
        raise NotImplementedError("Autoencoder reduction not yet implemented")

    def _transform_pool(self, data: np.ndarray) -> np.ndarray:
        """Transform using pooling (for image data)."""
        # Assume data is flattened images
        # Try to infer image shape
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # Try to reshape to square image
        img_size = int(np.sqrt(n_features))
        if img_size * img_size != n_features:
            raise ValueError(
                f"Pooling requires square images. "
                f"Got {n_features} features (not a perfect square)"
            )

        images = data.reshape(n_samples, img_size, img_size)

        # Calculate pooling size
        pool_size = img_size // int(np.sqrt(self.target_dim))
        if pool_size < 1:
            pool_size = 1

        # Apply average pooling
        pooled = self._average_pool(images, pool_size)

        # Flatten back
        reduced = pooled.reshape(n_samples, -1)

        # Truncate or pad to exact target_dim
        if reduced.shape[1] > self.target_dim:
            reduced = reduced[:, :self.target_dim]
        elif reduced.shape[1] < self.target_dim:
            padding = np.zeros((n_samples, self.target_dim - reduced.shape[1]))
            reduced = np.concatenate([reduced, padding], axis=1)

        return reduced

    def _average_pool(
        self,
        images: np.ndarray,
        pool_size: int
    ) -> np.ndarray:
        """Apply average pooling to images."""
        n_samples, h, w = images.shape
        new_h = h // pool_size
        new_w = w // pool_size

        pooled = np.zeros((n_samples, new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                pooled[:, i, j] = images[
                    :,
                    i*pool_size:(i+1)*pool_size,
                    j*pool_size:(j+1)*pool_size
                ].mean(axis=(1, 2))

        return pooled

    def _fit_random_projection(self, data: np.ndarray):
        """Fit random projection."""
        n_features = data.shape[1]

        # Generate random projection matrix
        rng = np.random.RandomState(self.kwargs.get('random_state', 42))
        self._components = rng.randn(n_features, self.target_dim)

        # Normalize columns
        self._components /= np.linalg.norm(self._components, axis=0)

        logger.info(f"Random projection: {n_features} -> {self.target_dim}")

    def _transform_random_projection(self, data: np.ndarray) -> np.ndarray:
        """Transform using random projection."""
        return data @ self._components


class QuantumImageAdapter:
    """
    Adapter for images for quantum processing.

    Handles:
    - Image resizing and normalization
    - Grayscale conversion
    - Feature extraction
    - Dimension reduction for quantum encoding

    Args:
        n_qubits: Number of qubits available
        resize_to: Target image size (height, width) or None
        grayscale: Convert to grayscale (default: True)
        feature_extraction: Feature extraction method (None, 'pca', 'pool')

    Example:
        >>> adapter = QuantumImageAdapter(n_qubits=8, resize_to=(16, 16))
        >>> adapted_images = adapter.transform(images)
        >>> print(f"Adapted {len(adapted_images)} images for quantum processing")
    """

    def __init__(
        self,
        n_qubits: int,
        resize_to: Optional[Tuple[int, int]] = None,
        grayscale: bool = True,
        feature_extraction: Optional[str] = None
    ):
        """
        Initialize quantum image adapter.

        Args:
            n_qubits: Number of qubits
            resize_to: Target image size
            grayscale: Convert to grayscale
            feature_extraction: Feature extraction method
        """
        self.n_qubits = n_qubits
        self.resize_to = resize_to
        self.grayscale = grayscale
        self.feature_extraction = feature_extraction

        # Calculate target dimensions
        if resize_to:
            self.target_pixels = resize_to[0] * resize_to[1]
        else:
            # Use maximum pixels for amplitude encoding
            self.target_pixels = 2 ** n_qubits

        # Initialize feature extractor if needed
        self.extractor = None
        if feature_extraction in ['pca', 'pool']:
            self.extractor = DimensionReducer(
                method=feature_extraction,
                target_dim=min(self.target_pixels, 2 ** n_qubits)
            )

        logger.info(
            f"Initialized QuantumImageAdapter: {n_qubits} qubits, "
            f"resize_to={resize_to}, grayscale={grayscale}"
        )

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform images for quantum processing.

        Args:
            images: Image array (n_samples, height, width) or
                   (n_samples, height, width, channels)

        Returns:
            Transformed images ready for quantum encoding
        """
        # Flatten and normalize
        processed = self.flatten_and_normalize(images)

        # Resize if needed
        if self.resize_to:
            processed = self.resize_for_qubits(processed)

        # Extract features if extractor configured
        if self.extractor:
            if not self.extractor._fitted:
                processed = self.extractor.fit_transform(processed)
            else:
                processed = self.extractor.transform(processed)

        logger.info(
            f"Transformed {len(images)} images to shape {processed.shape}"
        )
        return processed

    def flatten_and_normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Flatten and normalize images.

        Args:
            images: Image array

        Returns:
            Flattened and normalized images
        """
        # Convert to grayscale if needed
        if self.grayscale and images.ndim == 4:
            # Average across color channels
            images = images.mean(axis=-1)

        # Flatten
        n_samples = images.shape[0]
        flattened = images.reshape(n_samples, -1)

        # Normalize to [0, 1]
        flattened = flattened.astype(np.float32)
        if flattened.max() > 1.0:
            flattened /= 255.0

        return flattened

    def resize_for_qubits(
        self,
        images: np.ndarray
    ) -> np.ndarray:
        """
        Resize images to fit qubit count.

        Args:
            images: Flattened images (n_samples, n_pixels)

        Returns:
            Resized images
        """
        n_samples, n_pixels = images.shape

        if n_pixels > self.target_pixels:
            # Reduce dimensions (truncate or pool)
            if self.target_pixels == int(np.sqrt(self.target_pixels)) ** 2:
                # Square target, use pooling
                img_size = int(np.sqrt(n_pixels))
                target_size = int(np.sqrt(self.target_pixels))

                # Reshape to images
                imgs = images.reshape(n_samples, img_size, img_size)

                # Pool
                pool_size = img_size // target_size
                pooled = self._average_pool_2d(imgs, pool_size, target_size)

                # Flatten
                return pooled.reshape(n_samples, -1)
            else:
                # Just truncate
                return images[:, :self.target_pixels]

        elif n_pixels < self.target_pixels:
            # Pad with zeros
            padding = np.zeros((n_samples, self.target_pixels - n_pixels))
            return np.concatenate([images, padding], axis=1)

        return images

    def _average_pool_2d(
        self,
        images: np.ndarray,
        pool_size: int,
        target_size: int
    ) -> np.ndarray:
        """Apply 2D average pooling."""
        n_samples = images.shape[0]
        pooled = np.zeros((n_samples, target_size, target_size))

        for i in range(target_size):
            for j in range(target_size):
                pooled[:, i, j] = images[
                    :,
                    i*pool_size:(i+1)*pool_size,
                    j*pool_size:(j+1)*pool_size
                ].mean(axis=(1, 2))

        return pooled

    def extract_features(
        self,
        images: np.ndarray,
        method: str = 'pca'
    ) -> np.ndarray:
        """
        Extract features from images.

        Args:
            images: Image array
            method: Feature extraction method ('pca', 'pool')

        Returns:
            Extracted features
        """
        if method not in ['pca', 'pool']:
            raise ValueError(f"Unsupported feature extraction: {method}")

        # Flatten images first
        flattened = self.flatten_and_normalize(images)

        # Extract features
        extractor = DimensionReducer(
            method=method,
            target_dim=min(flattened.shape[1], 2 ** self.n_qubits)
        )
        features = extractor.fit_transform(flattened)

        logger.info(
            f"Extracted {features.shape[1]} features using {method}"
        )
        return features
