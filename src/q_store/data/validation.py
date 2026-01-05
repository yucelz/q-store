"""
Data Validation and Profiling for Quantum ML.

This module provides comprehensive data validation, quality checks, and profiling
utilities optimized for quantum machine learning datasets.

Key Components:
    - DataValidator: Validate data for quantum ML compatibility
    - DataProfiler: Profile dataset characteristics and statistics
    - OutlierDetector: Detect and handle outliers

Example:
    >>> from q_store.data.validation import DataValidator, DataProfiler
    >>>
    >>> # Validate data
    >>> validator = DataValidator()
    >>> is_valid, message = validator.validate_all(x_data, y_data, n_qubits=8)
    >>>
    >>> # Profile data
    >>> profiler = DataProfiler()
    >>> stats = profiler.profile(x_data, y_data)
    >>> print(stats)
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data for quantum machine learning.

    Performs comprehensive validation including:
    - Shape validation
    - Range validation
    - NaN/Inf detection
    - Quantum compatibility checks
    - Class distribution analysis

    Example:
        >>> validator = DataValidator()
        >>> is_valid, message = validator.validate_all(
        ...     x_data, y_data,
        ...     n_qubits=8,
        ...     encoding='amplitude'
        ... )
        >>> if not is_valid:
        ...     print(f"Validation failed: {message}")
    """

    @staticmethod
    def check_shape(
        data: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        ndim: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Check data shape.

        Args:
            data: Data to validate
            expected_shape: Expected shape (None for any)
            ndim: Expected number of dimensions

        Returns:
            Tuple of (is_valid, message)
        """
        if ndim is not None and data.ndim != ndim:
            return False, f"Expected {ndim}D array, got {data.ndim}D"

        if expected_shape is not None:
            # Check each dimension (None means any size)
            for i, (expected, actual) in enumerate(zip(expected_shape, data.shape)):
                if expected is not None and expected != actual:
                    return False, (
                        f"Shape mismatch at dimension {i}: "
                        f"expected {expected}, got {actual}"
                    )

        return True, "Shape is valid"

    @staticmethod
    def check_range(
        data: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check data value range.

        Args:
            data: Data to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Tuple of (is_valid, message)
        """
        data_min = data.min()
        data_max = data.max()

        if min_val is not None and data_min < min_val:
            return False, f"Data minimum {data_min} is below threshold {min_val}"

        if max_val is not None and data_max > max_val:
            return False, f"Data maximum {data_max} is above threshold {max_val}"

        return True, f"Data range [{data_min:.4f}, {data_max:.4f}] is valid"

    @staticmethod
    def check_nan(data: np.ndarray) -> Tuple[bool, str]:
        """
        Check for NaN values.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, message)
        """
        n_nan = np.isnan(data).sum()
        if n_nan > 0:
            return False, f"Data contains {n_nan} NaN values"
        return True, "No NaN values found"

    @staticmethod
    def check_inf(data: np.ndarray) -> Tuple[bool, str]:
        """
        Check for infinite values.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, message)
        """
        n_inf = np.isinf(data).sum()
        if n_inf > 0:
            return False, f"Data contains {n_inf} infinite values"
        return True, "No infinite values found"

    @staticmethod
    def check_quantum_compatibility(
        data: np.ndarray,
        n_qubits: int,
        encoding: str = 'amplitude'
    ) -> Tuple[bool, str]:
        """
        Check quantum encoding compatibility.

        Args:
            data: Data to validate
            n_qubits: Number of qubits
            encoding: Encoding type ('amplitude', 'angle', 'basis')

        Returns:
            Tuple of (is_valid, message)
        """
        if data.ndim != 2:
            return False, f"Data must be 2D for quantum encoding, got {data.ndim}D"

        n_samples, n_features = data.shape

        # Check dimension compatibility
        if encoding == 'amplitude':
            max_features = 2 ** n_qubits
            if n_features > max_features:
                return False, (
                    f"Amplitude encoding with {n_qubits} qubits supports max "
                    f"{max_features} features, got {n_features}"
                )
        elif encoding == 'angle':
            if n_features > n_qubits:
                return False, (
                    f"Angle encoding with {n_qubits} qubits supports max "
                    f"{n_qubits} features, got {n_features}"
                )
        elif encoding == 'basis':
            if n_features > n_qubits:
                return False, (
                    f"Basis encoding with {n_qubits} qubits supports max "
                    f"{n_qubits} features, got {n_features}"
                )
        else:
            return False, f"Unsupported encoding: {encoding}"

        return True, f"Data is compatible with {encoding} encoding on {n_qubits} qubits"

    @staticmethod
    def check_class_distribution(
        labels: np.ndarray,
        min_samples_per_class: int = 10
    ) -> Tuple[bool, str]:
        """
        Check class distribution.

        Args:
            labels: Class labels
            min_samples_per_class: Minimum samples per class

        Returns:
            Tuple of (is_valid, message)
        """
        unique, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique)

        # Check minimum samples
        min_count = counts.min()
        if min_count < min_samples_per_class:
            return False, (
                f"Class {unique[counts.argmin()]} has only {min_count} samples, "
                f"minimum required: {min_samples_per_class}"
            )

        # Check class balance
        max_count = counts.max()
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 10:
            return False, (
                f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                f"Consider using balanced sampling or class weights."
            )

        distribution = {cls: count for cls, count in zip(unique, counts)}
        return True, f"Class distribution is acceptable: {distribution}"

    def validate_all(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_qubits: Optional[int] = None,
        encoding: str = 'amplitude',
        min_samples_per_class: int = 10
    ) -> Tuple[bool, str]:
        """
        Perform all validation checks.

        Args:
            x_data: Input data
            y_data: Labels
            n_qubits: Number of qubits (for quantum compatibility check)
            encoding: Encoding type
            min_samples_per_class: Minimum samples per class

        Returns:
            Tuple of (is_valid, message)
        """
        checks = []

        # Shape checks
        is_valid, msg = self.check_shape(x_data, ndim=2)
        checks.append((is_valid, f"Shape check: {msg}"))

        is_valid, msg = self.check_shape(y_data, ndim=1)
        checks.append((is_valid, f"Labels shape check: {msg}"))

        # Sample count match
        if len(x_data) != len(y_data):
            checks.append((
                False,
                f"Sample count mismatch: x_data={len(x_data)}, y_data={len(y_data)}"
            ))
        else:
            checks.append((True, f"Sample counts match: {len(x_data)} samples"))

        # NaN/Inf checks
        is_valid, msg = self.check_nan(x_data)
        checks.append((is_valid, f"NaN check: {msg}"))

        is_valid, msg = self.check_inf(x_data)
        checks.append((is_valid, f"Inf check: {msg}"))

        # Quantum compatibility
        if n_qubits is not None:
            is_valid, msg = self.check_quantum_compatibility(x_data, n_qubits, encoding)
            checks.append((is_valid, f"Quantum compatibility: {msg}"))

        # Class distribution
        is_valid, msg = self.check_class_distribution(y_data, min_samples_per_class)
        checks.append((is_valid, f"Class distribution: {msg}"))

        # Combine results
        all_valid = all(valid for valid, _ in checks)
        messages = [msg for _, msg in checks]

        if all_valid:
            summary = "All validation checks passed:\n" + "\n".join(f"  ✓ {msg}" for msg in messages)
        else:
            failed = [msg for valid, msg in checks if not valid]
            passed = [msg for valid, msg in checks if valid]
            summary = "Validation failed:\n"
            summary += "\n".join(f"  ✗ {msg}" for msg in failed)
            if passed:
                summary += "\n\nPassed checks:\n"
                summary += "\n".join(f"  ✓ {msg}" for msg in passed)

        return all_valid, summary


class DataProfiler:
    """
    Profile dataset characteristics and statistics.

    Computes comprehensive statistics including:
    - Basic statistics (mean, std, min, max, quartiles)
    - Class distribution
    - Outlier detection
    - Feature correlations
    - Data quality metrics

    Example:
        >>> profiler = DataProfiler()
        >>> profile = profiler.profile(x_data, y_data)
        >>> print(f"Mean: {profile['statistics']['mean']}")
        >>> print(f"Class distribution: {profile['class_distribution']}")
    """

    @staticmethod
    def compute_statistics(data: np.ndarray) -> Dict[str, Any]:
        """
        Compute basic statistics.

        Args:
            data: Data array

        Returns:
            Dictionary with statistics
        """
        stats = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'n_samples': data.shape[0],
            'n_features': data.shape[1] if data.ndim > 1 else 1
        }

        # Per-feature statistics
        if data.ndim == 2:
            stats['feature_means'] = np.mean(data, axis=0).tolist()
            stats['feature_stds'] = np.std(data, axis=0).tolist()
            stats['feature_mins'] = np.min(data, axis=0).tolist()
            stats['feature_maxs'] = np.max(data, axis=0).tolist()

        return stats

    @staticmethod
    def compute_class_distribution(labels: np.ndarray) -> Dict[int, int]:
        """
        Compute class distribution.

        Args:
            labels: Class labels

        Returns:
            Dictionary mapping class to count
        """
        unique, counts = np.unique(labels, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}

    @staticmethod
    def detect_outliers(
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in data.

        Args:
            data: Data array
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Boolean array indicating outliers
        """
        if method == 'iqr':
            # IQR method
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr

            outliers = (data < lower_bound) | (data > upper_bound)
            return outliers.any(axis=1) if data.ndim > 1 else outliers

        elif method == 'zscore':
            # Z-score method
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-10))

            outliers = z_scores > threshold
            return outliers.any(axis=1) if data.ndim > 1 else outliers

        elif method == 'isolation_forest':
            # Isolation Forest (requires sklearn)
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError:
                raise ImportError(
                    "scikit-learn required for isolation_forest method. "
                    "Install with: pip install scikit-learn"
                )

            clf = IsolationForest(contamination=0.1, random_state=42)
            predictions = clf.fit_predict(data.reshape(-1, 1) if data.ndim == 1 else data)
            return predictions == -1

        else:
            raise ValueError(
                f"Unsupported outlier detection method: {method}. "
                "Use 'iqr', 'zscore', or 'isolation_forest'"
            )

    @staticmethod
    def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Compute feature correlation matrix.

        Args:
            data: Data array (n_samples, n_features)

        Returns:
            Correlation matrix (n_features, n_features)
        """
        if data.ndim != 2:
            raise ValueError("Correlation requires 2D data")

        return np.corrcoef(data.T)

    def profile(
        self,
        x_data: np.ndarray,
        y_data: Optional[np.ndarray] = None,
        include_outliers: bool = True,
        include_correlations: bool = False
    ) -> Dict[str, Any]:
        """
        Compute comprehensive data profile.

        Args:
            x_data: Input data
            y_data: Labels (optional)
            include_outliers: Whether to detect outliers
            include_correlations: Whether to compute feature correlations

        Returns:
            Dictionary with complete profile
        """
        profile = {
            'statistics': self.compute_statistics(x_data),
            'data_quality': self._assess_quality(x_data)
        }

        # Class distribution
        if y_data is not None:
            profile['class_distribution'] = self.compute_class_distribution(y_data)
            profile['num_classes'] = len(np.unique(y_data))

        # Outliers
        if include_outliers:
            outliers_iqr = self.detect_outliers(x_data, method='iqr')
            outliers_zscore = self.detect_outliers(x_data, method='zscore', threshold=3)

            profile['outliers'] = {
                'iqr_method': int(outliers_iqr.sum()),
                'zscore_method': int(outliers_zscore.sum()),
                'iqr_percentage': float(outliers_iqr.sum() / len(x_data) * 100),
                'zscore_percentage': float(outliers_zscore.sum() / len(x_data) * 100)
            }

        # Correlations
        if include_correlations and x_data.ndim == 2 and x_data.shape[1] > 1:
            corr_matrix = self.compute_correlation_matrix(x_data)
            profile['correlations'] = {
                'matrix': corr_matrix.tolist(),
                'max_correlation': float(np.max(np.abs(corr_matrix - np.eye(len(corr_matrix))))),
                'highly_correlated_pairs': self._find_highly_correlated(corr_matrix)
            }

        logger.info(
            f"Profiled dataset: {profile['statistics']['n_samples']} samples, "
            f"{profile['statistics']['n_features']} features"
        )

        return profile

    @staticmethod
    def _assess_quality(data: np.ndarray) -> Dict[str, Any]:
        """Assess data quality."""
        return {
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any()),
            'num_nan': int(np.isnan(data).sum()),
            'num_inf': int(np.isinf(data).sum()),
            'nan_percentage': float(np.isnan(data).sum() / data.size * 100),
            'inf_percentage': float(np.isinf(data).sum() / data.size * 100),
            'num_zeros': int((data == 0).sum()),
            'zero_percentage': float((data == 0).sum() / data.size * 100)
        }

    @staticmethod
    def _find_highly_correlated(
        corr_matrix: np.ndarray,
        threshold: float = 0.9
    ) -> List[Tuple[int, int, float]]:
        """Find highly correlated feature pairs."""
        pairs = []
        n_features = len(corr_matrix)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    pairs.append((i, j, float(corr)))

        # Sort by correlation strength
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[:10]  # Return top 10


class OutlierDetector:
    """
    Detect and handle outliers in datasets.

    Supports multiple detection methods and handling strategies.

    Args:
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Detection threshold
        action: Action to take ('flag', 'remove', 'clip')

    Example:
        >>> detector = OutlierDetector(method='iqr', action='clip')
        >>> x_clean = detector.fit_transform(x_data)
        >>> print(f"Detected {detector.n_outliers_} outliers")
    """

    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        action: str = 'flag'
    ):
        """
        Initialize outlier detector.

        Args:
            method: Detection method
            threshold: Detection threshold
            action: Action ('flag', 'remove', 'clip')
        """
        self.method = method
        self.threshold = threshold
        self.action = action

        self.outlier_mask_ = None
        self.n_outliers_ = 0
        self.bounds_ = None

    def fit(self, data: np.ndarray) -> 'OutlierDetector':
        """
        Fit outlier detector to data.

        Args:
            data: Training data

        Returns:
            Self
        """
        profiler = DataProfiler()
        self.outlier_mask_ = profiler.detect_outliers(
            data,
            method=self.method,
            threshold=self.threshold
        )
        self.n_outliers_ = int(self.outlier_mask_.sum())

        # Compute bounds for clipping
        if self.action == 'clip':
            if self.method == 'iqr':
                q75, q25 = np.percentile(data, [75, 25], axis=0)
                iqr = q75 - q25
                self.bounds_ = (
                    q25 - self.threshold * iqr,
                    q75 + self.threshold * iqr
                )
            elif self.method == 'zscore':
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                self.bounds_ = (
                    mean - self.threshold * std,
                    mean + self.threshold * std
                )

        logger.info(
            f"Detected {self.n_outliers_} outliers "
            f"({self.n_outliers_ / len(data) * 100:.2f}%)"
        )

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data based on outlier handling strategy.

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        if self.outlier_mask_ is None:
            raise RuntimeError("OutlierDetector must be fitted before transform")

        if self.action == 'flag':
            # Just return original data (outliers are flagged)
            return data

        elif self.action == 'remove':
            # Remove outlier samples
            return data[~self.outlier_mask_]

        elif self.action == 'clip':
            # Clip values to bounds
            if self.bounds_ is None:
                raise RuntimeError("Bounds not computed. Use 'iqr' or 'zscore' method.")
            return np.clip(data, self.bounds_[0], self.bounds_[1])

        else:
            raise ValueError(f"Unsupported action: {self.action}")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
