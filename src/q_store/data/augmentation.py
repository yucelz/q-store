"""
Data Augmentation for Quantum ML.

This module provides data augmentation techniques optimized for quantum machine learning,
including quantum-specific, classical, and hybrid augmentation strategies.

Key Components:
    - QuantumAugmentation: Quantum-specific augmentation (phase shifts, amplitude noise)
    - ClassicalAugmentation: Classical augmentation (wraps albumentations)
    - HybridAugmentation: Combined classical + quantum augmentation

Example:
    >>> from q_store.data.augmentation import QuantumAugmentation, HybridAugmentation
    >>>
    >>> # Quantum augmentation
    >>> aug = QuantumAugmentation(phase_shift_range=0.1, amplitude_noise=0.01)
    >>> augmented_data = aug.apply(data)
    >>>
    >>> # Hybrid augmentation
    >>> hybrid = HybridAugmentation(
    ...     classical_transforms=['horizontal_flip', 'rotate'],
    ...     quantum_augmentation=True
    ... )
    >>> augmented_data = hybrid.apply(images)
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class QuantumAugmentation:
    """
    Quantum-specific data augmentation.

    Applies quantum-aware transformations including:
    - Phase shifts
    - Amplitude noise
    - Random rotations
    - Quantum state perturbations

    Args:
        phase_shift_range: Maximum phase shift in radians (default: 0.1)
        amplitude_noise: Noise level for amplitude perturbation (default: 0.01)
        rotation_range: Maximum rotation angle (default: 0.1)
        probability: Probability of applying each transformation (default: 0.5)
        random_seed: Random seed for reproducibility (default: None)

    Example:
        >>> aug = QuantumAugmentation(phase_shift_range=0.1, amplitude_noise=0.01)
        >>> augmented = aug.apply(quantum_data)
        >>> print(f"Augmented {len(augmented)} samples")
    """

    def __init__(
        self,
        phase_shift_range: float = 0.1,
        amplitude_noise: float = 0.01,
        rotation_range: float = 0.1,
        probability: float = 0.5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize quantum augmentation.

        Args:
            phase_shift_range: Phase shift range
            amplitude_noise: Amplitude noise level
            rotation_range: Rotation angle range
            probability: Application probability
            random_seed: Random seed
        """
        self.phase_shift_range = phase_shift_range
        self.amplitude_noise = amplitude_noise
        self.rotation_range = rotation_range
        self.probability = probability

        self.rng = np.random.RandomState(random_seed)

        logger.info(
            f"Initialized QuantumAugmentation: "
            f"phase_shift={phase_shift_range}, "
            f"amplitude_noise={amplitude_noise}, "
            f"rotation={rotation_range}"
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply quantum augmentation to data.

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Augmented data
        """
        augmented = data.copy()

        # Apply phase shift
        if self.rng.rand() < self.probability:
            augmented = self._phase_shift(augmented)

        # Apply amplitude noise
        if self.rng.rand() < self.probability:
            augmented = self._amplitude_noise(augmented)

        # Apply random rotation
        if self.rng.rand() < self.probability:
            augmented = self._random_rotation(augmented)

        return augmented

    def _phase_shift(self, data: np.ndarray) -> np.ndarray:
        """Apply phase shift to quantum data."""
        # Generate random phase shifts
        shifts = self.rng.uniform(
            -self.phase_shift_range,
            self.phase_shift_range,
            size=data.shape
        )

        # Apply phase shift (assuming data represents amplitudes)
        # For quantum states, this corresponds to a global phase
        phase_factors = np.exp(1j * shifts)

        # If data is real, apply as additive shift
        if not np.iscomplexobj(data):
            return data + shifts
        else:
            return data * phase_factors

    def _amplitude_noise(self, data: np.ndarray) -> np.ndarray:
        """Add amplitude noise to quantum data."""
        noise = self.rng.normal(0, self.amplitude_noise, size=data.shape)
        noisy_data = data + noise

        # Renormalize to maintain quantum state normalization
        if data.ndim == 2:
            # L2 normalization per sample
            norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            noisy_data = noisy_data / norms

        return noisy_data

    def _random_rotation(self, data: np.ndarray) -> np.ndarray:
        """Apply random rotation to quantum data."""
        # Generate random rotation angles
        angles = self.rng.uniform(
            -self.rotation_range,
            self.rotation_range,
            size=data.shape[0]
        )

        # Apply rotation (simple 2D rotation for each sample)
        if data.ndim == 2 and data.shape[1] >= 2:
            rotated = data.copy()
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)

            # Rotate first two features
            x0 = data[:, 0]
            x1 = data[:, 1]

            rotated[:, 0] = cos_angles * x0 - sin_angles * x1
            rotated[:, 1] = sin_angles * x0 + cos_angles * x1

            return rotated

        return data


class ClassicalAugmentation:
    """
    Classical data augmentation (wraps albumentations for images).

    Provides classical augmentation techniques including:
    - Geometric transformations (flip, rotate, scale)
    - Color transformations (brightness, contrast, saturation)
    - Noise addition (Gaussian, salt-and-pepper)
    - Blur and sharpening

    Args:
        transforms: List of transformation names or albumentations config
        probability: Probability of applying transformations (default: 0.5)

    Example:
        >>> aug = ClassicalAugmentation(
        ...     transforms=['horizontal_flip', 'rotate', 'brightness']
        ... )
        >>> augmented_images = aug.apply(images)
    """

    # Default transformation configurations
    DEFAULT_TRANSFORMS = {
        'horizontal_flip': {'p': 0.5},
        'vertical_flip': {'p': 0.3},
        'rotate': {'limit': 15, 'p': 0.5},
        'brightness': {'limit': 0.2, 'p': 0.5},
        'contrast': {'limit': 0.2, 'p': 0.5},
        'gaussian_noise': {'var_limit': (10, 50), 'p': 0.3},
        'blur': {'blur_limit': 3, 'p': 0.3},
    }

    def __init__(
        self,
        transforms: Optional[List[str]] = None,
        albumentations_config: Optional[Dict[str, Any]] = None,
        probability: float = 0.5
    ):
        """
        Initialize classical augmentation.

        Args:
            transforms: List of transform names
            albumentations_config: Albumentations configuration dict
            probability: Application probability
        """
        self.transforms = transforms or ['horizontal_flip', 'rotate']
        self.albumentations_config = albumentations_config
        self.probability = probability
        self.pipeline = None

        # Try to import albumentations
        try:
            import albumentations as A
            self.albumentations_available = True
            self._build_pipeline(A)
        except ImportError:
            logger.warning(
                "albumentations not available. "
                "Install with: pip install albumentations"
            )
            self.albumentations_available = False

        logger.info(
            f"Initialized ClassicalAugmentation with transforms: {self.transforms}"
        )

    def _build_pipeline(self, A):
        """Build albumentations pipeline."""
        if self.albumentations_config:
            # Use custom configuration
            transforms_list = []
            for transform_name, params in self.albumentations_config.items():
                transform_cls = getattr(A, transform_name, None)
                if transform_cls:
                    transforms_list.append(transform_cls(**params))

            self.pipeline = A.Compose(transforms_list)

        else:
            # Build from transform names
            transforms_list = []

            for name in self.transforms:
                if name == 'horizontal_flip':
                    transforms_list.append(A.HorizontalFlip(p=self.probability))
                elif name == 'vertical_flip':
                    transforms_list.append(A.VerticalFlip(p=self.probability))
                elif name == 'rotate':
                    transforms_list.append(A.Rotate(limit=15, p=self.probability))
                elif name == 'brightness':
                    transforms_list.append(A.RandomBrightness(limit=0.2, p=self.probability))
                elif name == 'contrast':
                    transforms_list.append(A.RandomContrast(limit=0.2, p=self.probability))
                elif name == 'gaussian_noise':
                    transforms_list.append(A.GaussNoise(var_limit=(10, 50), p=self.probability))
                elif name == 'blur':
                    transforms_list.append(A.Blur(blur_limit=3, p=self.probability))

            self.pipeline = A.Compose(transforms_list)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply classical augmentation.

        Args:
            data: Input data (images)

        Returns:
            Augmented data
        """
        if not self.albumentations_available:
            logger.warning("Albumentations not available, returning original data")
            return data

        if self.pipeline is None:
            logger.warning("Pipeline not built, returning original data")
            return data

        # Apply to each image
        augmented = []
        for img in data:
            # Ensure uint8 format for albumentations
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Apply transformations
            transformed = self.pipeline(image=img)
            augmented_img = transformed['image']

            # Normalize back to [0, 1]
            if augmented_img.dtype == np.uint8:
                augmented_img = augmented_img.astype(np.float32) / 255.0

            augmented.append(augmented_img)

        return np.array(augmented)


class HybridAugmentation:
    """
    Combined classical + quantum augmentation.

    Applies both classical and quantum augmentation techniques
    in sequence for comprehensive data augmentation.

    Args:
        classical_transforms: List of classical transform names
        quantum_config: Configuration for quantum augmentation
        apply_classical_first: Whether to apply classical before quantum (default: True)
        classical_probability: Probability for classical augmentation (default: 0.5)
        quantum_probability: Probability for quantum augmentation (default: 0.5)

    Example:
        >>> hybrid = HybridAugmentation(
        ...     classical_transforms=['horizontal_flip', 'rotate'],
        ...     quantum_config={'phase_shift_range': 0.1, 'amplitude_noise': 0.01}
        ... )
        >>> augmented = hybrid.apply(data)
    """

    def __init__(
        self,
        classical_transforms: Optional[List[str]] = None,
        quantum_config: Optional[Dict[str, Any]] = None,
        apply_classical_first: bool = True,
        classical_probability: float = 0.5,
        quantum_probability: float = 0.5
    ):
        """
        Initialize hybrid augmentation.

        Args:
            classical_transforms: Classical transforms
            quantum_config: Quantum augmentation config
            apply_classical_first: Order of application
            classical_probability: Classical probability
            quantum_probability: Quantum probability
        """
        self.apply_classical_first = apply_classical_first

        # Initialize classical augmentation
        if classical_transforms:
            self.classical_aug = ClassicalAugmentation(
                transforms=classical_transforms,
                probability=classical_probability
            )
        else:
            self.classical_aug = None

        # Initialize quantum augmentation
        if quantum_config:
            self.quantum_aug = QuantumAugmentation(
                probability=quantum_probability,
                **quantum_config
            )
        else:
            self.quantum_aug = QuantumAugmentation(probability=quantum_probability)

        logger.info(
            f"Initialized HybridAugmentation: "
            f"classical_first={apply_classical_first}"
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply hybrid augmentation.

        Args:
            data: Input data

        Returns:
            Augmented data
        """
        if self.apply_classical_first:
            # Classical → Quantum
            if self.classical_aug:
                data = self.classical_aug.apply(data)
            data = self.quantum_aug.apply(data)
        else:
            # Quantum → Classical
            data = self.quantum_aug.apply(data)
            if self.classical_aug:
                data = self.classical_aug.apply(data)

        return data


class AugmentationPipeline:
    """
    Sequential augmentation pipeline.

    Allows chaining multiple augmentation techniques in a specific order.

    Args:
        augmentations: List of augmentation objects or configurations

    Example:
        >>> pipeline = AugmentationPipeline([
        ...     QuantumAugmentation(phase_shift_range=0.1),
        ...     ClassicalAugmentation(transforms=['horizontal_flip'])
        ... ])
        >>> augmented = pipeline.apply(data)
    """

    def __init__(self, augmentations: List[Any]):
        """
        Initialize augmentation pipeline.

        Args:
            augmentations: List of augmentation objects
        """
        self.augmentations = augmentations

        logger.info(
            f"Initialized AugmentationPipeline with {len(augmentations)} augmentations"
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply all augmentations in sequence.

        Args:
            data: Input data

        Returns:
            Augmented data
        """
        for aug in self.augmentations:
            data = aug.apply(data)
        return data

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Make pipeline callable."""
        return self.apply(data)


def create_augmentation(
    augmentation_type: str = 'quantum',
    **kwargs
) -> Any:
    """
    Convenience function to create an augmentation object.

    Args:
        augmentation_type: Type ('quantum', 'classical', 'hybrid')
        **kwargs: Configuration parameters

    Returns:
        Augmentation object

    Example:
        >>> aug = create_augmentation('quantum', phase_shift_range=0.1)
        >>> aug = create_augmentation('classical', transforms=['flip', 'rotate'])
        >>> aug = create_augmentation('hybrid', classical_transforms=['flip'])
    """
    if augmentation_type == 'quantum':
        return QuantumAugmentation(**kwargs)
    elif augmentation_type == 'classical':
        return ClassicalAugmentation(**kwargs)
    elif augmentation_type == 'hybrid':
        return HybridAugmentation(**kwargs)
    else:
        raise ValueError(
            f"Unsupported augmentation type: {augmentation_type}. "
            "Use 'quantum', 'classical', or 'hybrid'"
        )
