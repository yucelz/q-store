"""
Data Generators for Efficient Quantum ML Training.

This module provides data generators for efficient batch processing,
on-the-fly augmentation, and memory-efficient streaming for large datasets.

Key Components:
    - QuantumDataGenerator: Batch generator with augmentation
    - StreamingDataGenerator: Memory-efficient generator for large datasets
    - InfiniteDataGenerator: Infinite loop generator for continuous training

Example:
    >>> from q_store.data.generators import QuantumDataGenerator
    >>>
    >>> # Create generator
    >>> generator = QuantumDataGenerator(
    ...     x_data, y_data,
    ...     batch_size=32,
    ...     shuffle=True
    ... )
    >>>
    >>> # Iterate over batches
    >>> for x_batch, y_batch in generator:
    ...     # Train on batch
    ...     pass
"""

import logging
from typing import Optional, Callable, Tuple, Iterator
import numpy as np

logger = logging.getLogger(__name__)


class QuantumDataGenerator:
    """
    Data generator for quantum ML training.

    Provides efficient batching with optional shuffling, augmentation,
    and preprocessing for quantum machine learning.

    Args:
        x_data: Input data (n_samples, n_features)
        y_data: Labels (n_samples,)
        batch_size: Batch size (default: 32)
        shuffle: Whether to shuffle after each epoch (default: True)
        augmentation: Augmentation function (optional)
        preprocessing: Preprocessing function (optional)
        random_seed: Random seed for reproducibility (default: 42)

    Example:
        >>> generator = QuantumDataGenerator(
        ...     x_train, y_train,
        ...     batch_size=32,
        ...     shuffle=True
        ... )
        >>> for epoch in range(10):
        ...     for x_batch, y_batch in generator:
        ...         # Train on batch
        ...         model.train_step(x_batch, y_batch)
    """

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        random_seed: int = 42
    ):
        """
        Initialize quantum data generator.

        Args:
            x_data: Input data
            y_data: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            augmentation: Augmentation function
            preprocessing: Preprocessing function
            random_seed: Random seed
        """
        if len(x_data) != len(y_data):
            raise ValueError(
                f"x_data and y_data must have same length: "
                f"{len(x_data)} vs {len(y_data)}"
            )

        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.random_seed = random_seed

        self.n_samples = len(x_data)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

        self.rng = np.random.RandomState(random_seed)
        self.indices = np.arange(self.n_samples)
        self.current_index = 0

        if self.shuffle:
            self.rng.shuffle(self.indices)

        logger.info(
            f"Initialized QuantumDataGenerator: {self.n_samples} samples, "
            f"{self.n_batches} batches of size {batch_size}"
        )

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.n_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return iterator over batches."""
        self.current_index = 0
        if self.shuffle:
            self.rng.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_index >= self.n_samples:
            raise StopIteration

        # Get batch indices
        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Extract batch
        x_batch = self.x_data[batch_indices]
        y_batch = self.y_data[batch_indices]

        # Apply augmentation
        if self.augmentation:
            x_batch = self.augmentation(x_batch)

        # Apply preprocessing
        if self.preprocessing:
            x_batch = self.preprocessing(x_batch)

        self.current_index = end_idx

        return x_batch, y_batch

    def reset(self):
        """Reset generator to beginning."""
        self.current_index = 0
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        self.reset()


class StreamingDataGenerator:
    """
    Memory-efficient generator for large datasets.

    Loads data in chunks from disk, reducing memory footprint
    for datasets that don't fit in RAM.

    Args:
        data_path: Path to data file (.npz, .hdf5)
        batch_size: Batch size (default: 32)
        chunk_size: Number of samples to load at once (default: 10000)
        shuffle: Whether to shuffle chunks (default: True)
        data_key: Key for data in file (default: 'x')
        labels_key: Key for labels in file (default: 'y')

    Example:
        >>> generator = StreamingDataGenerator(
        ...     data_path='large_dataset.npz',
        ...     batch_size=32,
        ...     chunk_size=10000
        ... )
        >>> for x_batch, y_batch in generator:
        ...     model.train_step(x_batch, y_batch)
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        chunk_size: int = 10000,
        shuffle: bool = True,
        data_key: str = 'x',
        labels_key: str = 'y'
    ):
        """
        Initialize streaming data generator.

        Args:
            data_path: Path to data file
            batch_size: Batch size
            chunk_size: Chunk size for loading
            shuffle: Whether to shuffle
            data_key: Data key in file
            labels_key: Labels key in file
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.data_key = data_key
        self.labels_key = labels_key

        # Determine file format
        if data_path.endswith('.npz'):
            self.file_format = 'npz'
        elif data_path.endswith(('.h5', '.hdf5')):
            self.file_format = 'hdf5'
        else:
            raise ValueError(
                f"Unsupported file format. Use .npz or .hdf5: {data_path}"
            )

        # Get dataset size
        self.n_samples = self._get_dataset_size()
        self.n_chunks = int(np.ceil(self.n_samples / chunk_size))

        # Current state
        self.current_chunk = None
        self.current_chunk_idx = 0
        self.current_batch_idx = 0

        logger.info(
            f"Initialized StreamingDataGenerator: {self.n_samples} samples, "
            f"{self.n_chunks} chunks of size {chunk_size}"
        )

    def _get_dataset_size(self) -> int:
        """Get total number of samples in dataset."""
        if self.file_format == 'npz':
            data = np.load(self.data_path)
            size = len(data[self.data_key])
            data.close()
            return size
        elif self.file_format == 'hdf5':
            import h5py
            with h5py.File(self.data_path, 'r') as f:
                return len(f[self.data_key])

    def _load_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a chunk of data from disk."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_samples)

        if self.file_format == 'npz':
            data = np.load(self.data_path)
            x_chunk = data[self.data_key][start_idx:end_idx]
            y_chunk = data[self.labels_key][start_idx:end_idx]
            data.close()
        elif self.file_format == 'hdf5':
            import h5py
            with h5py.File(self.data_path, 'r') as f:
                x_chunk = f[self.data_key][start_idx:end_idx]
                y_chunk = f[self.labels_key][start_idx:end_idx]

        logger.debug(f"Loaded chunk {chunk_idx}: samples {start_idx}-{end_idx}")
        return x_chunk, y_chunk

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return iterator over batches."""
        self.current_chunk_idx = 0
        self.current_batch_idx = 0
        self.current_chunk = None
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch."""
        # Load new chunk if needed
        if self.current_chunk is None or self.current_batch_idx >= len(self.current_chunk[0]):
            if self.current_chunk_idx >= self.n_chunks:
                raise StopIteration

            self.current_chunk = self._load_chunk(self.current_chunk_idx)
            self.current_batch_idx = 0
            self.current_chunk_idx += 1

            # Shuffle chunk if requested
            if self.shuffle:
                indices = np.random.permutation(len(self.current_chunk[0]))
                self.current_chunk = (
                    self.current_chunk[0][indices],
                    self.current_chunk[1][indices]
                )

        # Extract batch from current chunk
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.batch_size, len(self.current_chunk[0]))

        x_batch = self.current_chunk[0][start_idx:end_idx]
        y_batch = self.current_chunk[1][start_idx:end_idx]

        self.current_batch_idx = end_idx

        return x_batch, y_batch

    def reset(self):
        """Reset generator to beginning."""
        self.current_chunk_idx = 0
        self.current_batch_idx = 0
        self.current_chunk = None


class InfiniteDataGenerator:
    """
    Infinite loop generator for continuous training.

    Wraps a finite generator and loops infinitely, useful for
    training with a fixed number of steps rather than epochs.

    Args:
        base_generator: Base generator to wrap
        shuffle_on_reset: Whether to shuffle when restarting (default: True)

    Example:
        >>> base_gen = QuantumDataGenerator(x_train, y_train, batch_size=32)
        >>> infinite_gen = InfiniteDataGenerator(base_gen)
        >>>
        >>> # Train for fixed number of steps
        >>> for step in range(10000):
        ...     x_batch, y_batch = next(infinite_gen)
        ...     model.train_step(x_batch, y_batch)
    """

    def __init__(
        self,
        base_generator: QuantumDataGenerator,
        shuffle_on_reset: bool = True
    ):
        """
        Initialize infinite data generator.

        Args:
            base_generator: Base generator to wrap
            shuffle_on_reset: Whether to shuffle on reset
        """
        self.base_generator = base_generator
        self.shuffle_on_reset = shuffle_on_reset
        self.iterator = iter(base_generator)

        logger.info("Initialized InfiniteDataGenerator")

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch, looping infinitely."""
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset base generator
            if self.shuffle_on_reset and hasattr(self.base_generator, 'on_epoch_end'):
                self.base_generator.on_epoch_end()
            self.iterator = iter(self.base_generator)
            return next(self.iterator)


class BalancedBatchGenerator:
    """
    Generator that ensures balanced class distribution in each batch.

    Useful for imbalanced datasets where you want each batch to have
    approximately equal representation of all classes.

    Args:
        x_data: Input data (n_samples, n_features)
        y_data: Labels (n_samples,)
        batch_size: Batch size (default: 32)
        shuffle: Whether to shuffle within classes (default: True)
        random_seed: Random seed (default: 42)

    Example:
        >>> generator = BalancedBatchGenerator(
        ...     x_train, y_train,
        ...     batch_size=32
        ... )
        >>> for x_batch, y_batch in generator:
        ...     # Each batch has balanced classes
        ...     print(np.bincount(y_batch))
    """

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize balanced batch generator.

        Args:
            x_data: Input data
            y_data: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            random_seed: Random seed
        """
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_seed)

        # Group indices by class
        self.classes = np.unique(y_data)
        self.class_indices = {
            cls: np.where(y_data == cls)[0]
            for cls in self.classes
        }

        # Shuffle class indices
        if self.shuffle:
            for cls in self.classes:
                self.rng.shuffle(self.class_indices[cls])

        # Calculate samples per class per batch
        self.n_classes = len(self.classes)
        self.samples_per_class = batch_size // self.n_classes
        self.remainder = batch_size % self.n_classes

        # Current position in each class
        self.class_positions = {cls: 0 for cls in self.classes}

        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.n_batches = min_class_size // self.samples_per_class

        logger.info(
            f"Initialized BalancedBatchGenerator: {self.n_classes} classes, "
            f"{self.samples_per_class} samples per class per batch"
        )

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return iterator over balanced batches."""
        self.class_positions = {cls: 0 for cls in self.classes}
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next balanced batch."""
        batch_indices = []

        # Check if we can create another batch
        for cls in self.classes:
            if self.class_positions[cls] + self.samples_per_class > len(self.class_indices[cls]):
                raise StopIteration

        # Sample from each class
        for i, cls in enumerate(self.classes):
            # Add extra sample to first few classes if there's remainder
            n_samples = self.samples_per_class + (1 if i < self.remainder else 0)

            start_pos = self.class_positions[cls]
            end_pos = start_pos + n_samples

            # Get indices for this class
            cls_batch_indices = self.class_indices[cls][start_pos:end_pos]
            batch_indices.extend(cls_batch_indices)

            # Update position
            self.class_positions[cls] = end_pos

        # Shuffle batch indices
        if self.shuffle:
            self.rng.shuffle(batch_indices)

        # Extract batch
        x_batch = self.x_data[batch_indices]
        y_batch = self.y_data[batch_indices]

        return x_batch, y_batch

    def reset(self):
        """Reset generator to beginning."""
        self.class_positions = {cls: 0 for cls in self.classes}
        if self.shuffle:
            for cls in self.classes:
                self.rng.shuffle(self.class_indices[cls])


def create_data_generator(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int = 32,
    generator_type: str = 'standard',
    **kwargs
) -> QuantumDataGenerator:
    """
    Convenience function to create a data generator.

    Args:
        x_data: Input data
        y_data: Labels
        batch_size: Batch size
        generator_type: Type of generator ('standard', 'balanced', 'infinite')
        **kwargs: Additional arguments for generator

    Returns:
        Data generator instance

    Example:
        >>> generator = create_data_generator(
        ...     x_train, y_train,
        ...     batch_size=32,
        ...     generator_type='balanced'
        ... )
    """
    if generator_type == 'standard':
        return QuantumDataGenerator(x_data, y_data, batch_size, **kwargs)
    elif generator_type == 'balanced':
        return BalancedBatchGenerator(x_data, y_data, batch_size, **kwargs)
    elif generator_type == 'infinite':
        base_gen = QuantumDataGenerator(x_data, y_data, batch_size, **kwargs)
        return InfiniteDataGenerator(base_gen)
    else:
        raise ValueError(
            f"Unsupported generator type: {generator_type}. "
            "Use 'standard', 'balanced', or 'infinite'"
        )
