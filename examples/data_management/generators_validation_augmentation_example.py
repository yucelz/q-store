"""
Data Generators, Validation, and Augmentation Examples.

This example demonstrates:
- Data generators for efficient batch processing
- Data validation and profiling
- Data augmentation techniques
"""

import numpy as np
from q_store.data.generators import (
    QuantumDataGenerator,
    StreamingDataGenerator,
    InfiniteDataGenerator
)
from q_store.data.validation import (
    DataValidator,
    DataProfiler,
    OutlierDetector
)
from q_store.data.augmentation import (
    QuantumAugmentation,
    ClassicalAugmentation,
    HybridAugmentation
)


def example_quantum_data_generator():
    """Basic data generator for batching."""
    print("\n" + "="*70)
    print("Example 1: Quantum Data Generator")
    print("="*70)

    # Create sample data
    x_data = np.random.randn(1000, 16)
    y_data = np.random.randint(0, 3, 1000)

    # Create generator
    generator = QuantumDataGenerator(
        x_data, y_data,
        batch_size=32,
        shuffle=True,
        random_seed=42
    )

    print(f"Total samples: {len(x_data)}")
    print(f"Batch size: {generator.batch_size}")
    print(f"Number of batches: {len(generator)}")

    # Iterate through batches
    print("\nFirst 3 batches:")
    for i, (x_batch, y_batch) in enumerate(generator):
        print(f"  Batch {i}: X shape {x_batch.shape}, Y shape {y_batch.shape}")
        if i >= 2:
            break

    print("✓ Data generator successful")

    return generator


def example_generator_with_augmentation():
    """Generator with on-the-fly augmentation."""
    print("\n" + "="*70)
    print("Example 2: Generator with Augmentation")
    print("="*70)

    x_data = np.random.randn(500, 16)
    y_data = np.random.randint(0, 2, 500)

    # Define augmentation function
    def augment_fn(x):
        noise = np.random.normal(0, 0.1, x.shape)
        return x + noise

    generator = QuantumDataGenerator(
        x_data, y_data,
        batch_size=32,
        shuffle=True,
        augmentation=augment_fn
    )

    # Get batch with augmentation
    x_batch, y_batch = next(iter(generator))

    print(f"Batch shape: {x_batch.shape}")
    print(f"Augmentation applied on-the-fly")
    print("✓ Augmentation generator successful")

    return generator


def example_streaming_generator():
    """Streaming generator for large datasets."""
    print("\n" + "="*70)
    print("Example 3: Streaming Data Generator")
    print("="*70)

    # Simulate large dataset stored in files
    x_data = np.random.randn(10000, 16)
    y_data = np.random.randint(0, 5, 10000)

    # Save to npz file (StreamingDataGenerator expects a file path)
    np.savez('large_data.npz', x=x_data, y=y_data)

    # Create streaming generator
    generator = StreamingDataGenerator(
        data_path='large_data.npz',
        batch_size=64,
        chunk_size=1000,
        shuffle=False,
        data_key='x',
        labels_key='y'
    )

    print(f"Streaming from file: large_data.npz")
    print(f"Batch size: {generator.batch_size}")
    print(f"Chunk size: {generator.chunk_size}")
    print(f"Total samples: {generator.n_samples}")
    print(f"Memory-efficient loading")

    # Get first batch
    x_batch, y_batch = next(iter(generator))
    print(f"Batch shape: {x_batch.shape}")
    print("✓ Streaming generator successful")

    # Cleanup
    import os
    os.remove('large_data.npz')


def example_data_validation():
    """Validate data for quantum ML."""
    print("\n" + "="*70)
    print("Example 4: Data Validation")
    print("="*70)

    validator = DataValidator()

    # Valid data
    x_valid = np.random.randn(100, 16)
    y_valid = np.random.randint(0, 3, 100)

    print("Testing valid data:")
    is_valid, message = validator.validate_all(
        x_valid, y_valid,
        n_qubits=4,
        encoding='amplitude'
    )
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    # Invalid data (NaN)
    x_invalid = x_valid.copy()
    x_invalid[0, 0] = np.nan

    print("\nTesting data with NaN:")
    is_valid, message = validator.validate_all(
        x_invalid, y_valid,
        n_qubits=4
    )
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    # Invalid data (wrong shape)
    x_wrong_shape = np.random.randn(100, 15)  # Should be 16

    print("\nTesting data with wrong shape:")
    is_valid, message = validator.validate_all(
        x_wrong_shape, y_valid,
        n_qubits=4,
        encoding='amplitude'
    )
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    print("\n✓ Data validation successful")


def example_data_profiling():
    """Profile dataset characteristics."""
    print("\n" + "="*70)
    print("Example 5: Data Profiling")
    print("="*70)

    # Create sample data
    x_data = np.random.randn(1000, 16)
    y_data = np.array([0]*400 + [1]*350 + [2]*250)  # Imbalanced

    profiler = DataProfiler()
    profile = profiler.profile(x_data, y_data)

    print("Dataset Profile:")
    stats = profile['statistics']
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Classes: {profile.get('num_classes', 'N/A')}")
    print(f"\nFeature Statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    
    if 'class_distribution' in profile:
        print(f"\nClass Distribution:")
        for cls, count in profile['class_distribution'].items():
            print(f"  Class {cls}: {count} ({count/stats['n_samples']:.1%})")
    
    print(f"\nData Quality:")
    quality = profile['data_quality']
    print(f"  Has NaN: {quality['has_nan']}")
    print(f"  Has Inf: {quality['has_inf']}")
    
    if 'outliers' in profile:
        print(f"\nOutliers:")
        print(f"  IQR method: {profile['outliers']['iqr_method']} ({profile['outliers']['iqr_percentage']:.1f}%)")
    
    print("\n✓ Data profiling successful")


def example_outlier_detection():
    """Detect and handle outliers."""
    print("\n" + "="*70)
    print("Example 6: Outlier Detection")
    print("="*70)

    # Data with outliers
    x_data = np.random.randn(100, 10)
    x_data[0, :] = 10  # Outlier
    x_data[1, :] = -10  # Outlier

    detector = OutlierDetector(method='zscore', threshold=3.0, action='flag')

    # Fit and detect outliers
    detector.fit(x_data)

    print(f"Total samples: {len(x_data)}")
    print(f"Outliers detected: {detector.n_outliers_}")
    print(f"Outlier indices: {np.where(detector.outlier_mask_)[0]}")

    # Remove outliers using action='remove'
    detector_remove = OutlierDetector(method='zscore', threshold=3.0, action='remove')
    x_cleaned = detector_remove.fit_transform(x_data)

    print(f"Cleaned data shape: {x_cleaned.shape}")
    print(f"Removed {len(x_data) - len(x_cleaned)} outliers")
    print("✓ Outlier detection successful")

    return x_cleaned


def example_quantum_augmentation():
    """Apply quantum-specific augmentation."""
    print("\n" + "="*70)
    print("Example 7: Quantum Augmentation")
    print("="*70)

    # Quantum-ready data
    x_data = np.random.randn(100, 16)

    # Normalize to unit norm (for amplitude encoding)
    x_data = x_data / np.linalg.norm(x_data, axis=1, keepdims=True)

    aug = QuantumAugmentation(
        phase_shift_range=0.1,
        amplitude_noise=0.01,
        probability=0.5,
        random_seed=42
    )

    # Apply augmentation
    x_augmented = aug.apply(x_data)

    print(f"Original shape: {x_data.shape}")
    print(f"Augmented shape: {x_augmented.shape}")
    print(f"Phase shift range: {aug.phase_shift_range}")
    print(f"Amplitude noise: {aug.amplitude_noise}")

    # Check norms are preserved
    norms_orig = np.linalg.norm(x_data, axis=1)
    norms_aug = np.linalg.norm(x_augmented, axis=1)
    print(f"Mean norm difference: {np.abs(norms_orig - norms_aug).mean():.6f}")

    print("✓ Quantum augmentation successful")

    return x_augmented


def example_classical_augmentation():
    """Apply classical image augmentation."""
    print("\n" + "="*70)
    print("Example 8: Classical Augmentation")
    print("="*70)

    # Image data
    images = np.random.rand(50, 28, 28, 1)

    try:
        aug = ClassicalAugmentation(
            transforms=[
                {'type': 'horizontal_flip', 'p': 0.5},
                {'type': 'rotate', 'limit': 15, 'p': 0.3},
                {'type': 'random_brightness_contrast', 'p': 0.5}
            ]
        )

        # Apply augmentation
        images_augmented = aug.apply(images)

        print(f"Original shape: {images.shape}")
        print(f"Augmented shape: {images_augmented.shape}")
        print(f"Transforms: {len(aug.transforms)}")
        print("✓ Classical augmentation successful")

        return images_augmented
    except ImportError:
        print("⚠ Albumentations not available")
        print("Install with: pip install albumentations")


def example_hybrid_augmentation():
    """Combine classical and quantum augmentation."""
    print("\n" + "="*70)
    print("Example 9: Hybrid Augmentation")
    print("="*70)

    # Image data
    images = np.random.rand(50, 28, 28, 1)

    try:
        aug = HybridAugmentation(
            classical_transforms=[
                {'type': 'horizontal_flip', 'p': 0.5},
                {'type': 'rotate', 'limit': 10, 'p': 0.3}
            ],
            quantum_config={
                'phase_shift_range': 0.1,
                'amplitude_noise': 0.01
            }
        )

        # Apply hybrid augmentation
        images_augmented = aug.apply(images)

        print(f"Original shape: {images.shape}")
        print(f"Augmented shape: {images_augmented.shape}")
        print(f"Classical transforms: {len(aug.classical_aug.transforms) if aug.classical_aug else 0}")
        print(f"Quantum augmentation: {aug.quantum_aug is not None}")
        print("✓ Hybrid augmentation successful")

        return images_augmented
    except ImportError:
        print("⚠ Albumentations not available")
        print("Install with: pip install albumentations")


def example_complete_pipeline():
    """Complete data preparation pipeline."""
    print("\n" + "="*70)
    print("Example 10: Complete Data Pipeline")
    print("="*70)

    # Step 1: Load data
    print("Step 1: Load data")
    x_data = np.random.randn(1000, 16)
    y_data = np.random.randint(0, 3, 1000)
    print(f"  Loaded: {x_data.shape}")

    # Step 2: Validate data
    print("\nStep 2: Validate data")
    validator = DataValidator()
    is_valid, message = validator.validate_all(x_data, y_data, n_qubits=4)
    print(f"  Valid: {is_valid}")
    if not is_valid:
        print(f"  Issue: {message}")
        return

    # Step 3: Profile data
    print("\nStep 3: Profile data")
    profiler = DataProfiler()
    profile = profiler.profile(x_data, y_data)
    print(f"  Samples: {profile['statistics']['n_samples']}")
    if 'class_distribution' in profile:
        # Check if balanced (simple check: all classes within 20% of each other)
        dist = profile['class_distribution']
        counts = list(dist.values())
        is_balanced = (max(counts) / min(counts)) < 1.5 if counts else True
        print(f"  Balanced: {is_balanced}")

    # Step 4: Detect outliers
    print("\nStep 4: Detect outliers")
    detector = OutlierDetector(method='zscore', threshold=3.0, action='flag')
    detector.fit(x_data)
    print(f"  Outliers: {detector.n_outliers_}")
    if detector.n_outliers_ > 0:
        detector_remove = OutlierDetector(method='zscore', threshold=3.0, action='remove')
        x_data = detector_remove.fit_transform(x_data)
        y_data = y_data[~detector.outlier_mask_]
        print(f"  Cleaned: {x_data.shape}")

    # Step 5: Augment data
    print("\nStep 5: Augment data")
    aug = QuantumAugmentation(phase_shift_range=0.1, amplitude_noise=0.01)
    x_augmented = aug.apply(x_data)
    print(f"  Augmented: {x_augmented.shape}")

    # Step 6: Create generator
    print("\nStep 6: Create data generator")
    generator = QuantumDataGenerator(
        x_augmented, y_data,
        batch_size=32,
        shuffle=True
    )
    print(f"  Batches: {len(generator)}")

    print("\n✓ Complete pipeline successful!")
    print("Data is ready for quantum ML training")

    return generator


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Data Generators, Validation & Augmentation Examples")
    print("="*70)

    # Generators
    try:
        example_quantum_data_generator()
    except Exception as e:
        print(f"⚠ Generator example failed: {e}")

    try:
        example_generator_with_augmentation()
    except Exception as e:
        print(f"⚠ Augmentation generator failed: {e}")

    try:
        example_streaming_generator()
    except Exception as e:
        print(f"⚠ Streaming generator failed: {e}")

    # Validation
    try:
        example_data_validation()
    except Exception as e:
        print(f"⚠ Validation example failed: {e}")

    try:
        example_data_profiling()
    except Exception as e:
        print(f"⚠ Profiling example failed: {e}")

    try:
        example_outlier_detection()
    except Exception as e:
        print(f"⚠ Outlier detection failed: {e}")

    # Augmentation
    try:
        example_quantum_augmentation()
    except Exception as e:
        print(f"⚠ Quantum augmentation failed: {e}")

    example_classical_augmentation()
    example_hybrid_augmentation()

    # Complete pipeline
    try:
        example_complete_pipeline()
    except Exception as e:
        print(f"⚠ Complete pipeline failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
