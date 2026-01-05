"""
Data Preprocessing Example - Demonstrates preprocessing for quantum ML.

This example shows various preprocessing techniques:
- Normalization methods (minmax, zscore, L1, L2, robust)
- Data splitting strategies
- Feature scaling for quantum compatibility
"""

import numpy as np
from q_store.data.preprocessing import (
    QuantumPreprocessor,
    DataSplitter,
    NormalizationMethod
)


def example_minmax_normalization():
    """Min-max normalization to [0, 1]."""
    print("\n" + "="*70)
    print("Example 1: Min-Max Normalization")
    print("="*70)

    # Random data with varying ranges
    data = np.random.randn(100, 10) * 10 + 5

    preprocessor = QuantumPreprocessor(
        method=NormalizationMethod.MINMAX,
        feature_range=(0, 1)
    )

    normalized = preprocessor.fit_transform(data)

    print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Target range: {preprocessor.feature_range}")
    print("✓ Min-max normalization successful")

    return normalized


def example_zscore_standardization():
    """Z-score standardization (mean=0, std=1)."""
    print("\n" + "="*70)
    print("Example 2: Z-Score Standardization")
    print("="*70)

    data = np.random.randn(100, 10) * 5 + 10

    preprocessor = QuantumPreprocessor(method=NormalizationMethod.ZSCORE)
    normalized = preprocessor.fit_transform(data)

    print(f"Original mean: {data.mean():.3f}, std: {data.std():.3f}")
    print(f"Normalized mean: {normalized.mean():.3f}, std: {normalized.std():.3f}")
    print("✓ Z-score standardization successful")

    return normalized


def example_l2_normalization():
    """L2 normalization (unit norm)."""
    print("\n" + "="*70)
    print("Example 3: L2 Normalization")
    print("="*70)

    data = np.random.randn(100, 10)

    preprocessor = QuantumPreprocessor(method=NormalizationMethod.L2)
    normalized = preprocessor.fit_transform(data)

    # Check norms
    norms = np.linalg.norm(normalized, axis=1)
    print(f"Sample norms (should be ~1.0):")
    for i in range(5):
        print(f"  Sample {i}: {norms[i]:.6f}")

    print("✓ L2 normalization successful")

    return normalized


def example_robust_scaling():
    """Robust scaling using median and IQR."""
    print("\n" + "="*70)
    print("Example 4: Robust Scaling")
    print("="*70)

    # Data with outliers
    data = np.random.randn(100, 10)
    data[0, :] = 100  # Add outlier

    preprocessor = QuantumPreprocessor(method=NormalizationMethod.ROBUST)
    normalized = preprocessor.fit_transform(data)

    print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Original median: {np.median(data):.3f}")
    print(f"Normalized median: {np.median(normalized):.3f}")
    print("✓ Robust scaling successful (outliers handled)")

    return normalized


def example_basic_data_splitting():
    """Basic train/test split."""
    print("\n" + "="*70)
    print("Example 5: Basic Data Splitting")
    print("="*70)

    x_data = np.random.randn(1000, 10)
    y_data = np.random.randint(0, 3, 1000)

    splitter = DataSplitter(split_ratio={'train': 0.8, 'test': 0.2})
    splits = splitter.split(x_data, y_data)

    print(f"Total samples: {len(x_data)}")
    print(f"Train samples: {len(splits['x_train'])}")
    print(f"Test samples: {len(splits['x_test'])}")
    print(f"Split ratio: {len(splits['x_train']) / len(x_data):.2f}")
    print("✓ Data splitting successful")

    return splits


def example_train_val_test_split():
    """Train/validation/test split."""
    print("\n" + "="*70)
    print("Example 6: Train/Validation/Test Split")
    print("="*70)

    x_data = np.random.randn(1000, 10)
    y_data = np.random.randint(0, 3, 1000)

    splitter = DataSplitter(
        split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15},
        shuffle=True,
        random_seed=42
    )

    splits = splitter.split(x_data, y_data)

    print(f"Total samples: {len(x_data)}")
    print(f"Train samples: {len(splits['x_train'])} ({len(splits['x_train'])/len(x_data):.1%})")
    print(f"Validation samples: {len(splits['x_val'])} ({len(splits['x_val'])/len(x_data):.1%})")
    print(f"Test samples: {len(splits['x_test'])} ({len(splits['x_test'])/len(x_data):.1%})")
    print("✓ Train/val/test split successful")

    return splits


def example_stratified_split():
    """Stratified split to maintain class distribution."""
    print("\n" + "="*70)
    print("Example 7: Stratified Split")
    print("="*70)

    # Imbalanced dataset
    x_data = np.random.randn(1000, 10)
    y_data = np.array([0]*700 + [1]*200 + [2]*100)  # Imbalanced

    splitter = DataSplitter(
        split_ratio={'train': 0.8, 'test': 0.2},
        stratify=True,
        random_seed=42
    )

    splits = splitter.split(x_data, y_data)

    # Check class distributions
    print("Original distribution:")
    for cls in range(3):
        count = (y_data == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(y_data):.1%})")

    print("\nTrain distribution:")
    for cls in range(3):
        count = (splits['y_train'] == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(splits['y_train']):.1%})")

    print("\nTest distribution:")
    for cls in range(3):
        count = (splits['y_test'] == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(splits['y_test']):.1%})")

    print("\n✓ Stratified split successful (proportions maintained)")

    return splits


def example_complete_pipeline():
    """Complete preprocessing pipeline."""
    print("\n" + "="*70)
    print("Example 8: Complete Preprocessing Pipeline")
    print("="*70)

    # Simulate raw data
    print("Step 1: Load raw data")
    x_data = np.random.randn(1000, 28*28) * 50 + 100
    y_data = np.random.randint(0, 10, 1000)
    print(f"  Shape: {x_data.shape}")
    print(f"  Range: [{x_data.min():.1f}, {x_data.max():.1f}]")

    # Split data
    print("\nStep 2: Split data (train/val/test)")
    splitter = DataSplitter(
        split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15},
        stratify=True,
        random_seed=42
    )
    splits = splitter.split(x_data, y_data)
    print(f"  Train: {len(splits['x_train'])}")
    print(f"  Val: {len(splits['x_val'])}")
    print(f"  Test: {len(splits['x_test'])}")

    # Normalize data
    print("\nStep 3: Normalize data")
    preprocessor = QuantumPreprocessor(
        method=NormalizationMethod.MINMAX,
        feature_range=(0, 1)
    )

    # Fit on training data
    x_train_norm = preprocessor.fit_transform(splits['x_train'])
    x_val_norm = preprocessor.transform(splits['x_val'])
    x_test_norm = preprocessor.transform(splits['x_test'])

    print(f"  Train range: [{x_train_norm.min():.3f}, {x_train_norm.max():.3f}]")
    print(f"  Val range: [{x_val_norm.min():.3f}, {x_val_norm.max():.3f}]")
    print(f"  Test range: [{x_test_norm.min():.3f}, {x_test_norm.max():.3f}]")

    print("\n✓ Complete pipeline successful")
    print("Data is ready for quantum ML training!")

    return {
        'x_train': x_train_norm,
        'y_train': splits['y_train'],
        'x_val': x_val_norm,
        'y_val': splits['y_val'],
        'x_test': x_test_norm,
        'y_test': splits['y_test']
    }


def example_inverse_transform():
    """Demonstrate inverse transformation."""
    print("\n" + "="*70)
    print("Example 9: Inverse Transform")
    print("="*70)

    data = np.random.randn(100, 10) * 5 + 10

    preprocessor = QuantumPreprocessor(method=NormalizationMethod.MINMAX)

    # Forward transform
    normalized = preprocessor.fit_transform(data)

    # Inverse transform
    reconstructed = preprocessor.inverse_transform(normalized)

    print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    print(f"Reconstruction error: {np.abs(data - reconstructed).mean():.6f}")
    print("✓ Inverse transform successful")

    return reconstructed


def example_feature_clipping():
    """Clip features to valid range."""
    print("\n" + "="*70)
    print("Example 10: Feature Clipping")
    print("="*70)

    # Data with outliers
    data = np.random.randn(100, 10)
    data[0, :] = 10  # Large outlier
    data[1, :] = -10  # Large outlier

    preprocessor = QuantumPreprocessor(
        method=NormalizationMethod.MINMAX,
        clip_range=(-3, 3)  # Clip before normalization
    )

    normalized = preprocessor.fit_transform(data)

    print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Clip range: {preprocessor.clip_range}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print("✓ Feature clipping successful")

    return normalized


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Data Preprocessing Examples")
    print("="*70)

    # Normalization examples
    try:
        example_minmax_normalization()
    except Exception as e:
        print(f"⚠ Min-max example failed: {e}")

    try:
        example_zscore_standardization()
    except Exception as e:
        print(f"⚠ Z-score example failed: {e}")

    try:
        example_l2_normalization()
    except Exception as e:
        print(f"⚠ L2 example failed: {e}")

    try:
        example_robust_scaling()
    except Exception as e:
        print(f"⚠ Robust scaling example failed: {e}")

    # Splitting examples
    try:
        example_basic_data_splitting()
    except Exception as e:
        print(f"⚠ Basic split example failed: {e}")

    try:
        example_train_val_test_split()
    except Exception as e:
        print(f"⚠ Train/val/test split example failed: {e}")

    try:
        example_stratified_split()
    except Exception as e:
        print(f"⚠ Stratified split example failed: {e}")

    # Pipeline examples
    try:
        example_complete_pipeline()
    except Exception as e:
        print(f"⚠ Pipeline example failed: {e}")

    try:
        example_inverse_transform()
    except Exception as e:
        print(f"⚠ Inverse transform example failed: {e}")

    try:
        example_feature_clipping()
    except Exception as e:
        print(f"⚠ Clipping example failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
