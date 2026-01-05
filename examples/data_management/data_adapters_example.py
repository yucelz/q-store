"""
Data Adapters Example - Demonstrates quantum data preparation and adaptation.

This example shows how to use adapters to prepare classical data for quantum processing:
- Dimension reduction (PCA, SVD, feature selection)
- Quantum encoding preparation
- Image adaptation for quantum circuits
"""

import numpy as np
from q_store.data.adapters import (
    QuantumDataAdapter,
    DimensionReducer,
    QuantumImageAdapter,
    EncodingType
)


def example_dimension_reduction_pca():
    """Reduce dimensions using PCA."""
    print("\n" + "="*70)
    print("Example 1: Dimension Reduction with PCA")
    print("="*70)

    # High-dimensional data (100 samples, 784 features)
    data = np.random.randn(100, 784)

    # Reduce to 8 dimensions for 8-qubit encoding
    reducer = DimensionReducer(method='pca', target_dim=8)
    reduced_data = reducer.fit_transform(data)

    print(f"Original shape: {data.shape}")
    print(f"Reduced shape: {reduced_data.shape}")
    print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
    print("✓ PCA reduction successful")

    return reduced_data


def example_dimension_reduction_svd():
    """Reduce dimensions using SVD."""
    print("\n" + "="*70)
    print("Example 2: Dimension Reduction with SVD")
    print("="*70)

    data = np.random.randn(100, 784)

    reducer = DimensionReducer(method='svd', target_dim=16)
    reduced_data = reducer.fit_transform(data)

    print(f"Original shape: {data.shape}")
    print(f"Reduced shape: {reduced_data.shape}")
    print(f"Method: {reducer.method}")
    print("✓ SVD reduction successful")

    return reduced_data


def example_quantum_adapter_amplitude():
    """Prepare data for amplitude encoding."""
    print("\n" + "="*70)
    print("Example 3: Quantum Adapter - Amplitude Encoding")
    print("="*70)

    # Data for 4-qubit amplitude encoding (2^4 = 16 features)
    data = np.random.randn(50, 16)

    adapter = QuantumDataAdapter(n_qubits=4, encoding=EncodingType.AMPLITUDE)
    quantum_data = adapter.prepare(data)

    print(f"Input shape: {data.shape}")
    print(f"Quantum shape: {quantum_data.shape}")
    print(f"Encoding: {adapter.encoding.value}")
    print(f"Normalized: {adapter.is_normalized(quantum_data)}")

    # Verify amplitude encoding requirements
    for i in range(5):
        norm = np.linalg.norm(quantum_data[i])
        print(f"Sample {i} norm: {norm:.6f}")

    print("✓ Amplitude encoding preparation successful")

    return quantum_data


def example_quantum_adapter_angle():
    """Prepare data for angle encoding."""
    print("\n" + "="*70)
    print("Example 4: Quantum Adapter - Angle Encoding")
    print("="*70)

    # Data for 8-qubit angle encoding (8 features)
    data = np.random.randn(50, 8)

    adapter = QuantumDataAdapter(n_qubits=8, encoding=EncodingType.ANGLE)
    quantum_data = adapter.prepare(data)

    print(f"Input shape: {data.shape}")
    print(f"Quantum shape: {quantum_data.shape}")
    print(f"Encoding: {adapter.encoding.value}")
    print(f"Data range: [{quantum_data.min():.3f}, {quantum_data.max():.3f}]")
    print("✓ Angle encoding preparation successful")

    return quantum_data


def example_image_adapter():
    """Adapt images for quantum processing."""
    print("\n" + "="*70)
    print("Example 5: Quantum Image Adapter")
    print("="*70)

    # Simulate image data (10 images, 28x28 pixels)
    images = np.random.rand(10, 28, 28)

    # Adapt for 8-qubit quantum circuit
    adapter = QuantumImageAdapter(
        n_qubits=8,
        reduction_method='pca',
        encoding=EncodingType.AMPLITUDE
    )

    quantum_images = adapter.prepare(images)

    print(f"Original images shape: {images.shape}")
    print(f"Quantum images shape: {quantum_images.shape}")
    print(f"Reduction method: {adapter.reduction_method}")
    print(f"Encoding: {adapter.encoding.value}")
    print("✓ Image adaptation successful")

    return quantum_images


def example_end_to_end_pipeline():
    """Complete pipeline: load, reduce, adapt."""
    print("\n" + "="*70)
    print("Example 6: End-to-End Data Preparation Pipeline")
    print("="*70)

    # Simulate Fashion MNIST-like data
    print("Step 1: Load data")
    x_train = np.random.rand(1000, 28, 28)
    y_train = np.random.randint(0, 10, 1000)
    print(f"  Loaded: {x_train.shape}")

    # Flatten images
    print("\nStep 2: Flatten images")
    x_flat = x_train.reshape(x_train.shape[0], -1)
    print(f"  Flattened: {x_flat.shape}")

    # Reduce dimensions
    print("\nStep 3: Reduce dimensions")
    reducer = DimensionReducer(method='pca', target_dim=16)
    x_reduced = reducer.fit_transform(x_flat)
    print(f"  Reduced: {x_reduced.shape}")
    print(f"  Variance retained: {reducer.explained_variance_ratio_.sum():.3f}")

    # Prepare for quantum encoding
    print("\nStep 4: Prepare for quantum encoding")
    adapter = QuantumDataAdapter(n_qubits=4, encoding=EncodingType.AMPLITUDE)
    x_quantum = adapter.prepare(x_reduced)
    print(f"  Quantum-ready: {x_quantum.shape}")
    print(f"  Encoding: {adapter.encoding.value}")

    print("\n✓ Complete pipeline successful")
    print(f"Final data ready for {adapter.n_qubits}-qubit quantum circuit")

    return x_quantum, y_train


def example_custom_encoding():
    """Use custom encoding with adapter."""
    print("\n" + "="*70)
    print("Example 7: Custom Encoding Function")
    print("="*70)

    data = np.random.randn(50, 8)

    # Define custom encoding function
    def custom_encoder(x):
        """Custom encoding: scale to [0, 2π] and apply sin."""
        scaled = (x - x.min()) / (x.max() - x.min()) * 2 * np.pi
        return np.sin(scaled)

    adapter = QuantumDataAdapter(
        n_qubits=8,
        encoding=EncodingType.ANGLE,
        custom_encoding_fn=custom_encoder
    )

    quantum_data = adapter.prepare(data)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {quantum_data.shape}")
    print(f"Output range: [{quantum_data.min():.3f}, {quantum_data.max():.3f}]")
    print("✓ Custom encoding successful")

    return quantum_data


def example_validation_and_checks():
    """Demonstrate data validation features."""
    print("\n" + "="*70)
    print("Example 8: Data Validation")
    print("="*70)

    # Valid data
    valid_data = np.random.randn(50, 16)
    adapter = QuantumDataAdapter(n_qubits=4, encoding=EncodingType.AMPLITUDE)

    print("Testing valid data:")
    is_valid, message = adapter.validate(valid_data)
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    # Invalid data (wrong dimensions)
    print("\nTesting invalid data (wrong shape):")
    invalid_data = np.random.randn(50, 15)  # Should be 16 for 4 qubits
    is_valid, message = adapter.validate(invalid_data)
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    # Data with NaN
    print("\nTesting data with NaN:")
    nan_data = valid_data.copy()
    nan_data[0, 0] = np.nan
    is_valid, message = adapter.validate(nan_data)
    print(f"  Valid: {is_valid}")
    print(f"  Message: {message}")

    print("\n✓ Validation checks complete")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Data Adapters Examples")
    print("="*70)

    # Example 1: PCA reduction
    try:
        reduced_pca = example_dimension_reduction_pca()
    except Exception as e:
        print(f"⚠ PCA example failed: {e}")

    # Example 2: SVD reduction
    try:
        reduced_svd = example_dimension_reduction_svd()
    except Exception as e:
        print(f"⚠ SVD example failed: {e}")

    # Example 3: Amplitude encoding
    try:
        quantum_amp = example_quantum_adapter_amplitude()
    except Exception as e:
        print(f"⚠ Amplitude encoding example failed: {e}")

    # Example 4: Angle encoding
    try:
        quantum_angle = example_quantum_adapter_angle()
    except Exception as e:
        print(f"⚠ Angle encoding example failed: {e}")

    # Example 5: Image adapter
    try:
        quantum_images = example_image_adapter()
    except Exception as e:
        print(f"⚠ Image adapter example failed: {e}")

    # Example 6: End-to-end pipeline
    try:
        x_quantum, y_train = example_end_to_end_pipeline()
    except Exception as e:
        print(f"⚠ Pipeline example failed: {e}")

    # Example 7: Custom encoding
    try:
        custom_quantum = example_custom_encoding()
    except Exception as e:
        print(f"⚠ Custom encoding example failed: {e}")

    # Example 8: Validation
    try:
        example_validation_and_checks()
    except Exception as e:
        print(f"⚠ Validation example failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
