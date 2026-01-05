"""
Data Loaders Example - Demonstrates unified dataset loading from multiple sources.

This example shows how to use the DatasetLoader with different data sources:
- Keras datasets
- HuggingFace datasets
- Backend API datasets
- Local files (CSV, NumPy)
"""

import numpy as np
from q_store.data.loaders import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
)


def example_keras_loading():
    """Load Fashion MNIST from Keras."""
    print("\n" + "="*70)
    print("Example 1: Loading Fashion MNIST from Keras")
    print("="*70)

    config = DatasetConfig(
        name='fashion_mnist',
        source=DatasetSource.KERAS,
        source_params={
            'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
        }
    )

    dataset = DatasetLoader.load(config)

    print(f"Dataset: {dataset.name}")
    print(f"Source: {dataset.source}")
    print(f"Samples: {dataset.num_samples}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Features: {dataset.num_features}")
    print(f"Train shape: {dataset.x_train.shape}")
    print(f"Test shape: {dataset.x_test.shape}")
    print(f"Metadata: {dataset.metadata}")

    return dataset


def example_huggingface_loading():
    """Load dataset from HuggingFace."""
    print("\n" + "="*70)
    print("Example 2: Loading MNIST from HuggingFace")
    print("="*70)

    config = DatasetConfig(
        name='mnist',
        source=DatasetSource.HUGGINGFACE,
        source_params={
            'dataset_name': 'mnist',
            'split_train': 'train',
            'split_test': 'test',
            'feature_column': 'image',
            'label_column': 'label'
        }
    )

    try:
        dataset = DatasetLoader.load(config)
        print(f"Dataset: {dataset.name}")
        print(f"Samples: {dataset.num_samples}")
        print(f"Classes: {dataset.num_classes}")
        print("✓ Successfully loaded from HuggingFace")
    except ImportError as e:
        print(f"⚠ HuggingFace datasets not available: {e}")
        print("Install with: pip install datasets")


def example_backend_api_loading():
    """Load dataset from Q-Store Backend API."""
    print("\n" + "="*70)
    print("Example 3: Loading from Backend API")
    print("="*70)

    config = DatasetConfig(
        name='custom_dataset',
        source=DatasetSource.BACKEND_API,
        source_params={
            'base_url': 'http://localhost:8000',
            'api_key': 'your_api_key',
            'dataset_id': 'dataset-uuid-123'
        }
    )

    try:
        dataset = DatasetLoader.load(config)
        print(f"Dataset: {dataset.name}")
        print(f"Samples: {dataset.num_samples}")
        print("✓ Successfully loaded from Backend API")
    except Exception as e:
        print(f"⚠ Backend API not available: {e}")
        print("Make sure the Q-Store Backend is running")


def example_local_files_loading():
    """Load dataset from local files."""
    print("\n" + "="*70)
    print("Example 4: Loading from Local Files")
    print("="*70)

    # Create sample data
    x_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    x_test = np.random.randn(20, 10)
    y_test = np.random.randint(0, 2, 20)

    # Save to files
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

    config = DatasetConfig(
        name='local_dataset',
        source=DatasetSource.LOCAL_FILES,
        source_params={
            'x_train_path': 'x_train.npy',
            'y_train_path': 'y_train.npy',
            'x_test_path': 'x_test.npy',
            'y_test_path': 'y_test.npy'
        }
    )

    dataset = DatasetLoader.load(config)

    print(f"Dataset: {dataset.name}")
    print(f"Train samples: {len(dataset.x_train)}")
    print(f"Test samples: {len(dataset.x_test)}")
    print("✓ Successfully loaded from local files")

    # Cleanup
    import os
    for f in ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']:
        if os.path.exists(f):
            os.remove(f)

    return dataset


def example_with_preprocessing():
    """Load dataset with automatic preprocessing."""
    print("\n" + "="*70)
    print("Example 5: Loading with Preprocessing")
    print("="*70)

    config = DatasetConfig(
        name='fashion_mnist',
        source=DatasetSource.KERAS,
        source_params={
            'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
        },
        preprocessing={
            'normalize': True,
            'flatten': True,
            'reshape_to': None  # Keep flattened
        }
    )

    dataset = DatasetLoader.load(config)

    print(f"Dataset: {dataset.name}")
    print(f"Original shape: (28, 28)")
    print(f"Processed shape: {dataset.x_train.shape}")
    print(f"Data range: [{dataset.x_train.min():.3f}, {dataset.x_train.max():.3f}]")
    print("✓ Preprocessing applied successfully")

    return dataset


def example_with_validation_split():
    """Load dataset with automatic validation split."""
    print("\n" + "="*70)
    print("Example 6: Loading with Validation Split")
    print("="*70)

    config = DatasetConfig(
        name='fashion_mnist',
        source=DatasetSource.KERAS,
        source_params={
            'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
        },
        split_validation=0.2  # 20% of training data for validation
    )

    dataset = DatasetLoader.load(config)

    print(f"Dataset: {dataset.name}")
    print(f"Train samples: {len(dataset.x_train)}")
    print(f"Validation samples: {len(dataset.x_val)}")
    print(f"Test samples: {len(dataset.x_test)}")
    print(f"Split ratio: {len(dataset.x_val) / len(dataset.x_train):.2f}")
    print("✓ Validation split created successfully")

    return dataset


def example_dataset_registry():
    """Use dataset registry for common datasets."""
    print("\n" + "="*70)
    print("Example 7: Using Dataset Registry")
    print("="*70)

    # Register a dataset configuration
    DatasetLoader.register(
        'my_fashion_mnist',
        DatasetConfig(
            name='fashion_mnist',
            source=DatasetSource.KERAS,
            source_params={
                'dataset_module': 'tensorflow.keras.datasets.fashion_mnist'
            },
            preprocessing={'normalize': True, 'flatten': True}
        )
    )

    # Load by name
    dataset = DatasetLoader.load_by_name('my_fashion_mnist')

    print(f"Dataset: {dataset.name}")
    print(f"Loaded from registry: my_fashion_mnist")
    print("✓ Registry loading successful")

    # List registered datasets
    registered = DatasetLoader.list_registered()
    print(f"Registered datasets: {registered}")

    return dataset


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Data Loaders Examples")
    print("="*70)

    # Example 1: Keras loading
    try:
        dataset1 = example_keras_loading()
    except Exception as e:
        print(f"⚠ Keras example failed: {e}")

    # Example 2: HuggingFace loading
    example_huggingface_loading()

    # Example 3: Backend API loading
    example_backend_api_loading()

    # Example 4: Local files loading
    try:
        dataset4 = example_local_files_loading()
    except Exception as e:
        print(f"⚠ Local files example failed: {e}")

    # Example 5: With preprocessing
    try:
        dataset5 = example_with_preprocessing()
    except Exception as e:
        print(f"⚠ Preprocessing example failed: {e}")

    # Example 6: With validation split
    try:
        dataset6 = example_with_validation_split()
    except Exception as e:
        print(f"⚠ Validation split example failed: {e}")

    # Example 7: Dataset registry
    try:
        dataset7 = example_dataset_registry()
    except Exception as e:
        print(f"⚠ Registry example failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
