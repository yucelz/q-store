"""
Unit tests for Q-Store v4.1.1 Data Management Layer.

Tests cover:
- Dataset loaders with all source adapters
- Quantum data adapters
- Preprocessing utilities
- Data generators
- Data validation
- Data augmentation
- Backend API client
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from q_store.data import (
    DatasetLoader,
    DatasetConfig,
    DatasetSource,
    Dataset,
    SourceAdapter,
    KerasSourceAdapter,
    HuggingFaceSourceAdapter,
    BackendAPISourceAdapter,
    LocalFilesSourceAdapter,
    QuantumDataAdapter,
    DimensionReducer,
    QuantumImageAdapter,
    QuantumPreprocessor,
    DataSplitter,
    QuantumDataGenerator,
    StreamingDataGenerator,
    DataValidator,
    DataProfiler,
    QuantumAugmentation,
    ClassicalAugmentation,
    HybridAugmentation,
    BackendAPIClient,
)


class TestDatasetLoader:
    """Test DatasetLoader and adapter registration."""

    def test_adapter_registration(self):
        """Test that adapters are registered correctly."""
        # Check that default adapters are registered
        assert DatasetSource.KERAS in DatasetLoader._adapters
        assert DatasetSource.HUGGINGFACE in DatasetLoader._adapters
        assert DatasetSource.BACKEND_API in DatasetLoader._adapters
        assert DatasetSource.LOCAL_FILES in DatasetLoader._adapters

    def test_custom_adapter_registration(self):
        """Test registering a custom adapter."""
        class CustomAdapter(SourceAdapter):
            def load(self, config, cache_dir):
                return Dataset(
                    name="custom",
                    x_train=np.array([1, 2, 3]),
                    y_train=np.array([0, 1, 0]),
                    x_test=np.array([4, 5]),
                    y_test=np.array([1, 0])
                )

            def list_datasets(self):
                return []

        custom_source = DatasetSource("custom")
        adapter = CustomAdapter()
        DatasetLoader.register_adapter(custom_source, adapter)

        assert custom_source in DatasetLoader._adapters

    def test_dataset_config_creation(self):
        """Test DatasetConfig creation."""
        config = DatasetConfig(
            name="test_dataset",
            source=DatasetSource.KERAS,
            source_params={"dataset_module": "test.module"}
        )

        assert config.name == "test_dataset"
        assert config.source == DatasetSource.KERAS
        assert config.source_params["dataset_module"] == "test.module"


class TestKerasSourceAdapter:
    """Test Keras dataset source adapter."""

    @patch('q_store.data.loaders.importlib.import_module')
    def test_keras_adapter_load(self, mock_import):
        """Test loading from Keras datasets."""
        # Mock Keras dataset
        mock_dataset = Mock()
        mock_dataset.load_data.return_value = (
            (np.random.rand(100, 28, 28), np.random.randint(0, 10, 100)),
            (np.random.rand(20, 28, 28), np.random.randint(0, 10, 20))
        )
        mock_import.return_value = mock_dataset

        adapter = KerasSourceAdapter()
        config = DatasetConfig(
            name="fashion_mnist",
            source=DatasetSource.KERAS,
            source_params={"dataset_module": "tensorflow.keras.datasets.fashion_mnist"}
        )

        dataset = adapter.load(config, cache_dir=None)

        assert dataset.name == "fashion_mnist"
        assert dataset.x_train.shape == (100, 28, 28)
        assert dataset.y_train.shape == (100,)
        assert dataset.x_test.shape == (20, 28, 28)
        assert dataset.y_test.shape == (20,)

    def test_keras_adapter_list_datasets(self):
        """Test listing available Keras datasets."""
        adapter = KerasSourceAdapter()
        datasets = adapter.list_datasets()

        assert len(datasets) > 0
        assert any(d['name'] == 'mnist' for d in datasets)
        assert any(d['name'] == 'fashion_mnist' for d in datasets)


class TestLocalFilesSourceAdapter:
    """Test local files source adapter."""

    def test_load_numpy_files(self):
        """Test loading from NumPy files."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            x_train = np.random.rand(100, 10)
            y_train = np.random.randint(0, 2, 100)
            x_test = np.random.rand(20, 10)
            y_test = np.random.randint(0, 2, 20)

            # Save to files
            np.save(f"{tmpdir}/x_train.npy", x_train)
            np.save(f"{tmpdir}/y_train.npy", y_train)
            np.save(f"{tmpdir}/x_test.npy", x_test)
            np.save(f"{tmpdir}/y_test.npy", y_test)

            adapter = LocalFilesSourceAdapter()
            config = DatasetConfig(
                name="custom_dataset",
                source=DatasetSource.LOCAL_FILES,
                source_params={
                    'format': 'numpy',
                    'train_data': f"{tmpdir}/x_train.npy",
                    'train_labels': f"{tmpdir}/y_train.npy",
                    'test_data': f"{tmpdir}/x_test.npy",
                    'test_labels': f"{tmpdir}/y_test.npy"
                }
            )

            dataset = adapter.load(config, cache_dir=None)

            assert dataset.name == "custom_dataset"
            assert dataset.x_train.shape == (100, 10)
            assert dataset.y_train.shape == (100,)
            np.testing.assert_array_equal(dataset.x_train, x_train)


class TestDataset:
    """Test Dataset container class."""

    def test_dataset_creation(self):
        """Test creating a Dataset object."""
        dataset = Dataset(
            name="test",
            x_train=np.random.rand(100, 10),
            y_train=np.random.randint(0, 2, 100),
            x_test=np.random.rand(20, 10),
            y_test=np.random.randint(0, 2, 20)
        )

        assert dataset.name == "test"
        assert dataset.num_samples == 120
        assert dataset.num_classes == 2
        assert dataset.input_shape == (10,)

    def test_dataset_with_validation_split(self):
        """Test dataset with validation split."""
        dataset = Dataset(
            name="test",
            x_train=np.random.rand(100, 10),
            y_train=np.random.randint(0, 2, 100),
            x_val=np.random.rand(20, 10),
            y_val=np.random.randint(0, 2, 20),
            x_test=np.random.rand(20, 10),
            y_test=np.random.randint(0, 2, 20)
        )

        assert dataset.x_val is not None
        assert dataset.y_val is not None

    def test_get_split(self):
        """Test getting specific data splits."""
        dataset = Dataset(
            name="test",
            x_train=np.random.rand(100, 10),
            y_train=np.random.randint(0, 2, 100),
            x_test=np.random.rand(20, 10),
            y_test=np.random.randint(0, 2, 20)
        )

        x, y = dataset.get_split('train')
        assert x.shape == (100, 10)
        assert y.shape == (100,)

        x, y = dataset.get_split('test')
        assert x.shape == (20, 10)
        assert y.shape == (20,)


class TestQuantumDataAdapter:
    """Test quantum data adapters."""

    def test_prepare_for_quantum_amplitude(self):
        """Test amplitude encoding preparation."""
        data = np.random.rand(10, 16)  # 16 features for 4 qubits

        prepared = QuantumDataAdapter.prepare_for_quantum(
            data,
            n_qubits=4,
            encoding='amplitude'
        )

        assert prepared.shape[0] == 10
        # Check normalization for amplitude encoding
        norms = np.linalg.norm(prepared, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10), decimal=5)

    def test_dimension_reducer_pca(self):
        """Test PCA dimension reduction."""
        data = np.random.rand(100, 50)

        reduced = DimensionReducer.pca(data, n_components=10)

        assert reduced.shape == (100, 10)

    def test_quantum_image_adapter(self):
        """Test quantum image adapter."""
        images = np.random.rand(10, 28, 28)

        # Flatten and normalize
        flat = QuantumImageAdapter.flatten_and_normalize(images)
        assert flat.shape == (10, 784)

        # Resize for qubits
        resized = QuantumImageAdapter.resize_for_qubits(images, n_qubits=4)
        assert resized.shape[1] == 16  # 2^4


class TestQuantumPreprocessor:
    """Test quantum preprocessing utilities."""

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        normalized = QuantumPreprocessor.normalize(
            data,
            method='minmax',
            feature_range=(0, 1)
        )

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_standardize(self):
        """Test standardization (z-score)."""
        data = np.random.rand(100, 10) * 10 + 5

        standardized = QuantumPreprocessor.standardize(data)

        # Check mean ≈ 0 and std ≈ 1
        assert abs(standardized.mean()) < 0.1
        assert abs(standardized.std() - 1.0) < 0.1

    def test_reduce_dimensions(self):
        """Test dimension reduction."""
        data = np.random.rand(100, 50)

        reduced = QuantumPreprocessor.reduce_dimensions(
            data,
            target_dim=10,
            method='pca'
        )

        assert reduced.shape == (100, 10)

    def test_validate_for_quantum(self):
        """Test quantum validation."""
        # Valid data for amplitude encoding (normalized)
        valid_data = np.random.rand(10, 16)
        valid_data = valid_data / np.linalg.norm(valid_data, axis=1, keepdims=True)

        is_valid = QuantumPreprocessor.validate_for_quantum(
            valid_data,
            n_qubits=4,
            encoding='amplitude'
        )

        assert is_valid


class TestDataSplitter:
    """Test data splitting utilities."""

    def test_train_val_test_split(self):
        """Test train/val/test splitting."""
        x = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        splits = DataSplitter.train_val_test_split(
            x, y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        x_train, y_train = splits['train']
        x_val, y_val = splits['val']
        x_test, y_test = splits['test']

        assert x_train.shape[0] == 70
        assert x_val.shape[0] == 15
        assert x_test.shape[0] == 15

    def test_k_fold_split(self):
        """Test k-fold cross-validation splitting."""
        x = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        folds = DataSplitter.k_fold_split(x, y, n_splits=5)

        assert len(folds) == 5

        for train_data, val_data in folds:
            x_train, y_train = train_data
            x_val, y_val = val_data

            assert x_train.shape[0] == 80
            assert x_val.shape[0] == 20


class TestQuantumDataGenerator:
    """Test quantum data generators."""

    def test_basic_generator(self):
        """Test basic data generator."""
        x = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        generator = QuantumDataGenerator(
            x, y,
            batch_size=10,
            shuffle=False
        )

        assert len(generator) == 10

        # Get first batch
        x_batch, y_batch = generator[0]
        assert x_batch.shape == (10, 10)
        assert y_batch.shape == (10,)

    def test_generator_with_augmentation(self):
        """Test generator with augmentation."""
        x = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        def augment_fn(batch_x):
            return batch_x + np.random.rand(*batch_x.shape) * 0.1

        generator = QuantumDataGenerator(
            x, y,
            batch_size=10,
            augmentation_fn=augment_fn
        )

        x_batch, y_batch = generator[0]
        # Augmented data should be different from original
        assert not np.allclose(x_batch, x[:10])

    def test_generator_shuffle(self):
        """Test generator shuffling."""
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        generator = QuantumDataGenerator(
            x, y,
            batch_size=10,
            shuffle=True,
            random_seed=42
        )

        # Trigger epoch end to shuffle
        generator.on_epoch_end()

        x_batch, _ = generator[0]
        # After shuffling, first batch should not be [0, 1, 2, ...]
        assert not np.array_equal(x_batch.flatten(), np.arange(10))


class TestStreamingDataGenerator:
    """Test streaming data generator."""

    def test_streaming_generator_basic(self):
        """Test streaming generator with data function."""
        def data_fn():
            for i in range(10):
                yield np.random.rand(10, 5), np.random.randint(0, 2, 10)

        generator = StreamingDataGenerator(
            data_fn=data_fn,
            batch_size=10
        )

        batches = list(generator)
        assert len(batches) == 10


class TestDataValidator:
    """Test data validation."""

    def test_validate_shape(self):
        """Test shape validation."""
        data = np.random.rand(100, 10)

        is_valid, message = DataValidator.validate_shape(
            data,
            expected_shape=(None, 10)
        )

        assert is_valid
        assert "valid" in message.lower()

    def test_validate_range(self):
        """Test range validation."""
        data = np.random.rand(100, 10)

        is_valid, message = DataValidator.validate_range(
            data,
            min_val=0.0,
            max_val=1.0
        )

        assert is_valid

    def test_validate_no_nan(self):
        """Test NaN validation."""
        data = np.random.rand(100, 10)

        is_valid, message = DataValidator.validate_no_nan(data)
        assert is_valid

        # Add NaN and test
        data[0, 0] = np.nan
        is_valid, message = DataValidator.validate_no_nan(data)
        assert not is_valid

    def test_validate_quantum_compatible(self):
        """Test quantum compatibility validation."""
        data = np.random.rand(100, 16)  # 16 = 2^4 for 4 qubits

        is_valid, message = DataValidator.validate_quantum_compatible(
            data,
            n_qubits=4,
            encoding='amplitude'
        )

        # Should warn about normalization but structure is valid
        assert "shape" in message.lower() or "valid" in message.lower()


class TestDataProfiler:
    """Test data profiling."""

    def test_compute_statistics(self):
        """Test computing data statistics."""
        data = np.random.rand(100, 10)

        stats = DataProfiler.compute_statistics(data)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'].shape == (10,)

    def test_detect_outliers(self):
        """Test outlier detection."""
        data = np.random.randn(100, 5)
        # Add outliers
        data[0] = 10.0
        data[1] = -10.0

        outliers = DataProfiler.detect_outliers(data, method='zscore', threshold=3.0)

        assert len(outliers) > 0
        assert 0 in outliers or 1 in outliers


class TestQuantumAugmentation:
    """Test quantum data augmentation."""

    def test_phase_shift(self):
        """Test phase shift augmentation."""
        data = np.random.rand(10, 16)

        augmented = QuantumAugmentation.phase_shift(data, shift_range=0.1)

        assert augmented.shape == data.shape
        assert not np.allclose(augmented, data)

    def test_amplitude_noise(self):
        """Test amplitude noise augmentation."""
        data = np.random.rand(10, 16)

        augmented = QuantumAugmentation.amplitude_noise(data, noise_level=0.01)

        assert augmented.shape == data.shape
        assert not np.allclose(augmented, data)

    def test_random_rotation(self):
        """Test random rotation augmentation."""
        data = np.random.rand(10, 16)

        augmented = QuantumAugmentation.random_rotation(data, angle_range=0.1)

        assert augmented.shape == data.shape


class TestBackendAPIClient:
    """Test Backend API client."""

    @patch('q_store.data.backend_client.requests.get')
    def test_list_datasets(self, mock_get):
        """Test listing datasets from backend."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'datasets': [
                {'id': '1', 'name': 'dataset1'},
                {'id': '2', 'name': 'dataset2'}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = BackendAPIClient(base_url="http://test.com", api_key="test_key")
        datasets = client.list_datasets()

        assert len(datasets) == 2
        assert datasets[0]['name'] == 'dataset1'

    @patch('q_store.data.backend_client.requests.get')
    def test_get_dataset_info(self, mock_get):
        """Test getting dataset info."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': '1',
            'name': 'test_dataset',
            'size': 1000
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = BackendAPIClient(base_url="http://test.com", api_key="test_key")
        info = client.get_dataset_info('1')

        assert info['name'] == 'test_dataset'
        assert info['size'] == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
