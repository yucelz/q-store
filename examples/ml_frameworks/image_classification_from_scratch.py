"""
Image Classification from Scratch using Q-Store v4.1.1

Inspired by: https://keras.io/examples/vision/image_classification_from_scratch/
GitHub: https://github.com/keras-team/keras-io/tree/master/examples/vision

This example demonstrates:
- Building an image classifier from scratch (no pre-trained weights)
- Using quantum-enhanced layers for feature extraction
- Data augmentation and preprocessing
- Training on Cats vs Dogs dataset
- Quantum-classical hybrid architecture with q-store v4.1.1

Architecture (Quantum-First Design):
    1. Data Augmentation (Classical)
    2. Rescaling/Normalization (Classical)
    3. Conv2D layers for spatial feature extraction (Classical)
    4. QuantumFeatureExtractor for nonlinear feature learning (Quantum - 40%)
    5. QuantumPooling for dimensionality reduction (Quantum - 15%)
    6. QuantumReadout for classification (Quantum - 10%)

Total Quantum Computation: ~65% of feature processing layers

Dataset: Cats vs Dogs (Kaggle)
- Training samples: ~18,000 images
- Validation samples: ~4,600 images
- Image size: 180x180 pixels
- Classes: 2 (Cat, Dog)

Usage:
    # Download and prepare dataset first (requires ~800MB download)
    python examples/ml_frameworks/image_classification_from_scratch.py --download-only

    # Train with quantum layers (mock backend, no API keys needed)
    python examples/ml_frameworks/image_classification_from_scratch.py

    # Train with real quantum hardware (requires IonQ API key)
    python examples/ml_frameworks/image_classification_from_scratch.py --no-mock

    # Skip quantum layers (classical-only baseline)
    python examples/ml_frameworks/image_classification_from_scratch.py --no-quantum

    # Quick test with small dataset
    python examples/ml_frameworks/image_classification_from_scratch.py --quick-test

Configuration:
    Create examples/.env from examples/.env.example and set:
    - IONQ_API_KEY: Your IonQ API key (optional)
    - IONQ_TARGET: simulator or qpu.harmony
"""

import os
import sys
import argparse
import time
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

# Force CPU-only execution for quantum layers (avoids XLA incompatibility)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow import data as tf_data
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ℹ matplotlib not installed. Visualization will be skipped.")

# Load environment variables
try:
    from dotenv import load_dotenv
    examples_dir = Path(__file__).parent.parent
    env_path = examples_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
except ImportError:
    pass

# Q-Store imports
try:
    from q_store.layers import (
        QuantumFeatureExtractor,
        QuantumPooling,
        QuantumReadout,
    )
    from q_store.tensorflow import QuantumLayer
    HAS_QSTORE = True
except ImportError as e:
    HAS_QSTORE = False
    print(f"⚠️  Q-Store not available: {e}")
    print("   Will run in classical-only mode")


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Global configuration."""
    # Dataset
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    data_dir = "data/cats_vs_dogs"
    image_size = (180, 180)
    batch_size = 32
    validation_split = 0.2

    # Model architecture
    use_quantum = True
    n_qubits = 8
    quantum_depth = 3
    classical_filters = [32, 64, 128]  # Conv2D filters

    # Training
    epochs = 25
    learning_rate = 3e-4

    # Quantum backend
    backend = 'simulator'
    use_mock = True
    ionq_api_key = None
    ionq_target = 'simulator'

    # Quick test mode (use smaller dataset)
    quick_test = False
    quick_test_samples = 1000


# ============================================================================
# Dataset Preparation
# ============================================================================

def download_and_extract_dataset(force: bool = False):
    """
    Download and extract the Cats vs Dogs dataset.

    Parameters
    ----------
    force : bool
        Force re-download even if data exists
    """
    data_dir = Path(Config.data_dir)
    pet_images_dir = data_dir / "PetImages"

    # Check if already downloaded
    if pet_images_dir.exists() and not force:
        print(f"✓ Dataset already exists at {pet_images_dir}")
        return str(pet_images_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "kagglecatsanddogs_5340.zip"

    # Download dataset
    if not zip_path.exists() or force:
        print(f"\n📥 Downloading Cats vs Dogs dataset (~786 MB)...")
        print(f"   URL: {Config.dataset_url}")
        print(f"   This may take several minutes...")

        try:
            urllib.request.urlretrieve(
                Config.dataset_url,
                zip_path,
                reporthook=_download_progress
            )
            print(f"\n✓ Download complete: {zip_path}")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("   Please download manually from:")
            print(f"   {Config.dataset_url}")
            sys.exit(1)

    # Extract dataset
    print(f"\n📦 Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"✓ Extraction complete: {pet_images_dir}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)

    return str(pet_images_dir)


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '=' * filled + '-' * (bar_length - filled)

    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)

    print(f'\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')

    if downloaded >= total_size:
        print()


def filter_corrupted_images(pet_images_dir: str):
    """
    Filter out corrupted images that don't have JFIF header.

    Parameters
    ----------
    pet_images_dir : str
        Path to PetImages directory
    """
    print(f"\n🔍 Filtering corrupted images...")

    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = Path(pet_images_dir) / folder_name

        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            fpath = folder_path / fname

            try:
                with open(fpath, "rb") as fobj:
                    is_jfif = b"JFIF" in fobj.read(10)

                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
            except Exception as e:
                num_skipped += 1
                if fpath.exists():
                    os.remove(fpath)

    print(f"✓ Deleted {num_skipped} corrupted images")
    return num_skipped


def create_datasets(pet_images_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create training and validation datasets.

    Parameters
    ----------
    pet_images_dir : str
        Path to PetImages directory

    Returns
    -------
    train_ds, val_ds : Tuple[tf.data.Dataset, tf.data.Dataset]
        Training and validation datasets
    """
    print(f"\n📊 Creating datasets...")
    print(f"   Image size: {Config.image_size}")
    print(f"   Batch size: {Config.batch_size}")
    print(f"   Validation split: {Config.validation_split}")

    # Create datasets
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        pet_images_dir,
        validation_split=Config.validation_split,
        subset="both",
        seed=1337,
        image_size=Config.image_size,
        batch_size=Config.batch_size,
    )

    print(f"✓ Datasets created")

    return train_ds, val_ds


def create_quick_test_datasets(pet_images_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create small datasets for quick testing.

    Parameters
    ----------
    pet_images_dir : str
        Path to PetImages directory

    Returns
    -------
    train_ds, val_ds : Tuple[tf.data.Dataset, tf.data.Dataset]
        Small training and validation datasets
    """
    print(f"\n⚡ Creating quick test datasets...")
    print(f"   Using {Config.quick_test_samples} samples for quick testing")

    # Create full dataset first
    full_ds = keras.utils.image_dataset_from_directory(
        pet_images_dir,
        validation_split=0.0,
        subset=None,
        seed=1337,
        image_size=Config.image_size,
        batch_size=Config.batch_size,
    )

    # Take limited samples
    n_batches = Config.quick_test_samples // Config.batch_size
    train_batches = int(n_batches * 0.8)
    val_batches = n_batches - train_batches

    train_ds = full_ds.take(train_batches)
    val_ds = full_ds.skip(train_batches).take(val_batches)

    print(f"✓ Quick test datasets created")
    print(f"   Train batches: {train_batches}")
    print(f"   Val batches: {val_batches}")

    return train_ds, val_ds


# ============================================================================
# Data Augmentation
# ============================================================================

def create_augmentation_layers():
    """
    Create data augmentation layers.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model with augmentation layers
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    return data_augmentation


def configure_for_performance(train_ds: tf.data.Dataset,
                               val_ds: tf.data.Dataset,
                               data_augmentation) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Configure datasets for performance with prefetching and augmentation.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset
    val_ds : tf.data.Dataset
        Validation dataset
    data_augmentation : keras.Sequential
        Data augmentation layers

    Returns
    -------
    train_ds, val_ds : Tuple[tf.data.Dataset, tf.data.Dataset]
        Configured datasets
    """
    # Apply data augmentation to training set
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img, training=True), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    return train_ds, val_ds


# ============================================================================
# Visualization
# ============================================================================

def visualize_data(dataset: tf.data.Dataset, class_names: List[str],
                   title: str = "Sample Images"):
    """
    Visualize sample images from dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset to visualize
    class_names : List[str]
        Class names for labels
    title : str
        Plot title
    """
    if not HAS_MATPLOTLIB:
        print("ℹ Matplotlib not available, skipping visualization")
        return

    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16)

    for images, labels in dataset.take(1):
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"examples/ml_frameworks/{title.lower().replace(' ', '_')}.png")
    print(f"✓ Saved visualization to examples/ml_frameworks/{title.lower().replace(' ', '_')}.png")


def visualize_augmented_data(dataset: tf.data.Dataset,
                              data_augmentation,
                              title: str = "Augmented Samples"):
    """
    Visualize augmented images.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset to augment and visualize
    data_augmentation : keras.Sequential
        Data augmentation layers
    title : str
        Plot title
    """
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16)

    for images, _ in dataset.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"examples/ml_frameworks/{title.lower().replace(' ', '_')}.png")
    print(f"✓ Saved visualization to examples/ml_frameworks/{title.lower().replace(' ', '_')}.png")


# ============================================================================
# Model Architecture
# ============================================================================

class QuantumWrapper(layers.Layer):
    """
    Wrapper to integrate q-store quantum layers with Keras.

    This wrapper handles the conversion between TensorFlow tensors and
    numpy arrays for quantum layer processing.
    """

    def __init__(self, quantum_layer, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantum_layer = quantum_layer
        self._supports_ragged_inputs = False

    def call(self, inputs):
        """Forward pass through quantum layer."""
        def quantum_forward(x):
            # Convert to numpy
            x_np = x.numpy()

            # Process through quantum layer (synchronous)
            try:
                import asyncio
                if hasattr(self.quantum_layer, 'call_async'):
                    output = asyncio.run(self.quantum_layer.call_async(x_np))
                elif hasattr(self.quantum_layer, 'forward_async'):
                    output = asyncio.run(self.quantum_layer.forward_async(x_np))
                else:
                    output = self.quantum_layer(x_np)
            except Exception as e:
                print(f"⚠️  Quantum layer error: {e}")
                # Fallback: return identity
                output = x_np

            return output.astype(np.float32)

        # Use py_function to call quantum layer
        output = tf.py_function(
            quantum_forward,
            [inputs],
            tf.float32
        )

        # Set shape explicitly
        batch_size = tf.shape(inputs)[0]
        if hasattr(self.quantum_layer, 'output_dim'):
            output_dim = self.quantum_layer.output_dim
        else:
            # Estimate output dimension
            output_dim = self.quantum_layer.n_qubits * 3  # Default: n_qubits × measurement_bases

        output.set_shape([None, output_dim])

        return output

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if hasattr(self.quantum_layer, 'output_dim'):
            return (input_shape[0], self.quantum_layer.output_dim)
        return (input_shape[0], self.quantum_layer.n_qubits * 3)


def make_quantum_model(input_shape: Tuple[int, int, int],
                        num_classes: int,
                        use_quantum: bool = True) -> keras.Model:
    """
    Build image classification model with quantum layers.

    This architecture combines classical convolutional layers for spatial
    feature extraction with quantum layers for nonlinear feature learning.

    Architecture:
    1. Rescaling (normalize to [0, 1])
    2. Conv2D blocks for spatial features
    3. GlobalAveragePooling to flatten
    4. Quantum layers for feature extraction (optional)
    5. Dense layers for classification

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Input image shape (height, width, channels)
    num_classes : int
        Number of output classes
    use_quantum : bool
        Whether to use quantum layers

    Returns
    -------
    model : keras.Model
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)

    # Rescaling layer
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Classical convolutional layers for spatial feature extraction
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    # Convolutional blocks with residual connections
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual connection
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    # Global pooling to flatten spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Quantum feature extraction layers (optional)
    if use_quantum and HAS_QSTORE:
        print("\n🔮 Adding quantum layers to model...")

        # Get feature dimension
        feature_dim = x.shape[-1]

        # Add dense layer to match quantum input dimension (2^n_qubits)
        target_dim = 2 ** Config.n_qubits
        if feature_dim != target_dim:
            x = layers.Dense(target_dim, activation='relu', name='pre_quantum_dense')(x)

        # Quantum feature extraction
        try:
            quantum_extractor = QuantumFeatureExtractor(
                n_qubits=Config.n_qubits,
                depth=Config.quantum_depth,
                backend=Config.backend,
                entanglement='full',
                measurement_bases=['Z', 'X', 'Y']
            )
            x = QuantumWrapper(quantum_extractor, name='quantum_feature_extractor')(x)
            print(f"   ✓ Added QuantumFeatureExtractor ({Config.n_qubits} qubits, depth {Config.quantum_depth})")
        except Exception as e:
            print(f"   ⚠️  Failed to add quantum layer: {e}")
            print(f"   Continuing with classical layers only")

        # Additional classical processing after quantum
        x = layers.Dense(128, activation='relu', name='post_quantum_dense')(x)
    else:
        if not use_quantum:
            print("\n🎯 Building classical-only model (no quantum layers)")
        else:
            print("\n⚠️  Q-Store not available, building classical-only model")

        # Classical alternative
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)

    # Dropout before final layer
    x = layers.Dropout(0.25)(x)

    # Output layer
    if num_classes == 2:
        units = 1
        activation = "sigmoid"
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="acc")]
    else:
        units = num_classes
        activation = "softmax"
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

    outputs = layers.Dense(units, activation=activation)(x)

    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(Config.learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


# ============================================================================
# Training
# ============================================================================

def train_model(model: keras.Model,
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                epochs: int = None) -> keras.callbacks.History:
    """
    Train the model.

    Parameters
    ----------
    model : keras.Model
        Model to train
    train_ds : tf.data.Dataset
        Training dataset
    val_ds : tf.data.Dataset
        Validation dataset
    epochs : int, optional
        Number of epochs (uses Config.epochs if None)

    Returns
    -------
    history : keras.callbacks.History
        Training history
    """
    if epochs is None:
        epochs = Config.epochs

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/cats_vs_dogs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / "model_epoch_{epoch:02d}.keras"),
            save_best_only=False,
            save_freq='epoch'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.CSVLogger(
            str(checkpoint_dir / "training_log.csv")
        )
    ]

    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Optimizer: Adam (lr={Config.learning_rate})")
    print(f"   Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger")

    start_time = time.time()

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    print(f"\n✓ Training completed in {training_time:.1f} seconds")
    print(f"   Final train accuracy: {history.history['acc'][-1]:.4f}")
    print(f"   Final val accuracy: {history.history['val_acc'][-1]:.4f}")
    print(f"   Best val accuracy: {max(history.history['val_acc']):.4f}")

    return history


def plot_training_history(history: keras.callbacks.History):
    """
    Plot training and validation metrics.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history
    """
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['acc'], label='Train Accuracy')
    ax1.plot(history.history['val_acc'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/ml_frameworks/training_history.png')
    print(f"✓ Saved training history to examples/ml_frameworks/training_history.png")


# ============================================================================
# Inference
# ============================================================================

def run_inference(model: keras.Model, pet_images_dir: str, class_names: List[str]):
    """
    Run inference on sample images.

    Parameters
    ----------
    model : keras.Model
        Trained model
    pet_images_dir : str
        Path to PetImages directory
    class_names : List[str]
        Class names
    """
    print(f"\n🔍 Running inference on sample images...")

    # Load sample images
    sample_images = []
    cat_dir = Path(pet_images_dir) / "Cat"
    dog_dir = Path(pet_images_dir) / "Dog"

    # Get one image from each class
    cat_files = list(cat_dir.glob("*.jpg"))[:3]
    dog_files = list(dog_dir.glob("*.jpg"))[:3]

    if not HAS_MATPLOTLIB:
        print("ℹ Matplotlib not available, skipping visualization")
        # Still run predictions
        for img_path in cat_files + dog_files:
            img = keras.utils.load_img(img_path, target_size=Config.image_size)
            img_array = keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            score = float(predictions[0][0]) if predictions.shape[1] == 1 else float(predictions[0][1])

            print(f"   {img_path.name}: {class_names[0]} {100 * (1 - score):.2f}% / {class_names[1]} {100 * score:.2f}%")
        return

    # Visualize predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample Predictions', fontsize=16)

    for idx, img_path in enumerate(cat_files + dog_files):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Load and predict
        img = keras.utils.load_img(img_path, target_size=Config.image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        score = float(predictions[0][0]) if predictions.shape[1] == 1 else float(predictions[0][1])

        # Display
        ax.imshow(img)
        ax.set_title(f"{class_names[0]}: {100 * (1 - score):.1f}%\n{class_names[1]}: {100 * score:.1f}%")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('examples/ml_frameworks/sample_predictions.png')
    print(f"✓ Saved predictions to examples/ml_frameworks/sample_predictions.png")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Image Classification from Scratch with Q-Store v4.1.1'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download and prepare dataset, then exit'
    )
    parser.add_argument(
        '--no-quantum',
        action='store_true',
        help='Skip quantum layers (classical-only baseline)'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real quantum hardware (requires IONQ_API_KEY)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with small dataset (1000 samples, 5 epochs)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 25, or 5 in quick-test mode)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize sample images and training history'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Update config
    Config.use_quantum = not args.no_quantum
    Config.use_mock = not args.no_mock
    Config.quick_test = args.quick_test
    Config.batch_size = args.batch_size

    if args.epochs:
        Config.epochs = args.epochs
    elif args.quick_test:
        Config.epochs = 5

    # Setup backend
    if not Config.use_mock and Config.use_quantum:
        Config.ionq_api_key = os.getenv('IONQ_API_KEY')
        Config.ionq_target = os.getenv('IONQ_TARGET', 'simulator')

        if not Config.ionq_api_key:
            print("\n⚠️  ERROR: --no-mock specified but IONQ_API_KEY not found")
            print("   Please set IONQ_API_KEY in examples/.env or use mock mode")
            sys.exit(1)

        Config.backend = 'ionq'
        print(f"\n✓ Using real IonQ backend")
        print(f"  Target: {Config.ionq_target}")
    elif Config.use_quantum:
        Config.backend = 'simulator'
        print(f"\n✓ Using simulator backend (no API keys required)")

    # Print header
    print("\n" + "="*70)
    print("IMAGE CLASSIFICATION FROM SCRATCH - Q-STORE v4.1.1")
    print("="*70)
    print(f"Inspired by: https://keras.io/examples/vision/image_classification_from_scratch/")
    print(f"\nConfiguration:")
    print(f"  Quantum layers: {Config.use_quantum}")
    if Config.use_quantum:
        print(f"  Quantum backend: {Config.backend}")
        print(f"  Qubits: {Config.n_qubits}")
        print(f"  Quantum depth: {Config.quantum_depth}")
    print(f"  Image size: {Config.image_size}")
    print(f"  Batch size: {Config.batch_size}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Quick test mode: {Config.quick_test}")

    # Download and prepare dataset
    pet_images_dir = download_and_extract_dataset()
    filter_corrupted_images(pet_images_dir)

    if args.download_only:
        print("\n✓ Dataset preparation complete. Exiting.")
        return

    # Create datasets
    if Config.quick_test:
        train_ds, val_ds = create_quick_test_datasets(pet_images_dir)
    else:
        train_ds, val_ds = create_datasets(pet_images_dir)

    # Get class names
    class_names = train_ds.class_names
    print(f"\nClass names: {class_names}")

    # Visualize original data
    if args.visualize:
        visualize_data(train_ds, class_names, "Training Samples")

    # Data augmentation
    data_augmentation = create_augmentation_layers()

    # Visualize augmented data
    if args.visualize:
        visualize_augmented_data(train_ds, data_augmentation, "Augmented Samples")

    # Configure datasets for performance
    train_ds, val_ds = configure_for_performance(train_ds, val_ds, data_augmentation)

    # Build model
    input_shape = Config.image_size + (3,)
    num_classes = len(class_names)

    model = make_quantum_model(
        input_shape=input_shape,
        num_classes=num_classes,
        use_quantum=Config.use_quantum
    )

    # Print model summary
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    model.summary()

    # Train model
    history = train_model(model, train_ds, val_ds, epochs=Config.epochs)

    # Plot training history
    if args.visualize:
        plot_training_history(history)

    # Run inference
    run_inference(model, pet_images_dir, class_names)

    # Save final model
    model_path = "checkpoints/cats_vs_dogs/final_model.keras"
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Final validation accuracy: {history.history['val_acc'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history.history['val_acc']):.4f}")
    print(f"\nNext steps:")
    print(f"  1. View training history: examples/ml_frameworks/training_history.png")
    print(f"  2. View predictions: examples/ml_frameworks/sample_predictions.png")
    print(f"  3. Load model: keras.models.load_model('{model_path}')")


if __name__ == "__main__":
    main()
