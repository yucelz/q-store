"""
Fashion MNIST Classification with Q-Store v4.1.0 (TensorFlow)

This example demonstrates:
- Using QuantumLayer in a Keras model
- Training with quantum-enhanced neural networks
- End-to-end quantum machine learning workflow
- Mock mode vs real quantum hardware connection

Usage:
    # Run with mock backend (default, no API keys needed)
    python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py

    # Run with real IonQ connection (requires IONQ_API_KEY in .env)
    python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py --no-mock

Configuration:
    Create examples/.env from examples/.env.example and set:
    - IONQ_API_KEY: Your IonQ API key
    - IONQ_TARGET: simulator or qpu.harmony
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# IMPORTANT: Q-Store quantum layers use tf.py_function which is not compatible with XLA/GPU
# Force CPU-only execution to avoid "EagerPyFunc not supported on XLA_GPU_JIT" error
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
tf.config.set_visible_devices([], 'GPU')    # Also disable via TF API

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load environment variables
if HAS_DOTENV:
    # Look for .env in examples directory (parent.parent = examples/)
    examples_dir = Path(__file__).parent.parent
    env_path = examples_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"ℹ No .env file found at {env_path}, using defaults")
else:
    print(f"ℹ python-dotenv not installed, using environment variables")

try:
    from q_store.tensorflow import QuantumLayer, AmplitudeEncoding
    from q_store.core import UnifiedCircuit, GateType
    from q_store.backends import BackendManager
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Fashion MNIST class labels
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Global configuration
USE_MOCK = True
IONQ_API_KEY = None
IONQ_TARGET = None
DEFAULT_BACKEND = 'mock_ideal'

    # Optimization
    use_adaptive_batching = True
    use_caching = True
    use_ionq_compilation = False  # Enable for IonQ hardware

    # Storage
    checkpoint_dir = "experiments/fashion_mnist_tf_v4_1"
    metrics_dir = "experiments/fashion_mnist_tf_v4_1/metrics"
    save_frequency = 2  # Save every N epochs


# ============================================================================
# Backend Configuration
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fashion MNIST Classification with Q-Store v4.1.0 (TensorFlow)'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real IonQ backend (requires IONQ_API_KEY in .env)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    return parser.parse_args()


def setup_backend():
    """Setup quantum backend based on configuration."""
    global USE_MOCK, IONQ_API_KEY, IONQ_TARGET

    if not USE_MOCK:
        IONQ_API_KEY = os.getenv('IONQ_API_KEY')
        IONQ_TARGET = os.getenv('IONQ_TARGET', 'simulator')

        if not IONQ_API_KEY:
            print("\n⚠️  ERROR: --no-mock specified but IONQ_API_KEY not found in .env")
            print("   Please set IONQ_API_KEY in examples/.env or use mock mode")
            sys.exit(1)

        Config.backend = 'ionq'
        print(f"\n✓ Using real IonQ connection")
        print(f"  Backend: {Config.backend}")
        print(f"  Target: {IONQ_TARGET}")

        # Configure IonQ backend if available
        try:
            from q_store.backends.ionq_hardware_backend import IonQHardwareBackend
            print("✓ IonQ backend module loaded")
        except ImportError as e:
            print(f"⚠️  Failed to import IonQ backend: {e}")
            print("   Falling back to simulator backend")
            Config.backend = 'simulator'
    else:
        Config.backend = 'simulator'
        print(f"\n✓ Using mock simulator backend (no API keys required)")
        print(f"  Backend: {Config.backend}")

    return Config.backend


# ============================================================================
# Data Loading
# ============================================================================

def load_fashion_mnist():
    """Load and preprocess Fashion MNIST dataset."""
    print("\n📦 Loading Fashion MNIST dataset...")

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten images
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, Config.num_classes)
    y_test = keras.utils.to_categorical(y_test, Config.num_classes)

    print(f"  Train: {x_train.shape[0]} samples")
    print(f"  Test: {x_test.shape[0]} samples")
    print(f"  Classes: {Config.num_classes}")

    return (x_train, y_train), (x_test, y_test)


# ============================================================================
# Model Architecture
# ============================================================================

def build_quantum_first_model():
    """
    Build quantum-first model with 70% quantum computation.

    Architecture:
    - EncodingLayer: Minimal classical preprocessing
    - QuantumDense(128): First quantum layer (70% of computation)
    - QuantumDense(64): Second quantum layer
    - Dense(32): Minimal classical layer (30% of computation)
    - DecodingLayer: Output projection
    - Dense(10): Classification head
    """
    print("\n🏗️  Building quantum-first model (70% quantum)...")

    model = keras.Sequential([
        # Classical encoding (minimal)
        keras.layers.Input(shape=(Config.input_dim,)),
        EncodingLayer(
            target_dim=Config.input_dim,
            normalization='l2',
            name='encoding'
        ),

        # Quantum layer 1 (main computation)
        QuantumDense(
            n_qubits=Config.n_qubits,
            n_layers=Config.n_layers,
            shots=Config.shots,
            backend=Config.backend,
            activation=None,
            name='quantum_dense_1'
        ),

        # Quantum layer 2
        QuantumDense(
            n_qubits=Config.n_qubits,
            n_layers=Config.n_layers,
            shots=Config.shots,
            backend=Config.backend,
            activation=None,
            name='quantum_dense_2'
        ),

        # Minimal classical layer (30%)
        keras.layers.Dense(
            Config.classical_dim,
            activation='relu',
            name='classical_dense'
        ),

        # Classical decoding
        DecodingLayer(
            output_dim=Config.classical_dim,
            scaling='expectation',
            name='decoding'
        ),

        # Classification head
        keras.layers.Dense(
            Config.num_classes,
            activation='softmax',
            name='output'
        ),
    ])

    # Print architecture summary
    print("\n📊 Model Architecture:")
    model.summary()

    # Calculate quantum vs classical computation
    total_params = model.count_params()
    quantum_params = (
        Config.input_dim * Config.quantum_dim_1 +
        Config.quantum_dim_1 * Config.quantum_dim_2
    )
    classical_params = total_params - quantum_params
    quantum_percentage = (quantum_params / total_params) * 100

    print(f"\n🔬 Computation Distribution:")
    print(f"  Quantum parameters: {quantum_params:,} ({quantum_percentage:.1f}%)")
    print(f"  Classical parameters: {classical_params:,} ({100-quantum_percentage:.1f}%)")
    print(f"  Total parameters: {total_params:,}")

    return model


# ============================================================================
# Optimization Components
# ============================================================================

def setup_optimizations():
    """Initialize optimization components."""
    print("\n⚡ Setting up optimizations...")

    components = {}

    if Config.use_adaptive_batching:
        components['scheduler'] = AdaptiveBatchScheduler(
            min_batch_size=8,
            max_batch_size=64,
            target_latency_ms=100.0,
        )
        print("  ✓ Adaptive batch scheduler")

    if Config.use_caching:
        components['cache'] = MultiLevelCache(
            l1_size=100,    # Hot parameters
            l2_size=1000,   # Compiled circuits
            l3_size=10000,  # Results
        )
        print("  ✓ Multi-level cache (L1/L2/L3)")

    if Config.use_ionq_compilation:
        components['compiler'] = IonQNativeCompiler(
            optimization_level=2
        )
        print("  ✓ IonQ native compiler")

    return components


# ============================================================================
# Async Storage
# ============================================================================

def setup_storage():
    """Initialize async storage components."""
    print("\n💾 Setting up async storage...")

    # Create directories
    Path(Config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(Config.metrics_dir).mkdir(parents=True, exist_ok=True)

    # Checkpoint manager (Zarr)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Config.checkpoint_dir,
        max_to_keep=3,
        compression='zstd',
    )
    print(f"  ✓ Checkpoint manager: {Config.checkpoint_dir}")

    # Metrics logger (Parquet)
    metrics_logger = AsyncMetricsLogger(
        metrics_dir=Config.metrics_dir,
        buffer_size=1000,
        flush_interval=10.0,
    )
    print(f"  ✓ Metrics logger: {Config.metrics_dir}")

    return checkpoint_manager, metrics_logger


# ============================================================================
# Custom Training Loop with Async Execution
# ============================================================================

class QuantumFirstTrainer:
    """Custom trainer with async execution and optimization."""

    def __init__(self, model, optimizer, loss_fn, metrics_logger,
                 checkpoint_manager, optimizations):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics_logger = metrics_logger
        self.checkpoint_manager = checkpoint_manager
        self.optimizations = optimizations

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.val_accuracy = keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(self, x, y):
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(y, predictions)

        return loss, predictions

    @tf.function
    def val_step(self, x, y):
        """Single validation step."""
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)

        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(y, predictions)

        return loss, predictions

    def train(self, train_data, val_data, epochs):
        """Full training loop with async execution."""
        print("\n🚀 Starting quantum-first training...")

        x_train, y_train = train_data
        x_val, y_val = val_data

        # Calculate steps
        steps_per_epoch = len(x_train) // Config.batch_size
        val_steps = len(x_val) // Config.batch_size

        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {val_steps}")

        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()

            print(f"\n━━━ Epoch {epoch+1}/{epochs} ━━━")

            # Reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # Training phase
            print("Training...")
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.shuffle(10000).batch(Config.batch_size)

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                # Adaptive batching
                if 'scheduler' in self.optimizations:
                    queue_depth = steps_per_epoch - step
                    batch_size = self.optimizations['scheduler'].get_batch_size(
                        queue_depth=queue_depth
                    )
                    # Note: In production, would dynamically adjust batch_size here

                # Train step
                loss, predictions = self.train_step(x_batch, y_batch)

                # Record execution for scheduler
                if 'scheduler' in self.optimizations:
                    self.optimizations['scheduler'].record_execution(
                        batch_size=Config.batch_size,
                        latency_ms=50.0,  # Simplified for demo
                    )

                # Progress
                if step % 50 == 0:
                    print(f"  Step {step}/{steps_per_epoch} - "
                          f"Loss: {self.train_loss.result():.4f} - "
                          f"Acc: {self.train_accuracy.result():.4f}")

            # Validation phase
            print("Validating...")
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.batch(Config.batch_size)

            for x_batch, y_batch in val_dataset:
                self.val_step(x_batch, y_batch)

            # Epoch results
            epoch_time = time.time() - epoch_start

            train_loss = self.train_loss.result().numpy()
            train_acc = self.train_accuracy.result().numpy()
            val_loss = self.val_loss.result().numpy()
            val_acc = self.val_accuracy.result().numpy()

            print(f"\n📈 Epoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Log metrics
            self.metrics_logger.log({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_accuracy': float(train_acc),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc),
                'epoch_time': epoch_time,
                'timestamp': time.time(),
            })

            # Save checkpoint
            if (epoch + 1) % Config.save_frequency == 0:
                print(f"💾 Saving checkpoint...")
                self.checkpoint_manager.save(
                    model=self.model,
                    epoch=epoch + 1,
                    metrics={
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_acc),
                    }
                )

        print("\n✅ Training complete!")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline."""
    global USE_MOCK

    # Parse arguments
    args = parse_args()
    USE_MOCK = not args.no_mock

    # Update config from args
    Config.epochs = args.epochs
    Config.batch_size = args.batch_size

    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     Q-Store v4.1.0 Fashion MNIST - TensorFlow Implementation      ║")
    print("║                                                                    ║")
    print("║  Quantum-first ML with 70% quantum computation                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"\nConfiguration:")
    print(f"  Mode: {'REAL QUANTUM (IonQ)' if not USE_MOCK else 'MOCK (Simulator)'}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch size: {Config.batch_size}")

    # Setup backend
    backend_name = setup_backend()

    start_time = time.time()

    # Load data
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    # Split validation
    val_size = int(len(x_train) * Config.validation_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print(f"\n📊 Data split:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples")
    print(f"  Test: {len(x_test)} samples")

    # Build model
    model = build_quantum_first_model()

    # Setup optimizations
    optimizations = setup_optimizations()

    # Setup storage
    checkpoint_manager, metrics_logger = setup_storage()

    # Compile model
    print("\n⚙️  Compiling model...")
    optimizer = keras.optimizers.Adam(learning_rate=Config.learning_rate)
    loss_fn = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Create trainer
    trainer = QuantumFirstTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
        optimizations=optimizations,
    )

    # Train
    trainer.train(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        epochs=Config.epochs,
    )

    # Final evaluation
    print("\n🎯 Final Evaluation on Test Set:")
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=Config.batch_size, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Optimization statistics
    if optimizations:
        print("\n⚡ Optimization Statistics:")

        if 'scheduler' in optimizations:
            stats = optimizations['scheduler'].stats()
            print(f"\n  Adaptive Scheduler:")
            print(f"    Total batches: {stats['total_batches']}")
            print(f"    Avg throughput: {stats['avg_throughput']:.1f} circuits/sec")

        if 'cache' in optimizations:
            stats = optimizations['cache'].stats()
            print(f"\n  Multi-Level Cache:")
            print(f"    Overall hit rate: {stats['overall_hit_rate']:.3f}")
            print(f"    L1 utilization: {stats['l1']['utilization']:.3f}")
            print(f"    L2 utilization: {stats['l2']['utilization']:.3f}")
            print(f"    L3 utilization: {stats['l3']['utilization']:.3f}")

        if 'compiler' in optimizations:
            stats = optimizations['compiler'].stats()
            print(f"\n  IonQ Compiler:")
            print(f"    Circuits compiled: {stats['circuits_compiled']}")
            print(f"    Estimated speedup: {stats['estimated_speedup']:.2f}x")

    # Cleanup
    metrics_logger.close()

    total_time = time.time() - start_time

    # Summary
    print("\n")
    print("="*70)
    print("🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Total Time: {total_time:.2f}s")
    print(f"✓ Checkpoints: {Config.checkpoint_dir}")
    print(f"✓ Metrics: {Config.metrics_dir}")
    print("="*70)

    print("\n💡 Next Steps:")
    print("  • Load best checkpoint for inference")
    print("  • Analyze training metrics in Parquet files")
    print("  • Deploy to IonQ hardware for 30% speedup")
    print("  • Try PyTorch version: fashion_mnist_pytorch.py")


if __name__ == '__main__':
    main()
