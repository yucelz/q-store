"""
Fashion MNIST Classification Example using Q-Store v4.0 with TensorFlow/Keras.

This example demonstrates:
- Using QuantumLayer in a Keras model
- Training with quantum-enhanced neural networks
- End-to-end quantum machine learning workflow
- Mock mode vs real quantum hardware connection

Usage:
    # Run with mock backend (default, no API keys needed)
    python examples/tensorflow/fashion_mnist.py

    # Run with real IonQ connection (requires IONQ_API_KEY in .env)
    python examples/tensorflow/fashion_mnist.py --no-mock

Configuration:
    Create examples/.env from examples/.env.example and set:
    - IONQ_API_KEY: Your IonQ API key
    - IONQ_TARGET: simulator or qpu.harmony
    - DEFAULT_BACKEND: mock_ideal (default) or ionq_simulator
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
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

# Load environment variables from examples/.env if dotenv is available
if HAS_DOTENV:
    examples_dir = Path(__file__).parent.parent
    env_path = examples_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"ℹ No .env file found at {env_path}, using defaults")
else:
    print(f"ℹ python-dotenv not installed, using environment variables or defaults")

try:
    from q_store.tensorflow import QuantumLayer, AmplitudeEncoding
    from q_store.core import UnifiedCircuit, GateType
    from q_store.backends import BackendManager
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Global configuration
USE_MOCK = True
IONQ_API_KEY = None
IONQ_TARGET = None
DEFAULT_BACKEND = 'mock_ideal'


def setup_backend():
    """Setup quantum backend based on configuration.

    Returns:
        Tuple of (backend_name, backend_manager)
    """
    global USE_MOCK, IONQ_API_KEY, IONQ_TARGET, DEFAULT_BACKEND

    backend_name = DEFAULT_BACKEND
    backend_manager = None

    if not USE_MOCK:
        # Real quantum connection
        if not IONQ_API_KEY:
            print("\n⚠️  ERROR: --no-mock specified but IONQ_API_KEY not found in .env")
            print("   Please set IONQ_API_KEY in examples/.env or use mock mode")
            sys.exit(1)

        backend_name = 'ionq_simulator'
        print(f"\n✓ Using real IonQ connection")
        print(f"  Backend: {backend_name}")
        print(f"  Target: {IONQ_TARGET or 'simulator'}")

        # Configure backend manager with IonQ credentials
        from q_store.tensorflow.layers import get_backend_manager
        from q_store.backends.ionq_hardware_backend import IonQHardwareBackend
        backend_manager = get_backend_manager()

        # Register IonQ backend
        try:
            # Create IonQ backend instance
            ionq_backend = IonQHardwareBackend(
                api_key=IONQ_API_KEY,
                target=IONQ_TARGET or 'simulator',
                use_native_gates=True,
                timeout=300
            )
            # Register the backend instance
            backend_manager.register_backend(
                'ionq_simulator',
                ionq_backend,
                set_as_default=True
            )
            print("✓ IonQ backend registered successfully")
        except Exception as e:
            print(f"⚠️  Failed to register IonQ backend: {e}")
            print("   Falling back to mock_ideal backend")
            backend_name = 'mock_ideal'
    else:
        print(f"\n✓ Using mock backend (no API keys required)")
        print(f"  Backend: {backend_name}")

    return backend_name, backend_manager


def create_quantum_model(n_qubits=4, depth=2, backend='mock_ideal'):
    """Create a hybrid classical-quantum model for Fashion MNIST.

    Args:
        n_qubits: Number of qubits in the quantum circuit
        depth: Circuit depth (number of repeated layers)
        backend: Backend to use for quantum layers

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Classical preprocessing
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(n_qubits, activation='relu'),
        keras.layers.BatchNormalization(),

        # Quantum layer with amplitude encoding
        AmplitudeEncoding(n_qubits=n_qubits, name='quantum_encoding'),
        QuantumLayer(
            n_qubits=n_qubits,
            depth=depth,
            backend=backend,
            name='quantum_layer_1'
        ),

        # Classical postprocessing
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_fashion_mnist(num_samples=1000):
    """Load and preprocess Fashion MNIST dataset.

    Args:
        num_samples: Number of samples to use for training

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Use subset for faster training
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:200]
    y_test = y_test[:200]

    return x_train, y_train, x_test, y_test


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fashion MNIST Classification with Q-Store Quantum Layers'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real quantum hardware/simulator (requires IONQ_API_KEY in .env)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of training samples to use (default: 1000)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    global USE_MOCK, IONQ_API_KEY, IONQ_TARGET, DEFAULT_BACKEND

    if not HAS_DEPENDENCIES:
        print("Cannot run example - missing dependencies")
        return

    # Parse arguments
    args = parse_args()
    USE_MOCK = not args.no_mock

    # Load configuration from environment
    IONQ_API_KEY = os.getenv('IONQ_API_KEY')
    IONQ_TARGET = os.getenv('IONQ_TARGET', 'simulator')
    DEFAULT_BACKEND = os.getenv('DEFAULT_BACKEND', 'mock_ideal')

    print("=" * 80)
    print("Fashion MNIST Classification with Q-Store v4.0 (TensorFlow)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Mode: {'REAL QUANTUM' if not USE_MOCK else 'MOCK (Testing)'}")
    print(f"  Training samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Setup backend
    backend_name, backend_manager = setup_backend()

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading Fashion MNIST dataset...")
    x_train, y_train, x_test, y_test = load_fashion_mnist(num_samples=args.samples)
    print(f"✓ Training samples: {len(x_train)}")
    print(f"✓ Test samples: {len(x_test)}")

    # Create model
    print("\nCreating quantum-classical hybrid model...")
    n_qubits = 4
    depth = 2
    model = create_quantum_model(n_qubits=n_qubits, depth=depth, backend=backend_name)
    model.summary()

    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTraining:")
    print(f"  Duration: {training_time:.2f} seconds")
    print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    print(f"\nTest Set:")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

    print(f"\nBackend:")
    print(f"  Mode: {'Real Quantum ({})'.format(backend_name) if not USE_MOCK else 'Mock (Testing)'}")
    print(f"  Backend: {backend_name}")

    # Save model
    model_path = '/tmp/fashion_mnist_quantum_tf.keras'
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("✓ Model saved successfully!")

    print("\n" + "=" * 80)

    return model, history


if __name__ == '__main__':
    main()
