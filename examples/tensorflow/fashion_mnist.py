"""
Fashion MNIST Classification Example using Q-Store v4.0 with TensorFlow/Keras.

This example demonstrates:
- Using QuantumLayer in a Keras model
- Training with quantum-enhanced neural networks
- End-to-end quantum machine learning workflow
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

try:
    from q_store.tensorflow import QuantumLayer, AmplitudeEncoding
    from q_store.core import UnifiedCircuit, GateType
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    HAS_DEPENDENCIES = False


def create_quantum_model(n_qubits=4, depth=2):
    """Create a hybrid classical-quantum model for Fashion MNIST.

    Args:
        n_qubits: Number of qubits in the quantum circuit
        depth: Circuit depth (number of repeated layers)

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
            name=f'quantum_layer_1'
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


def main():
    """Main training function."""
    if not HAS_DEPENDENCIES:
        print("Cannot run example - missing dependencies")
        return

    print("=" * 60)
    print("Fashion MNIST Classification with Q-Store v4.0 (TensorFlow)")
    print("=" * 60)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading Fashion MNIST dataset...")
    x_train, y_train, x_test, y_test = load_fashion_mnist(num_samples=1000)
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Create model
    print("\nCreating quantum-classical hybrid model...")
    n_qubits = 4
    depth = 2
    model = create_quantum_model(n_qubits=n_qubits, depth=depth)
    model.summary()

    # Training
    print("\nTraining model...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluation
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Display training history
    print("\nTraining History:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Save model
    model_path = '/tmp/fashion_mnist_quantum_tf.keras'
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")

    return model, history


if __name__ == '__main__':
    main()
