"""
Simple Classification Task Example for Q-Store v4.0.

This example uses a toy dataset to quickly validate both TensorFlow
and PyTorch implementations work correctly for classification tasks.
"""

import numpy as np


def generate_toy_dataset(n_samples=200, n_features=4, n_classes=2, seed=42):
    """Generate a simple toy classification dataset.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        n_classes: Number of output classes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create linearly separable classes with some noise
    weights = np.random.randn(n_features)
    logits = X @ weights
    y = (logits > np.median(logits)).astype(np.int64)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    return X, y


def tensorflow_classification():
    """Run classification with TensorFlow."""
    print("=" * 60)
    print("TensorFlow Simple Classification")
    print("=" * 60)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from q_store.tensorflow import QuantumLayer

        # Generate data
        X, y = generate_toy_dataset(n_samples=200, n_features=4)

        # Split into train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")

        # Create model
        n_qubits = 4
        model = keras.Sequential([
            keras.layers.Dense(n_qubits, activation='relu', input_shape=(4,)),
            QuantumLayer(n_qubits=n_qubits, depth=2),
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel architecture:")
        model.summary()

        # Train
        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=10,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        if test_acc > 0.5:  # Random chance for binary classification
            print("  ✓ TensorFlow classification PASSED!")
            return True
        else:
            print("  ✗ TensorFlow classification performance too low")
            return False

    except ImportError as e:
        print(f"Skipping TensorFlow test: {e}")
        return None
    except Exception as e:
        print(f"Error in TensorFlow test: {e}")
        import traceback
        traceback.print_exc()
        return False


def pytorch_classification():
    """Run classification with PyTorch."""
    print("\n" + "=" * 60)
    print("PyTorch Simple Classification")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from q_store.torch import QuantumLayer

        # Generate data
        X, y = generate_toy_dataset(n_samples=200, n_features=4)

        # Convert to tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        # Split into train/test
        split = int(0.8 * len(X))
        X_train = X_tensor[:split]
        y_train = y_tensor[:split]
        X_test = X_tensor[split:]
        y_test = y_tensor[split:]

        print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")

        # Create model
        n_qubits = 4

        class QuantumClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = nn.Sequential(
                    nn.Linear(4, n_qubits),
                    nn.ReLU()
                )
                self.quantum = QuantumLayer(n_qubits=n_qubits, depth=2)
                self.post = nn.Linear(n_qubits, 2)

            def forward(self, x):
                x = self.pre(x)
                x = self.quantum(x)
                x = self.post(x)
                return x

        model = QuantumClassifier()

        print("\nModel architecture:")
        print(model)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train
        print("\nTraining...")
        model.train()
        for epoch in range(20):  # More epochs for better convergence
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_outputs = model(X_train)
            train_preds = train_outputs.argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            # Test accuracy
            test_outputs = model(X_test)
            test_preds = test_outputs.argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        if test_acc > 0.5:  # Random chance for binary classification
            print("  ✓ PyTorch classification PASSED!")
            return True
        else:
            print("  ✗ PyTorch classification performance too low")
            return False

    except ImportError as e:
        print(f"Skipping PyTorch test: {e}")
        return None
    except Exception as e:
        print(f"Error in PyTorch test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all classification tests."""
    print("\n" + "=" * 60)
    print("Q-Store v4.0 Simple Classification Validation")
    print("=" * 60)

    results = {}

    # Test TensorFlow
    tf_result = tensorflow_classification()
    if tf_result is not None:
        results['TensorFlow'] = tf_result

    # Test PyTorch
    pytorch_result = pytorch_classification()
    if pytorch_result is not None:
        results['PyTorch'] = pytorch_result

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for framework, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{framework}: {status}")

    if all(results.values()):
        print("\nAll classification tests PASSED! 🎉")
    else:
        print("\nSome tests failed - please investigate")

    return results


if __name__ == '__main__':
    main()
