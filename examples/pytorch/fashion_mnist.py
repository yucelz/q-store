"""
Fashion MNIST Classification Example using Q-Store v4.0 with PyTorch.

This example demonstrates:
- Using QuantumLayer in a PyTorch model
- Training with quantum-enhanced neural networks
- End-to-end quantum machine learning workflow with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from q_store.torch import QuantumLayer, AmplitudeEncoding
    from q_store.core import UnifiedCircuit, GateType
    HAS_QSTORE = True
except ImportError as e:
    print(f"Missing Q-Store dependencies: {e}")
    HAS_QSTORE = False


class HybridQuantumNet(nn.Module):
    """Hybrid classical-quantum neural network for Fashion MNIST."""

    def __init__(self, n_qubits=4, depth=2):
        """Initialize the hybrid model.

        Args:
            n_qubits: Number of qubits in quantum circuit
            depth: Circuit depth (number of repeated layers)
        """
        super().__init__()

        # Classical preprocessing layers
        self.classical_pre = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, n_qubits),
            nn.ReLU(),
            nn.BatchNorm1d(n_qubits)
        )

        # Quantum layers
        self.quantum_encoding = AmplitudeEncoding(n_qubits=n_qubits)
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            depth=depth
        )

        # Classical postprocessing layers
        self.classical_post = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """Forward pass through the hybrid network."""
        x = self.classical_pre(x)
        x = self.quantum_encoding(x)
        x = self.quantum_layer(x)
        x = self.classical_post(x)
        return x


def load_fashion_mnist(num_samples=1000, batch_size=32):
    """Load and preprocess Fashion MNIST dataset.

    Args:
        num_samples: Number of samples to use for training
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if not HAS_TORCHVISION:
        # Fallback to manual data creation for testing
        print("torchvision not available, creating dummy data...")
        x_train = torch.randn(num_samples, 1, 28, 28)
        y_train = torch.randint(0, 10, (num_samples,))
        x_test = torch.randn(200, 1, 28, 28)
        y_test = torch.randint(0, 10, (200,))

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
    else:
        # Use real Fashion MNIST data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST(
            root='/tmp/data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.FashionMNIST(
            root='/tmp/data',
            train=False,
            download=True,
            transform=transform
        )

        # Subset for faster training
        train_dataset = torch.utils.data.Subset(train_dataset, range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, range(200))

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    """Main training function."""
    if not HAS_QSTORE:
        print("Cannot run example - missing Q-Store dependencies")
        return

    print("=" * 60)
    print("Fashion MNIST Classification with Q-Store v4.0 (PyTorch)")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading Fashion MNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist(
        num_samples=1000,
        batch_size=32
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating quantum-classical hybrid model...")
    n_qubits = 4
    depth = 2
    model = HybridQuantumNet(n_qubits=n_qubits, depth=depth).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nTraining model...")
    start_time = time.time()
    epochs = 5

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save model
    model_path = '/tmp/fashion_mnist_quantum_pytorch.pt'
    print(f"\nSaving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'n_qubits': n_qubits,
        'depth': depth
    }, model_path)
    print("Model saved successfully!")

    return model


if __name__ == '__main__':
    main()
