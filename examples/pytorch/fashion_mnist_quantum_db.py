"""
Fashion MNIST with Full Quantum Database Integration (PyTorch)

This comprehensive example demonstrates:
1. Quantum-enhanced neural network layers (QuantumLayer)
2. Training with quantum circuits in the model
3. Storing learned embeddings in Quantum Database (Pinecone)
4. Quantum superposition for multi-context embeddings
5. Context-aware similarity search with quantum enhancements
6. Complete workflow: Train → Store → Query with quantum features

This combines BOTH quantum machine learning AND quantum database features.

Usage:
    # Run with mock backends (no API keys needed)
    python examples/fashion_mnist_quantum_db.py

    # Run with real IonQ + Pinecone (requires API keys in .env)
    python examples/fashion_mnist_quantum_db.py --no-mock

Configuration:
    Create examples/.env from examples/.env.example and set:
    - IONQ_API_KEY: Your IonQ API key for quantum circuits
    - IONQ_TARGET: simulator or qpu.harmony
    - PINECONE_API_KEY: Your Pinecone API key for vector storage
    - PINECONE_ENVIRONMENT: Your Pinecone environment (e.g., us-east-1)
"""

import os
import sys
import argparse
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load environment variables from examples/.env
if HAS_DOTENV:
    # Look for .env in parent examples directory
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
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("⚠️  torchvision not available, will use dummy data")

try:
    from q_store.torch import QuantumLayer, AmplitudeEncoding
    from q_store.core import UnifiedCircuit, GateType
    from q_store.backends import BackendManager
    from q_store.core import QuantumDatabase, DatabaseConfig, QueryMode
    HAS_QSTORE = True
except ImportError as e:
    print(f"⚠️  Missing Q-Store dependencies: {e}")
    HAS_QSTORE = False

# Fashion MNIST class labels
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Global configuration
USE_MOCK = True
IONQ_API_KEY = None
IONQ_TARGET = None
PINECONE_API_KEY = None
PINECONE_ENVIRONMENT = None
DEFAULT_BACKEND = 'mock_ideal'


class HybridQuantumNet(nn.Module):
    """Hybrid classical-quantum neural network with embedding extraction."""

    def __init__(self, n_qubits=4, depth=2, backend='mock_ideal', embedding_dim=64):
        """Initialize the hybrid model.

        Args:
            n_qubits: Number of qubits in quantum circuit
            depth: Circuit depth (number of repeated layers)
            backend: Backend to use for quantum layers
            embedding_dim: Dimension of the embedding layer (for database storage)
        """
        super().__init__()
        self.embedding_dim = embedding_dim

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
            depth=depth,
            backend=backend
        )

        # Embedding layer (for quantum database storage)
        self.embedding_layer = nn.Sequential(
            nn.Linear(n_qubits, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x, return_embedding=False):
        """Forward pass through the hybrid network."""
        x = self.classical_pre(x)
        x = self.quantum_encoding(x)
        x = self.quantum_layer(x)
        embedding = self.embedding_layer(x)

        if return_embedding:
            return embedding

        x = self.classifier(embedding)
        return x

    def get_embedding(self, x):
        """Extract embedding vector for database storage."""
        with torch.no_grad():
            return self.forward(x, return_embedding=True)


def setup_backend():
    """Setup quantum backend based on configuration.

    Returns:
        Tuple of (backend_name, backend_manager)
    """
    global USE_MOCK, IONQ_API_KEY, IONQ_TARGET, DEFAULT_BACKEND

    backend_name = DEFAULT_BACKEND
    backend_manager = None

    if not USE_MOCK:
        if not IONQ_API_KEY:
            print("\n⚠️  ERROR: --no-mock specified but IONQ_API_KEY not found in .env")
            print("   Please set IONQ_API_KEY in examples/.env or use mock mode")
            sys.exit(1)

        backend_name = 'ionq_simulator'
        print(f"\n✓ Using real IonQ connection")
        print(f"  Backend: {backend_name}")
        print(f"  Target: {IONQ_TARGET or 'simulator'}")

        from q_store.torch.layers import get_backend_manager
        from q_store.backends.ionq_hardware_backend import IonQHardwareBackend
        backend_manager = get_backend_manager()

        try:
            ionq_backend = IonQHardwareBackend(
                api_key=IONQ_API_KEY,
                target=IONQ_TARGET or 'simulator',
                use_native_gates=True,
                timeout=300
            )
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


def load_fashion_mnist(num_samples=1000, batch_size=32):
    """Load and preprocess Fashion MNIST dataset."""
    if not HAS_TORCHVISION:
        print("torchvision not available, creating dummy data...")
        x_train = torch.randn(num_samples, 1, 28, 28)
        y_train = torch.randint(0, 10, (num_samples,))
        x_test = torch.randn(200, 1, 28, 28)
        y_test = torch.randint(0, 10, (200,))

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
    else:
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


async def store_embeddings_in_quantum_db(
    model: nn.Module,
    data_loader: DataLoader,
    db: QuantumDatabase,
    device: torch.device,
    max_items: int = 100
) -> Dict[str, List]:
    """
    Extract embeddings from trained model and store in Quantum Database.

    Args:
        model: Trained quantum-classical hybrid model
        data_loader: DataLoader with images
        db: Initialized QuantumDatabase
        device: Torch device
        max_items: Maximum number of items to store

    Returns:
        Dictionary with stored item metadata
    """
    print("\n" + "=" * 80)
    print("STORING EMBEDDINGS IN QUANTUM DATABASE")
    print("=" * 80)

    model.eval()
    stored_items = []

    item_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            if item_count >= max_items:
                break

            inputs = inputs.to(device)
            embeddings = model.get_embedding(inputs)

            # Store each item in the batch
            for i in range(embeddings.shape[0]):
                if item_count >= max_items:
                    break

                embedding = embeddings[i].cpu().numpy()
                label = int(labels[i].item())
                class_name = CLASS_NAMES[label]
                item_id = f"fashion_mnist_{item_count}"

                # Create contexts for quantum superposition
                # Each item has multiple contexts: class, category, style
                contexts = [
                    (f"class_{class_name}", 0.6),
                    (f"category_{get_category(class_name)}", 0.3),
                    (f"style_{get_style(class_name)}", 0.1)
                ]

                # Insert into quantum database with superposition
                await db.insert(
                    id=item_id,
                    vector=embedding,
                    contexts=contexts,
                    metadata={
                        'class': class_name,
                        'label': label,
                        'category': get_category(class_name),
                        'style': get_style(class_name),
                        'source': 'fashion_mnist'
                    }
                )

                stored_items.append({
                    'id': item_id,
                    'class': class_name,
                    'label': label,
                    'embedding_shape': embedding.shape
                })

                item_count += 1

                if item_count % 20 == 0:
                    print(f"  Stored {item_count}/{max_items} items...")

    print(f"\n✓ Successfully stored {item_count} embeddings in Quantum Database")
    print(f"  - With quantum superposition across {len(contexts)} contexts per item")
    print(f"  - Metadata includes: class, category, style")

    return {
        'stored_items': stored_items,
        'total_count': item_count
    }


async def query_quantum_db(
    db: QuantumDatabase,
    query_embedding: np.ndarray,
    context: str = None,
    top_k: int = 5
) -> List:
    """
    Query the Quantum Database with context-aware search.

    Args:
        db: Initialized QuantumDatabase
        query_embedding: Query vector
        context: Optional context for quantum collapse
        top_k: Number of results to return

    Returns:
        List of query results
    """
    results = await db.query(
        vector=query_embedding,
        context=context,
        mode=QueryMode.BALANCED,
        top_k=top_k
    )

    return results


def get_category(class_name: str) -> str:
    """Map class names to broader categories."""
    clothing_upper = ['T-shirt/top', 'Pullover', 'Coat', 'Shirt']
    clothing_lower = ['Trouser', 'Dress']
    footwear = ['Sandal', 'Sneaker', 'Ankle boot']
    accessories = ['Bag']

    if class_name in clothing_upper:
        return 'upper_body'
    elif class_name in clothing_lower:
        return 'lower_body'
    elif class_name in footwear:
        return 'footwear'
    elif class_name in accessories:
        return 'accessories'
    return 'other'


def get_style(class_name: str) -> str:
    """Map class names to style categories."""
    casual = ['T-shirt/top', 'Trouser', 'Sneaker', 'Bag']
    formal = ['Shirt', 'Dress', 'Coat']
    athletic = ['Pullover', 'Ankle boot']
    casual_footwear = ['Sandal']

    if class_name in casual:
        return 'casual'
    elif class_name in formal:
        return 'formal'
    elif class_name in athletic:
        return 'athletic'
    return 'casual'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fashion MNIST with Full Quantum Database Integration (PyTorch)'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real quantum hardware/simulator and Pinecone (requires API keys in .env)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of training samples to use (default: 500)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--store-items',
        type=int,
        default=100,
        help='Number of items to store in quantum database (default: 100)'
    )
    return parser.parse_args()


async def main_async():
    """Main training and database integration function."""
    global USE_MOCK, IONQ_API_KEY, IONQ_TARGET, PINECONE_API_KEY, PINECONE_ENVIRONMENT

    if not HAS_QSTORE:
        print("❌ Cannot run example - missing Q-Store dependencies")
        return

    # Parse arguments
    args = parse_args()
    USE_MOCK = not args.no_mock

    # Load configuration from environment
    IONQ_API_KEY = os.getenv('IONQ_API_KEY')
    IONQ_TARGET = os.getenv('IONQ_TARGET', 'simulator')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')

    print("\n" + "=" * 80)
    print("Fashion MNIST with Full Quantum Database Integration")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Mode: {'REAL QUANTUM + PINECONE' if not USE_MOCK else 'MOCK (Testing)'}")
    print(f"  Training samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Items to store in DB: {args.store_items}")

    # Setup quantum backend for neural network
    backend_name, backend_manager = setup_backend()

    # Setup Quantum Database configuration
    if not USE_MOCK:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            print("\n⚠️  ERROR: --no-mock requires PINECONE_API_KEY and PINECONE_ENVIRONMENT")
            print("   Please set them in examples/.env or use mock mode")
            sys.exit(1)

        print(f"\n✓ Pinecone Configuration:")
        print(f"  API Key: {'*' * 20}")
        print(f"  Environment: {PINECONE_ENVIRONMENT}")
    else:
        # Use mock credentials for testing
        PINECONE_API_KEY = "mock-test-key-12345"
        PINECONE_ENVIRONMENT = "us-east-1"
        print(f"\n✓ Using mock Pinecone backend (no real API calls)")

    # Create database config
    db_config = DatabaseConfig(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        pinecone_index_name="fashion-mnist-quantum",
        pinecone_dimension=64,  # Match embedding_dim
        ionq_api_key=IONQ_API_KEY if not USE_MOCK else None,
        ionq_target=IONQ_TARGET,
        enable_quantum=True,
        enable_superposition=True,
        quantum_sdk='mock' if USE_MOCK else 'cirq',
        use_mock_pinecone=USE_MOCK  # Use mock Pinecone when in mock mode
    )    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train_loader, val_loader, test_loader = load_fashion_mnist(
        num_samples=args.samples,
        batch_size=args.batch_size
    )
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 80)
    print("CREATING QUANTUM-CLASSICAL HYBRID MODEL")
    print("=" * 80)
    device = torch.device('cpu')  # Force CPU for quantum layer compatibility
    n_qubits = 4
    depth = 2
    embedding_dim = 64

    model = HybridQuantumNet(
        n_qubits=n_qubits,
        depth=depth,
        backend=backend_name,
        embedding_dim=embedding_dim
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {embedding_dim}-dimensional embeddings")
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING WITH QUANTUM LAYERS")
    print("=" * 80)
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    training_time = time.time() - start_time

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Initialize Quantum Database
    print("\n" + "=" * 80)
    print("INITIALIZING QUANTUM DATABASE")
    print("=" * 80)

    db = QuantumDatabase(db_config)
    async with db.connect():
        print("✓ Quantum Database connected")
        print(f"  Pinecone Index: {db_config.pinecone_index_name}")
        print(f"  Quantum Features: {'Enabled' if db_config.enable_quantum else 'Disabled'}")
        print(f"  Superposition: {'Enabled' if db_config.enable_superposition else 'Disabled'}")

        # Store embeddings in quantum database
        stored_data = await store_embeddings_in_quantum_db(
            model=model,
            data_loader=test_loader,
            db=db,
            device=device,
            max_items=args.store_items
        )

        # Demonstrate quantum-enhanced queries
        print("\n" + "=" * 80)
        print("QUANTUM-ENHANCED SIMILARITY SEARCH")
        print("=" * 80)

        # Get a test item
        test_iter = iter(test_loader)
        test_images, test_labels = next(test_iter)
        test_image = test_images[0:1].to(device)
        test_label = int(test_labels[0].item())
        test_class = CLASS_NAMES[test_label]

        # Extract embedding
        test_embedding = model.get_embedding(test_image).cpu().numpy()[0]

        print(f"\nQuery: {test_class}")
        print(f"Searching for similar items...")

        # Query 1: Classical search (no context)
        print(f"\n1. Classical Search (no quantum context):")
        results_classical = await query_quantum_db(
            db=db,
            query_embedding=test_embedding,
            context=None,
            top_k=5
        )

        for i, result in enumerate(results_classical, 1):
            metadata = result.metadata
            print(f"   {i}. {metadata.get('class', 'Unknown')} "
                  f"(score: {result.score:.4f}, "
                  f"quantum: {result.quantum_enhanced})")

        # Query 2: Quantum search with class context
        context_class = f"class_{test_class}"
        print(f"\n2. Quantum Search with Context: '{context_class}'")
        results_quantum = await query_quantum_db(
            db=db,
            query_embedding=test_embedding,
            context=context_class,
            top_k=5
        )

        for i, result in enumerate(results_quantum, 1):
            metadata = result.metadata
            print(f"   {i}. {metadata.get('class', 'Unknown')} "
                  f"(score: {result.score:.4f}, "
                  f"quantum: {result.quantum_enhanced})")

        # Query 3: Quantum search with category context
        category = get_category(test_class)
        context_category = f"category_{category}"
        print(f"\n3. Quantum Search with Context: '{context_category}'")
        results_category = await query_quantum_db(
            db=db,
            query_embedding=test_embedding,
            context=context_category,
            top_k=5
        )

        for i, result in enumerate(results_category, 1):
            metadata = result.metadata
            print(f"   {i}. {metadata.get('class', 'Unknown')} "
                  f"(category: {metadata.get('category', 'N/A')}, "
                  f"score: {result.score:.4f}, "
                  f"quantum: {result.quantum_enhanced})")

        # Display database statistics
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        stats = db.get_stats()
        print(f"  Quantum states: {stats['quantum_states']}")
        print(f"  Total queries: {stats['metrics']['total_queries']}")
        print(f"  Quantum queries: {stats['metrics']['quantum_queries']}")
        print(f"  Cache hit rate: {stats['metrics']['cache_hit_rate']:.2%}")
        print(f"  Avg latency: {stats['metrics']['avg_latency_ms']:.2f}ms")

    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n1. Training:")
    print(f"   Duration: {training_time:.2f} seconds")
    print(f"   Test accuracy: {test_acc:.2f}%")

    print(f"\n2. Quantum Database:")
    print(f"   Stored items: {stored_data['total_count']}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Contexts per item: 3 (class, category, style)")
    print(f"   Quantum superposition: Enabled")

    print(f"\n3. Backend:")
    print(f"   Mode: {'Real Quantum + Pinecone' if not USE_MOCK else 'Mock (Testing)'}")
    print(f"   Quantum Backend: {backend_name}")
    if not USE_MOCK:
        print(f"   Pinecone Index: {db_config.pinecone_index_name}")

    print("\n" + "=" * 80)
    print("✓ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThis example demonstrated:")
    print("  ✓ Quantum-enhanced neural network training")
    print("  ✓ Embedding extraction from trained model")
    print("  ✓ Storage in Quantum Database with Pinecone")
    print("  ✓ Quantum superposition across multiple contexts")
    print("  ✓ Context-aware similarity search")
    print("  ✓ Quantum vs classical search comparison")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
