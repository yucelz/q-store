"""
Fashion MNIST with Full Quantum Database Integration (TensorFlow)

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
    python examples/fashion_mnist_quantum_db_tf.py

    # Run with real IonQ + Pinecone (requires API keys in .env)
    python examples/fashion_mnist_quantum_db_tf.py --no-mock

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
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from pathlib import Path
from typing import List, Dict

# Force CPU execution (quantum layers use py_function, not XLA-compatible)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load environment variables
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
    print(f"ℹ python-dotenv not installed, using environment variables")

try:
    from q_store.tensorflow import QuantumLayer, AmplitudeEncoding
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


def create_hybrid_quantum_model(n_qubits=4, depth=2, backend='mock_ideal', embedding_dim=64):
    """Create hybrid quantum-classical model with embedding extraction."""

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1), name='input')

    # Classical preprocessing
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(n_qubits, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    # Quantum layers
    x = AmplitudeEncoding(n_qubits=n_qubits)(x)
    x = QuantumLayer(n_qubits=n_qubits, depth=depth, backend=backend)(x)

    # Embedding layer (for database storage)
    embedding = keras.layers.Dense(embedding_dim, activation='relu', name='embedding')(x)
    embedding = keras.layers.BatchNormalization()(embedding)

    # Classification head
    x = keras.layers.Dense(64, activation='relu')(embedding)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(10, activation='softmax', name='output')(x)

    # Full model
    model = keras.Model(inputs=inputs, outputs=outputs, name='hybrid_quantum_model')

    # Embedding extraction model
    embedding_model = keras.Model(inputs=inputs, outputs=embedding, name='embedding_extractor')

    return model, embedding_model


def setup_backend():
    """Setup quantum backend based on configuration."""
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

        from q_store.tensorflow.layers import get_backend_manager
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


def load_fashion_mnist(num_samples=500):
    """Load and preprocess Fashion MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Subset for faster training
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:200]
    y_test = y_test[:200]

    # Split train into train/val
    split_idx = int(0.8 * len(x_train))
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


async def store_embeddings_in_quantum_db(
    embedding_model: keras.Model,
    x_data: np.ndarray,
    y_data: np.ndarray,
    db: QuantumDatabase,
    max_items: int = 100
) -> Dict:
    """Extract embeddings and store in Quantum Database."""
    print("\n" + "=" * 80)
    print("STORING EMBEDDINGS IN QUANTUM DATABASE")
    print("=" * 80)

    stored_items = []
    num_items = min(max_items, len(x_data))

    # Extract embeddings in batches
    batch_size = 32
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        batch_x = x_data[start_idx:end_idx]
        batch_y = y_data[start_idx:end_idx]

        # Get embeddings
        embeddings = embedding_model.predict(batch_x, verbose=0)

        # Store each item
        for i in range(len(embeddings)):
            item_idx = start_idx + i
            embedding = embeddings[i]
            label = int(batch_y[i])
            class_name = CLASS_NAMES[label]
            item_id = f"fashion_mnist_{item_idx}"

            # Create contexts for quantum superposition
            contexts = [
                (f"class_{class_name}", 0.6),
                (f"category_{get_category(class_name)}", 0.3),
                (f"style_{get_style(class_name)}", 0.1)
            ]

            # Insert into quantum database
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
                'label': label
            })

        if end_idx % 20 == 0 or end_idx == num_items:
            print(f"  Stored {end_idx}/{num_items} items...")

    print(f"\n✓ Successfully stored {num_items} embeddings in Quantum Database")
    print(f"  - With quantum superposition across 3 contexts per item")

    return {
        'stored_items': stored_items,
        'total_count': num_items
    }


async def query_quantum_db(
    db: QuantumDatabase,
    query_embedding: np.ndarray,
    context: str = None,
    top_k: int = 5
) -> List:
    """Query the Quantum Database with context-aware search."""
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
        description='Fashion MNIST with Full Quantum Database Integration (TensorFlow)'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real quantum hardware and Pinecone (requires API keys)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of training samples (default: 500)'
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
        help='Number of items to store in database (default: 100)'
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

    # Load configuration
    IONQ_API_KEY = os.getenv('IONQ_API_KEY')
    IONQ_TARGET = os.getenv('IONQ_TARGET', 'simulator')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')

    print("\n" + "=" * 80)
    print("Fashion MNIST with Full Quantum Database Integration (TensorFlow)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Mode: {'REAL QUANTUM + PINECONE' if not USE_MOCK else 'MOCK (Testing)'}")
    print(f"  Training samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Items to store: {args.store_items}")

    # Setup backend
    backend_name, backend_manager = setup_backend()

    # Database configuration
    if not USE_MOCK:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            print("\n⚠️  ERROR: --no-mock requires PINECONE_API_KEY and PINECONE_ENVIRONMENT")
            sys.exit(1)
        print(f"\n✓ Pinecone: {PINECONE_ENVIRONMENT}")
    else:
        # Use mock credentials for testing
        PINECONE_API_KEY = "mock-test-key-12345"
        PINECONE_ENVIRONMENT = "us-east-1"
        print(f"\n✓ Using mock Pinecone backend (no real API calls)")

    db_config = DatabaseConfig(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        pinecone_index_name="fashion-mnist-quantum-tf",
        pinecone_dimension=64,
        ionq_api_key=IONQ_API_KEY if not USE_MOCK else None,
        ionq_target=IONQ_TARGET,
        enable_quantum=True,
        enable_superposition=True,
        quantum_sdk='mock' if USE_MOCK else 'cirq',
        use_mock_pinecone=USE_MOCK  # Use mock Pinecone when in mock mode
    )

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist(args.samples)
    print(f"✓ Train: {len(x_train)} samples")
    print(f"✓ Val: {len(x_val)} samples")
    print(f"✓ Test: {len(x_test)} samples")

    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    n_qubits = 4
    depth = 2
    embedding_dim = 64

    model, embedding_model = create_hybrid_quantum_model(
        n_qubits=n_qubits,
        depth=depth,
        backend=backend_name,
        embedding_dim=embedding_dim
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✓ Model created with {embedding_dim}-dim embeddings")
    print(model.summary())

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

    # Initialize Quantum Database
    print("\n" + "=" * 80)
    print("INITIALIZING QUANTUM DATABASE")
    print("=" * 80)

    db = QuantumDatabase(db_config)
    async with db.connect():
        print("✓ Quantum Database connected")
        print(f"  Index: {db_config.pinecone_index_name}")
        print(f"  Quantum Features: Enabled")

        # Store embeddings
        stored_data = await store_embeddings_in_quantum_db(
            embedding_model=embedding_model,
            x_data=x_test,
            y_data=y_test,
            db=db,
            max_items=args.store_items
        )

        # Demonstrate queries
        print("\n" + "=" * 80)
        print("QUANTUM-ENHANCED SIMILARITY SEARCH")
        print("=" * 80)

        # Get test item
        test_idx = 0
        test_image = x_test[test_idx:test_idx+1]
        test_label = int(y_test[test_idx])
        test_class = CLASS_NAMES[test_label]
        test_embedding = embedding_model.predict(test_image, verbose=0)[0]

        print(f"\nQuery: {test_class}")

        # Classical search
        print(f"\n1. Classical Search:")
        results = await query_quantum_db(db, test_embedding, context=None, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r.metadata.get('class', 'N/A')} (score: {r.score:.4f})")

        # Quantum search with class context
        context_class = f"class_{test_class}"
        print(f"\n2. Quantum Search (context: '{context_class}'):")
        results = await query_quantum_db(db, test_embedding, context=context_class, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r.metadata.get('class', 'N/A')} "
                  f"(score: {r.score:.4f}, quantum: {r.quantum_enhanced})")

        # Quantum search with category context
        category = get_category(test_class)
        context_cat = f"category_{category}"
        print(f"\n3. Quantum Search (context: '{context_cat}'):")
        results = await query_quantum_db(db, test_embedding, context=context_cat, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r.metadata.get('class', 'N/A')} "
                  f"({r.metadata.get('category', 'N/A')}, "
                  f"score: {r.score:.4f})")

        # Statistics
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        stats = db.get_stats()
        print(f"  Quantum states: {stats['quantum_states']}")
        print(f"  Queries: {stats['metrics']['total_queries']}")
        print(f"  Quantum queries: {stats['metrics']['quantum_queries']}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTraining:")
    print(f"  Duration: {training_time:.2f}s")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print(f"\nQuantum Database:")
    print(f"  Stored items: {stored_data['total_count']}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Quantum superposition: Enabled")
    print(f"\n✓ DEMONSTRATION COMPLETE")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
