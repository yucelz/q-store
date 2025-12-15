"""
Quantum-Native Database v3.2 - ML Training Examples
Comprehensive examples of quantum ML training with hardware abstraction
"""

import asyncio
import numpy as np
import logging
import os
import argparse
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import v3.2 components
from q_store.core import QuantumDatabase, DatabaseConfig
from q_store.ml import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    QuantumLayer,
    QuantumDataEncoder,
    QuantumFeatureMap
)
from q_store.backends import BackendManager, create_default_backend_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
PINECONE_API_KEY = None
PINECONE_ENVIRONMENT = None
IONQ_API_KEY = None
IONQ_TARGET = None
USE_MOCK = True


# ============================================================================
# Example 1: Basic Quantum Neural Network Training
# ============================================================================

async def example_1_basic_training():
    """
    Train a simple quantum neural network on synthetic data
    Uses mock backend for testing without quantum hardware
    """
    print("\n" + "="*70)
    print("Example 1: Basic Quantum Neural Network Training")
    print("="*70 + "\n")

    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 8

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)  # Binary classification

    # Configure training
    quantum_sdk = "mock" if USE_MOCK else "ionq"
    quantum_target = "simulator" if USE_MOCK else (IONQ_TARGET or "simulator")

    config = TrainingConfig(
        pinecone_api_key=PINECONE_API_KEY or "mock-key",
        pinecone_environment=PINECONE_ENVIRONMENT or "us-east-1",
        pinecone_index_name="quantum-ml-training",
        quantum_sdk=quantum_sdk,
        quantum_target=quantum_target,
        quantum_api_key=IONQ_API_KEY if not USE_MOCK else None,
        learning_rate=0.01,
        batch_size=10,
        epochs=5,
        n_qubits=8,
        circuit_depth=2,
        entanglement='linear'
    )

    print(f"Configuration: SDK={quantum_sdk}, Target={quantum_target}")
    print(f"Pinecone: {PINECONE_API_KEY[:10] + '...' if PINECONE_API_KEY else 'mock'}")
    if not USE_MOCK and IONQ_API_KEY:
        print(f"IonQ: {IONQ_API_KEY[:10] + '...'}\n")

    # Create backend manager
    backend_manager = create_default_backend_manager()

    # Configure IonQ backend if not using mock
    if not USE_MOCK and IONQ_API_KEY:
        from q_store.backends import setup_ionq_backends
        backend_manager = await setup_ionq_backends(
            backend_manager,
            api_key=IONQ_API_KEY,
            use_cirq=True
        )
        # Set IonQ simulator as default
        backend_manager.set_default_backend('ionq_sim_cirq')
        print("✓ IonQ backend configured")

    # Create trainer
    trainer = QuantumTrainer(config, backend_manager)

    # Create model
    model = QuantumModel(
        input_dim=n_features,
        n_qubits=8,
        output_dim=2,
        backend=backend_manager.get_backend(),
        depth=2
    )

    # Simple data loader
    class SimpleDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        async def __aiter__(self):
            for i in range(0, len(self.X), self.batch_size):
                batch_x = self.X[i:i+self.batch_size]
                batch_y = np.eye(2)[self.y[i:i+self.batch_size]]  # One-hot encode
                yield batch_x, batch_y

    train_loader = SimpleDataLoader(X_train, y_train, config.batch_size)

    # Train
    print("Starting training...")
    await trainer.train(
        model=model,
        train_loader=train_loader,
        epochs=config.epochs
    )

    print("\nTraining complete!")
    print(f"Final loss: {trainer.training_history[-1].loss:.4f}")
    print(f"Total epochs: {len(trainer.training_history)}")


# ============================================================================
# Example 2: Quantum Data Encoding Strategies
# ============================================================================

async def example_2_data_encoding():
    """
    Demonstrate different quantum data encoding methods
    """
    print("\n" + "="*70)
    print("Example 2: Quantum Data Encoding Strategies")
    print("="*70 + "\n")

    # Sample data
    data = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4])

    # 1. Amplitude Encoding
    print("1. Amplitude Encoding")
    amplitude_encoder = QuantumDataEncoder('amplitude')
    amplitude_circuit = amplitude_encoder.encode(data)
    print(f"   Circuit qubits: {amplitude_circuit.n_qubits}")
    print(f"   Circuit depth: {amplitude_circuit.depth()}")
    print(f"   Gate count: {len(amplitude_circuit.gates)}")

    # 2. Angle Encoding
    print("\n2. Angle Encoding")
    angle_encoder = QuantumDataEncoder('angle')
    angle_circuit = angle_encoder.encode(data, n_qubits=8)
    print(f"   Circuit qubits: {angle_circuit.n_qubits}")
    print(f"   Circuit depth: {angle_circuit.depth()}")
    print(f"   Gate count: {len(angle_circuit.gates)}")

    # 3. Feature Map
    print("\n3. ZZ Feature Map")
    feature_map = QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
    feature_circuit = feature_map.map_features(data)
    print(f"   Circuit qubits: {feature_circuit.n_qubits}")
    print(f"   Circuit depth: {feature_circuit.depth()}")
    print(f"   Gate count: {len(feature_circuit.gates)}")

    print("\nEncoding strategies showcase complete!")


# ============================================================================
# Example 3: Transfer Learning with Quantum Models
# ============================================================================

async def example_3_transfer_learning():
    """
    Pre-train on one task, fine-tune on another
    """
    print("\n" + "="*70)
    print("Example 3: Transfer Learning with Quantum Models")
    print("="*70 + "\n")

    # Create backend
    backend_manager = create_default_backend_manager()

    # Configure IonQ backend if not using mock
    if not USE_MOCK and IONQ_API_KEY:
        from q_store.backends import setup_ionq_backends
        backend_manager = await setup_ionq_backends(
            backend_manager,
            api_key=IONQ_API_KEY,
            use_cirq=True
        )
        backend_manager.set_default_backend('ionq_sim_cirq')

    # Task 1: Pre-training data
    print("Phase 1: Pre-training on Task A")
    X_pretrain = np.random.randn(50, 8)
    y_pretrain = np.random.randint(0, 2, 50)

    quantum_sdk = "mock" if USE_MOCK else "ionq"
    quantum_target = "simulator" if USE_MOCK else (IONQ_TARGET or "simulator")

    pretrain_config = TrainingConfig(
        pinecone_api_key=PINECONE_API_KEY or "mock-key",
        pinecone_environment=PINECONE_ENVIRONMENT or "us-east-1",
        pinecone_index_name="quantum-transfer-learning",
        quantum_sdk=quantum_sdk,
        quantum_target=quantum_target,
        quantum_api_key=IONQ_API_KEY if not USE_MOCK else None,
        learning_rate=0.01,
        batch_size=10,
        epochs=3,
        n_qubits=8,
        circuit_depth=3
    )

    trainer = QuantumTrainer(pretrain_config, backend_manager)
    model = QuantumModel(
        input_dim=8,
        n_qubits=8,
        output_dim=2,
        backend=backend_manager.get_backend(),
        depth=3
    )

    class SimpleDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        async def __aiter__(self):
            for i in range(0, len(self.X), self.batch_size):
                batch_x = self.X[i:i+self.batch_size]
                batch_y = np.eye(2)[self.y[i:i+self.batch_size]]
                yield batch_x, batch_y

    await trainer.train(
        model=model,
        train_loader=SimpleDataLoader(X_pretrain, y_pretrain, 10),
        epochs=3
    )

    print(f"Pre-training complete. Final loss: {trainer.training_history[-1].loss:.4f}")

    # Task 2: Fine-tuning
    print("\nPhase 2: Fine-tuning on Task B")
    X_finetune = np.random.randn(30, 8)
    y_finetune = np.random.randint(0, 2, 30)

    # Freeze first layer parameters
    model.quantum_layer.freeze_parameters(list(range(0, 8)))
    print("Frozen first 8 parameters")

    # Fine-tune with lower learning rate
    finetune_config = pretrain_config
    finetune_config.learning_rate = 0.001
    finetune_config.epochs = 2

    finetune_trainer = QuantumTrainer(finetune_config, backend_manager)

    await finetune_trainer.train(
        model=model,
        train_loader=SimpleDataLoader(X_finetune, y_finetune, 10),
        epochs=2
    )

    print(f"Fine-tuning complete. Final loss: {finetune_trainer.training_history[-1].loss:.4f}")
    print("\nTransfer learning example complete!")


# ============================================================================
# Example 4: Multiple Backend Comparison
# ============================================================================

async def example_4_backend_comparison():
    """
    Compare training on different quantum backends
    """
    print("\n" + "="*70)
    print("Example 4: Multiple Backend Comparison")
    print("="*70 + "\n")

    # Create backend manager with multiple backends
    backend_manager = BackendManager()

    from q_store.backends.backend_manager import MockQuantumBackend

    # Register different mock backends (simulating different QPUs)
    backend_manager.register_backend(
        "mock_fast",
        MockQuantumBackend(name="fast", max_qubits=10, noise_level=0.0),
        metadata={'description': 'Fast ideal simulator'}
    )

    backend_manager.register_backend(
        "mock_noisy",
        MockQuantumBackend(name="noisy", max_qubits=10, noise_level=0.1),
        metadata={'description': 'Noisy simulator'}
    )

    # Training data
    X_train = np.random.randn(40, 8)
    y_train = np.random.randint(0, 2, 40)

    class SimpleDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        async def __aiter__(self):
            for i in range(0, len(self.X), self.batch_size):
                batch_x = self.X[i:i+self.batch_size]
                batch_y = np.eye(2)[self.y[i:i+self.batch_size]]
                yield batch_x, batch_y

    results = {}

    for backend_name in ["mock_fast", "mock_noisy"]:
        print(f"\nTraining on backend: {backend_name}")

        # Set backend
        backend_manager.set_default_backend(backend_name)

        # Configure
        quantum_sdk = "mock" if USE_MOCK else "ionq"
        quantum_target = "simulator" if USE_MOCK else (IONQ_TARGET or "simulator")

        config = TrainingConfig(
            pinecone_api_key=PINECONE_API_KEY or "mock-key",
            pinecone_environment=PINECONE_ENVIRONMENT or "us-east-1",
            pinecone_index_name=f"quantum-backend-{backend_name}",
            quantum_sdk=quantum_sdk,
            quantum_target=quantum_target,
            quantum_api_key=IONQ_API_KEY if not USE_MOCK else None,
            learning_rate=0.01,
            batch_size=10,
            epochs=3,
            n_qubits=8,
            circuit_depth=2
        )

        # Create trainer and model
        trainer = QuantumTrainer(config, backend_manager)
        model = QuantumModel(
            input_dim=8,
            n_qubits=8,
            output_dim=2,
            backend=backend_manager.get_backend(),
            depth=2
        )

        # Train
        await trainer.train(
            model=model,
            train_loader=SimpleDataLoader(X_train, y_train, 10),
            epochs=3
        )

        # Collect metrics
        final_loss = trainer.training_history[-1].loss
        avg_time = np.mean([m.epoch_time_ms for m in trainer.training_history])

        results[backend_name] = {
            'final_loss': final_loss,
            'avg_epoch_time': avg_time
        }

        print(f"Final loss: {final_loss:.4f}")
        print(f"Avg epoch time: {avg_time:.2f}ms")

    print("\nBackend Comparison Summary:")
    print("-" * 50)
    for backend, metrics in results.items():
        print(f"{backend}:")
        print(f"  Loss: {metrics['final_loss']:.4f}")
        print(f"  Time: {metrics['avg_epoch_time']:.2f}ms")


# ============================================================================
# Example 5: Quantum Database with ML Training Integration
# ============================================================================

async def example_5_database_ml_integration():
    """
    Use quantum database for training data management
    """
    print("\n" + "="*70)
    print("Example 5: Quantum Database with ML Training Integration")
    print("="*70 + "\n")

    # Note: This is a simplified example showing the integration pattern
    # Full implementation requires a configured Pinecone instance

    print("Database-ML integration demonstrates:")
    print("1. Store training datasets in quantum database")
    print("2. Load data efficiently for ML training")
    print("3. Track model checkpoints in database")
    print("4. Version control for quantum circuits")
    print("\nFor full database functionality, configure Pinecone credentials.")


# ============================================================================
# Example 6: Advanced: Quantum Autoencoder
# ============================================================================

async def example_6_quantum_autoencoder():
    """
    Train a quantum autoencoder for dimensionality reduction
    """
    print("\n" + "="*70)
    print("Example 6: Quantum Autoencoder")
    print("="*70 + "\n")

    backend_manager = create_default_backend_manager()

    # Configure IonQ backend if not using mock
    if not USE_MOCK and IONQ_API_KEY:
        from q_store.backends import setup_ionq_backends
        backend_manager = await setup_ionq_backends(
            backend_manager,
            api_key=IONQ_API_KEY,
            use_cirq=True
        )
        backend_manager.set_default_backend('ionq_sim_cirq')

    # High-dimensional input
    input_dim = 16
    latent_dim = 4

    # Generate data
    X = np.random.randn(50, input_dim)

    # Encoder: input_dim -> latent_dim
    encoder = QuantumModel(
        input_dim=input_dim,
        n_qubits=4,
        output_dim=latent_dim,
        backend=backend_manager.get_backend(),
        depth=3
    )

    # Decoder: latent_dim -> input_dim
    decoder = QuantumModel(
        input_dim=latent_dim,
        n_qubits=4,
        output_dim=input_dim,
        backend=backend_manager.get_backend(),
        depth=3
    )

    print(f"Encoder: {input_dim} -> {latent_dim}")
    print(f"Decoder: {latent_dim} -> {input_dim}")
    print("\nAutoencoder architecture defined!")
    print("Training would minimize reconstruction loss: ||x - decoder(encoder(x))||²")


# ============================================================================
# Main: Run All Examples
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Quantum-Native Database v3.2 - ML Training Examples")
    print("="*70)

    examples = [
        ("Basic Training", example_1_basic_training),
        ("Data Encoding", example_2_data_encoding),
        ("Transfer Learning", example_3_transfer_learning),
        ("Backend Comparison", example_4_backend_comparison),
        ("Database Integration", example_5_database_ml_integration),
        ("Quantum Autoencoder", example_6_quantum_autoencoder),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}", exc_info=True)

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Q-Store v3.2 Examples')
    parser.add_argument('--no-mock', action='store_true',
                        help='Use real backends (IonQ, Pinecone) instead of mock')
    parser.add_argument('--pinecone-api-key', type=str,
                        help='Pinecone API key (or set PINECONE_API_KEY env var)')
    parser.add_argument('--pinecone-env', type=str,
                        help='Pinecone environment (or set PINECONE_ENVIRONMENT env var, default: us-east-1)')
    parser.add_argument('--ionq-api-key', type=str,
                        help='IonQ API key (or set IONQ_API_KEY env var)')
    parser.add_argument('--ionq-target', type=str,
                        help='IonQ target: simulator, ionq_simulator, ionq_qpu (or set IONQ_TARGET env var, default: simulator)')

    args = parser.parse_args()

    # Set global configuration - prioritize command line args, then env vars, then defaults
    USE_MOCK = not args.no_mock
    PINECONE_API_KEY = args.pinecone_api_key or os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = args.pinecone_env or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    IONQ_API_KEY = args.ionq_api_key or os.getenv('IONQ_API_KEY')
    IONQ_TARGET = args.ionq_target or os.getenv('IONQ_TARGET', 'simulator')

    if not USE_MOCK:
        print("\n" + "="*70)
        print("Running with REAL backends (not mock)")
        print("="*70)
        print(f"Pinecone API Key: {'✓ Set' if PINECONE_API_KEY else '✗ Missing'}")
        print(f"Pinecone Environment: {PINECONE_ENVIRONMENT}")
        print(f"IonQ API Key: {'✓ Set' if IONQ_API_KEY else '✗ Missing'}")
        print(f"IonQ Target: {IONQ_TARGET}")
        print("="*70 + "\n")

        if not PINECONE_API_KEY:
            print("WARNING: PINECONE_API_KEY not set. Examples may fail.")
        if not IONQ_API_KEY:
            print("WARNING: IONQ_API_KEY not set. Quantum features may not work.")
    else:
        print("\n" + "="*70)
        print("Running with MOCK backends (for testing)")
        print("="*70 + "\n")

    asyncio.run(main())
