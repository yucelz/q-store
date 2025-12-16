"""
Test script to verify Pinecone and IonQ connection setup
"""
import asyncio
import os
import logging
import pytest
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.integration
async def test_pinecone_connection():
    """Test Pinecone connection and index creation"""
    print("\n" + "="*70)
    print("TEST 1: Pinecone Connection")
    print("="*70)

    pinecone_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_key or pinecone_key.startswith('mock'):
        pytest.skip("PINECONE_API_KEY not set or is mock - skipping")

    try:
        from pinecone.grpc import PineconeGRPC as Pinecone
        from pinecone import ServerlessSpec

        pc = Pinecone(api_key=pinecone_key)
        print(f"✓ Pinecone client initialized")
        print(f"  API Key: {pinecone_key[:10]}...")

        # List existing indexes
        indexes = [index.name for index in pc.list_indexes()]
        print(f"✓ Existing indexes: {indexes if indexes else 'none'}")

        # Test index creation (with a test name)
        test_index_name = "q-store-test-index"
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')

        if test_index_name in indexes:
            print(f"✓ Test index '{test_index_name}' already exists")
        else:
            print(f"Creating test index '{test_index_name}'...")
            pc.create_index(
                name=test_index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )
            print(f"✓ Test index created successfully")

        # Get index
        index = pc.Index(test_index_name)
        print(f"✓ Connected to index: {test_index_name}")

        return True

    except Exception as e:
        print(f"✗ Pinecone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.integration
async def test_ionq_connection():
    """Test IonQ backend configuration"""
    print("\n" + "="*70)
    print("TEST 2: IonQ Backend Connection")
    print("="*70)

    ionq_key = os.getenv('IONQ_API_KEY')
    if not ionq_key:
        pytest.skip("IONQ_API_KEY not set - skipping")

    try:
        from q_store.backends import BackendManager, setup_ionq_backends

        print(f"✓ IonQ API Key: {ionq_key[:10]}...")

        # Create backend manager
        manager = BackendManager()
        print("✓ Backend manager created")

        # Setup IonQ backends
        manager = await setup_ionq_backends(
            manager,
            api_key=ionq_key,
            use_cirq=True
        )
        print("✓ IonQ backends configured")

        # List backends
        backends = manager.list_backends()
        print(f"✓ Available backends: {backends}")

        # Check for IonQ simulator
        if 'ionq_sim_cirq' in backends:
            print("✓ IonQ simulator backend registered")
            manager.set_default_backend('ionq_sim_cirq')
            backend = manager.get_backend()
            info = backend.get_backend_info()
            print(f"✓ Backend info: {info}")
            return True
        else:
            print("✗ IonQ simulator backend not found")
            return False

    except Exception as e:
        print(f"✗ IonQ test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trainer_initialization():
    """Test QuantumTrainer with Pinecone and IonQ"""
    print("\n" + "="*70)
    print("TEST 3: QuantumTrainer Initialization")
    print("="*70)

    from q_store.ml import QuantumTrainer, TrainingConfig, QuantumModel
    from q_store.backends import BackendManager, setup_ionq_backends
    import numpy as np

    pinecone_key = os.getenv('PINECONE_API_KEY')
    ionq_key = os.getenv('IONQ_API_KEY')

    # Determine if we're using real backends
    use_real_pinecone = pinecone_key and not pinecone_key.startswith('mock')
    use_real_ionq = ionq_key is not None

    try:
        # Create configuration
        config = TrainingConfig(
            pinecone_api_key=pinecone_key if use_real_pinecone else 'mock-key',
            pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
            pinecone_index_name='quantum-ml-test',
            quantum_sdk='ionq' if use_real_ionq else 'mock',
            quantum_target='simulator',
            quantum_api_key=ionq_key if use_real_ionq else None,
            learning_rate=0.01,
            batch_size=2,
            epochs=1,
            n_qubits=4,
            circuit_depth=1
        )
        print("✓ TrainingConfig created")
        print(f"  Pinecone: {'Real' if use_real_pinecone else 'Mock'}")
        print(f"  IonQ: {'Real' if use_real_ionq else 'Mock'}")

        # Create backend manager
        backend_manager = BackendManager()

        if use_real_ionq:
            backend_manager = await setup_ionq_backends(
                backend_manager,
                api_key=ionq_key,
                use_cirq=True
            )
            backend_manager.set_default_backend('ionq_sim_cirq')
            print("✓ IonQ backend configured")
        else:
            from q_store.backends.backend_manager import MockQuantumBackend
            mock = MockQuantumBackend(name="test", max_qubits=10, noise_level=0.0)
            backend_manager.register_backend("mock", mock, set_as_default=True)
            print("✓ Mock backend configured")

        # Create trainer
        trainer = QuantumTrainer(config, backend_manager)
        print("✓ QuantumTrainer created")

        # Create model
        model = QuantumModel(
            input_dim=4,
            n_qubits=4,
            output_dim=2,
            backend=backend_manager.get_backend(),
            depth=1
        )
        print("✓ QuantumModel created")

        # Create simple data
        X = np.random.randn(4, 4)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        class SimpleLoader:
            def __init__(self, X, y, batch_size):
                self.X = X
                self.y = y
                self.batch_size = batch_size

            async def __aiter__(self):
                for i in range(0, len(self.X), self.batch_size):
                    yield self.X[i:i+self.batch_size], self.y[i:i+self.batch_size]

        loader = SimpleLoader(X, y, 2)
        print("✓ DataLoader created")

        # Test training (this will call _init_pinecone)
        print("Starting training (1 epoch)...")
        await trainer.train(model=model, train_loader=loader, epochs=1)
        print("✓ Training completed")

        if use_real_pinecone:
            if trainer._pinecone_initialized:
                print("✓ Pinecone was initialized during training")
                print(f"  Index: {config.pinecone_index_name}")
            else:
                print("⚠️  Pinecone was not initialized")

        return True

    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PINECONE & IONQ CONNECTION TESTS")
    print("="*70)

    results = {}

    # Test 1: Pinecone
    results['pinecone'] = await test_pinecone_connection()

    # Test 2: IonQ
    results['ionq'] = await test_ionq_connection()

    # Test 3: Trainer
    results['trainer'] = await test_trainer_initialization()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL (or skipped)"
        print(f"{test_name.upper()}: {status}")

    print("\n" + "="*70)

    if not any(results.values()):
        print("\n⚠️  All tests were skipped or failed.")
        print("To test real connections, set environment variables:")
        print("  PINECONE_API_KEY=your-key")
        print("  PINECONE_ENVIRONMENT=us-east-1")
        print("  IONQ_API_KEY=your-key")

    print()


if __name__ == "__main__":
    asyncio.run(main())
