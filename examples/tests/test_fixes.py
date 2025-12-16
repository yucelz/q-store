#!/usr/bin/env python3
"""
Quick test to verify the fixes for:
1. zfill AttributeError (integer bitstring issue)
2. Pinecone vector storage integration
"""

import asyncio
import numpy as np
from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig
from q_store.backends import create_default_backend_manager

async def test_fixes():
    print("Testing Fixes for v3.2")
    print("=" * 70)

    # Test 1: Verify zfill fix with mock backend
    print("\n1. Testing zfill fix with mock backend...")
    config = TrainingConfig(
        pinecone_api_key="mock-key",
        quantum_sdk="mock",
        learning_rate=0.01,
        batch_size=2,
        epochs=1,
        n_qubits=4,
        circuit_depth=1
    )

    backend_manager = create_default_backend_manager()
    trainer = QuantumTrainer(config, backend_manager)
    model = QuantumModel(4, 4, 2, backend_manager.get_backend(), depth=1)

    # Create simple data loader
    class SimpleDataLoader:
        def __init__(self):
            self.X = np.random.randn(4, 4)
            self.y = np.eye(2)[[0, 1, 0, 1]]

        async def __aiter__(self):
            yield self.X[:2], self.y[:2]

    try:
        await trainer.train(model, SimpleDataLoader(), epochs=1)
        print("✓ No zfill error - FIX SUCCESSFUL!")
    except AttributeError as e:
        if 'zfill' in str(e):
            print(f"✗ zfill error still exists: {e}")
        else:
            raise

    # Test 2: Verify Pinecone integration
    print("\n2. Testing Pinecone vector storage integration...")
    pinecone_stats = trainer.get_pinecone_stats()
    print(f"   Pinecone enabled: {pinecone_stats['enabled']}")
    print(f"   Pinecone initialized: {pinecone_stats['initialized']}")
    print(f"   Vectors stored: {pinecone_stats['vectors_stored']}")
    print(f"   Index name: {pinecone_stats.get('index_name', 'N/A')}")

    if pinecone_stats['enabled'] and pinecone_stats['vectors_stored'] > 0:
        print("✓ Pinecone integration working - vectors being stored!")
    elif not pinecone_stats['enabled']:
        print("ℹ Pinecone disabled (mock mode or no API key)")
    else:
        print("⚠ Pinecone enabled but no vectors stored yet")

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_fixes())
