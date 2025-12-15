#!/usr/bin/env python3
"""
Quick Start Guide for Q-Store v3.2 ML Training
Demonstrates the simplest way to get started with quantum ML training
"""

import asyncio
import numpy as np
from q_store.ml import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
)
from q_store.backends import create_default_backend_manager


async def quickstart():
    """
    Minimal working example of quantum ML training
    """
    print("\n" + "="*70)
    print("Q-Store v3.2 Quick Start - Quantum ML Training")
    print("="*70 + "\n")
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic training data...")
    np.random.seed(42)
    n_samples = 50
    n_features = 4
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)  # Binary classification
    
    print(f"  Created {n_samples} samples with {n_features} features")
    
    # Step 2: Configure training
    print("\nStep 2: Configuring quantum training...")
    config = TrainingConfig(
        pinecone_api_key="mock-key",  # Use mock for testing
        quantum_sdk="mock",            # Use mock quantum backend
        learning_rate=0.01,
        batch_size=10,
        epochs=5,
        n_qubits=4,
        circuit_depth=2,
        entanglement='linear'
    )
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Qubits: {config.n_qubits}")
    print(f"  Circuit depth: {config.circuit_depth}")
    
    # Step 3: Create backend manager
    print("\nStep 3: Initializing quantum backend...")
    backend_manager = create_default_backend_manager()
    backend_info = backend_manager.get_backend().get_backend_info()
    print(f"  Backend: {backend_info['name']}")
    print(f"  Type: {backend_info['type']}")
    print(f"  Max qubits: {backend_info['max_qubits']}")
    
    # Step 4: Create trainer
    print("\nStep 4: Creating quantum trainer...")
    trainer = QuantumTrainer(config, backend_manager)
    print("  Trainer initialized with optimizer:", config.optimizer)
    
    # Step 5: Create quantum model
    print("\nStep 5: Building quantum model...")
    model = QuantumModel(
        input_dim=n_features,
        n_qubits=config.n_qubits,
        output_dim=2,
        backend=backend_manager.get_backend(),
        depth=config.circuit_depth
    )
    print(f"  Model: {n_features} inputs -> {config.n_qubits} qubits -> 2 outputs")
    print(f"  Total trainable parameters: {len(model.parameters)}")
    
    # Step 6: Create data loader
    print("\nStep 6: Preparing data loader...")
    class SimpleDataLoader:
        """Simple data loader for training"""
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size
        
        async def __aiter__(self):
            """Iterate over batches"""
            for i in range(0, len(self.X), self.batch_size):
                batch_x = self.X[i:i+self.batch_size]
                # One-hot encode labels
                batch_y = np.eye(2)[self.y[i:i+self.batch_size]]
                yield batch_x, batch_y
    
    train_loader = SimpleDataLoader(X_train, y_train, config.batch_size)
    print(f"  Data loader ready with {len(X_train) // config.batch_size} batches")
    
    # Step 7: Train the model
    print("\nStep 7: Training quantum model...")
    print("-" * 70)
    
    await trainer.train(
        model=model,
        train_loader=train_loader,
        epochs=config.epochs
    )
    
    # Step 8: Display results
    print("-" * 70)
    print("\nStep 8: Training complete!")
    print(f"  Final loss: {trainer.training_history[-1].loss:.4f}")
    print(f"  Total epochs: {len(trainer.training_history)}")
    
    # Display training progression
    print("\n  Training progression:")
    for i, metrics in enumerate(trainer.training_history):
        print(f"    Epoch {i}: Loss={metrics.loss:.4f}, "
              f"Grad Norm={metrics.gradient_norm:.4f}, "
              f"Time={metrics.epoch_time_ms/1000:.2f}s")
    
    # Step 9: Test prediction
    print("\nStep 9: Testing prediction...")
    test_sample = X_train[0]
    prediction = await model.forward(test_sample, shots=1000)
    predicted_class = np.argmax(prediction)
    print(f"  Test sample: {test_sample}")
    print(f"  Prediction: {prediction}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Actual class: {y_train[0]}")
    
    print("\n" + "="*70)
    print("Quick Start Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try different hyperparameters (learning_rate, circuit_depth)")
    print("  2. Experiment with different entanglement patterns")
    print("  3. Run full examples: python -m q_store_examples.examples_v3_2")
    print("  4. Check documentation: docs/README_v3_2.md")
    print("\n")


if __name__ == "__main__":
    asyncio.run(quickstart())
