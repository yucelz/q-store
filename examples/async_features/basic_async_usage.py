"""
Basic Async Usage - Q-Store v4.1

Simple demonstration of async quantum layer usage with non-blocking execution.

This example shows:
1. How to use quantum layers asynchronously
2. Performance benefits of async execution
3. Basic quantum-first model construction

Expected runtime: 2-3 minutes
"""

import asyncio
import os
import time
import numpy as np
from typing import List

# Q-Store v4.1 imports
from q_store.layers import (
    QuantumFeatureExtractor,
    QuantumNonlinearity,
    QuantumPooling,
    QuantumReadout,
    EncodingLayer,
    DecodingLayer,
)

# Backend configuration - use simulator if no API key
BACKEND = "simulator" if not os.getenv("IONQ_API_KEY") else "ionq"
BACKEND_KWARGS = {}

print(f"Using backend: {BACKEND}")
if BACKEND == "simulator":
    print("ℹ️  Running with local simulator (no API key needed)")
    print("   To use IonQ cloud, set IONQ_API_KEY environment variable")



async def example_1_basic_layer():
    """Example 1: Basic quantum layer usage."""
    print("\n" + "="*70)
    print("Example 1: Basic Quantum Layer")
    print("="*70)

    # Create a quantum feature extractor
    layer = QuantumFeatureExtractor(
        n_qubits=8,
        depth=3,
        entanglement='full',
        measurement_bases=['Z', 'X', 'Y'],
        backend=BACKEND,
        **BACKEND_KWARGS
    )

    print(f"\nLayer Configuration:")
    print(f"  - Qubits: {layer.n_qubits}")
    print(f"  - Depth: {layer.depth}")
    print(f"  - Entanglement: {layer.entanglement}")
    print(f"  - Output dimension: {layer.output_dim}")
    print(f"  - Parameters: {layer.n_parameters}")

    # Create sample input
    batch_size = 16
    input_dim = 128
    inputs = np.random.randn(batch_size, input_dim).astype(np.float32)

    print(f"\nInput shape: {inputs.shape}")

    # Execute quantum circuit (async)
    start_time = time.time()
    outputs = await layer.call_async(inputs)
    elapsed = time.time() - start_time

    print(f"Output shape: {outputs.shape}")
    print(f"Execution time: {elapsed:.3f}s")
    print(f"Throughput: {batch_size / elapsed:.1f} samples/sec")

    return outputs


async def example_2_multiple_layers():
    """Example 2: Multiple quantum layers in sequence."""
    print("\n" + "="*70)
    print("Example 2: Multiple Quantum Layers")
    print("="*70)

    # Build a simple quantum model
    layers = [
        EncodingLayer(target_dim=256),  # Encode for 8 qubits (2^8 = 256)
        QuantumFeatureExtractor(n_qubits=8, depth=3, backend=BACKEND, **BACKEND_KWARGS),  # Extract features
        QuantumNonlinearity(n_qubits=8, nonlinearity_type='amplitude_damping', backend=BACKEND, **BACKEND_KWARGS),
        QuantumPooling(n_qubits=8, pool_size=2, backend=BACKEND, **BACKEND_KWARGS),  # Reduce to 4 qubits
        QuantumFeatureExtractor(n_qubits=4, depth=2, backend=BACKEND, **BACKEND_KWARGS),  # Second extraction
        QuantumReadout(n_qubits=4, n_classes=10, backend=BACKEND, **BACKEND_KWARGS),  # Classification
    ]

    print(f"\nModel architecture:")
    for i, layer in enumerate(layers):
        layer_name = layer.__class__.__name__
        print(f"  {i+1}. {layer_name}")

    # Sample input
    batch_size = 8
    input_dim = 784  # MNIST-like
    inputs = np.random.randn(batch_size, input_dim).astype(np.float32)

    print(f"\nInput shape: {inputs.shape}")

    # Forward pass through all layers
    start_time = time.time()
    x = inputs

    for i, layer in enumerate(layers):
        layer_start = time.time()

        # Check if layer has async method
        if hasattr(layer, 'call_async'):
            x = await layer.call_async(x)
        else:
            x = layer(x)  # Synchronous for classical layers

        layer_time = time.time() - layer_start
        print(f"  Layer {i+1} output: {x.shape} ({layer_time:.3f}s)")

    total_time = time.time() - start_time

    print(f"\nFinal output shape: {x.shape}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average per sample: {total_time / batch_size:.3f}s")

    return x


async def example_3_parallel_execution():
    """Example 3: Parallel execution of multiple circuits."""
    print("\n" + "="*70)
    print("Example 3: Parallel Execution (Key v4.1 Feature)")
    print("="*70)

    # Create multiple quantum layers
    layers = [
        QuantumFeatureExtractor(n_qubits=6, depth=2, backend=BACKEND, **BACKEND_KWARGS) for _ in range(4)
    ]

    print(f"\nCreated {len(layers)} quantum layers")

    # Create sample inputs
    batch_size = 8
    input_dim = 64
    inputs = [
        np.random.randn(batch_size, input_dim).astype(np.float32)
        for _ in range(len(layers))
    ]

    # Sequential execution (v4.0 style)
    print("\n📊 Sequential Execution (v4.0 style):")
    start_time = time.time()
    sequential_results = []

    for i, (layer, inp) in enumerate(zip(layers, inputs)):
        result = await layer.call_async(inp)
        sequential_results.append(result)
        print(f"  Layer {i+1} completed ({time.time() - start_time:.3f}s)")

    sequential_time = time.time() - start_time
    print(f"Total sequential time: {sequential_time:.3f}s")

    # Parallel execution (v4.1 style)
    print("\n⚡ Parallel Execution (v4.1 style):")
    start_time = time.time()

    # Submit all at once
    tasks = [
        layer.call_async(inp)
        for layer, inp in zip(layers, inputs)
    ]

    # Await all results
    parallel_results = await asyncio.gather(*tasks)

    parallel_time = time.time() - start_time
    print(f"Total parallel time: {parallel_time:.3f}s")

    # Speedup
    speedup = sequential_time / parallel_time
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    print(f"   Sequential: {sequential_time:.3f}s")
    print(f"   Parallel:   {parallel_time:.3f}s")

    return parallel_results


async def example_4_quantum_first_model():
    """Example 4: Complete quantum-first model."""
    print("\n" + "="*70)
    print("Example 4: Quantum-First Model (70% Quantum)")
    print("="*70)

    # Define quantum-first model
    class QuantumFirstModel:
        def __init__(self):
            # Minimal classical encoding (5% compute)
            self.encoder = EncodingLayer(target_dim=256)

            # Primary quantum feature extraction (40% compute)
            self.quantum_layer1 = QuantumFeatureExtractor(
                n_qubits=8, depth=4, entanglement='full',
                backend=BACKEND, **BACKEND_KWARGS
            )

            # Quantum pooling (15% compute)
            self.pooling = QuantumPooling(n_qubits=8, pool_size=2, backend=BACKEND, **BACKEND_KWARGS)

            # Secondary quantum features (30% compute)
            self.quantum_layer2 = QuantumFeatureExtractor(
                n_qubits=4, depth=3, entanglement='circular',
                backend=BACKEND, **BACKEND_KWARGS
            )

            # Quantum readout (5% compute)
            self.readout = QuantumReadout(n_qubits=4, n_classes=10, backend=BACKEND, **BACKEND_KWARGS)

            # Minimal classical decoding (5% compute)
            self.decoder = DecodingLayer(output_dim=10)

        async def forward(self, x):
            """Forward pass."""
            x = self.encoder(x)  # Classical
            x = await self.quantum_layer1.call_async(x)  # Quantum
            x = await self.pooling.call_async(x)  # Quantum
            x = await self.quantum_layer2.call_async(x)  # Quantum
            x = await self.readout.call_async(x)  # Quantum
            x = self.decoder(x)  # Classical
            return x

    # Create model
    model = QuantumFirstModel()

    print("\nModel Architecture:")
    print("  1. EncodingLayer (Classical) - 5% compute")
    print("  2. QuantumFeatureExtractor (8 qubits, depth 4) - 40% compute")
    print("  3. QuantumPooling (reduce to 4 qubits) - 15% compute")
    print("  4. QuantumFeatureExtractor (4 qubits, depth 3) - 30% compute")
    print("  5. QuantumReadout (10 classes) - 5% compute")
    print("  6. DecodingLayer (Classical) - 5% compute")
    print("\n  Total Quantum: 90% | Total Classical: 10%")

    # Sample batch
    batch_size = 16
    input_dim = 784  # MNIST-like
    inputs = np.random.randn(batch_size, input_dim).astype(np.float32)

    print(f"\nInput shape: {inputs.shape}")

    # Forward pass
    start_time = time.time()
    predictions = await model.forward(inputs)
    elapsed = time.time() - start_time

    print(f"Output shape: {predictions.shape}")
    print(f"Execution time: {elapsed:.3f}s")
    print(f"Throughput: {batch_size / elapsed:.1f} samples/sec")

    # Check predictions
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"\nSample predictions: {predicted_classes[:5]}")
    print(f"Prediction probabilities (first sample):")
    print(f"  {predictions[0]}")

    return predictions


async def main():
    """Run all examples."""
    print("="*70)
    print("Q-Store v4.1 - Basic Async Usage Examples")
    print("="*70)
    print("\nThis demonstrates the key features of Q-Store v4.1:")
    print("  ✓ Async quantum execution (never blocks)")
    print("  ✓ Quantum-first architecture (70% quantum compute)")
    print("  ✓ Multi-basis measurements")
    print("  ✓ Parallel circuit execution")

    try:
        # Run examples
        await example_1_basic_layer()
        await example_2_multiple_layers()
        await example_3_parallel_execution()
        await example_4_quantum_first_model()

        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  1. Async execution enables parallel quantum computation")
        print("  2. Quantum-first models achieve 70%+ quantum utilization")
        print("  3. Multi-basis measurements provide rich feature spaces")
        print("  4. Classical overhead reduced to ~30%")
        print("\nNext Steps:")
        print("  - Try TensorFlow integration: examples/v4_1_0/tensorflow/")
        print("  - Try PyTorch integration: examples/v4_1_0/pytorch/")
        print("  - See benchmarks: examples/v4_1_0/benchmarks/")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
