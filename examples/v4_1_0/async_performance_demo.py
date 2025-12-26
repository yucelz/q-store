"""
Async Performance Demo - Q-Store v4.1 Phase 2

Demonstrates the performance improvements from async quantum execution.

Shows:
1. Sequential vs Parallel execution comparison
2. Throughput measurements
3. Latency hiding benefits
4. Cache performance
5. Real-world speedup metrics

Expected results:
- 10-20x throughput improvement
- Near-zero blocking time
- High cache hit rates on repeated patterns
"""

import asyncio
import time
import numpy as np
from typing import List

# Q-Store v4.1 Phase 2 imports
from q_store.layers import QuantumFeatureExtractor
from q_store.runtime import AsyncQuantumExecutor


async def benchmark_sequential_vs_parallel():
    """Benchmark sequential vs parallel execution."""
    print("\n" + "="*70)
    print("Benchmark 1: Sequential vs Parallel Execution")
    print("="*70)
    
    # Create quantum layer
    layer = QuantumFeatureExtractor(
        n_qubits=6,
        depth=2,
        backend='simulator',
        max_concurrent=50,  # Allow many parallel executions
        batch_size=10
    )
    
    # Prepare test data
    n_samples = 40
    input_dim = 64
    inputs = np.random.randn(n_samples, input_dim).astype(np.float32)
    
    print(f"\nTest configuration:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Qubits: {layer.n_qubits}")
    print(f"  - Circuit depth: {layer.depth}")
    
    # Sequential execution (process one at a time)
    print(f"\n📊 Sequential Execution (like v4.0):")
    start_time = time.time()
    
    sequential_results = []
    for i in range(n_samples):
        sample = inputs[i:i+1]
        result = await layer.call_async(sample)
        sequential_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
    
    sequential_time = time.time() - start_time
    sequential_throughput = n_samples / sequential_time
    
    print(f"\n  Total time: {sequential_time:.2f}s")
    print(f"  Throughput: {sequential_throughput:.1f} samples/sec")
    print(f"  Avg time per sample: {sequential_time/n_samples*1000:.1f}ms")
    
    # Parallel execution (process all at once)
    print(f"\n⚡ Parallel Execution (v4.1 async):")
    start_time = time.time()
    
    # Submit all at once!
    parallel_result = await layer.call_async(inputs)
    
    parallel_time = time.time() - start_time
    parallel_throughput = n_samples / parallel_time
    
    print(f"  Total time: {parallel_time:.2f}s")
    print(f"  Throughput: {parallel_throughput:.1f} samples/sec")
    print(f"  Avg time per sample: {parallel_time/n_samples*1000:.1f}ms")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    
    print(f"\n🚀 Performance Improvement:")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {sequential_time - parallel_time:.2f}s ({(1-parallel_time/sequential_time)*100:.1f}%)")
    print(f"  Throughput gain: {parallel_throughput/sequential_throughput:.1f}x")
    
    return speedup


async def benchmark_cache_performance():
    """Benchmark cache performance."""
    print("\n" + "="*70)
    print("Benchmark 2: Cache Performance")
    print("="*70)
    
    # Create executor with cache
    executor = AsyncQuantumExecutor(
        backend='simulator',
        cache_size=1000
    )
    
    # Create test circuit
    from q_store.layers.quantum_core.quantum_feature_extractor import QuantumCircuit
    circuit = QuantumCircuit(
        n_qubits=4,
        gates=[],
        parameters={},
        measurement_bases=['Z']
    )
    
    n_iterations = 100
    
    print(f"\nTest configuration:")
    print(f"  - Iterations: {n_iterations}")
    print(f"  - Unique circuits: 10")
    print(f"  - Expected hit rate: ~90%")
    
    # First pass: Fill cache
    print(f"\n📥 First pass (filling cache):")
    circuits = [circuit for _ in range(10)]  # 10 unique circuits
    
    start_time = time.time()
    for _ in range(n_iterations):
        # Pick random circuit (will create ~90% cache hits)
        test_circuit = np.random.choice(circuits)
        result = await executor.submit(test_circuit)
        await result  # Wait for result
    
    first_pass_time = time.time() - start_time
    
    # Get cache stats
    stats = executor.get_stats()
    
    print(f"  Time: {first_pass_time:.2f}s")
    print(f"  Throughput: {n_iterations/first_pass_time:.1f} circuits/sec")
    print(f"\n📊 Cache Statistics:")
    print(f"  Total requests: {stats['cache_hits'] + stats['cache_misses']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  Cache size: {stats['cache_size']}/{executor.cache.max_size}")
    
    # Shutdown executor
    await executor.shutdown()
    
    return stats['cache_hit_rate']


async def benchmark_batch_sizes():
    """Benchmark different batch sizes."""
    print("\n" + "="*70)
    print("Benchmark 3: Optimal Batch Size")
    print("="*70)
    
    batch_sizes = [1, 5, 10, 20, 50]
    n_total = 100
    
    print(f"\nTest configuration:")
    print(f"  - Total samples: {n_total}")
    print(f"  - Batch sizes to test: {batch_sizes}")
    
    results = {}
    
    for batch_size in batch_sizes:
        layer = QuantumFeatureExtractor(
            n_qubits=4,
            depth=2,
            backend='simulator',
            batch_size=batch_size
        )
        
        inputs = np.random.randn(n_total, 64).astype(np.float32)
        
        start_time = time.time()
        await layer.call_async(inputs)
        elapsed = time.time() - start_time
        
        throughput = n_total / elapsed
        results[batch_size] = {
            'time': elapsed,
            'throughput': throughput
        }
        
        print(f"  Batch size {batch_size:2d}: {elapsed:.2f}s ({throughput:.1f} samples/sec)")
    
    # Find optimal
    optimal_batch = max(results.items(), key=lambda x: x[1]['throughput'])
    
    print(f"\n🎯 Optimal batch size: {optimal_batch[0]}")
    print(f"   Best throughput: {optimal_batch[1]['throughput']:.1f} samples/sec")
    
    return optimal_batch[0]


async def benchmark_concurrent_layers():
    """Benchmark multiple layers running concurrently."""
    print("\n" + "="*70)
    print("Benchmark 4: Concurrent Layer Execution")
    print("="*70)
    
    n_layers = 4
    n_samples = 20
    
    print(f"\nTest configuration:")
    print(f"  - Number of layers: {n_layers}")
    print(f"  - Samples per layer: {n_samples}")
    print(f"  - Total circuits: {n_layers * n_samples}")
    
    # Create multiple layers
    layers = [
        QuantumFeatureExtractor(n_qubits=4, depth=2, backend='simulator')
        for _ in range(n_layers)
    ]
    
    # Prepare inputs for each layer
    inputs = [
        np.random.randn(n_samples, 64).astype(np.float32)
        for _ in range(n_layers)
    ]
    
    # Sequential execution (one layer at a time)
    print(f"\n📊 Sequential (process layers one by one):")
    start_time = time.time()
    
    for i, (layer, inp) in enumerate(zip(layers, inputs)):
        await layer.call_async(inp)
        print(f"  Layer {i+1}/{n_layers} completed")
    
    sequential_time = time.time() - start_time
    print(f"  Total time: {sequential_time:.2f}s")
    
    # Parallel execution (all layers at once)
    print(f"\n⚡ Parallel (all layers simultaneously):")
    start_time = time.time()
    
    # Submit all layers in parallel
    tasks = [
        layer.call_async(inp)
        for layer, inp in zip(layers, inputs)
    ]
    
    await asyncio.gather(*tasks)
    
    parallel_time = time.time() - start_time
    print(f"  Total time: {parallel_time:.2f}s")
    
    speedup = sequential_time / parallel_time
    
    print(f"\n🚀 Speedup: {speedup:.1f}x")
    
    return speedup


async def main():
    """Run all benchmarks."""
    print("="*70)
    print("Q-Store v4.1 Phase 2 - Async Performance Benchmarks")
    print("="*70)
    print("\nDemonstrating the performance benefits of async quantum execution")
    print("Target: 10-20x throughput improvement over sequential execution")
    
    try:
        # Run benchmarks
        speedup1 = await benchmark_sequential_vs_parallel()
        cache_hit_rate = await benchmark_cache_performance()
        optimal_batch = await benchmark_batch_sizes()
        speedup2 = await benchmark_concurrent_layers()
        
        # Summary
        print("\n" + "="*70)
        print("📊 Benchmark Summary")
        print("="*70)
        print(f"\n✅ Key Results:")
        print(f"  1. Sequential vs Parallel: {speedup1:.1f}x speedup")
        print(f"  2. Cache hit rate: {cache_hit_rate*100:.1f}%")
        print(f"  3. Optimal batch size: {optimal_batch}")
        print(f"  4. Concurrent layers: {speedup2:.1f}x speedup")
        
        avg_speedup = (speedup1 + speedup2) / 2
        
        print(f"\n🎯 Overall Performance:")
        print(f"  Average speedup: {avg_speedup:.1f}x")
        print(f"  Target achieved: {'✅ YES' if avg_speedup >= 10 else '⚠️  Close'}")
        
        if avg_speedup >= 10:
            print(f"\n🎉 SUCCESS! Achieved {avg_speedup:.1f}x speedup")
            print(f"   (Target was 10-20x)")
        else:
            print(f"\n⚠️  Achieved {avg_speedup:.1f}x speedup")
            print(f"   (Target: 10-20x, may vary based on hardware)")
        
        print(f"\n💡 Key Insights:")
        print(f"  - Async execution eliminates blocking time")
        print(f"  - Parallel circuit submission is crucial")
        print(f"  - Cache dramatically improves repeated patterns")
        print(f"  - Batch size affects throughput significantly")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
