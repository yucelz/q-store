"""
Q-Store v3.3 Example - Hardware-Efficient ML Training
Demonstrates 24-48x faster training through algorithmic optimization

Key v3.3 Features:
- SPSA gradient estimation (2 circuits vs 96)
- Hardware-efficient quantum layers (33% fewer parameters)
- Circuit batching and caching
- Adaptive gradient optimization
- Performance tracking
"""

import asyncio
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run v3.3 optimized quantum ML training"""

    print("=" * 70)
    print("Q-Store v3.3: Hardware-Efficient Quantum ML Training")
    print("=" * 70)
    print()

    # Import q_store components
    from q_store.backends.backend_manager import BackendManager
    from q_store.ml import (
        QuantumTrainer,
        QuantumModel,
        TrainingConfig,
        # v3.3 components
        HardwareEfficientQuantumLayer,
        SPSAGradientEstimator,
        AdaptiveGradientOptimizer,
        PerformanceTracker
    )

    # Create sample dataset
    print("📊 Creating sample dataset...")
    n_samples = 100
    n_features = 8

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, size=(n_samples, 2))  # Binary classification

    print(f"  - Training samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print()

    # Initialize backend
    print("🔧 Initializing quantum backend...")
    backend_manager = BackendManager()
    await backend_manager.initialize(
        sdk='mock',  # Use 'cirq' for real IonQ execution
        api_key='mock-key',
        target='simulator'
    )
    backend = backend_manager.get_backend()
    print(f"  - Backend: {backend.get_backend_info()['provider']}")
    print(f"  - Type: {backend.get_backend_info()['backend_type']}")
    print()

    # v3.3 Configuration with optimizations
    print("⚙️  Configuring v3.3 optimizations...")
    config = TrainingConfig(
        # Backend
        quantum_sdk='mock',
        quantum_api_key='mock-key',
        quantum_target='simulator',

        # Model
        n_qubits=8,
        circuit_depth=2,

        # Training
        learning_rate=0.01,
        batch_size=10,
        epochs=5,
        shots_per_circuit=1000,

        # v3.3 NEW: Performance optimizations
        gradient_method='spsa',  # 🔥 SPSA for 48x speedup
        hardware_efficient_ansatz=True,  # 🔥 33% fewer parameters
        enable_circuit_cache=True,  # 🔥 Cache compiled circuits
        enable_batch_execution=True,  # 🔥 Batch circuit submission
        enable_performance_tracking=True,  # 🔥 Track performance

        # SPSA parameters
        spsa_c_initial=0.1,
        spsa_a_initial=0.01,

        # Pinecone (optional)
        pinecone_api_key='mock-key',
        pinecone_environment='us-east-1',
        pinecone_index_name='quantum-ml-v33'
    )

    print("  ✓ SPSA gradient estimation enabled (2 circuits per batch)")
    print("  ✓ Hardware-efficient ansatz enabled (33% fewer params)")
    print("  ✓ Circuit caching enabled")
    print("  ✓ Batch execution enabled")
    print("  ✓ Performance tracking enabled")
    print()

    # Create trainer
    print("🎯 Creating quantum trainer...")
    trainer = QuantumTrainer(config, backend_manager)
    print(f"  - Gradient method: {config.gradient_method}")
    print(f"  - Circuit cache: {'Enabled' if trainer.circuit_cache else 'Disabled'}")
    print(f"  - Batch manager: {'Enabled' if trainer.batch_manager else 'Disabled'}")
    print()

    # Create model with hardware-efficient layer
    print("🧠 Creating quantum model...")
    model = QuantumModel(
        input_dim=n_features,
        n_qubits=config.n_qubits,
        output_dim=2,
        backend=backend,
        depth=config.circuit_depth,
        hardware_efficient=config.hardware_efficient_ansatz  # v3.3
    )

    if hasattr(model.quantum_layer, 'n_parameters'):
        print(f"  - Model parameters: {model.quantum_layer.n_parameters}")
        print(f"  - Layer type: {type(model.quantum_layer).__name__}")
    else:
        print(f"  - Model parameters: {len(model.parameters)}")
    print()

    # Simple data loader
    class SimpleDataLoader:
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        async def __aiter__(self):
            for i in range(0, len(self.X), self.batch_size):
                batch_X = self.X[i:i + self.batch_size]
                batch_y = self.y[i:i + self.batch_size]
                yield batch_X, batch_y

    # Train model
    print("🚀 Starting training (v3.3 optimized)...")
    print("=" * 70)

    for epoch in range(config.epochs):
        data_loader = SimpleDataLoader(X_train, y_train, config.batch_size)

        metrics = await trainer.train_epoch(model, data_loader, epoch)

        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print(f"  Loss: {metrics.loss:.4f}")
        print(f"  Gradient norm: {metrics.gradient_norm:.4f}")
        print(f"  Circuits executed: {metrics.n_circuit_executions}")
        print(f"  Time: {metrics.epoch_time_ms/1000:.2f}s")

        # Show performance tracker stats
        if trainer.performance_tracker:
            stats = trainer.performance_tracker.get_statistics()
            if stats:
                print(f"  Avg circuits/batch: {stats.get('avg_circuits_per_batch', 'N/A'):.1f}")
                if 'avg_cache_hit_rate' in stats:
                    print(f"  Cache hit rate: {stats['avg_cache_hit_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print()

    # Show final performance statistics
    if trainer.performance_tracker:
        print("📈 Final Performance Statistics")
        print("-" * 70)

        stats = trainer.performance_tracker.get_statistics()

        print(f"Total batches:        {stats.get('total_batches', 0)}")
        print(f"Total circuits:       {stats.get('total_circuits', 0)}")
        print(f"Total time:           {stats.get('total_runtime_s', 0):.2f}s")
        print(f"Circuits/second:      {stats.get('circuits_per_second', 0):.1f}")
        print(f"Average loss:         {stats.get('avg_loss', 0):.4f}")
        print(f"Final loss:           {stats.get('final_loss', 0):.4f}")

        # Estimate speedup vs v3.2
        speedup = trainer.performance_tracker.estimate_speedup(
            baseline_circuits_per_batch=96  # v3.2 with parameter shift
        )

        if speedup:
            print("\n🎉 v3.3 Performance Gains")
            print("-" * 70)
            print(f"Circuit reduction:    {speedup['circuit_reduction_factor']:.1f}x")
            print(f"Est. time speedup:    {speedup['estimated_time_speedup']:.1f}x")
            print(f"Circuits saved/batch: {speedup['circuits_saved_per_batch']:.0f}")

        # Cache statistics
        if trainer.circuit_cache:
            cache_stats = trainer.circuit_cache.get_stats()
            print("\n💾 Cache Performance")
            print("-" * 70)
            print(f"Hit rate:             {cache_stats['hit_rate']*100:.1f}%")
            print(f"Total hits:           {cache_stats['hits']}")
            print(f"Total misses:         {cache_stats['misses']}")
            print(f"Cached circuits:      {cache_stats['compiled_circuits']}")

        print()

    # Show convergence data
    if trainer.performance_tracker:
        convergence = trainer.performance_tracker.get_convergence_data()
        if convergence['batch_losses']:
            print("📊 Training Convergence")
            print("-" * 70)
            print(f"Initial loss:         {convergence['batch_losses'][0]:.4f}")
            print(f"Final loss:           {convergence['batch_losses'][-1]:.4f}")
            improvement = (convergence['batch_losses'][0] - convergence['batch_losses'][-1])
            improvement_pct = (improvement / convergence['batch_losses'][0]) * 100
            print(f"Improvement:          {improvement:.4f} ({improvement_pct:.1f}%)")
            print()

    print("🎓 v3.3 Key Features Demonstrated:")
    print("  ✓ SPSA gradient estimation (2 circuits instead of 96)")
    print("  ✓ Hardware-efficient quantum layer (reduced parameters)")
    print("  ✓ Circuit caching and batching")
    print("  ✓ Performance tracking and analysis")
    print()
    print("For production use with real quantum hardware:")
    print("  - Set quantum_sdk='cirq'")
    print("  - Provide IonQ API key")
    print("  - Set quantum_target='qpu.aria-1' or 'qpu.forte-1'")
    print("  - Use gradient_method='adaptive' for best results")
    print()
    print("Expected performance gain: 24-48x faster than v3.2!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
