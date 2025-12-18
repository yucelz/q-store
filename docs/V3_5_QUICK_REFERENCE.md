# Q-Store v3.5 Quick Reference

## Installation
```bash
pip install q-store==3.5.0
```

## Quick Start - Enable All v3.5 Features

```python
from q_store import QuantumDatabase, TrainingConfig, QuantumTrainer

# Configuration with all v3.5 optimizations
config = TrainingConfig(
    # Required
    pinecone_api_key="your-key",
    quantum_api_key="your-ionq-key",
    
    # Enable ALL v3.5 features (recommended)
    enable_all_v35_features=True,
    
    # Basic training params
    n_qubits=8,
    circuit_depth=4,
    batch_size=10,
    epochs=20,
    learning_rate=0.01,
)

# Train model
trainer = QuantumTrainer(config)
await trainer.train(model, train_loader, val_loader)
```

## Individual Feature Configuration

### 1. Multi-Backend Orchestration

**Benefit**: 2-3x throughput by distributing across backends

```python
from q_store.ml import BackendConfig, MultiBackendOrchestrator
from q_store.backends import IonQQuantumBackend

# Configure multiple backends
config = TrainingConfig(
    enable_multi_backend=True,
    backend_configs=[
        # Primary IonQ simulator
        {
            'provider': 'ionq',
            'target': 'simulator',
            'api_key': 'key1',
        },
        # Secondary IonQ simulator (if available)
        {
            'provider': 'ionq',
            'target': 'simulator',
            'api_key': 'key2',
        },
        # Local GPU simulator as fallback
        {
            'provider': 'local',
            'simulator': 'qiskit_aer',
            'device': 'GPU',
        },
    ],
)
```

### 2. Adaptive Circuit Optimization

**Benefit**: 30-40% faster execution with simplified circuits

```python
config = TrainingConfig(
    # Enable adaptive depth
    adaptive_circuit_depth=True,
    
    # Depth schedule
    circuit_depth_schedule='exponential',  # 'linear', 'exponential', 'step'
    min_circuit_depth=2,
    max_circuit_depth=4,
    
    # Optimization techniques
    enable_circuit_optimization=True,
    gate_merging=True,           # Combine consecutive rotations
    identity_removal=True,        # Remove near-zero gates
    entanglement_pruning=True,    # Reduce CNOT depth
)
```

### 3. Adaptive Shot Allocation

**Benefit**: 20-30% time savings from smart shot management

```python
config = TrainingConfig(
    # Enable adaptive shots
    adaptive_shot_allocation=True,
    
    # Shot range
    min_shots=500,    # Early training (fast, noisy)
    base_shots=1000,  # Mid training (balanced)
    max_shots=2000,   # Late training (precise)
)
```

### 4. Natural Gradient Descent

**Benefit**: 2-3x fewer iterations for convergence

```python
config = TrainingConfig(
    # Enable natural gradient
    use_natural_gradient=True,
    
    # QFIM parameters
    natural_gradient_regularization=0.01,
    qfim_cache_size=100,
)
```

## Performance Comparison

| Configuration | Circuits/sec | Batch Time (20) | Speedup |
|--------------|--------------|-----------------|---------|
| v3.3 Baseline | 0.3 | 65s | 1x |
| v3.4 (concurrent) | 0.57 | 35s | 2x |
| v3.5 (all features) | 1.2-1.5 | 15-20s | 4-5x |

## Component APIs

### MultiBackendOrchestrator

```python
from q_store.ml import MultiBackendOrchestrator, BackendConfig

# Create orchestrator
orchestrator = MultiBackendOrchestrator([
    BackendConfig(backend=backend1, name="ionq1", priority=1),
    BackendConfig(backend=backend2, name="ionq2", priority=0),
])

# Execute circuits
results = await orchestrator.execute_distributed(
    circuits=[...],
    shots=1000,
    preserve_order=True
)

# Get statistics
stats = orchestrator.get_statistics()
print(f"Total circuits: {stats['total_circuits_executed']}")
print(f"Backends: {stats['backends']}")
```

### AdaptiveCircuitOptimizer

```python
from q_store.ml import AdaptiveCircuitOptimizer

# Create optimizer
optimizer = AdaptiveCircuitOptimizer(
    initial_depth=4,
    min_depth=2,
    adaptation_schedule='exponential'
)

# Get depth for current epoch
depth = optimizer.get_depth_for_epoch(epoch=10, total_epochs=100)

# Optimize circuit
optimized = optimizer.optimize_circuit(
    circuit={'qubits': 4, 'circuit': [...]},
    target_depth=depth
)
```

### AdaptiveShotAllocator

```python
from q_store.ml import AdaptiveShotAllocator

# Create allocator
allocator = AdaptiveShotAllocator(
    min_shots=500,
    max_shots=2000,
    base_shots=1000
)

# Get shots for current batch
shots = allocator.get_shots_for_batch(
    epoch=10,
    total_epochs=100,
    recent_gradients=[...]  # Optional
)

# Update history
allocator.update_gradient_history(gradient)
```

### NaturalGradientEstimator

```python
from q_store.ml import NaturalGradientEstimator

# Create estimator
estimator = NaturalGradientEstimator(
    backend=backend,
    regularization=0.01,
    use_qfim_cache=True
)

# Estimate natural gradient
result = await estimator.estimate_natural_gradient(
    circuit_fn=lambda p: build_circuit(p),
    parameters=params,
    loss_fn=lambda p: compute_loss(p),
    shots=1000
)

# Access results
gradient = result.gradients
circuits_used = result.n_circuit_executions
```

## Backward Compatibility

### Deprecated Config (v3.4)

```python
# Old way (still works with warning)
config = TrainingConfig(
    use_batch_api=True  # DEPRECATED
)

# New way (recommended)
config = TrainingConfig(
    use_concurrent_submission=True
)
```

### Legacy Client Import

```python
# Old import (deprecated)
from q_store.ml import IonQBatchClient  # Warning

# New import (recommended)
from q_store.ml import IonQConcurrentClient
```

## Troubleshooting

### Check Feature Availability

```python
from q_store.ml import V3_5_AVAILABLE

if V3_5_AVAILABLE:
    print("✅ All v3.5 features available")
else:
    print("❌ v3.5 features not available, using v3.4 fallback")
```

### Verify Configuration

```python
config = TrainingConfig(enable_all_v35_features=True)

# Check what's enabled
print(f"Multi-backend: {config.enable_multi_backend}")
print(f"Adaptive circuit: {config.adaptive_circuit_depth}")
print(f"Adaptive shots: {config.adaptive_shot_allocation}")
print(f"Natural gradient: {config.use_natural_gradient}")
```

### Monitor Performance

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check orchestrator stats
stats = orchestrator.get_statistics()

# Check optimizer stats
opt_stats = optimizer.get_statistics()

# Check allocator stats
shot_stats = allocator.get_statistics()

# Check estimator stats
grad_stats = estimator.get_statistics()
```

## Common Patterns

### Full Training Loop with v3.5

```python
import asyncio
from q_store import TrainingConfig, QuantumTrainer, QuantumModel

async def train():
    # Setup
    config = TrainingConfig(
        pinecone_api_key="...",
        quantum_api_key="...",
        enable_all_v35_features=True,
        n_qubits=8,
        epochs=20,
    )
    
    # Create model
    model = QuantumModel(
        input_dim=64,
        n_qubits=8,
        output_dim=10,
        backend=backend,
        hardware_efficient=True
    )
    
    # Train
    trainer = QuantumTrainer(config)
    metrics = await trainer.train(model, train_loader, val_loader)
    
    # Results
    print(f"Final loss: {metrics[-1].loss:.4f}")
    print(f"Final accuracy: {metrics[-1].accuracy:.2%}")

# Run
asyncio.run(train())
```

## Best Practices

1. **Start with all features enabled**: `enable_all_v35_features=True`
2. **Use multi-backend when available**: Configure 2-3 backends for best throughput
3. **Monitor statistics**: Check component stats to verify optimizations are working
4. **Adaptive schedules**: Use exponential for slower depth reduction
5. **Natural gradient**: Best for small-medium parameter counts (<100)

## Version Information

```python
import q_store
print(q_store.__version__)  # Should be '3.5.0'
```

## Getting Help

- **Documentation**: `/docs/Architecture_v3_5_DESIGN.md`
- **Implementation**: `/docs/V3_5_IMPLEMENTATION_SUMMARY.md`
- **Tests**: `/tests/test_v3_5_features.py`
- **Issues**: Create issue with v3.5 tag

---

**Last Updated**: December 18, 2024  
**Version**: 3.5.0
