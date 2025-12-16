# Q-Store v3.3 Quick Reference

## 🚀 One-Minute Setup

```python
from q_store.backends import BackendManager
from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig

# Configure with v3.3 optimizations (just 3 lines!)
config = TrainingConfig(
    n_qubits=8, 
    gradient_method='spsa',  # 🔥 24-48x faster
    hardware_efficient_ansatz=True  # 🔥 33% fewer params
)

# Initialize and train
backend_manager = BackendManager()
await backend_manager.initialize(sdk='mock')
trainer = QuantumTrainer(config, backend_manager)
# Now training is 24x faster! 🚀
```

---

## 🎯 Key v3.3 Features

### SPSA Gradient Estimation

```python
# Enable SPSA (2 circuits instead of 96)
config = TrainingConfig(
    gradient_method='spsa',  # Use SPSA
    spsa_c_initial=0.1,      # Perturbation size
    spsa_a_initial=0.01      # Step size
)
```

**When to use**: Always, unless you need exact gradients

### Adaptive Gradient Optimization

```python
# Auto-switch between SPSA and parameter shift
config = TrainingConfig(
    gradient_method='adaptive'  # Best of both worlds
)
```

**When to use**: Production training, best performance/accuracy tradeoff

### Hardware-Efficient Layers

```python
# 33% fewer parameters, same expressivity
model = QuantumModel(
    input_dim=8,
    n_qubits=8,
    output_dim=2,
    backend=backend,
    hardware_efficient=True  # Enable HE layer
)
```

**When to use**: Always, unless you specifically need 3-rotation gates

### Circuit Caching

```python
# Cache compiled circuits and results
config = TrainingConfig(
    enable_circuit_cache=True,  # Enable caching
    cache_size=1000             # Cache 1000 circuits
)
```

**When to use**: Training with repeated circuits, small parameter changes

### Performance Tracking

```python
# Monitor training performance
config = TrainingConfig(
    enable_performance_tracking=True,
    performance_log_dir='./logs'
)

# After training
stats = trainer.performance_tracker.get_statistics()
print(f"Speedup: {stats['circuits_per_second']:.1f} circuits/sec")
```

**When to use**: Always, for performance analysis

---

## 📊 Performance Comparison

### Before (v3.2)
```python
config = TrainingConfig(
    gradient_method='parameter_shift'  # 96 circuits/batch
)
# Time: 240s per batch
```

### After (v3.3)
```python
config = TrainingConfig(
    gradient_method='spsa'  # 2 circuits/batch
)
# Time: 10s per batch → 24x faster! 🎉
```

---

## 🔧 Configuration Cheat Sheet

### Minimal (use defaults)
```python
config = TrainingConfig(
    n_qubits=8,
    gradient_method='spsa'
)
```

### Recommended (production)
```python
config = TrainingConfig(
    # Model
    n_qubits=8,
    circuit_depth=2,
    
    # Optimization
    gradient_method='adaptive',
    hardware_efficient_ansatz=True,
    
    # Performance
    enable_circuit_cache=True,
    enable_batch_execution=True,
    enable_performance_tracking=True,
    
    # Training
    learning_rate=0.01,
    batch_size=10,
    epochs=100
)
```

### Advanced (fine-tuning)
```python
config = TrainingConfig(
    # Model
    n_qubits=16,
    circuit_depth=3,
    
    # Optimization
    gradient_method='spsa',
    spsa_c_initial=0.2,      # Larger perturbations
    spsa_a_initial=0.02,     # Larger steps
    
    # Performance
    enable_circuit_cache=True,
    cache_size=5000,         # Larger cache
    batch_timeout=120.0,     # Longer timeout
    
    # Training
    learning_rate=0.001,     # Lower LR with SPSA
    batch_size=32,
    use_gradient_clipping=True,
    gradient_clip_value=0.5
)
```

---

## 🎓 Common Patterns

### Pattern 1: Fast Prototyping

```python
# Quick and dirty training
config = TrainingConfig(
    n_qubits=4,
    gradient_method='spsa',
    epochs=10
)
```

### Pattern 2: Production Training

```python
# Full optimization stack
config = TrainingConfig(
    n_qubits=8,
    gradient_method='adaptive',
    hardware_efficient_ansatz=True,
    enable_circuit_cache=True,
    enable_batch_execution=True,
    enable_performance_tracking=True,
    performance_log_dir='./training_logs'
)
```

### Pattern 3: Debugging/Validation

```python
# Use accurate gradients for debugging
config = TrainingConfig(
    n_qubits=4,
    gradient_method='parameter_shift',  # Exact
    enable_performance_tracking=True
)
```

---

## 🐛 Troubleshooting

### Issue: SPSA not converging

```python
# Solution 1: Increase perturbation
config.spsa_c_initial = 0.2  # Default: 0.1

# Solution 2: Use adaptive
config.gradient_method = 'adaptive'

# Solution 3: Lower learning rate
config.learning_rate = 0.001  # Default: 0.01
```

### Issue: Training still slow

```python
# Check if optimizations enabled
assert config.gradient_method == 'spsa'
assert config.hardware_efficient_ansatz == True
assert config.enable_circuit_cache == True

# Monitor performance
stats = trainer.performance_tracker.get_statistics()
print(f"Circuits/batch: {stats['avg_circuits_per_batch']}")
# Should be ~2 for SPSA
```

### Issue: Memory errors

```python
# Reduce cache size
config.cache_size = 500  # Default: 1000

# Or disable caching
config.enable_circuit_cache = False
```

---

## 📈 Performance Metrics

### Expected Speedups

| Configuration | Speedup | Use Case |
|---------------|---------|----------|
| SPSA only | 24x | Fast training |
| SPSA + HE layer | 32x | Better efficiency |
| SPSA + HE + cache | 40x | Repeated circuits |
| Full v3.3 stack | 48x | Production |

### Monitoring

```python
# Get performance stats
stats = trainer.performance_tracker.get_statistics()

# Key metrics
print(f"Total circuits: {stats['total_circuits']}")
print(f"Time: {stats['total_runtime_s']:.1f}s")
print(f"Throughput: {stats['circuits_per_second']:.1f} c/s")

# Speedup estimate
speedup = trainer.performance_tracker.estimate_speedup()
print(f"Speedup vs v3.2: {speedup['estimated_time_speedup']:.1f}x")

# Cache performance
if trainer.circuit_cache:
    cache = trainer.circuit_cache.get_stats()
    print(f"Cache hit rate: {cache['hit_rate']*100:.1f}%")
```

---

## 🎯 Best Practices

### ✅ Do

- Use `gradient_method='spsa'` for most training
- Enable `hardware_efficient_ansatz=True`
- Enable caching for repeated circuits
- Track performance to identify bottlenecks
- Use adaptive method for production

### ❌ Don't

- Use parameter shift for large models (slow!)
- Disable caching without reason
- Ignore performance metrics
- Use large perturbations with SPSA
- Forget to enable v3.3 features

---

## 🔗 Quick Links

- [Full Documentation](README_v3_3.md)
- [Architecture Details](docs/Quantum-Native_Database_Architecture_v3_3.md)
- [Implementation Summary](docs/V3_3_IMPLEMENTATION_SUMMARY.md)
- [Example Code](examples_v3_3.py)

---

## 💡 Tips

1. **Start with SPSA** - It's almost always the right choice
2. **Enable all optimizations** - They work together
3. **Monitor performance** - Use PerformanceTracker
4. **Use adaptive for production** - Best overall results
5. **Cache compiled circuits** - Huge speedup for training

---

**Questions?** Check the [full documentation](README_v3_3.md) or run `python examples_v3_3.py`
