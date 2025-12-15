# Quantum-Native Database Architecture v3.2
## Complete ML Training Integration with Hardware Abstraction

---

## 📦 What's New in v3.2

This release adds **complete machine learning training capabilities** to the quantum-native database while maintaining full hardware abstraction from v3.1.

### Major New Features

1. **Quantum Neural Network Layers** (`quantum_layer.py`)
   - Variational quantum circuits for ML
   - Trainable parameters with freezing support
   - Multiple entanglement patterns
   - Convolutional quantum layers

2. **Quantum Gradient Computation** (`gradient_computer.py`)
   - Parameter shift rule implementation
   - Natural gradient computation
   - Finite difference fallback
   - Parallel gradient computation

3. **Quantum Training Engine** (`quantum_trainer.py`)
   - Complete training orchestration
   - Multiple optimizer support (Adam, SGD, Natural Gradient)
   - Checkpointing and resumption
   - Comprehensive metrics tracking

4. **Quantum Data Encoding** (`data_encoder.py`)
   - Amplitude encoding
   - Angle encoding
   - Basis encoding
   - Advanced feature maps (ZZ, Pauli)

5. **Comprehensive Examples** (`examples_v3_2.py`)
   - Basic training workflows
   - Transfer learning
   - Backend comparison
   - Quantum autoencoder

---

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install numpy pinecone-client

# For Cirq + IonQ (optional)
pip install cirq-ionq

# For Qiskit + IonQ (optional)
pip install qiskit-ionq
```

### Basic Training Example

```python
import asyncio
import numpy as np
from quantum_database import QuantumDatabase, DatabaseConfig
from quantum_trainer import QuantumTrainer, QuantumModel, TrainingConfig
from backends.backend_manager import create_default_backend_manager

async def train_quantum_model():
    # Prepare data
    X_train = np.random.randn(100, 8)  # 100 samples, 8 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels
    
    # Configure training
    config = TrainingConfig(
        pinecone_api_key="your-key",
        pinecone_environment="us-east-1",
        quantum_sdk="mock",  # Use 'cirq' or 'qiskit' for real hardware
        learning_rate=0.01,
        batch_size=10,
        epochs=50,
        n_qubits=8,
        circuit_depth=4
    )
    
    # Create components
    backend_manager = create_default_backend_manager()
    trainer = QuantumTrainer(config, backend_manager)
    model = QuantumModel(
        input_dim=8,
        n_qubits=8,
        output_dim=2,
        backend=backend_manager.get_backend(),
        depth=4
    )
    
    # Simple data loader
    class DataLoader:
        async def __aiter__(self):
            for i in range(0, len(X_train), config.batch_size):
                yield X_train[i:i+config.batch_size], y_train[i:i+config.batch_size]
    
    # Train
    await trainer.train(model, DataLoader(), epochs=50)
    
    print(f"Training complete! Final loss: {trainer.training_history[-1].loss:.4f}")

# Run
asyncio.run(train_quantum_model())
```

---

## 📚 File Structure

```
quantum_db_v3.2/
├── Architecture Document
│   └── Quantum-Native_Database_Architecture_v3_2.md
│
├── Core Quantum Components (from v3.1)
│   ├── quantum_backend_interface.py    # Hardware abstraction
│   ├── backend_manager.py              # Backend management
│   ├── cirq_ionq_adapter.py           # Cirq adapter
│   ├── qiskit_ionq_adapter.py         # Qiskit adapter
│   ├── ionq_backend.py                # Legacy IonQ
│   ├── quantum_database.py            # Main database
│   ├── state_manager.py               # Quantum state management
│   ├── entanglement_registry.py       # Entanglement tracking
│   └── tunneling_engine.py            # Quantum tunneling
│
├── NEW: Machine Learning Components
│   ├── quantum_layer.py               # Quantum NN layers
│   ├── gradient_computer.py           # Gradient computation
│   ├── quantum_trainer.py             # Training orchestration
│   └── data_encoder.py                # Data encoding
│
└── Examples
    └── examples_v3_2.py               # Comprehensive examples
```

---

## 🎓 Core Concepts

### 1. Quantum Neural Network Layer

A quantum layer implements a parametrized quantum circuit:

```
Input → Encoding → Variational Circuit → Measurement → Output
```

**Key Components:**
- **Encoding Layer**: Maps classical data to quantum states
- **Variational Layers**: Trainable rotations + entanglement
- **Measurement**: Extracts classical output

### 2. Quantum Gradient Computation

Uses the **parameter shift rule**:

```
∂⟨O⟩/∂θᵢ = [⟨O⟩(θᵢ + π/2) - ⟨O⟩(θᵢ - π/2)] / 2
```

**Advantages:**
- Exact gradients (no approximation)
- Works on real quantum hardware
- Hardware-agnostic

**Cost**: 2 circuit executions per parameter

### 3. Training Pipeline

```python
for epoch in range(epochs):
    for batch in data_loader:
        # 1. Forward pass
        output = await model.forward(batch_x)
        
        # 2. Compute loss
        loss = loss_function(output, batch_y)
        
        # 3. Compute quantum gradients
        gradients = await compute_gradients(model, loss)
        
        # 4. Update parameters
        model.parameters -= learning_rate * gradients
```

---

## 🔬 Advanced Features

### Transfer Learning

```python
# Pre-train on Task A
await trainer.train(model, task_a_loader, epochs=50)

# Freeze early layers
model.quantum_layer.freeze_parameters([0, 1, 2, 3])

# Fine-tune on Task B with lower learning rate
config.learning_rate = 0.001
await trainer.train(model, task_b_loader, epochs=20)
```

### Multiple Backend Training

```python
# Register multiple backends
backend_manager.register_backend("ionq_sim", cirq_backend)
backend_manager.register_backend("ionq_qpu", qpu_backend)

# Train on simulator
backend_manager.set_default_backend("ionq_sim")
await trainer.train(model, train_loader, epochs=10)

# Fine-tune on QPU
backend_manager.set_default_backend("ionq_qpu")
await trainer.train(model, train_loader, epochs=5)
```

### Custom Data Encoding

```python
from data_encoder import QuantumDataEncoder, QuantumFeatureMap

# Amplitude encoding (default)
encoder = QuantumDataEncoder('amplitude')
circuit = encoder.encode(data)

# Angle encoding
encoder = QuantumDataEncoder('angle')
circuit = encoder.encode(data)

# Advanced feature map
feature_map = QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
circuit = feature_map.map_features(data)
```

### Quantum Autoencoder

```python
# Encoder: High-dim → Low-dim
encoder = QuantumModel(
    input_dim=64,
    n_qubits=64,
    output_dim=8,  # Compressed
    backend=backend
)

# Decoder: Low-dim → High-dim
decoder = QuantumModel(
    input_dim=8,
    n_qubits=8,
    output_dim=64,
    backend=backend
)

# Train for reconstruction
for x in data:
    latent = await encoder.forward(x)
    reconstructed = await decoder.forward(latent)
    loss = mse_loss(reconstructed, x)
    # ... update both models
```

---

## 📊 Performance Considerations

### Circuit Execution Costs

| Component | Circuit Executions |
|-----------|-------------------|
| Forward pass | 1 |
| Gradient (N params) | 2N |
| **Total per batch** | **1 + 2N** |

### Optimization Strategies

1. **Gradient Batching**: Average gradients over multiple samples
2. **Stochastic Gradients**: Compute gradients for subset of parameters
3. **Circuit Caching**: Reuse circuits across batches
4. **Backend Selection**: Use simulator for development, QPU for final training

### Example: Training Time Estimate

```python
# For a model with 100 parameters, 1000 samples, batch_size=10
n_params = 100
n_samples = 1000
batch_size = 10
circuit_time = 200  # ms on IonQ simulator

batches_per_epoch = n_samples / batch_size  # 100
circuits_per_batch = 1 + 2 * n_params  # 201
total_circuits = batches_per_epoch * circuits_per_batch  # 20,100

time_per_epoch = total_circuits * circuit_time / 1000  # 4,020 seconds ≈ 67 minutes
```

**Optimization**: Use gradient batching to reduce to ~10 minutes per epoch

---

## 🔧 Configuration Options

### TrainingConfig

```python
TrainingConfig(
    # Database
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    
    # Quantum Backend
    quantum_sdk: str = "mock"  # 'cirq', 'qiskit', 'mock'
    quantum_api_key: Optional[str] = None
    quantum_target: str = "simulator"
    
    # Hyperparameters
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    
    # Model Architecture
    n_qubits: int = 10
    circuit_depth: int = 4
    entanglement: str = 'linear'  # 'linear', 'circular', 'full'
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'sgd', 'natural_gradient'
    gradient_method: str = 'parameter_shift'
    use_gradient_clipping: bool = True
    
    # Execution
    shots_per_circuit: int = 1000
    max_concurrent_circuits: int = 5
    
    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_directory: str = './checkpoints'
)
```

---

## 🧪 Testing

### Run All Examples

```bash
python examples_v3_2.py
```

### Run Individual Examples

```python
import asyncio
from examples_v3_2 import example_1_basic_training

asyncio.run(example_1_basic_training())
```

### Unit Testing

```python
import pytest
from quantum_layer import QuantumLayer
from backends.backend_manager import MockQuantumBackend

@pytest.mark.asyncio
async def test_quantum_layer():
    backend = MockQuantumBackend(max_qubits=10)
    await backend.initialize()
    
    layer = QuantumLayer(n_qubits=8, depth=2, backend=backend)
    
    # Test forward pass
    input_data = np.random.randn(8)
    output = await layer.forward(input_data)
    
    assert output.shape == (8,)
    assert not np.isnan(output).any()
```

---

## 📖 API Reference

### QuantumLayer

```python
layer = QuantumLayer(
    n_qubits=8,
    depth=4,
    backend=backend,
    entanglement='linear'
)

# Forward pass
output = await layer.forward(input_data, shots=1000)

# Parameter management
params = layer.get_parameters()
layer.update_parameters(new_params)
layer.freeze_parameters([0, 1, 2])
layer.unfreeze_parameters()

# Save/Load
state = layer.save_state()
layer.load_state(state)
```

### QuantumGradientComputer

```python
grad_computer = QuantumGradientComputer(backend)

result = await grad_computer.compute_gradients(
    circuit_builder=lambda p: build_circuit(p),
    loss_function=lambda r: compute_loss(r),
    parameters=current_params
)

# Access results
gradients = result.gradients
loss = result.function_value
n_circuits = result.n_circuit_executions
```

### QuantumTrainer

```python
trainer = QuantumTrainer(config, backend_manager)

# Train model
await trainer.train(model, train_loader, val_loader, epochs=100)

# Checkpointing
await trainer.save_checkpoint(model, epoch, metrics)
await trainer.load_checkpoint(path, model)

# History
history = trainer.get_training_history()
```

---

## 🔮 Roadmap

### v3.3 (Next Release)
- [ ] Quantum federated learning
- [ ] Quantum continual learning  
- [ ] Advanced error mitigation
- [ ] Multi-QPU training orchestration

### v4.0 (Future)
- [ ] Quantum neural architecture search
- [ ] Automated circuit optimization
- [ ] Real-time quantum training
- [ ] Integration with quantum ML frameworks (PennyLane, TensorFlow Quantum)

---

## 🤝 Migration from v3.1

### Step 1: Update Imports

```python
# Add new imports
from quantum_trainer import QuantumTrainer, QuantumModel, TrainingConfig
from quantum_layer import QuantumLayer
from data_encoder import QuantumDataEncoder
```

### Step 2: Create Training Config

```python
# v3.1 database config
db_config = DatabaseConfig(
    pinecone_api_key="key",
    quantum_sdk="cirq"
)

# v3.2 training config (extends database config)
train_config = TrainingConfig(
    **db_config.__dict__,
    learning_rate=0.01,
    batch_size=32,
    n_qubits=10,
    circuit_depth=4
)
```

### Step 3: Use Trainer Instead of Direct Backend

```python
# v3.1 - Direct backend usage
backend = backend_manager.get_backend()
circuit = build_circuit()
result = await backend.execute_circuit(circuit)

# v3.2 - Trainer orchestration
trainer = QuantumTrainer(config, backend_manager)
model = QuantumModel(...)
await trainer.train(model, data_loader, epochs=100)
```

---

## 📝 Citation

If you use this quantum ML training framework in your research, please cite:

```bibtex
@software{quantum_db_v32,
  title={Quantum-Native Database Architecture v3.2},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo/quantum-db}
}
```

---

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: See `Quantum-Native_Database_Architecture_v3_2.md`
- **Examples**: Run `examples_v3_2.py`

---

## 📄 License

Apache 2.0

---

**Built with ❤️ for quantum-accelerated machine learning**
