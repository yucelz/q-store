# Quantum Machine Learning (ML) Module

This module provides complete machine learning training capabilities for quantum neural networks with hardware abstraction.

## 📁 File Structure

```
q_store/ml/
├── __init__.py                 # Module exports
├── quantum_layer.py            # Quantum neural network layers
├── gradient_computer.py        # Gradient computation algorithms
├── data_encoder.py            # Classical-to-quantum data encoding
└── quantum_trainer.py         # Training orchestration
```

## 🧩 Components

### 1. Quantum Layers (`quantum_layer.py`)

**Classes:**
- `QuantumLayer`: Variational quantum circuit layer
- `QuantumConvolutionalLayer`: Convolutional quantum layer
- `QuantumPoolingLayer`: Quantum pooling layer
- `LayerConfig`: Layer configuration dataclass

**Features:**
- Trainable parameters (3 rotations per qubit per layer)
- Multiple entanglement patterns (linear, circular, full)
- Parameter freezing for transfer learning
- State save/load for checkpointing

**Example:**
```python
from q_store.ml import QuantumLayer

layer = QuantumLayer(
    n_qubits=8,
    depth=4,
    backend=backend,
    entanglement='linear'
)

output = await layer.forward(input_data, shots=1000)
```

### 2. Gradient Computation (`gradient_computer.py`)

**Classes:**
- `QuantumGradientComputer`: Parameter shift rule implementation
- `FiniteDifferenceGradient`: Finite difference fallback
- `NaturalGradientComputer`: Quantum Fisher information optimization
- `GradientResult`: Gradient computation results

**Features:**
- Exact gradients via parameter shift rule
- Parallel gradient computation
- Stochastic gradient estimation
- Hessian diagonal computation

**Example:**
```python
from q_store.ml import QuantumGradientComputer

grad_computer = QuantumGradientComputer(backend)
result = await grad_computer.compute_gradients(
    circuit_builder=circuit_builder,
    loss_function=loss_fn,
    parameters=params
)
```

### 3. Data Encoding (`data_encoder.py`)

**Classes:**
- `QuantumDataEncoder`: Multi-strategy encoding
- `QuantumFeatureMap`: Advanced feature mapping

**Encoding Strategies:**
- Amplitude encoding (N-dim → log₂N qubits)
- Angle encoding (features → rotation angles)
- Basis encoding (binary features)
- ZZ Feature Map (second-order Pauli)

**Example:**
```python
from q_store.ml import QuantumDataEncoder, QuantumFeatureMap

# Amplitude encoding
encoder = QuantumDataEncoder('amplitude')
circuit = encoder.encode(data)

# Feature map
feature_map = QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
circuit = feature_map.map_features(data)
```

### 4. Training Orchestration (`quantum_trainer.py`)

**Classes:**
- `QuantumTrainer`: Complete training orchestration
- `QuantumModel`: Base quantum ML model
- `TrainingConfig`: Training configuration
- `TrainingMetrics`: Per-epoch metrics

**Features:**
- Multiple optimizers (Adam, SGD with momentum)
- Gradient clipping
- Checkpoint management (save/load)
- Training history tracking
- Validation support

**Example:**
```python
from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig

config = TrainingConfig(
    pinecone_api_key="key",
    quantum_sdk="mock",
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    n_qubits=8,
    circuit_depth=4
)

trainer = QuantumTrainer(config, backend_manager)
model = QuantumModel(8, 8, 2, backend)

await trainer.train(model, train_loader, epochs=100)
```

## 🔄 Import Paths

All ML components can be imported from the `q_store.ml` module:

```python
# Import all at once
from q_store.ml import (
    QuantumLayer,
    QuantumGradientComputer,
    QuantumDataEncoder,
    QuantumTrainer,
    QuantumModel,
    TrainingConfig
)

# Or import individually
from q_store.ml import QuantumLayer
from q_store.ml import QuantumTrainer
```

## 📊 Dependencies

### Internal Dependencies
- `q_store.backends`: Quantum backend interface and management
- `q_store.core`: Core database components (for integration)

### External Dependencies
- `numpy`: Numerical computations
- `asyncio`: Asynchronous execution
- Standard library: `logging`, `json`, `time`, `pathlib`

## 🎓 Usage Patterns

### Basic Training Workflow

```python
import asyncio
import numpy as np
from q_store.ml import QuantumTrainer, QuantumModel, TrainingConfig
from q_store.backends import create_default_backend_manager

async def train():
    # 1. Setup
    config = TrainingConfig(
        pinecone_api_key="key",
        quantum_sdk="mock",
        learning_rate=0.01,
        batch_size=10,
        epochs=10,
        n_qubits=4
    )
    
    backend_manager = create_default_backend_manager()
    trainer = QuantumTrainer(config, backend_manager)
    
    # 2. Create model
    model = QuantumModel(
        input_dim=4,
        n_qubits=4,
        output_dim=2,
        backend=backend_manager.get_backend()
    )
    
    # 3. Train
    await trainer.train(model, data_loader, epochs=10)

asyncio.run(train())
```

### Transfer Learning

```python
# Pre-train
await trainer.train(model, task_a_loader, epochs=50)

# Freeze early layers
model.quantum_layer.freeze_parameters([0, 1, 2, 3])

# Fine-tune
config.learning_rate = 0.001
await trainer.train(model, task_b_loader, epochs=20)
```

### Custom Encoding

```python
from q_store.ml import QuantumDataEncoder, QuantumFeatureMap

# Try different encodings
encoders = {
    'amplitude': QuantumDataEncoder('amplitude'),
    'angle': QuantumDataEncoder('angle'),
    'zz_map': QuantumFeatureMap(n_qubits=8, feature_map_type='ZZFeatureMap')
}

for name, encoder in encoders.items():
    circuit = encoder.encode(data) if isinstance(encoder, QuantumDataEncoder) else encoder.map_features(data)
    print(f"{name}: {circuit.n_qubits} qubits, {circuit.depth()} depth")
```

## 🧪 Testing

Run the verification script to test all components:

```bash
python verify_v3_2.py
```

Run the quickstart guide:

```bash
python quickstart_v3_2.py
```

Run comprehensive examples:

```bash
python -m q_store_examples.examples_v3_2
```

## 📖 Documentation

- **Quick Start**: `/docs/README_v3_2.md`
- **Architecture**: `/docs/Quantum-Native_Database_Architecture_v3_2.md`
- **Examples**: `/examples/src/q_store_examples/examples_v3_2.py`

## 🔮 Future Enhancements

### Planned Features
- Quantum federated learning
- Quantum continual learning
- Advanced error mitigation
- Multi-QPU orchestration
- Neural architecture search
- Automated circuit optimization

## 📝 Notes

- All components use async/await for quantum circuit execution
- Hardware abstraction maintained through backend interface
- Compatible with mock, Cirq, and Qiskit backends
- Full type hints and documentation
- Production-ready error handling

---

**Module Version**: 3.2.0  
**Last Updated**: December 2025
