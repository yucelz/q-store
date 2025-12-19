# Q-Store v4.0 Examples

This directory contains example scripts demonstrating Q-Store v4.0 capabilities.

## Directory Structure

```
examples/
├── basic_usage.py                    # Core operations and workflows
├── advanced_features.py              # Verification, profiling, visualization
├── qml_examples.py                   # Quantum machine learning
├── chemistry_examples.py             # Quantum chemistry simulations
├── error_correction_examples.py     # Error correction workflows
├── tensorflow/                       # TensorFlow/Keras examples
│   └── fashion_mnist.py
├── pytorch/                          # PyTorch examples
│   └── fashion_mnist.py
└── validation/                       # Validation scripts
    ├── gradient_validation.py
    └── simple_classification.py
```

## Core Examples

### Basic Usage (`basic_usage.py`)
Fundamental Q-Store operations:
- **Bell State Creation**: Entangled states with H and CNOT gates
- **Parameterized Circuits**: Rotation gates with parameters
- **Circuit Optimization**: Reduce gate count
- **Backend Conversion**: Convert to Qiskit and Cirq
- **State Visualization**: Visualize quantum states

Run with:
```bash
python examples/basic_usage.py
```

### Advanced Features (`advanced_features.py`)
Advanced capabilities:
- **Circuit Verification**: Check equivalence and properties
- **Property Verification**: Unitarity, reversibility, commutativity
- **Performance Profiling**: Gate-level execution metrics
- **Performance Analysis**: Optimization suggestions
- **Optimization Profiling**: Benchmark optimization strategies
- **Bloch Sphere**: State visualization on Bloch sphere

Run with:
```bash
python examples/advanced_features.py
```

## Domain-Specific Examples

### Quantum Machine Learning (`qml_examples.py`)
QML workflows:
- **Feature Maps**: Encode classical data into quantum states
- **Quantum Kernels**: Compute quantum kernel matrices
- **Quantum Models**: Variational quantum models
- **Variational Training**: Train parameterized circuits
- **Data Encoding**: Multiple encoding strategies
- **Complete QML Workflow**: End-to-end ML pipeline

Run with:
```bash
python examples/qml_examples.py
```

### Quantum Chemistry (`chemistry_examples.py`)
Molecular simulations:
- **Molecule Creation**: Define molecular systems (H2, etc.)
- **Pauli Strings**: Pauli operators and Hamiltonians
- **VQE Ansatz**: Variational quantum eigensolver circuits
- **VQE Energy**: Calculate molecular energies
- **UCCSD Ansatz**: Unitary coupled-cluster construction
- **Complete Chemistry Workflow**: Full VQE simulation

Run with:
```bash
python examples/chemistry_examples.py
```

### Error Correction (`error_correction_examples.py`)
Quantum error correction:
- **Surface Codes**: Create surface code lattices
- **Stabilizer Measurements**: X and Z stabilizers
- **Error Syndromes**: Extract and analyze syndromes
- **Syndrome Decoding**: Decode syndromes to identify errors
- **Error Detection**: Detect errors via stabilizers
- **Logical Operations**: Gates on encoded qubits
- **Complete Workflow**: Full error correction cycle

Run with:
```bash
python examples/error_correction_examples.py
```

## TensorFlow Examples

### Fashion MNIST Classification

Train a hybrid quantum-classical model on Fashion MNIST using Keras:

```bash
cd examples/tensorflow
python fashion_mnist.py
```

This example demonstrates:
- Using `QuantumLayer` in a Keras Sequential model
- `AmplitudeEncoding` for quantum state preparation
- End-to-end training with TensorFlow's optimizers
- Model saving and evaluation

## PyTorch Examples

### Fashion MNIST Classification

Train a hybrid quantum-classical model on Fashion MNIST using PyTorch:

```bash
cd examples/pytorch
python fashion_mnist.py
```

This example demonstrates:
- Using `QuantumLayer` as a PyTorch `nn.Module`
- Integration with PyTorch's autograd system
- Training with standard PyTorch optimizers
- Model checkpointing

## Validation Scripts

### Gradient Validation

Validate gradient computation correctness using numerical gradient checking:

```bash
cd examples/validation
python gradient_validation.py
```

This script:
- Compares analytical gradients (from autograd) to numerical gradients
- Tests both TensorFlow and PyTorch implementations
- Reports maximum and mean gradient differences
- Passes if differences are below threshold (< 1e-3)

### Simple Classification

Quick validation using a toy dataset:

```bash
cd examples/validation
python simple_classification.py
```

This script:
- Creates a simple binary classification problem
- Tests both frameworks on the same data
- Validates that models can learn (>60% accuracy)
- Provides fast smoke tests for CI/CD

## Running All Examples

To run core examples:

```bash
# Core Q-Store examples
python examples/basic_usage.py
python examples/advanced_features.py

# Domain-specific examples
python examples/qml_examples.py
python examples/chemistry_examples.py
python examples/error_correction_examples.py
```

To run ML framework examples:

```bash
# TensorFlow/Keras
cd examples/tensorflow && python fashion_mnist.py

# PyTorch
cd examples/pytorch && python fashion_mnist.py

# Validation
cd examples/validation
python gradient_validation.py
python simple_classification.py
```

## Learning Path

Recommended order for learning Q-Store:

1. **Start**: `basic_usage.py` - Learn core concepts
2. **Explore**: `advanced_features.py` - See verification and profiling
3. **Specialize**: Choose domain examples:
   - Machine Learning → `qml_examples.py` or ML framework examples
   - Quantum Chemistry → `chemistry_examples.py`
   - Error Correction → `error_correction_examples.py`
4. **Validate**: Run validation scripts to understand testing

## Requirements

### Core Requirements
- q-store v4.0
- numpy

### Framework-Specific Requirements

**TensorFlow:**
```bash
pip install tensorflow
```

**PyTorch:**
```bash
pip install torch torchvision
```

### Installing All Dependencies

```bash
pip install q-store[tensorflow,torch]
```

## Expected Output

All examples should run successfully and produce reasonable results:

- **Fashion MNIST**: 60-75% test accuracy (varies with quantum circuit depth)
- **Gradient Validation**: Max gradient difference < 1e-3
- **Simple Classification**: >60% test accuracy

## Performance Notes

The examples use small datasets and simple quantum circuits for fast execution:

- Fashion MNIST examples use 1000 training samples (vs 60,000 in full dataset)
- Test sets use 200 samples
- Quantum circuits use 4 qubits and 2 layers

For production use, scale up:
- Increase training samples
- Add more quantum layers
- Tune hyperparameters
- Use more qubits if available

## Troubleshooting

**Import errors**: Make sure q-store v4.0 is installed:
```bash
pip install -e .
```

**TensorFlow warnings**: TF may show compilation warnings - these are normal

**Slow execution**: Quantum simulation is computationally intensive. Consider:
- Using GPU backends (when available in Phase 3)
- Reducing circuit depth
- Reducing batch size

**Low accuracy**: This is expected for toy examples. To improve:
- Increase training data
- Add more quantum layers
- Tune learning rate
- Train for more epochs

## Next Steps

After running these examples, explore:

1. **Custom quantum circuits**: Modify circuit architectures
2. **Different datasets**: Try CIFAR-10, custom datasets
3. **Hyperparameter tuning**: Experiment with learning rates, depths
4. **Backend optimization**: Use qsim or Lightning backends (Phase 3)
5. **Distributed training**: Scale to multiple GPUs/nodes (Phase 4)

## Questions or Issues?

Please open an issue on GitHub if you encounter problems or have questions about these examples.
