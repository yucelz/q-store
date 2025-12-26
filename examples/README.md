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

├── 
├── tensorflow/                       # TensorFlow/Keras examples
│   └── fashion_mnist.py             # Quantum layers only
│   └── fashion_mnist_quantum_db_tf.py   # **NEW** Full integration: QML + Quantum DB (TensorFlow)
├── pytorch/                          # PyTorch examples
│   └── fashion_mnist.py             # Quantum layers only
├── fashion_mnist_quantum_db.py      # **NEW** Full integration: QML + Quantum DB (PyTorch)
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

### Fashion MNIST Classification (Quantum Layers Only)

Train a hybrid quantum-classical model on Fashion MNIST using Keras.

**Quick Start (Mock Mode - No API Keys Required):**
```bash
python examples/tensorflow/fashion_mnist.py
```

**With Custom Parameters:**
```bash
# Train with 100 samples for 2 epochs
python examples/tensorflow/fashion_mnist.py --samples 100 --epochs 2 --batch-size 32

# Use real IonQ quantum backend (requires IONQ_API_KEY in examples/.env)
python examples/tensorflow/fashion_mnist.py --no-mock --samples 50 --epochs 1
```

**Command-Line Options:**
- `--no-mock`: Use real IonQ quantum backend instead of mock (requires API key)
- `--samples N`: Number of training samples to use (default: 1000)
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size for training (default: 32)

This example demonstrates:
- Using `QuantumLayer` in a Keras Sequential model
- `AmplitudeEncoding` for quantum state preparation
- Mock mode for testing without quantum hardware
- Real quantum backend integration with IonQ
- End-to-end training with TensorFlow's optimizers
- Model saving and evaluation

**Note:** This example focuses on quantum layers in neural networks only.

### Fashion MNIST + Quantum Database (Full Integration)

**NEW!** Comprehensive example combining quantum neural networks AND quantum database features.

**Quick Start (Mock Mode - No API Keys Required):**
```bash
python examples/tensorflow/fashion_mnist_quantum_db_tf.py
```

**With Real Backends (Requires Both IONQ_API_KEY and PINECONE_API_KEY):**
```bash
# Configure your API keys in examples/.env first
python examples/tensorflow/fashion_mnist_quantum_db_tf.py --no-mock --samples 500 --epochs 3 --store-items 100
```

**Command-Line Options:**
- `--no-mock`: Use real IonQ quantum backend + Pinecone database (requires API keys)
- `--samples N`: Number of training samples (default: 500)
- `--epochs N`: Number of training epochs (default: 3)
- `--batch-size N`: Batch size for training (default: 32)
- `--store-items N`: Number of embeddings to store in database (default: 100)

**Configuration:**
Create `examples/.env` with both quantum and database credentials:
```bash
cd examples
cp .env.example .env
# Edit .env and add:
#   IONQ_API_KEY=your_ionq_key
#   PINECONE_API_KEY=your_pinecone_key
#   PINECONE_ENVIRONMENT=us-east-1
```

This comprehensive example demonstrates:
1. **Quantum Neural Networks**: Train with quantum circuit layers
2. **Embedding Extraction**: Extract learned embeddings from trained model
3. **Quantum Database Storage**: Store embeddings in Pinecone with quantum superposition
4. **Multi-Context Superposition**: Each embedding stored with multiple contexts (class, category, style)
5. **Context-Aware Search**: Query database with quantum-enhanced similarity search
6. **Classical vs Quantum Comparison**: Compare search results with and without quantum features

**This is the only example that demonstrates the complete Q-Store workflow: Train → Store → Query**

**Note:** TensorFlow examples run on CPU only (quantum layers use `tf.py_function` which is not XLA-compatible).

## PyTorch Examples

### Fashion MNIST Classification (Quantum Layers Only)

Train a hybrid quantum-classical model on Fashion MNIST using PyTorch.

**Quick Start (Mock Mode - No API Keys Required):**
```bash
python examples/pytorch/fashion_mnist.py
```

**With Custom Parameters:**
```bash
# Train with 100 samples for 2 epochs
python examples/pytorch/fashion_mnist.py --samples 100 --epochs 2 --batch-size 32

# Use real IonQ quantum backend (requires IONQ_API_KEY in examples/.env)
python examples/pytorch/fashion_mnist.py --no-mock --samples 50 --epochs 1
```

**Command-Line Options:**
- `--no-mock`: Use real IonQ quantum backend instead of mock (requires API key)
- `--samples N`: Number of training samples to use (default: 1000)
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size for training (default: 32)

This example demonstrates:
- Using `QuantumLayer` as a PyTorch `nn.Module`
- Integration with PyTorch's autograd system
- Mock mode for rapid prototyping and testing
- Real quantum backend integration with IonQ
- Training with standard PyTorch optimizers
- Model checkpointing

**Note:** This example focuses on quantum layers in neural networks only.

### Fashion MNIST + Quantum Database (Full Integration)

**NEW!** Comprehensive example combining quantum neural networks AND quantum database features.

**Quick Start (Mock Mode - No API Keys Required):**
```bash
python examples/pytorch/fashion_mnist_quantum_db.py
```

**With Real Backends (Requires Both IONQ_API_KEY and PINECONE_API_KEY):**
```bash
# Configure your API keys in examples/.env first
python examples/pytorch/fashion_mnist_quantum_db.py --no-mock --samples 500 --epochs 3 --store-items 100
```

**Command-Line Options:**
- `--no-mock`: Use real IonQ quantum backend + Pinecone database (requires API keys)
- `--samples N`: Number of training samples (default: 500)
- `--epochs N`: Number of training epochs (default: 3)
- `--batch-size N`: Batch size for training (default: 32)
- `--store-items N`: Number of embeddings to store in database (default: 100)

**Configuration:**
Create `examples/.env` with both quantum and database credentials:
```bash
cd examples
cp .env.example .env
# Edit .env and add:
#   IONQ_API_KEY=your_ionq_key
#   PINECONE_API_KEY=your_pinecone_key
#   PINECONE_ENVIRONMENT=us-east-1
```

This comprehensive example demonstrates:
1. **Quantum Neural Networks**: Train with quantum circuit layers
2. **Embedding Extraction**: Extract learned embeddings from trained model
3. **Quantum Database Storage**: Store embeddings in Pinecone with quantum superposition
4. **Multi-Context Superposition**: Each embedding stored with multiple contexts (class, category, style)
5. **Context-Aware Search**: Query database with quantum-enhanced similarity search
6. **Classical vs Quantum Comparison**: Compare search results with and without quantum features

**This is the only example that demonstrates the complete Q-Store workflow: Train → Store → Query**

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
# TensorFlow/Keras - Quantum Layers Only (mock mode - fast, no API keys)
python examples/tensorflow/fashion_mnist.py --samples 100 --epochs 2

# TensorFlow - Quantum Layers Only with real quantum backend (requires IONQ_API_KEY)
python examples/tensorflow/fashion_mnist.py --no-mock --samples 50 --epochs 1

# TensorFlow - Full Quantum Database Integration (mock mode)
python examples/tensorflow/fashion_mnist_quantum_db_tf.py --samples 500 --epochs 3

# TensorFlow - Full Integration with Real Backends (requires IONQ_API_KEY + PINECONE_API_KEY)
python examples/tensorflow/fashion_mnist_quantum_db_tf.py --no-mock --samples 500 --epochs 3 --store-items 100

# PyTorch - Quantum Layers Only (mock mode - fast, no API keys)
python examples/pytorch/fashion_mnist.py --samples 100 --epochs 2

# PyTorch - Quantum Layers Only with real quantum backend (requires IONQ_API_KEY)
python examples/pytorch/fashion_mnist.py --no-mock --samples 50 --epochs 1

# PyTorch - Full Quantum Database Integration (mock mode)
python examples/pytorch/fashion_mnist_quantum_db.py --samples 500 --epochs 3

# PyTorch - Full Integration with Real Backends (requires IONQ_API_KEY + PINECONE_API_KEY)
python examples/pytorch/fashion_mnist_quantum_db.py --no-mock --samples 500 --epochs 3 --store-items 100

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

### Optional Dependencies

**pinecone (for pinecone vector DB):**
```bash
pip install pinecone
```

**python-dotenv (for .env configuration):**
```bash
pip install python-dotenv
```
*Note: Examples work without this - they'll fall back to environment variables or defaults.*

### Installing All Dependencies

```bash
pip install q-store[tensorflow,torch]
pip install python-dotenv  # Optional but recommended
```

### Real Quantum Backend Setup

To use real quantum hardware/simulators (IonQ):

1. **Copy the environment template:**
   ```bash
   cd examples
   cp .env.example .env
   ```

2. **Edit `examples/.env` and add your API keys:**
   ```bash
   # IonQ Configuration
   IONQ_API_KEY=your_ionq_api_key_here
   IONQ_TARGET=simulator  # or qpu.harmony for real hardware
   
   # Backend Selection
   DEFAULT_BACKEND=mock_ideal  # Use mock by default
   ```

3. **Run with real backend:**
   ```bash
   python examples/tensorflow/fashion_mnist.py --no-mock
   ```

**Getting API Keys:**
- **IonQ**: Sign up at [ionq.com](https://ionq.com/) for quantum simulator/hardware access

## Expected Output

All examples should run successfully and produce reasonable results:

- **Fashion MNIST (Mock Mode)**: ~10-20% test accuracy (mock backend returns random results)
- **Fashion MNIST (Real Backend)**: 60-75% test accuracy (varies with quantum circuit depth)
- **Gradient Validation**: Max gradient difference < 1e-3
- **Simple Classification**: >60% test accuracy

**Note:** Mock mode is designed for testing workflows, not for achieving high accuracy. Use `--no-mock` with a real quantum backend for actual ML performance.

## Performance Notes

The examples use small datasets and simple quantum circuits for fast execution:

- Fashion MNIST examples default to 1000 training samples (configurable with `--samples`)
- Test sets use 200 samples
- Quantum circuits use 4 qubits and 2 layers
- Mock mode executes instantly (no real quantum simulation)
- Real backends may take longer due to API calls and queue times

For production use, scale up:
- Increase training samples: `--samples 10000`
- Increase epochs: `--epochs 50`
- Add more quantum layers
- Tune hyperparameters (learning rate, batch size)
- Use more qubits if available

**Quick Testing:**
```bash
# Fast test run (completes in ~10 seconds)
python examples/tensorflow/fashion_mnist.py --samples 50 --epochs 1 --batch-size 16
```

## Troubleshooting

**Import errors**: Make sure q-store v4.0 is installed:
```bash
pip install -e .
```

**TensorFlow warnings**: TF may show compilation warnings - these are normal and can be ignored

**TensorFlow GPU disabled warning**: This is expected - quantum layers run on CPU only (tf.py_function not XLA-compatible)

**"python-dotenv not installed" message**: This is informational - examples work without it using environment variables

**"IONQ_API_KEY not found" error**: 
- You used `--no-mock` but didn't configure API keys
- Either remove `--no-mock` flag (use mock mode) or add IONQ_API_KEY to `examples/.env`

**Slow execution**: 
- Use mock mode for fast testing (default)
- Real quantum backends involve API calls and queue times
- Reduce samples/epochs for faster iteration: `--samples 50 --epochs 1`
- Reduce batch size: `--batch-size 16`

**Low accuracy in mock mode**: This is expected - mock backend returns random quantum results (~10% accuracy for 10-class problem). Use `--no-mock` with real backend for actual performance.

**"Gradients do not exist" warning**: This is expected - the quantum encoding layer blocks gradient flow by design (using `tf.stop_gradient`). Only the quantum layer parameters and classical layers receive gradients.

**Real backend errors**: 
- Check your API key is valid
- Verify you have credits/access
- Some backends have qubit limits
- Check IonQ service status

## Next Steps

After running these examples, explore:

1. **Custom quantum circuits**: Modify circuit architectures
2. **Different datasets**: Try CIFAR-10, custom datasets
3. **Hyperparameter tuning**: Experiment with learning rates, depths
4. **Backend optimization**: Use qsim or Lightning backends (Phase 3)
5. **Distributed training**: Scale to multiple GPUs/nodes (Phase 4)

## Questions or Issues?

Please open an issue on GitHub if you encounter problems or have questions about these examples.
