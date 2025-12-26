# Q-Store v4.1 Examples

Complete examples demonstrating Q-Store's quantum-first architecture with async execution, production-ready storage, and ML framework integration.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Directory Structure](#-directory-structure)
- [Example Categories](#-example-categories)
- [Running Examples](#-running-examples)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv q-store-env

# Activate virtual environment
# On Linux/macOS:
source q-store-env/bin/activate
# On Windows:
q-store-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Q-Store

```bash
# Install from PyPI (recommended)
pip install q-store

# OR install from source with all dependencies
cd /path/to/q-store
pip install -e ".[all]"
```

### 3. Run Your First Example

```bash
# Basic quantum operations
python examples/basic_usage.py

# Or try async features
python examples/async_features/basic_async_usage.py
```

---

## 📦 Installation

### Minimum Requirements

- **Python**: 3.11 or higher
- **pip**: 23.0 or higher
- **Operating System**: Linux, macOS, or Windows

### Core Installation

```bash
pip install q-store
```

### Optional Dependencies

#### For Machine Learning Examples
```bash
# TensorFlow integration
pip install "q-store[ml]" tensorflow>=2.13.0

# PyTorch integration  
pip install "q-store[ml]" torch>=2.0.0

# Both frameworks
pip install "q-store[ml]" tensorflow>=2.13.0 torch>=2.0.0
```

#### For Async & Storage Features (v4.1)
```bash
pip install "q-store[v4.1]"
# Includes: zarr, pyarrow, aiohttp, pandas
```

#### For Development
```bash
pip install "q-store[dev]"
# Includes: pytest, black, ruff, mypy, etc.
```

#### Install Everything
```bash
pip install "q-store[all]"
```

### Verify Installation

```bash
python -c "import q_store; print(f'Q-Store version: {q_store.__version__}')"
```

---

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── .env.example                       # Configuration template
├── basic_usage.py                     # Start here - core operations
├── basic_pinecone_setup.py           # Vector database setup
├── run_all_validation.py             # Run all validation tests
│
├── ml_frameworks/                     # Machine Learning Integration
│   ├── fashion_mnist_plain.py         # v4.1 plain Python (no Keras/PyTorch)
│   ├── run_fashion_mnist_plain.sh     # Convenience script for plain version
│   ├── framework_integration_demo.py  # Compare TF vs PyTorch
│   ├── tensorflow/
│   │   ├── fashion_mnist_quantum_layers.py      # Basic quantum layers
│   │   ├── fashion_mnist_quantum_db.py          # Full quantum DB integration
│   │   └── fashion_mnist_tensorflow.py          # v4.1 quantum-first (70% quantum)
│   └── pytorch/
│       ├── fashion_mnist_quantum_layers.py      # Basic quantum layers
│       ├── fashion_mnist_quantum_db.py          # Full quantum DB integration
│       └── fashion_mnist_pytorch.py             # v4.1 quantum-first (70% quantum)
│
├── async_features/                    # Async Execution & Storage (v4.1)
│   ├── basic_async_usage.py          # Introduction to async patterns
│   ├── async_performance_demo.py     # Async vs sync benchmarks
│   └── storage_demo.py               # Zarr checkpoints & Parquet metrics
│
├── optimization/                      # Performance Optimization (v4.1)
│   └── optimization_demo.py          # Adaptive batching, caching, compilation
│
├── quantum_algorithms/                # Core Quantum Computing
│   ├── advanced_features.py          # Verification, profiling, visualization
│   ├── qml_examples.py               # Quantum machine learning patterns
│   ├── chemistry_examples.py         # Quantum chemistry simulations
│   └── error_correction_examples.py  # Error mitigation techniques
│
└── validation/                        # Testing & Validation
    ├── gradient_validation.py        # Gradient computation tests
    └── simple_classification.py      # Basic classification benchmark
```

---

## 🎯 Example Categories

### 1. Getting Started

**Start with `basic_usage.py`** - Covers fundamental concepts:
- Creating quantum circuits
- Bell states and entanglement
- Parameterized circuits
- Circuit optimization
- State visualization

```bash
python examples/basic_usage.py
```

### 2. Machine Learning Integration

#### Plain Python (v4.1 Quantum-First Architecture)

**Best for learning the v4.1 architecture without ML framework complexity:**

```bash
# Run with default settings (100 samples, batch size 8)
./examples/ml_frameworks/run_fashion_mnist_plain.sh

# Run with custom parameters
./examples/ml_frameworks/run_fashion_mnist_plain.sh --samples 50 --batch-size 4

# Run with real quantum hardware (auto-confirm with --yes)
./examples/ml_frameworks/run_fashion_mnist_plain.sh --real --samples 10 --yes
```

**Key Features:**
- Pure v4.1 quantum layers (no Keras/PyTorch wrapper)
- 70% quantum computation architecture
- Async execution demonstration
- Inference-only (no training)

**Architecture:**
```python
Flatten()                                    # Classical (5%)
QuantumFeatureExtractor(n_qubits=8, depth=4) # Quantum (40%)
QuantumPooling(n_qubits=4)                   # Quantum (15%)
QuantumFeatureExtractor(n_qubits=4, depth=3) # Quantum (30%)
QuantumReadout(n_qubits=4, n_classes=10)     # Quantum (5%)
# Classical decoding is implicit (5%)
```

#### TensorFlow Examples

```bash
# Basic quantum layers (v4.0 style)
python examples/ml_frameworks/tensorflow/fashion_mnist_quantum_layers.py

# Full quantum database integration
python examples/ml_frameworks/tensorflow/fashion_mnist_quantum_db.py

# v4.1 quantum-first architecture (70% quantum computation)
python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py
```

**Expected Results:**
- ~85% test accuracy on Fashion MNIST
- 8.4x speedup vs v4.0 (with async execution)
- Checkpoints saved to `experiments/fashion_mnist_tf_v4_1/`

#### PyTorch Examples

```bash
# Basic quantum layers (v4.0 style)
python examples/ml_frameworks/pytorch/fashion_mnist_quantum_layers.py

# Full quantum database integration
python examples/ml_frameworks/pytorch/fashion_mnist_quantum_db.py

# v4.1 quantum-first architecture (70% quantum computation)
python examples/ml_frameworks/pytorch/fashion_mnist_pytorch.py
```

**Expected Results:**
- ~85% test accuracy on Fashion MNIST
- 8.4x speedup vs v4.0 (with async execution)
- GPU acceleration support
- Checkpoints saved to `experiments/fashion_mnist_torch_v4_1/`

#### Framework Comparison

```bash
# Compare TensorFlow vs PyTorch implementations
python examples/ml_frameworks/framework_integration_demo.py
```

### 3. Async Features (v4.1)

```bash
# Learn async patterns
python examples/async_features/basic_async_usage.py

# Benchmark async vs sync
python examples/async_features/async_performance_demo.py

# Production storage (Zarr + Parquet)
python examples/async_features/storage_demo.py
```

**Key Features:**
- Async circuit execution
- Concurrent batch processing
- Zarr checkpointing
- Parquet metrics logging

### 4. Performance Optimization (v4.1)

```bash
python examples/optimization/optimization_demo.py
```

**Demonstrates:**
- Adaptive batch scheduling (2-3x throughput)
- Multi-level caching (90%+ hit rate)
- IonQ native compilation (30% speedup)
- Circuit complexity estimation

### 5. Quantum Algorithms

```bash
# Advanced quantum features
python examples/quantum_algorithms/advanced_features.py

# Quantum machine learning patterns
python examples/quantum_algorithms/qml_examples.py

# Quantum chemistry
python examples/quantum_algorithms/chemistry_examples.py

# Error correction
python examples/quantum_algorithms/error_correction_examples.py
```

### 6. Validation & Testing

```bash
# Run all validation examples
python examples/run_all_validation.py

# Individual validation tests
python examples/validation/gradient_validation.py
python examples/validation/simple_classification.py
```

---

## 🏃 Running Examples

### Convenience Scripts

For easier execution, use the provided shell scripts:

```bash
# Fashion MNIST Plain Python (v4.1)
./examples/ml_frameworks/run_fashion_mnist_plain.sh --help

# Quick examples:
./examples/ml_frameworks/run_fashion_mnist_plain.sh                    # Mock mode, defaults
./examples/ml_frameworks/run_fashion_mnist_plain.sh --samples 200     # More samples
./examples/ml_frameworks/run_fashion_mnist_plain.sh --real --yes      # Real hardware, auto-confirm
```

**Script Options:**
- `--mock` / `--real` - Backend selection (default: mock)
- `--samples N` - Number of test samples (default: 100)
- `--batch-size N` - Batch size for inference (default: 8)
- `-y, --yes` - Skip confirmation prompt for real hardware
- `-h, --help` - Show help message

### Mock Mode (Default - No API Keys Required)

All examples work with mock quantum backends:

```bash
python examples/basic_usage.py
python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py
```

### Real Quantum Hardware/Simulators

#### 1. Create Configuration File

```bash
cp .env.example .env
```

#### 2. Add Your API Keys to `.env`

```bash
# IonQ Configuration
IONQ_API_KEY=your_ionq_api_key_here
IONQ_TARGET=simulator  # Options: simulator, qpu.harmony, qpu.aria-1

# Pinecone Configuration (optional)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Backend Selection
DEFAULT_BACKEND=ionq_simulator  # Change from mock_ideal to use IonQ
```

#### 3. Get API Keys

**IonQ** (for quantum hardware/simulators):
- Sign up at https://cloud.ionq.com/
- Navigate to API Keys section
- Generate a new API key
- Copy to your `.env` file

**Pinecone** (optional, for vector database features):
- Sign up at https://www.pinecone.io/
- Get API key from dashboard
- Copy to your `.env` file

#### 4. Run with Real Backend

```bash
# Install python-dotenv for .env support
pip install python-dotenv

# Run examples - they'll automatically use your .env configuration
python examples/basic_usage.py
python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `IONQ_API_KEY` | IonQ API authentication | (none) | Your API key |
| `IONQ_TARGET` | IonQ target device | `simulator` | `simulator`, `qpu.harmony`, `qpu.aria-1` |
| `PINECONE_API_KEY` | Pinecone API key | (none) | Your API key |
| `PINECONE_ENVIRONMENT` | Pinecone region | (none) | `us-east-1`, etc. |
| `DEFAULT_BACKEND` | Quantum backend | `mock_ideal` | `mock_ideal`, `ionq_simulator`, `cirq` |

### Backend Options

- **mock_ideal**: Perfect quantum simulator (no noise, no API needed)
- **mock_noisy**: Realistic noise simulation (no API needed)
- **ionq_simulator**: IonQ cloud simulator (API key required)
- **qpu.harmony**: IonQ Harmony QPU (API key + credits required)
- **qpu.aria-1**: IonQ Aria QPU (API key + credits required)
- **cirq**: Google Cirq simulator (local execution)

---

## 🔧 Troubleshooting

### Common Issues

#### ImportError: No module named 'q_store'

**Solution:**
```bash
pip install q-store
# OR
pip install -e .  # if installing from source
```

#### ImportError: No module named 'tensorflow'

**Solution:**
```bash
pip install "q-store[ml]" tensorflow>=2.13.0
```

#### ImportError: No module named 'torch'

**Solution:**
```bash
pip install "q-store[ml]" torch>=2.0.0
```

#### ImportError: No module named 'zarr' or 'aiohttp'

**Solution:**
```bash
pip install "q-store[v4.1]"
```

#### API Key Errors

**Symptoms:**
- `401 Unauthorized`
- `Invalid API key`

**Solution:**
1. Verify `.env` file exists in `examples/` directory
2. Check API keys are correct (no extra spaces)
3. Install python-dotenv: `pip install python-dotenv`
4. Try mock mode first to verify code works

#### Out of Memory Errors

**Solution:**
- Reduce batch sizes in ML examples
- Use smaller datasets
- Close other applications
- Try examples one at a time

#### Slow Performance

**For ML Examples:**
- Reduce number of epochs
- Reduce training samples
- Try mock backend first
- Check internet connection for cloud backends

**For Production:**
- Use adaptive batching (examples/optimization/optimization_demo.py)
- Enable caching
- Consider IonQ native compilation
- Use async execution patterns

### Getting Help

1. **Check logs**: Most examples print detailed progress information
2. **Start simple**: Run `examples/basic_usage.py` first to verify setup
3. **Use mock mode**: Test code without API keys
4. **Check version**: Ensure Python 3.11+ and latest q-store
5. **GitHub Issues**: https://github.com/yucelz/q-store/issues

---

## 📚 Learning Path

### Beginner
1. `examples/basic_usage.py` - Core concepts
2. `examples/basic_pinecone_setup.py` - Vector database setup
3. `examples/ml_frameworks/fashion_mnist_plain.py` - v4.1 architecture (plain Python)
4. `examples/async_features/basic_async_usage.py` - Async patterns

### Intermediate
4. `examples/ml_frameworks/tensorflow/fashion_mnist_quantum_layers.py` - TensorFlow integration
5. `examples/ml_frameworks/pytorch/fashion_mnist_quantum_layers.py` - PyTorch integration
6. `examples/quantum_algorithms/qml_examples.py` - QML patterns

### Advanced
7. `examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py` - v4.1 quantum-first
8. `examples/optimization/optimization_demo.py` - Performance tuning
9. `examples/async_features/storage_demo.py` - Production storage
10. `examples/quantum_algorithms/chemistry_examples.py` - Domain applications

---

## 🎓 Additional Resources

- **Website**: http://www.q-store.tech
- **Documentation**: https://github.com/yucelz/q-store/tree/main/docs
- **Repository**: https://github.com/yucelz/q-store
- **Issues**: https://github.com/yucelz/q-store/issues

---

## 📄 License

Q-Store is licensed under the AGPL-3.0 License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

Built with support from:
- IonQ trapped-ion quantum computers
- Cirq quantum computing framework
- TensorFlow and PyTorch ML frameworks
- Pinecone vector database

---

**Happy Quantum Computing! 🚀**
