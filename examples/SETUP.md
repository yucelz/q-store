# Q-Store Examples Setup Guide

This guide explains how to set up and run Q-Store examples with both mock and real quantum backends.

## Quick Start (Mock Mode)

All examples work out-of-the-box with mock backends (no API keys required):

```bash
# Basic examples
python examples/basic_usage.py
python examples/qml_examples.py
python examples/chemistry_examples.py

# Machine Learning examples
python examples/tensorflow/fashion_mnist.py
python examples/pytorch/fashion_mnist.py
```

## Prerequisites

### Required Dependencies

```bash
# Core Q-Store installation
pip install q-store

# OR install from source
pip install -e .
```

### Optional Dependencies

For TensorFlow examples:
```bash
pip install tensorflow keras
```

For PyTorch examples:
```bash
pip install torch torchvision
```

For advanced features (optional):
```bash
pip install python-dotenv  # For .env file support
```

## Configuration for Real Quantum Hardware

### 1. Create Environment File

Copy the example environment file:

```bash
cd examples
cp .env.example .env
```

### 2. Edit Configuration

Edit `examples/.env` with your credentials:

```bash
# IonQ Configuration (for real quantum hardware/simulator)
IONQ_API_KEY=your_actual_api_key_here
IONQ_TARGET=simulator  # Options: simulator, qpu.harmony, qpu.aria-1

# Pinecone Configuration (for quantum database features)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Backend Configuration
DEFAULT_BACKEND=mock_ideal  # or ionq_simulator for real connections
```

### 3. Get API Keys

**IonQ API Key:**
- Sign up at https://cloud.ionq.com/
- Navigate to API Keys section
- Generate a new API key

**Pinecone API Key** (optional, for database features):
- Sign up at https://www.pinecone.io/
- Get your API key from the dashboard

## Running Examples

### Mock Mode (Default)

Run examples without API keys:

```bash
# TensorFlow example (mock mode)
python examples/tensorflow/fashion_mnist.py

# PyTorch example (mock mode)
python examples/pytorch/fashion_mnist.py

# With custom parameters
python examples/tensorflow/fashion_mnist.py --samples 500 --epochs 3 --batch-size 16
```

### Real Quantum Hardware Mode

Run with real IonQ connection (requires API key in .env):

```bash
# TensorFlow example (real quantum)
python examples/tensorflow/fashion_mnist.py --no-mock

# PyTorch example (real quantum)
python examples/pytorch/fashion_mnist.py --no-mock

# With custom parameters
python examples/pytorch/fashion_mnist.py --no-mock --samples 100 --epochs 2
```

## Command Line Arguments

Both TensorFlow and PyTorch examples support these arguments:

```bash
--no-mock              Use real quantum hardware/simulator (requires IONQ_API_KEY)
--samples N            Number of training samples (default: 1000)
--epochs N             Number of training epochs (default: 5)
--batch-size N         Training batch size (default: 32)
--help                 Show help message
```

### Examples:

```bash
# Quick test with 100 samples
python examples/tensorflow/fashion_mnist.py --samples 100 --epochs 2

# Full training run
python examples/tensorflow/fashion_mnist.py --samples 5000 --epochs 10

# Real quantum with reduced dataset
python examples/pytorch/fashion_mnist.py --no-mock --samples 200 --epochs 3
```

## Available Examples

### Basic Examples

1. **basic_usage.py** - Core Q-Store functionality
   - Circuit creation and manipulation
   - Gate operations
   - Parameter handling
   - Backend usage

2. **qml_examples.py** - Quantum Machine Learning
   - Quantum kernels
   - Feature maps
   - QML workflows

3. **chemistry_examples.py** - Quantum Chemistry
   - Molecular Hamiltonians (H2, LiH)
   - VQE (Variational Quantum Eigensolver)
   - Energy calculations

4. **error_mitigation_examples.py** - Error Mitigation
   - Zero-noise extrapolation (ZNE)
   - Probabilistic error cancellation (PEC)
   - Measurement error mitigation

### Machine Learning Examples

1. **tensorflow/fashion_mnist.py** - TensorFlow/Keras Integration
   - Hybrid classical-quantum model
   - QuantumLayer usage
   - Training with Fashion MNIST
   - Model saving/loading

2. **pytorch/fashion_mnist.py** - PyTorch Integration
   - Hybrid quantum neural network
   - Custom quantum layers
   - Training loop
   - Model checkpointing

## Troubleshooting

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
pip install tensorflow keras
```

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch torchvision
```

**Problem:** `ModuleNotFoundError: No module named 'dotenv'`

**Solution:** (Optional - examples work without it)
```bash
pip install python-dotenv
```

### API Connection Issues

**Problem:** `ERROR: --no-mock specified but IONQ_API_KEY not found`

**Solution:**
1. Create `examples/.env` from `examples/.env.example`
2. Add your IonQ API key to the file
3. Make sure you're running from the project root

**Problem:** `ValueError: Backend 'qsim' not found`

**Solution:** The qsim backend is not available. Use `mock_ideal` instead:
```bash
# In examples/.env, set:
DEFAULT_BACKEND=mock_ideal
```

### Backend Issues

**Problem:** Backend initialization errors

**Solution:**
1. Start with mock mode to verify installation
2. Check that Q-Store is properly installed
3. For real quantum: verify API keys are correct

## Environment Variables

You can also set environment variables directly instead of using .env:

```bash
# Linux/Mac
export IONQ_API_KEY="your_key_here"
export IONQ_TARGET="simulator"
export DEFAULT_BACKEND="mock_ideal"

# Windows (PowerShell)
$env:IONQ_API_KEY="your_key_here"
$env:IONQ_TARGET="simulator"
$env:DEFAULT_BACKEND="mock_ideal"
```

Then run examples:
```bash
python examples/tensorflow/fashion_mnist.py --no-mock
```

## Performance Tips

1. **Start Small:** Begin with `--samples 100 --epochs 2` for testing
2. **Mock Mode:** Use mock mode for development and debugging
3. **Batch Size:** Larger batch sizes can speed up training on GPUs
4. **Real Quantum:** Real quantum execution is slower - use smaller datasets

## Example Workflow

Typical development workflow:

```bash
# 1. Test with mock backend (fast)
python examples/tensorflow/fashion_mnist.py --samples 100 --epochs 2

# 2. Verify with more data (mock)
python examples/tensorflow/fashion_mnist.py --samples 1000 --epochs 5

# 3. Test real quantum with small dataset
python examples/tensorflow/fashion_mnist.py --no-mock --samples 50 --epochs 1

# 4. Production run with real quantum
python examples/tensorflow/fashion_mnist.py --no-mock --samples 500 --epochs 3
```

## Additional Resources

- **Q-Store Documentation:** See main README.md
- **API Reference:** Check docstrings in source code
- **Examples README:** See examples/README.md for example descriptions
- **Issue Tracker:** Report issues on GitHub

## Support

For issues or questions:
1. Check this setup guide
2. Review error messages carefully
3. Start with mock mode for debugging
4. Check API key configuration for real quantum mode
5. Open an issue on GitHub with error details
