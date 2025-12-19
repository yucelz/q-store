# Q-Store v4.0 Examples

This directory contains example scripts demonstrating Q-Store v4.0 capabilities with both TensorFlow and PyTorch.

## Directory Structure

```
examples/
├── tensorflow/          # TensorFlow/Keras examples
│   └── fashion_mnist.py # Fashion MNIST classification
├── pytorch/             # PyTorch examples
│   └── fashion_mnist.py # Fashion MNIST classification  
└── validation/          # Validation scripts
    ├── gradient_validation.py      # Gradient correctness tests
    └── simple_classification.py    # Simple classification validation
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
