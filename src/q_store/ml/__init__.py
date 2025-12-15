"""
Quantum Machine Learning Training Components

This module provides complete ML training capabilities for quantum neural networks
with hardware abstraction.
"""

from .quantum_layer import (
    QuantumLayer,
    QuantumConvolutionalLayer,
    QuantumPoolingLayer,
    LayerConfig
)
from .gradient_computer import (
    QuantumGradientComputer,
    FiniteDifferenceGradient,
    NaturalGradientComputer,
    GradientResult
)
from .data_encoder import (
    QuantumDataEncoder,
    QuantumFeatureMap
)
from .quantum_trainer import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    TrainingMetrics
)

__all__ = [
    # Quantum Layers
    "QuantumLayer",
    "QuantumConvolutionalLayer",
    "QuantumPoolingLayer",
    "LayerConfig",
    # Gradient Computation
    "QuantumGradientComputer",
    "FiniteDifferenceGradient",
    "NaturalGradientComputer",
    "GradientResult",
    # Data Encoding
    "QuantumDataEncoder",
    "QuantumFeatureMap",
    # Training
    "QuantumTrainer",
    "QuantumModel",
    "TrainingConfig",
    "TrainingMetrics",
]
