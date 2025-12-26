"""
Q-Store v4.1 Quantum-First Layers

This module provides quantum-first neural network layers that maximize
quantum computation and minimize classical overhead.

Architecture:
- quantum_core: Primary quantum operations (70% of compute)
- classical_minimal: Minimal classical pre/post processing (30% of compute)
- hybrid: Adaptive layers that switch between quantum and classical

Key Features:
- Async execution (never blocks on quantum hardware)
- Multi-basis measurements for rich feature spaces
- Native gate compilation for performance
- Framework-agnostic design (TF, PyTorch, JAX compatible)
"""

from q_store.layers.quantum_core import (
    QuantumFeatureExtractor,
    QuantumNonlinearity,
    QuantumPooling,
    QuantumReadout,
)

from q_store.layers.classical_minimal import (
    EncodingLayer,
    DecodingLayer,
)

__all__ = [
    'QuantumFeatureExtractor',
    'QuantumNonlinearity',
    'QuantumPooling',
    'QuantumReadout',
    'EncodingLayer',
    'DecodingLayer',
]
