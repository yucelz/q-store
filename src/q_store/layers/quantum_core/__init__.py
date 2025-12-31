"""
Quantum Core Layers - v4.1

Primary quantum operations that form the backbone of quantum-first ML models.
These layers maximize quantum computation and achieve 60-70% quantum utilization.
"""

from q_store.layers.quantum_core.quantum_feature_extractor import QuantumFeatureExtractor
from q_store.layers.quantum_core.quantum_nonlinearity import QuantumNonlinearity
from q_store.layers.quantum_core.quantum_pooling import QuantumPooling
from q_store.layers.quantum_core.quantum_readout import QuantumReadout
from q_store.layers.quantum_core.quantum_regularization import (
    QuantumDropout,
    QuantumRegularization,
    apply_quantum_regularization,
)

__all__ = [
    'QuantumFeatureExtractor',
    'QuantumNonlinearity',
    'QuantumPooling',
    'QuantumReadout',
    # v4.1 Enhanced: Quantum Regularization
    'QuantumDropout',
    'QuantumRegularization',
    'apply_quantum_regularization',
]
