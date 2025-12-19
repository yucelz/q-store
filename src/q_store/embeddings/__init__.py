"""
Quantum data embedding strategies.

This module provides various data encoding methods for quantum machine learning:
- Feature maps (ZZ, IQP, Pauli)
- Amplitude encoding
- Angle encoding
- Basis encoding
"""

from q_store.embeddings.feature_maps import (
    ZZFeatureMap,
    PauliFeatureMap,
    IQPFeatureMap,
)
from q_store.embeddings.amplitude_encoding import AmplitudeEncoding
from q_store.embeddings.angle_encoding import AngleEncoding
from q_store.embeddings.basis_encoding import BasisEncoding

__all__ = [
    'ZZFeatureMap',
    'PauliFeatureMap',
    'IQPFeatureMap',
    'AmplitudeEncoding',
    'AngleEncoding',
    'BasisEncoding',
]
