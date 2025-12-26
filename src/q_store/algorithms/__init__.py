"""
Variational quantum algorithms.

This module provides implementations of variational quantum algorithms including:
- VQE (Variational Quantum Eigensolver)
- QAOA (Quantum Approximate Optimization Algorithm)
"""

from q_store.algorithms.vqe import VQE, VQEResult
from q_store.algorithms.qaoa import QAOA, QAOAResult

__all__ = [
    'VQE',
    'VQEResult',
    'QAOA',
    'QAOAResult',
]
