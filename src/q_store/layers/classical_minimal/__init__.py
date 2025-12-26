"""
Classical Minimal Layers - v4.1

Minimal classical pre/post processing layers.
These layers perform only essential operations (~30% of total compute):
- Data normalization and encoding
- Output decoding and scaling

Design Philosophy:
- No heavy computation
- No activations (use quantum nonlinearity instead)
- No BatchNorm (use quantum-aware normalization if needed)
- Minimal parameter count
"""

from q_store.layers.classical_minimal.encoding_layer import EncodingLayer
from q_store.layers.classical_minimal.decoding_layer import DecodingLayer

__all__ = [
    'EncodingLayer',
    'DecodingLayer',
]
