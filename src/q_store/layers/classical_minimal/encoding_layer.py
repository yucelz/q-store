"""
Encoding Layer - v4.1

Minimal classical preprocessing before quantum computation.
Performs only essential operations: normalization and dimension adjustment.

Key Features:
- L2 normalization for stable quantum encoding
- Dimension padding/truncation to match qubit requirements
- No learnable parameters (purely preprocessing)
- Fast CPU operation (~1-5% of total compute)
"""

import numpy as np
from typing import Dict, Any, Optional


class EncodingLayer:
    """
    Minimal encoding layer for quantum circuits.

    Does ONLY:
    - L2 normalization (ensures unit vectors for amplitude encoding)
    - Dimension padding/truncation (match 2^n_qubits requirement)

    No heavy compute! No learnable parameters!

    Parameters
    ----------
    target_dim : int
        Target dimension (typically 2^n_qubits for amplitude encoding)
    normalization : str, default='l2'
        Normalization type: 'l2', 'l1', or 'none'
    padding_value : float, default=0.0
        Value to use for padding if input dimension < target dimension

    Examples
    --------
    >>> layer = EncodingLayer(target_dim=256)  # For 8 qubits (2^8 = 256)
    >>> encoded = layer(inputs)
    >>> print(encoded.shape)  # (batch_size, 256)
    """

    def __init__(
        self,
        target_dim: int,
        normalization: str = 'l2',
        padding_value: float = 0.0,
        **kwargs
    ):
        self.target_dim = target_dim
        self.normalization = normalization
        self.padding_value = padding_value

        # Validate target_dim is power of 2
        if target_dim > 0 and (target_dim & (target_dim - 1)) != 0:
            import warnings
            warnings.warn(
                f"target_dim={target_dim} is not a power of 2. "
                "This may not align with qubit requirements."
            )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Encode classical data for quantum circuits.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch_size, input_dim)

        Returns
        -------
        encoded : np.ndarray
            Encoded data of shape (batch_size, target_dim)
        """
        # Step 1: Normalize
        if self.normalization == 'l2':
            norms = np.linalg.norm(inputs, axis=1, keepdims=True)
            normalized = inputs / (norms + 1e-8)  # Avoid division by zero
        elif self.normalization == 'l1':
            norms = np.sum(np.abs(inputs), axis=1, keepdims=True)
            normalized = inputs / (norms + 1e-8)
        elif self.normalization == 'none':
            normalized = inputs
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        # Step 2: Adjust dimension
        current_dim = normalized.shape[1]

        if current_dim < self.target_dim:
            # Pad with zeros (or specified value)
            padding = np.full(
                (normalized.shape[0], self.target_dim - current_dim),
                self.padding_value,
                dtype=normalized.dtype
            )
            encoded = np.concatenate([normalized, padding], axis=1)
        elif current_dim > self.target_dim:
            # Truncate (keep first target_dim features)
            encoded = normalized[:, :self.target_dim]
        else:
            # Exact match
            encoded = normalized

        return encoded

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        return {
            'target_dim': self.target_dim,
            'normalization': self.normalization,
            'padding_value': self.padding_value,
        }
