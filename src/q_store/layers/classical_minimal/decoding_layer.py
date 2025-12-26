"""
Decoding Layer - v4.1

Minimal classical postprocessing after quantum computation.
Performs only essential operations: scaling and optional projection.

Key Features:
- Scale expectation values from [-1, 1] to [0, 1]
- Optional linear projection (no activation!)
- Minimal parameter count
- Fast CPU operation (~1-5% of total compute)
"""

import numpy as np
from typing import Dict, Any, Optional


class DecodingLayer:
    """
    Minimal decoding layer after quantum measurements.

    Does ONLY:
    - Scale expectation values to [0, 1]
    - Optional linear projection (no bias, no activation)

    No activations! No BatchNorm! Minimal compute!

    Parameters
    ----------
    output_dim : int, optional
        Output dimension for linear projection. If None, no projection.
    scaling : str, default='expectation'
        Scaling method: 'expectation' ([-1,1] to [0,1]) or 'none'

    Examples
    --------
    >>> layer = DecodingLayer(output_dim=10)  # Project to 10 classes
    >>> decoded = layer(quantum_features)
    >>> print(decoded.shape)  # (batch_size, 10)
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        scaling: str = 'expectation',
        **kwargs
    ):
        self.output_dim = output_dim
        self.scaling = scaling

        # Linear projection weights (if output_dim specified)
        self.projection_weights = None
        if output_dim is not None:
            # Will be initialized on first call
            self._initialized = False
        else:
            self._initialized = True

    def _initialize_projection(self, input_dim: int):
        """Initialize projection weights."""
        if self.output_dim is not None and not self._initialized:
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (input_dim + self.output_dim))
            self.projection_weights = np.random.randn(input_dim, self.output_dim) * scale
            self._initialized = True

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Decode quantum measurements to classical outputs.

        Parameters
        ----------
        inputs : np.ndarray
            Quantum measurement results of shape (batch_size, n_features)

        Returns
        -------
        decoded : np.ndarray
            Decoded outputs of shape (batch_size, output_dim) or (batch_size, n_features)
        """
        # Initialize projection if needed
        if not self._initialized:
            self._initialize_projection(inputs.shape[1])

        # Step 1: Scale
        if self.scaling == 'expectation':
            # Expectation values are in [-1, 1], scale to [0, 1]
            scaled = (inputs + 1.0) / 2.0
        elif self.scaling == 'probability':
            # Already in [0, 1], clip to be safe
            scaled = np.clip(inputs, 0.0, 1.0)
        elif self.scaling == 'none':
            scaled = inputs
        else:
            raise ValueError(f"Unknown scaling: {self.scaling}")

        # Step 2: Optional linear projection
        if self.projection_weights is not None:
            output = np.dot(scaled, self.projection_weights)
        else:
            output = scaled

        return output

    def set_weights(self, weights: np.ndarray):
        """
        Set projection weights manually.

        Parameters
        ----------
        weights : np.ndarray
            Projection weights of shape (input_dim, output_dim)
        """
        if self.output_dim is None:
            raise ValueError("Cannot set weights when output_dim is None")

        expected_shape = (weights.shape[0], self.output_dim)
        if weights.shape[1] != self.output_dim:
            raise ValueError(f"Expected weights shape (*, {self.output_dim}), got {weights.shape}")

        self.projection_weights = weights
        self._initialized = True

    def get_weights(self) -> Optional[np.ndarray]:
        """Get current projection weights."""
        return self.projection_weights

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        return {
            'output_dim': self.output_dim,
            'scaling': self.scaling,
        }
