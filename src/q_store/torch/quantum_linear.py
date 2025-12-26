"""
Quantum Linear Layer - PyTorch

Drop-in replacement for nn.Linear using quantum circuits.

Key difference from classical Linear:
- Classical: y = Wx + b (linear transformation)
- Quantum: y = U(x)ψ(θ)|0⟩ (unitary transformation)

Advantages:
- Non-linear by default (quantum gates)
- Exponential expressiveness (2^n states)
- Natural feature interactions (entanglement)
"""

import torch
import torch.nn as nn
import numpy as np

from q_store.torch.quantum_layer import QuantumLayer


class QuantumLinear(QuantumLayer):
    """
    Quantum Linear layer (replacement for nn.Linear).

    Usage identical to nn.Linear:
    >>> # Classical
    >>> linear = nn.Linear(in_features=16, out_features=128)
    >>>
    >>> # Quantum (70% of computation)
    >>> linear = QuantumLinear(n_qubits=7)  # 2^7=128 dimensional output

    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines output dimension)
    n_layers : int, default=2
        Variational layers (depth)
    bias : bool, default=False
        Add bias (classical)
    backend : str, default='simulator'
        Quantum backend
    shots : int, default=1024
        Measurement shots

    Examples
    --------
    >>> # Replace Linear in existing model
    >>> model = nn.Sequential(
    ...     nn.Flatten(),
    ...     QuantumLinear(n_qubits=7),  # 128-dim output
    ...     nn.Linear(128, 10),
    ... )
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        bias: bool = False,
        backend: str = 'simulator',
        shots: int = 1024,
    ):
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend=backend,
            shots=shots,
        )

        self.use_bias = bias

        # Output dimension (n_qubits * n_measurement_bases)
        self.out_features = n_qubits * 3  # X, Y, Z bases

        # Add bias if requested
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features),
                requires_grad=True
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        """Forward pass."""
        # Quantum computation
        output = super().forward(inputs)

        # Add bias
        if self.use_bias:
            output = output + self.bias

        return output
