"""
PyTorch Quantum Layer - Base Class

nn.Module compatible quantum layer with async execution.

Key Features:
- torch.autograd.Function for backprop
- SPSA gradient estimation
- Async quantum execution
- Parameter management
- GPU tensor support
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import Optional

from q_store.layers.quantum_core import QuantumFeatureExtractor
from q_store.runtime import AsyncQuantumExecutor


class QuantumFunction(torch.autograd.Function):
    """
    Custom autograd function for quantum circuits.

    Implements forward and backward passes with SPSA gradients.
    """

    @staticmethod
    def forward(ctx, inputs, params, quantum_layer, epsilon):
        """
        Forward pass.

        Parameters
        ----------
        ctx : context
            Context for backward pass
        inputs : torch.Tensor
            Input data (batch_size, features)
        params : torch.Tensor
            Quantum parameters
        quantum_layer : QuantumLayer
            Quantum layer instance
        epsilon : float
            SPSA epsilon

        Returns
        -------
        outputs : torch.Tensor
            Quantum features
        """
        # Save for backward
        ctx.save_for_backward(inputs, params)
        ctx.quantum_layer = quantum_layer
        ctx.epsilon = epsilon

        # Execute quantum circuits
        with torch.no_grad():
            outputs = quantum_layer._forward_pass(inputs, params)

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with SPSA gradients.

        Parameters
        ----------
        ctx : context
            Saved context
        grad_output : torch.Tensor
            Upstream gradients

        Returns
        -------
        grad_inputs : None
            No gradients for inputs (quantum encoding)
        grad_params : torch.Tensor
            Parameter gradients
        grad_layer : None
            No gradients for layer
        grad_epsilon : None
            No gradients for epsilon
        """
        inputs, params = ctx.saved_tensors
        quantum_layer = ctx.quantum_layer
        epsilon = ctx.epsilon

        # Compute SPSA gradients
        with torch.no_grad():
            grad_params = quantum_layer._spsa_gradient(
                inputs, params, grad_output, epsilon
            )

        # Return gradients (None for non-parameter inputs)
        return None, grad_params, None, None


class QuantumLayer(nn.Module):
    """
    Base quantum layer for PyTorch.

    Compatible with nn.Module API.
    Uses SPSA for gradient estimation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    n_layers : int, default=1
        Number of variational layers
    backend : str, default='simulator'
        Quantum backend
    shots : int, default=1024
        Measurement shots
    gradient_method : str, default='spsa'
        Gradient estimation method
    spsa_epsilon : float, default=0.01
        SPSA perturbation size
    async_execution : bool, default=True
        Use async execution

    Examples
    --------
    >>> layer = QuantumLayer(n_qubits=4, n_layers=2)
    >>> x = torch.randn(32, 16)  # batch_size=32, features=16
    >>> y = layer(x)  # (32, 4*3) - 4 qubits, 3 bases
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        backend: str = 'simulator',
        shots: int = 1024,
        gradient_method: str = 'spsa',
        spsa_epsilon: float = 0.01,
        async_execution: bool = True,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_name = backend
        self.shots = shots
        self.gradient_method = gradient_method
        self.spsa_epsilon = spsa_epsilon
        self.async_execution = async_execution

        # Create quantum feature extractor
        self.quantum_extractor = QuantumFeatureExtractor(
            n_qubits=n_qubits,
            depth=n_layers,
            entanglement='full',
            measurement_bases=['X', 'Y', 'Z'],
            backend=backend,
        )

        # Create async executor if needed
        if self.async_execution:
            self.executor = AsyncQuantumExecutor(
                backend=backend,
                max_concurrent=10,
            )
            self.quantum_extractor.executor = self.executor

        # Create trainable parameters
        n_params = self.quantum_extractor.n_parameters
        self.quantum_params = nn.Parameter(
            torch.rand(n_params) * 2 * np.pi,
            requires_grad=True
        )

    def forward(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (batch_size, features)

        Returns
        -------
        outputs : torch.Tensor
            Quantum features (batch_size, n_qubits * n_bases)
        """
        return QuantumFunction.apply(
            inputs,
            self.quantum_params,
            self,
            self.spsa_epsilon
        )

    def _forward_pass(self, inputs, params):
        """Execute forward pass."""
        # Convert to numpy
        x_np = inputs.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()

        # Update quantum parameters
        self.quantum_extractor.params = params_np

        # Execute quantum circuits
        if self.async_execution:
            # Async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                output_np = loop.run_until_complete(
                    self.quantum_extractor.call_async(x_np)
                )
            finally:
                loop.close()
        else:
            # Sync execution
            output_np = self.quantum_extractor(x_np)

        # Convert back to PyTorch
        device = inputs.device
        return torch.tensor(output_np, dtype=torch.float32, device=device)

    def _spsa_gradient(self, inputs, params, grad_output, epsilon):
        """
        Estimate gradient using SPSA.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data
        params : torch.Tensor
            Current parameters
        grad_output : torch.Tensor
            Upstream gradients
        epsilon : float
            Perturbation size

        Returns
        -------
        grad_params : torch.Tensor
            Parameter gradients
        """
        # Random perturbation direction
        delta = torch.rand_like(params)
        delta = torch.where(delta < 0.5, -1.0, 1.0)

        # Perturbed parameters
        params_plus = params + epsilon * delta
        params_minus = params - epsilon * delta

        # Forward passes
        output_plus = self._forward_pass(inputs, params_plus)
        output_minus = self._forward_pass(inputs, params_minus)

        # Finite difference
        output_diff = output_plus - output_minus

        # Chain rule
        grad_output_flat = grad_output.sum(dim=0)  # Sum over batch
        grad_params = (grad_output_flat * output_diff.sum(dim=0)) / (2 * epsilon * delta)

        # Average
        grad_params = grad_params.mean()

        return grad_params
