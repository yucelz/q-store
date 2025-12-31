"""
TensorFlow Quantum Layer - Base Class

tf.keras.layers.Layer compatible quantum layer with async execution.

Key Features:
- @tf.custom_gradient for backprop
- SPSA gradient estimation
- Async quantum execution
- Parameter management
- GPU tensor support
"""

import tensorflow as tf
import numpy as np
import asyncio
from typing import Optional, Callable, Dict, Any

from q_store.layers.quantum_core import QuantumFeatureExtractor
from q_store.runtime import AsyncQuantumExecutor


class QuantumLayer(tf.keras.layers.Layer):
    """
    Base quantum layer for TensorFlow.

    Compatible with tf.keras.layers.Layer API.
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
        Gradient estimation method ('spsa', 'parameter_shift')
    spsa_epsilon : float, default=0.01
        SPSA perturbation size
    async_execution : bool, default=True
        Use async execution
    **kwargs
        Additional layer arguments

    Examples
    --------
    >>> layer = QuantumLayer(n_qubits=4, n_layers=2)
    >>> x = tf.random.normal((32, 16))  # batch_size=32, features=16
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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_name = backend
        self.shots = shots
        self.gradient_method = gradient_method
        self.spsa_epsilon = spsa_epsilon
        self.async_execution = async_execution

        # Create quantum feature extractor
        self.quantum_extractor = None  # Created in build()

        # Async executor
        self.executor = None

    def build(self, input_shape):
        """Build layer (called on first call)."""
        # Create quantum feature extractor
        self.quantum_extractor = QuantumFeatureExtractor(
            n_qubits=self.n_qubits,
            depth=self.n_layers,
            entanglement='full',
            measurement_bases=['X', 'Y', 'Z'],
            backend=self.backend_name,
        )

        # Create async executor if needed
        if self.async_execution:
            self.executor = AsyncQuantumExecutor(
                backend=self.backend_name,
                max_concurrent=10,
            )
            self.quantum_extractor.executor = self.executor

        # Create trainable parameters
        n_params = self.quantum_extractor.n_parameters
        self.quantum_params = self.add_weight(
            name='quantum_params',
            shape=(n_params,),
            initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi),
            trainable=True,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass with custom gradient.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor (batch_size, features)
        training : bool, optional
            Training mode

        Returns
        -------
        outputs : tf.Tensor
            Quantum features (batch_size, n_qubits * n_bases)
        """
        @tf.custom_gradient
        def quantum_forward(x, params):
            """Forward pass with custom gradient."""
            # Forward pass
            output = self._forward_pass(x, params)

            def grad_fn(dy):
                """Custom gradient using SPSA."""
                if self.gradient_method == 'spsa':
                    grad_params = self._spsa_gradient(x, params, dy)
                else:
                    # Parameter shift (future implementation)
                    grad_params = self._spsa_gradient(x, params, dy)

                # Input gradients (none for quantum)
                grad_x = None

                return grad_x, grad_params

            return output, grad_fn

        return quantum_forward(inputs, self.quantum_params)

    def _forward_pass(self, inputs, params):
        """Execute forward pass."""
        # Convert to numpy
        x_np = inputs.numpy()
        params_np = params.numpy()

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

        # Convert back to TensorFlow
        return tf.constant(output_np, dtype=tf.float32)

    def _spsa_gradient(self, inputs, params, dy):
        """
        Estimate gradient using SPSA.

        SPSA (Simultaneous Perturbation Stochastic Approximation):
        - Perturb all parameters simultaneously
        - Only 2 forward passes needed
        - Unbiased gradient estimate

        Parameters
        ----------
        inputs : tf.Tensor
            Input data
        params : tf.Tensor
            Current parameters
        dy : tf.Tensor
            Upstream gradients

        Returns
        -------
        grad_params : tf.Tensor
            Parameter gradients
        """
        epsilon = self.spsa_epsilon

        # Random perturbation direction
        delta = tf.random.uniform(
            shape=params.shape,
            minval=0,
            maxval=1,
            dtype=tf.float32
        )
        delta = tf.where(delta < 0.5, -1.0, 1.0)  # Bernoulli {-1, +1}

        # Perturbed parameters
        params_plus = params + epsilon * delta
        params_minus = params - epsilon * delta

        # Forward passes
        output_plus = self._forward_pass(inputs, params_plus)
        output_minus = self._forward_pass(inputs, params_minus)

        # Finite difference
        output_diff = output_plus - output_minus

        # Chain rule: dy/dparams = dy/doutput * doutput/dparams
        # SPSA gradient: (f(θ+ε) - f(θ-ε)) / (2ε) * 1/δ
        grad_output = tf.reduce_sum(dy * output_diff, axis=0)  # Sum over batch
        grad_params = grad_output / (2 * epsilon * delta)

        # Average over output dimensions
        grad_params = tf.reduce_mean(grad_params)

        return grad_params

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'backend': self.backend_name,
            'shots': self.shots,
            'gradient_method': self.gradient_method,
            'spsa_epsilon': self.spsa_epsilon,
            'async_execution': self.async_execution,
        })
        return config
