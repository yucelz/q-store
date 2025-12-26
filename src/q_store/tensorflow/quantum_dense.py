"""
Quantum Dense Layer - TensorFlow

Drop-in replacement for tf.keras.layers.Dense using quantum circuits.

Key difference from classical Dense:
- Classical: y = Wx + b (linear transformation)
- Quantum: y = U(x)ψ(θ)|0⟩ (unitary transformation)

Advantages:
- Non-linear by default (quantum gates)
- Exponential expressiveness (2^n states)
- Natural feature interactions (entanglement)
"""

import tensorflow as tf
import numpy as np

from q_store.tensorflow.quantum_layer import QuantumLayer


class QuantumDense(QuantumLayer):
    """
    Quantum Dense layer (replacement for Dense).

    Usage identical to tf.keras.layers.Dense:
    >>> # Classical
    >>> dense = tf.keras.layers.Dense(128, activation='relu')
    >>>
    >>> # Quantum (70% of computation)
    >>> dense = QuantumDense(n_qubits=7)  # 2^7=128 dimensional output

    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines output dimension)
    n_layers : int, default=2
        Variational layers (depth)
    activation : str or callable, optional
        Classical activation (applied after quantum)
    use_bias : bool, default=False
        Add bias (classical)
    backend : str, default='simulator'
        Quantum backend
    shots : int, default=1024
        Measurement shots
    **kwargs
        Additional layer arguments

    Examples
    --------
    >>> # Replace Dense in existing model
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Flatten(input_shape=(28, 28)),
    ...     QuantumDense(n_qubits=7),  # 128-dim output
    ...     tf.keras.layers.Dense(10, activation='softmax')
    ... ])
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        activation=None,
        use_bias: bool = False,
        backend: str = 'simulator',
        shots: int = 1024,
        **kwargs
    ):
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend=backend,
            shots=shots,
            **kwargs
        )

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.bias = None

        # Output dimension (n_qubits * n_measurement_bases)
        self.output_dim = n_qubits * 3  # X, Y, Z bases

    def build(self, input_shape):
        """Build layer."""
        super().build(input_shape)

        # Add bias if requested
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.output_dim,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32,
            )

    def call(self, inputs, training=None):
        """Forward pass."""
        # Quantum computation
        output = super().call(inputs, training=training)

        # Add bias
        if self.use_bias:
            output = output + self.bias

        # Apply activation
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        """Get configuration."""
        config = super().get_config()
        config.update({
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
        })
        return config

    @property
    def units(self):
        """Output units (for Dense compatibility)."""
        return self.output_dim
