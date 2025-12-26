"""
Q-Store TensorFlow Integration (v4.0 + v4.1)

Provides Keras-compatible quantum layers that integrate seamlessly
with TensorFlow/Keras models, training loops, and distributed strategies.

v4.1 Additions:
- Quantum-first layers (QuantumDense for Dense replacement)
- Async execution support
- SPSA gradient estimation
- Storage integration

Example:
    >>> import tensorflow as tf
    >>> from q_store.tensorflow import QuantumDense
    >>>
    >>> # v4.1: Quantum-first architecture (70% quantum)
    >>> model = tf.keras.Sequential([
    >>>     tf.keras.layers.Flatten(input_shape=(28, 28)),
    >>>     QuantumDense(n_qubits=7),  # Replaces Dense(128)
    >>>     tf.keras.layers.Dense(10, activation='softmax')
    >>> ])
    >>>
    >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    >>> model.fit(x_train, y_train, epochs=10)
"""

# v4.0 components
from .layers import QuantumLayer as QuantumLayerV4, AmplitudeEncoding, AngleEncoding
from .circuit_executor import TensorFlowCircuitExecutor
from .gradients import ParameterShiftGradient, AdjointGradient

# v4.1 components (new)
try:
    from .quantum_layer import QuantumLayer
    from .quantum_dense import QuantumDense
    from .spsa_gradients import spsa_gradient
    HAS_V4_1 = True
except ImportError:
    HAS_V4_1 = False
    QuantumLayer = None
    QuantumDense = None
    spsa_gradient = None

__all__ = [
    # v4.0
    'QuantumLayerV4',
    'AmplitudeEncoding',
    'AngleEncoding',
    'TensorFlowCircuitExecutor',
    'ParameterShiftGradient',
    'AdjointGradient',
    # v4.1
    'QuantumLayer',
    'QuantumDense',
    'spsa_gradient',
]
