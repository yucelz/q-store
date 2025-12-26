"""
Q-Store TensorFlow Integration (v4.0)

Provides Keras-compatible quantum layers that integrate seamlessly
with TensorFlow/Keras models, training loops, and distributed strategies.

Example:
    >>> import tensorflow as tf
    >>> import q_store.tensorflow as qs_tf
    >>>
    >>> model = tf.keras.Sequential([
    >>>     qs_tf.QuantumLayer(n_qubits=4, depth=2),
    >>>     tf.keras.layers.Dense(10, activation='softmax')
    >>> ])
    >>>
    >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    >>> model.fit(x_train, y_train, epochs=10)
"""

from .layers import QuantumLayer, AmplitudeEncoding, AngleEncoding
from .circuit_executor import TensorFlowCircuitExecutor
from .gradients import ParameterShiftGradient, AdjointGradient

__all__ = [
    'QuantumLayer',
    'AmplitudeEncoding',
    'AngleEncoding',
    'TensorFlowCircuitExecutor',
    'ParameterShiftGradient',
    'AdjointGradient',
]
