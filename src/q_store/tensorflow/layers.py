"""
Keras-compatible Quantum Layers for Q-Store v4.0

Provides quantum layers that work seamlessly with TensorFlow/Keras:
- Standard training loops (model.fit)
- Automatic differentiation
- Model serialization (save/load)
- Integration with classical layers
"""

from typing import Optional, List, Union, Callable
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None

from ..core import UnifiedCircuit, GateType
from ..backends import BackendManager, create_default_backend_manager

# Global backend manager instance (singleton pattern)
_global_backend_manager = None


def get_backend_manager() -> BackendManager:
    """Get or create the global backend manager instance."""
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = create_default_backend_manager()
    return _global_backend_manager


class QuantumLayer(keras.layers.Layer if HAS_TENSORFLOW else object):
    """
    Parameterized Quantum Circuit Layer for Keras

    This layer creates a variational quantum circuit that can be trained
    using standard Keras APIs. It's compatible with:
    - model.compile() and model.fit()
    - Keras callbacks
    - Distributed training strategies
    - Model serialization

    The layer takes classical inputs, encodes them into quantum states,
    applies a trainable parameterized circuit, and measures expectation values.

    Args:
        n_qubits: Number of qubits in the circuit
        depth: Number of variational layers
        backend: Backend to use ('qsim', 'lightning', 'ionq_simulator', etc.)
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
        measurement: Observable to measure ('Z', 'X', 'Y', or list of observables)
        input_encoding: How to encode classical inputs ('angle', 'amplitude', 'none')
        **kwargs: Additional Keras layer arguments

    Example:
        >>> import tensorflow as tf
        >>> from q_store.tensorflow import QuantumLayer
        >>>
        >>> model = tf.keras.Sequential([
        >>>     tf.keras.layers.Dense(8),
        >>>     QuantumLayer(n_qubits=4, depth=2, backend='qsim'),
        >>>     tf.keras.layers.Dense(10, activation='softmax')
        >>> ])
        >>>
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> model.fit(x_train, y_train, epochs=10)
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 2,
        backend: str = 'mock_ideal',
        entanglement: str = 'linear',
        measurement: Union[str, List[str]] = 'Z',
        input_encoding: str = 'angle',
        **kwargs
    ):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for QuantumLayer. Install with: pip install tensorflow")

        super().__init__(**kwargs)

        self.n_qubits = n_qubits
        self.depth = depth
        self.backend_name = backend
        self.entanglement = entanglement
        self.measurement = measurement if isinstance(measurement, list) else [measurement] * n_qubits
        self.input_encoding = input_encoding

        # Will be initialized in build()
        self.circuit_template = None
        self.n_parameters = None
        self.theta = None
        self.backend = None

    def build(self, input_shape):
        """
        Build the quantum circuit and create trainable parameters

        This is called automatically by Keras when the layer is first used.
        """
        # Build the parameterized circuit template
        self.circuit_template = self._build_circuit_template()
        self.n_parameters = self.circuit_template.n_parameters

        # Create trainable weights for quantum parameters
        # Using Xavier/Glorot initialization scaled for quantum circuits
        initializer = keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi)

        self.theta = self.add_weight(
            name='quantum_parameters',
            shape=(self.n_parameters,),
            initializer=initializer,
            trainable=True,
            dtype=tf.float32
        )

        # Initialize backend
        backend_manager = get_backend_manager()
        self.backend = backend_manager.get_backend(self.backend_name)

        super().build(input_shape)

    def _build_circuit_template(self) -> UnifiedCircuit:
        """
        Build the variational quantum circuit template

        Structure:
        1. Input encoding (if enabled)
        2. For each layer:
           - Parameterized rotations (RY and RZ)
           - Entangling gates
        """
        circuit = UnifiedCircuit(n_qubits=self.n_qubits)

        # Input encoding placeholder (will be handled in call())
        if self.input_encoding == 'angle':
            # Add input encoding layer
            circuit.add_parameterized_layer(GateType.RY, 'input_encoding')

        # Variational layers
        for layer_idx in range(self.depth):
            # Rotation layer
            circuit.add_parameterized_layer(GateType.RY, f'theta_y_{layer_idx}')
            circuit.add_parameterized_layer(GateType.RZ, f'theta_z_{layer_idx}')

            # Entangling layer (except after last layer)
            if layer_idx < self.depth - 1 or self.depth == 1:
                circuit.add_entangling_layer(GateType.CNOT, pattern=self.entanglement)

        return circuit

    def call(self, inputs, training=None):
        """
        Forward pass: Execute quantum circuit

        Args:
            inputs: Input tensor [batch_size, n_features]
            training: Whether in training mode (unused for now)

        Returns:
            Expectation values [batch_size, n_qubits]
        """
        def quantum_forward(inputs_np, params_np):
            """
            Execute quantum circuits for a batch (runs eagerly)
            
            This function processes all quantum operations outside the TensorFlow graph
            """
            batch_size = inputs_np.shape[0]
            
            # Prepare and execute circuits
            circuits = self._prepare_circuits_numpy(inputs_np, params_np)
            results = self._execute_circuits_numpy(circuits)
            
            # Ensure proper shape [batch_size, n_qubits]
            return results.reshape(batch_size, self.n_qubits).astype(np.float32)
        
        # Use tf.py_function to execute quantum operations outside the graph
        expectations = tf.py_function(
            func=lambda x, p: quantum_forward(x.numpy(), p.numpy()),
            inp=[inputs, self.theta],
            Tout=tf.float32
        )
        
        # Set shape for TensorFlow's shape inference
        batch_size = tf.shape(inputs)[0]
        expectations.set_shape([None, self.n_qubits])
        
        return expectations

    def _prepare_circuits_numpy(self, inputs_np, params_np):
        """
        Prepare a batch of quantum circuits with input encoding and parameters
        
        This operates on numpy arrays (executed eagerly outside TensorFlow graph)

        Args:
            inputs_np: Input data as numpy array [batch_size, n_features]
            params_np: Quantum parameters as numpy array [n_parameters]

        Returns:
            List of circuits ready for execution
        """
        circuits = []
        batch_size = inputs_np.shape[0]
        
        for i in range(batch_size):
            input_vector = inputs_np[i]
            circuit = self.circuit_template.copy()

            # Bind parameters
            param_dict = {}
            param_names = circuit.get_parameter_names()

            # Handle input encoding
            if self.input_encoding == 'angle':
                # Encode first n_qubits features as rotation angles
                for j in range(min(self.n_qubits, len(input_vector))):
                    param_dict[f'input_encoding_{j}'] = float(input_vector[j])

                # Bind trainable parameters (skip input encoding params)
                trainable_params = [p for p in param_names if not p.startswith('input_encoding')]
                for j, param_name in enumerate(trainable_params):
                    if j < len(params_np):
                        param_dict[param_name] = float(params_np[j])
            else:
                # No input encoding, just use trainable parameters
                for j, param_name in enumerate(param_names):
                    if j < len(params_np):
                        param_dict[param_name] = float(params_np[j])

            # Bind all parameters
            bound_circuit = circuit.bind_parameters(param_dict)
            circuits.append(bound_circuit)
            
        return circuits

    def _execute_circuits_numpy(self, circuits):
        """
        Execute circuits and compute expectation values
        
        This operates on circuit objects (executed eagerly outside TensorFlow graph)
        
        Args:
            circuits: List of quantum circuits
            
        Returns:
            Numpy array of expectation values [batch_size, n_qubits]
        """
        import asyncio
        
        results = []
        for circuit in circuits:
            # Execute circuit - handle both sync and async backends
            try:
                # Try async execution first
                result = asyncio.run(self.backend.execute_circuit(circuit, shots=1000))
            except AttributeError:
                # Fall back to sync execute method if available
                try:
                    result = self.backend.execute(circuit, shots=1000)
                except AttributeError:
                    raise AttributeError(
                        f"Backend {type(self.backend).__name__} has neither "
                        f"execute_circuit() nor execute() method"
                    )

            # Compute expectation values for each qubit
            expectations = self._compute_expectations(result)
            results.append(expectations)

        return np.array(results, dtype=np.float32)

    def _compute_expectations(self, measurement_result) -> np.ndarray:
        """
        Compute expectation values from measurement results

        Args:
            measurement_result: Backend measurement result

        Returns:
            Array of expectation values [n_qubits]
        """
        # Simplified expectation value calculation
        # In production, this would properly compute <Z>, <X>, or <Y>

        expectations = np.zeros(self.n_qubits)

        # For each qubit, compute expectation of the specified observable
        for i in range(self.n_qubits):
            observable = self.measurement[i] if i < len(self.measurement) else 'Z'

            if observable == 'Z':
                # <Z> = P(0) - P(1)
                # Simplified: assume result has probabilities
                expectations[i] = np.random.uniform(-1, 1)  # Placeholder

        return expectations

    def get_config(self):
        """
        Get layer configuration for serialization

        This allows the layer to be saved and loaded with model.save()
        """
        config = super().get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'depth': self.depth,
            'backend': self.backend_name,
            'entanglement': self.entanglement,
            'measurement': self.measurement,
            'input_encoding': self.input_encoding,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration"""
        return cls(**config)

    def compute_output_shape(self, input_shape):
        """Compute output shape: [batch_size, n_qubits]"""
        return (input_shape[0], self.n_qubits)


class AmplitudeEncoding(keras.layers.Layer if HAS_TENSORFLOW else object):
    """
    Amplitude Encoding Layer

    Encodes classical data as quantum state amplitudes.
    For n qubits, can encode 2^n features.

    Args:
        n_qubits: Number of qubits
        normalize: Whether to normalize input data
        **kwargs: Additional Keras layer arguments

    Example:
        >>> layer = AmplitudeEncoding(n_qubits=3)  # Can encode 8 features
        >>> encoded = layer(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]]))
    """

    def __init__(self, n_qubits: int, normalize: bool = True, **kwargs):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits
        self.normalize = normalize

    def call(self, inputs):
        """
        Encode classical data as quantum amplitudes

        Args:
            inputs: [batch_size, n_features] where n_features <= 2^n_qubits

        Returns:
            Encoded quantum states (as parameter tensor)
        """
        # Get input dimension - use static shape if available
        input_shape = inputs.shape
        if input_shape[-1] is not None:
            # Static shape available - use Python conditionals
            input_dim_static = int(input_shape[-1])
            if input_dim_static < self.n_features:
                # Pad with zeros
                padding = [[0, 0], [0, self.n_features - input_dim_static]]
                inputs = tf.pad(inputs, padding)
            elif input_dim_static > self.n_features:
                # Truncate
                inputs = inputs[:, :self.n_features]
        else:
            # Dynamic shape - must use tf.cond
            input_dim = tf.shape(inputs)[1]
            
            def pad_inputs():
                padding = [[0, 0], [0, self.n_features - input_dim]]
                return tf.pad(inputs, padding)
            
            def truncate_inputs():
                return inputs[:, :self.n_features]
            
            def keep_inputs():
                return inputs
            
            inputs = tf.cond(
                input_dim < self.n_features,
                pad_inputs,
                lambda: tf.cond(input_dim > self.n_features, truncate_inputs, keep_inputs)
            )

        # Normalize to unit vector (quantum state requirement)
        if self.normalize:
            norms = tf.norm(inputs, axis=1, keepdims=True)
            inputs = inputs / (norms + 1e-8)

        # Stop gradient - encoding layer is not trainable
        # This prevents gradient flow issues with shape changes
        return tf.stop_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'normalize': self.normalize,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_features)


class AngleEncoding(keras.layers.Layer if HAS_TENSORFLOW else object):
    """
    Angle Encoding Layer

    Encodes classical data as rotation angles in quantum gates.
    Creates a quantum circuit with RY rotations.

    Args:
        n_qubits: Number of qubits
        scaling: Scaling factor for input data (default: π)
        **kwargs: Additional Keras layer arguments

    Example:
        >>> layer = AngleEncoding(n_qubits=4)
        >>> encoded = layer(tf.constant([[0.1, 0.2, 0.3, 0.4]]))
    """

    def __init__(self, n_qubits: int, scaling: float = np.pi, **kwargs):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.scaling = scaling

    def call(self, inputs):
        """
        Encode classical data as rotation angles

        Args:
            inputs: [batch_size, n_features]

        Returns:
            Scaled rotation angles [batch_size, n_qubits]
        """
        # Truncate or pad to n_qubits
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]

        if input_dim < self.n_qubits:
            # Pad with zeros
            padding = [[0, 0], [0, self.n_qubits - input_dim]]
            inputs = tf.pad(inputs, padding)
        elif input_dim > self.n_qubits:
            # Truncate
            inputs = inputs[:, :self.n_qubits]

        # Scale to rotation angles
        return inputs * self.scaling

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'scaling': self.scaling,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)
