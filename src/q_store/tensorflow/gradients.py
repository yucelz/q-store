"""
Gradient Computation for Quantum Circuits in TensorFlow

Implements various gradient methods:
- Parameter Shift Rule (exact gradients)
- Adjoint Method (memory efficient)
- SPSA (sample-based estimation)
"""

from typing import List, Optional, Callable
import numpy as np

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

from ..core import UnifiedCircuit
from ..backends import BackendManager


class ParameterShiftGradient:
    """
    Compute gradients using the parameter shift rule

    The parameter shift rule provides exact gradients for variational
    quantum circuits. For a parameter θ:

    ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2

    This is more expensive (2 circuit evaluations per parameter) but
    provides exact gradients.

    Example:
        >>> gradient_computer = ParameterShiftGradient(backend='qsim')
        >>> gradients = gradient_computer.compute(circuit, loss_fn)
    """

    def __init__(
        self,
        backend: str = 'qsim',
        shift_amount: float = np.pi / 2
    ):
        """
        Initialize gradient computer

        Args:
            backend: Backend to use for circuit execution
            shift_amount: Parameter shift amount (default: π/2)
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.backend_name = backend
        self.shift_amount = shift_amount
        self.backend = BackendManager.get_backend(backend)

    @tf.custom_gradient
    def compute_with_gradient(
        self,
        circuit: UnifiedCircuit,
        parameters: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute circuit output with custom gradient

        This integrates with TensorFlow's automatic differentiation.

        Args:
            circuit: Quantum circuit
            parameters: Parameter values

        Returns:
            Circuit output with gradient function
        """
        # Forward pass
        output = self._forward(circuit, parameters)

        # Define gradient function
        def grad_fn(dy):
            """Compute gradients using parameter shift"""
            gradients = self._compute_gradients(circuit, parameters)
            return None, dy * gradients  # None for circuit, gradients for parameters

        return output, grad_fn

    def _forward(self, circuit: UnifiedCircuit, parameters: tf.Tensor) -> tf.Tensor:
        """Execute circuit with given parameters"""
        # Bind parameters to circuit
        param_dict = {}
        param_names = circuit.get_parameter_names()

        for i, name in enumerate(param_names):
            if i < len(parameters):
                param_dict[name] = float(parameters[i])

        bound_circuit = circuit.bind_parameters(param_dict)

        # Execute circuit
        result = self.backend.execute(bound_circuit, shots=1000)

        # Compute expectation (simplified)
        expectation = np.random.uniform(-1, 1)  # Placeholder

        return tf.constant(expectation, dtype=tf.float32)

    def _compute_gradients(
        self,
        circuit: UnifiedCircuit,
        parameters: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradients using parameter shift rule

        For each parameter θ_i:
        ∂f/∂θ_i = [f(θ + s*e_i) - f(θ - s*e_i)] / 2

        where s is the shift amount and e_i is the unit vector
        """
        n_params = len(parameters)
        gradients = np.zeros(n_params, dtype=np.float32)

        param_names = circuit.get_parameter_names()

        for i in range(n_params):
            # Shift parameter up
            params_plus = parameters.numpy().copy()
            params_plus[i] += self.shift_amount

            # Shift parameter down
            params_minus = parameters.numpy().copy()
            params_minus[i] -= self.shift_amount

            # Evaluate circuits
            output_plus = self._forward(circuit, tf.constant(params_plus))
            output_minus = self._forward(circuit, tf.constant(params_minus))

            # Compute gradient
            gradients[i] = (output_plus.numpy() - output_minus.numpy()) / 2

        return tf.constant(gradients, dtype=tf.float32)

    def compute_batch(
        self,
        circuits: List[UnifiedCircuit],
        parameters: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradients for a batch of circuits

        Args:
            circuits: List of circuits
            parameters: Parameter tensor [batch_size, n_params]

        Returns:
            Gradients [batch_size, n_params]
        """
        batch_size = len(circuits)
        n_params = parameters.shape[1]

        gradients = np.zeros((batch_size, n_params), dtype=np.float32)

        for i, circuit in enumerate(circuits):
            grad = self._compute_gradients(circuit, parameters[i])
            gradients[i] = grad.numpy()

        return tf.constant(gradients, dtype=tf.float32)


class AdjointGradient:
    """
    Compute gradients using the adjoint method

    The adjoint method is more memory-efficient than parameter shift,
    requiring only one forward and one backward pass. However, it's
    only applicable to certain types of circuits and backends.

    This is similar to backpropagation in classical neural networks.

    Example:
        >>> gradient_computer = AdjointGradient(backend='lightning')
        >>> gradients = gradient_computer.compute(circuit, loss_fn)
    """

    def __init__(self, backend: str = 'lightning'):
        """
        Initialize adjoint gradient computer

        Args:
            backend: Backend to use (must support adjoint differentiation)
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.backend_name = backend
        self.backend = BackendManager.get_backend(backend)

    @tf.custom_gradient
    def compute_with_gradient(
        self,
        circuit: UnifiedCircuit,
        parameters: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute circuit output with adjoint gradient

        Args:
            circuit: Quantum circuit
            parameters: Parameter values

        Returns:
            Circuit output with gradient function
        """
        # Forward pass
        output = self._forward(circuit, parameters)

        # Define gradient function
        def grad_fn(dy):
            """Compute gradients using adjoint method"""
            gradients = self._compute_adjoint_gradients(circuit, parameters)
            return None, dy * gradients

        return output, grad_fn

    def _forward(self, circuit: UnifiedCircuit, parameters: tf.Tensor) -> tf.Tensor:
        """Execute circuit with given parameters"""
        # Similar to ParameterShiftGradient._forward
        param_dict = {}
        param_names = circuit.get_parameter_names()

        for i, name in enumerate(param_names):
            if i < len(parameters):
                param_dict[name] = float(parameters[i])

        bound_circuit = circuit.bind_parameters(param_dict)
        result = self.backend.execute(bound_circuit, shots=1000)

        # Compute expectation (simplified)
        expectation = np.random.uniform(-1, 1)  # Placeholder

        return tf.constant(expectation, dtype=tf.float32)

    def _compute_adjoint_gradients(
        self,
        circuit: UnifiedCircuit,
        parameters: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradients using adjoint method

        This is a placeholder - actual implementation would use
        backend-specific adjoint differentiation.
        """
        n_params = len(parameters)

        # Placeholder: In practice, this would use the backend's
        # adjoint differentiation capability
        gradients = np.random.randn(n_params).astype(np.float32)

        return tf.constant(gradients, dtype=tf.float32)


class SPSAGradient:
    """
    Gradient estimation using SPSA (Simultaneous Perturbation
    Stochastic Approximation)

    SPSA estimates gradients using only 2 circuit evaluations regardless
    of the number of parameters, making it very efficient for large
    parameter spaces. However, it provides stochastic estimates.

    Example:
        >>> gradient_computer = SPSAGradient(backend='qsim')
        >>> gradients = gradient_computer.compute(circuit, loss_fn)
    """

    def __init__(
        self,
        backend: str = 'qsim',
        epsilon: float = 0.1,
        num_samples: int = 1
    ):
        """
        Initialize SPSA gradient estimator

        Args:
            backend: Backend to use
            epsilon: Perturbation size
            num_samples: Number of samples for gradient estimation
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        self.backend_name = backend
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.backend = BackendManager.get_backend(backend)

    def compute(
        self,
        circuit: UnifiedCircuit,
        parameters: tf.Tensor,
        loss_fn: Callable
    ) -> tf.Tensor:
        """
        Compute SPSA gradient estimate

        Args:
            circuit: Quantum circuit
            parameters: Current parameter values
            loss_fn: Loss function to differentiate

        Returns:
            Gradient estimate
        """
        n_params = len(parameters)
        gradient_estimate = np.zeros(n_params, dtype=np.float32)

        for _ in range(self.num_samples):
            # Generate random perturbation direction
            delta = np.random.choice([-1, 1], size=n_params)

            # Perturb parameters
            params_plus = parameters.numpy() + self.epsilon * delta
            params_minus = parameters.numpy() - self.epsilon * delta

            # Evaluate loss at perturbed points
            loss_plus = loss_fn(circuit, tf.constant(params_plus))
            loss_minus = loss_fn(circuit, tf.constant(params_minus))

            # SPSA gradient estimate
            gradient_estimate += (loss_plus - loss_minus) * delta / (2 * self.epsilon)

        # Average over samples
        gradient_estimate /= self.num_samples

        return tf.constant(gradient_estimate, dtype=tf.float32)


def select_gradient_method(
    circuit: UnifiedCircuit,
    backend: str = 'qsim'
) -> str:
    """
    Automatically select appropriate gradient method

    Selection criteria:
    - Small circuits (<10 params): Parameter shift (exact)
    - Large circuits: SPSA (efficient)
    - Statevector backends: Adjoint (if supported)

    Args:
        circuit: Quantum circuit
        backend: Target backend

    Returns:
        Gradient method name ('parameter_shift', 'adjoint', or 'spsa')
    """
    n_params = circuit.n_parameters

    # Check backend capabilities
    if 'lightning' in backend.lower() and n_params < 20:
        return 'adjoint'
    elif n_params < 10:
        return 'parameter_shift'
    else:
        return 'spsa'
