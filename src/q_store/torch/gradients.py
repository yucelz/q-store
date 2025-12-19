"""
Custom autograd functions for quantum circuit gradients in PyTorch.

This module provides custom autograd.Function implementations that enable
automatic differentiation for quantum circuits using parameter shift rule
and adjoint methods.
"""

from typing import Dict, Optional, Tuple, Any
import math

try:
    import torch
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from ..core import UnifiedCircuit
    from ..backends import BackendManager
    from .circuit_executor import PyTorchCircuitExecutor


class QuantumExecution(Function):
    """
    Custom autograd function for quantum circuit execution.

    This function implements the forward and backward passes for quantum
    circuit execution, using the parameter shift rule for gradient computation.

    The parameter shift rule states that for a gate with parameter θ:
    ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2

    Example:
        >>> circuit = UnifiedCircuit(n_qubits=4)
        >>> # Build circuit...
        >>> params = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        >>> result = QuantumExecution.apply(circuit, params, executor)
        >>> loss = result.sum()
        >>> loss.backward()  # Computes gradients via parameter shift
    """

    @staticmethod
    def forward(
        ctx: Any,
        circuit: UnifiedCircuit,
        parameters: torch.Tensor,
        executor: PyTorchCircuitExecutor,
        param_names: list[str],
    ) -> torch.Tensor:
        """
        Forward pass: execute quantum circuit.

        Args:
            ctx: Context for saving information for backward pass
            circuit: Quantum circuit to execute
            parameters: Tensor of parameter values
            executor: Circuit executor
            param_names: List of parameter names

        Returns:
            Measurement results
        """
        # Save for backward
        ctx.save_for_backward(parameters)
        ctx.circuit = circuit
        ctx.executor = executor
        ctx.param_names = param_names

        # Build parameter dictionary
        param_dict = {
            name: parameters[i]
            for i, name in enumerate(param_names)
        }

        # Execute circuit
        result = executor.execute(circuit, param_dict)

        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        """
        Backward pass: compute gradients using parameter shift rule.

        Args:
            ctx: Context with saved information from forward pass
            grad_output: Gradient of loss with respect to output

        Returns:
            Tuple of gradients (None for non-tensor inputs, gradient for parameters)
        """
        parameters, = ctx.saved_tensors
        circuit = ctx.circuit
        executor = ctx.executor
        param_names = ctx.param_names

        # Compute gradients for each parameter
        param_grads = []

        for i, param_name in enumerate(param_names):
            # Create shifted parameter dictionaries
            params_plus = {
                name: parameters[j] for j, name in enumerate(param_names)
            }
            params_minus = {
                name: parameters[j] for j, name in enumerate(param_names)
            }

            # Apply parameter shift
            shift = math.pi / 2
            params_plus[param_name] = parameters[i] + shift
            params_minus[param_name] = parameters[i] - shift

            # Execute circuits with shifted parameters
            result_plus = executor.execute(circuit, params_plus)
            result_minus = executor.execute(circuit, params_minus)

            # Compute gradient using parameter shift rule
            param_gradient = (result_plus - result_minus) / 2.0

            # Apply chain rule with grad_output
            grad = torch.sum(grad_output * param_gradient)
            param_grads.append(grad)

        # Stack gradients
        if param_grads:
            grads = torch.stack(param_grads)
        else:
            grads = torch.zeros_like(parameters)

        # Return gradients (None for circuit, executor, param_names)
        return None, grads, None, None


class ParameterShiftGradient:
    """
    Parameter shift gradient computation for quantum circuits.

    This class provides explicit parameter shift gradient computation
    with support for batched execution.

    Args:
        executor: Circuit executor for running quantum circuits

    Example:
        >>> gradient = ParameterShiftGradient(executor)
        >>> circuit = UnifiedCircuit(n_qubits=4)
        >>> params = {'theta_0': torch.tensor(0.5), 'theta_1': torch.tensor(0.3)}
        >>> result, grads = gradient.compute_with_gradient(circuit, params)
    """

    def __init__(self, executor: PyTorchCircuitExecutor):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ParameterShiftGradient")

        self.executor = executor

    def compute_with_gradient(
        self,
        circuit: UnifiedCircuit,
        parameters: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute circuit output and gradients using parameter shift rule.

        Args:
            circuit: Circuit to execute
            parameters: Dictionary of parameter values

        Returns:
            Tuple of (output, gradients) where gradients is a dict
        """
        # Forward pass
        output = self.executor.execute(circuit, parameters)

        # Compute gradients
        gradients = {}
        shift = math.pi / 2

        for param_name, param_value in parameters.items():
            # Create shifted parameters
            params_plus = parameters.copy()
            params_minus = parameters.copy()

            params_plus[param_name] = param_value + shift
            params_minus[param_name] = param_value - shift

            # Execute with shifted parameters
            result_plus = self.executor.execute(circuit, params_plus)
            result_minus = self.executor.execute(circuit, params_minus)

            # Compute gradient
            gradient = (result_plus - result_minus) / 2.0
            gradients[param_name] = gradient

        return output, gradients

    def compute_batch(
        self,
        circuits: list[UnifiedCircuit],
        parameters_list: list[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, list[Dict[str, torch.Tensor]]]:
        """
        Compute outputs and gradients for a batch of circuits.

        Args:
            circuits: List of circuits to execute
            parameters_list: List of parameter dictionaries

        Returns:
            Tuple of (batched outputs, list of gradient dicts)
        """
        outputs = []
        gradients_list = []

        for circuit, parameters in zip(circuits, parameters_list):
            output, gradients = self.compute_with_gradient(circuit, parameters)
            outputs.append(output)
            gradients_list.append(gradients)

        batched_outputs = torch.stack(outputs)

        return batched_outputs, gradients_list


class AdjointGradient:
    """
    Adjoint method for efficient gradient computation.

    The adjoint method computes gradients more efficiently than parameter shift
    by utilizing the quantum state during circuit execution. This is particularly
    efficient for state vector simulators.

    Args:
        executor: Circuit executor for running quantum circuits

    Note:
        This requires backend support for adjoint differentiation.
        Falls back to parameter shift if not available.
    """

    def __init__(self, executor: PyTorchCircuitExecutor):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AdjointGradient")

        self.executor = executor
        self._fallback_gradient = ParameterShiftGradient(executor)

    def compute_with_gradient(
        self,
        circuit: UnifiedCircuit,
        parameters: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute circuit output and gradients using adjoint method.

        Args:
            circuit: Circuit to execute
            parameters: Dictionary of parameter values

        Returns:
            Tuple of (output, gradients)
        """
        # Check if backend supports adjoint differentiation
        backend = self.executor.backend

        if hasattr(backend, 'adjoint_gradient'):
            # Use backend's adjoint gradient
            try:
                output = self.executor.execute(circuit, parameters)
                gradients = backend.adjoint_gradient(circuit, parameters)

                # Convert gradients to tensors if needed
                tensor_gradients = {}
                for key, value in gradients.items():
                    if not isinstance(value, torch.Tensor):
                        tensor_gradients[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        tensor_gradients[key] = value

                return output, tensor_gradients
            except Exception:
                # Fall back to parameter shift
                pass

        # Fall back to parameter shift rule
        return self._fallback_gradient.compute_with_gradient(circuit, parameters)


def select_gradient_method(
    circuit: UnifiedCircuit,
    backend_name: str,
    prefer_adjoint: bool = True,
) -> str:
    """
    Select the best gradient method for a given circuit and backend.

    Args:
        circuit: The quantum circuit
        backend_name: Name of the backend
        prefer_adjoint: Whether to prefer adjoint method when available

    Returns:
        Gradient method name ('parameter_shift' or 'adjoint')
    """
    # Check circuit size
    n_params = len([g for g in circuit.gates if g.parameters])

    # For small circuits, parameter shift is fine
    if n_params <= 10:
        return 'parameter_shift'

    # For state vector simulators, adjoint is more efficient
    if prefer_adjoint and 'simulator' in backend_name.lower():
        return 'adjoint'

    # Default to parameter shift
    return 'parameter_shift'


# Export only if PyTorch is available
if not HAS_TORCH:
    QuantumExecution = None
    ParameterShiftGradient = None
    AdjointGradient = None
    select_gradient_method = None
