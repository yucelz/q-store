"""
SPSA Gradient Estimation for Quantum Circuits - PyTorch

Quantum circuits are not differentiable in the classical sense.
We need special methods to estimate gradients.

Methods:
1. SPSA (Simultaneous Perturbation Stochastic Approximation)
   - Fast: Only 2 forward passes
   - Unbiased gradient estimate
   - Works for any circuit

2. Parameter Shift Rule
   - Exact gradients for certain gates
   - Requires 2n forward passes
   - More accurate but slower

3. Finite Difference
   - Simple but noisy
   - Good for debugging
"""

import torch
import numpy as np
from typing import Callable


def spsa_gradient(
    forward_fn: Callable,
    params: torch.Tensor,
    inputs: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    SPSA gradient estimation.
    
    Algorithm:
    1. Sample random perturbation δ ~ Bernoulli({-1, +1})
    2. Compute f(θ + εδ) and f(θ - εδ)
    3. Gradient ≈ [f(θ+εδ) - f(θ-εδ)] / (2ε) * 1/δ
    
    Advantages:
    - Only 2 forward passes (vs 2n for parameter shift)
    - Unbiased estimate
    - Efficient for high-dimensional parameters
    
    Parameters
    ----------
    forward_fn : callable
        Forward function: (inputs, params) -> outputs
    params : torch.Tensor
        Current parameters (shape: [n_params])
    inputs : torch.Tensor
        Input data (shape: [batch_size, ...])
    epsilon : float, default=0.01
        Perturbation size
    
    Returns
    -------
    gradients : torch.Tensor
        Estimated gradients (shape: [n_params])
    
    References
    ----------
    Spall, J. C. (1992). Multivariate stochastic approximation using a 
    simultaneous perturbation gradient approximation. IEEE TAC.
    """
    # Random perturbation direction (Bernoulli)
    delta = torch.rand_like(params)
    delta = torch.where(delta < 0.5, -1.0, 1.0)
    
    # Perturbed parameters
    params_plus = params + epsilon * delta
    params_minus = params - epsilon * delta
    
    # Forward passes
    with torch.no_grad():
        output_plus = forward_fn(inputs, params_plus)
        output_minus = forward_fn(inputs, params_minus)
    
    # Finite difference
    output_diff = output_plus - output_minus
    
    # SPSA gradient estimate
    # g = [f(θ+εδ) - f(θ-εδ)] / (2ε) * 1/δ
    grad = output_diff / (2 * epsilon * delta)
    
    # Average over batch and output dimensions
    grad = grad.mean(dim=0)
    
    return grad


def parameter_shift_gradient(
    forward_fn: Callable,
    params: torch.Tensor,
    inputs: torch.Tensor,
    shift: float = np.pi/2,
) -> torch.Tensor:
    """
    Parameter shift rule for quantum gradients.
    
    For gates U(θ) = exp(-iθG) with G² = I:
    ∂⟨ψ|U†(θ)OU(θ)|ψ⟩/∂θ = [⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}] / 2
    
    Requires 2n forward passes (n = number of parameters).
    More accurate than SPSA but slower.
    
    Parameters
    ----------
    forward_fn : callable
        Forward function: (inputs, params) -> outputs
    params : torch.Tensor
        Current parameters (shape: [n_params])
    inputs : torch.Tensor
        Input data
    shift : float, default=π/2
        Parameter shift amount
    
    Returns
    -------
    gradients : torch.Tensor
        Exact gradients (shape: [n_params])
    
    References
    ----------
    Schuld et al. (2019). Evaluating analytic gradients on quantum hardware.
    Physical Review A.
    """
    n_params = params.shape[0]
    gradients = []
    
    with torch.no_grad():
        for i in range(n_params):
            # Shift parameter i
            params_plus = params.clone()
            params_plus[i] += shift
            
            params_minus = params.clone()
            params_minus[i] -= shift
            
            # Forward passes
            output_plus = forward_fn(inputs, params_plus)
            output_minus = forward_fn(inputs, params_minus)
            
            # Gradient for parameter i
            grad_i = (output_plus - output_minus) / 2
            grad_i = grad_i.mean()  # Average over batch
            
            gradients.append(grad_i)
    
    return torch.stack(gradients)


def finite_difference_gradient(
    forward_fn: Callable,
    params: torch.Tensor,
    inputs: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Simple finite difference gradient (for debugging).
    
    ∂f/∂θᵢ ≈ [f(θ + εeᵢ) - f(θ)] / ε
    
    Parameters
    ----------
    forward_fn : callable
        Forward function
    params : torch.Tensor
        Parameters
    inputs : torch.Tensor
        Input data
    epsilon : float, default=1e-5
        Finite difference step
    
    Returns
    -------
    gradients : torch.Tensor
        Approximate gradients
    """
    n_params = params.shape[0]
    gradients = []
    
    with torch.no_grad():
        # Base output
        output_base = forward_fn(inputs, params)
        
        for i in range(n_params):
            # Perturb parameter i
            params_pert = params.clone()
            params_pert[i] += epsilon
            
            # Forward pass
            output_pert = forward_fn(inputs, params_pert)
            
            # Gradient
            grad_i = (output_pert - output_base) / epsilon
            grad_i = grad_i.mean()
            
            gradients.append(grad_i)
    
    return torch.stack(gradients)
