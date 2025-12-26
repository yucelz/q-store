"""
PyTorch integration for Q-Store v4.0.

This module provides PyTorch nn.Module compatible quantum layers that integrate
seamlessly with PyTorch's training infrastructure, autograd system, and distributed
training capabilities.

Key Components:
    - QuantumLayer: PyTorch nn.Module for parameterized quantum circuits
    - AmplitudeEncoding: Encode classical data as quantum amplitudes
    - AngleEncoding: Encode classical data as rotation angles
    - PyTorchCircuitExecutor: Execute quantum circuits within PyTorch computation graph
    - QuantumExecution: Custom autograd.Function for gradient computation

Example:
    >>> import torch
    >>> from q_store.torch import QuantumLayer, AngleEncoding
    >>>
    >>> # Create a quantum model
    >>> model = torch.nn.Sequential(
    ...     AngleEncoding(n_qubits=4),
    ...     QuantumLayer(n_qubits=4, depth=2),
    ...     torch.nn.Linear(4, 2)
    ... )
    >>>
    >>> # Train with PyTorch
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> for x, y in dataloader:
    ...     optimizer.zero_grad()
    ...     output = model(x)
    ...     loss = torch.nn.functional.mse_loss(output, y)
    ...     loss.backward()
    ...     optimizer.step()
"""

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .layers import (
        QuantumLayer,
        AmplitudeEncoding,
        AngleEncoding,
    )
    from .circuit_executor import PyTorchCircuitExecutor
    from .gradients import (
        QuantumExecution,
        ParameterShiftGradient,
        AdjointGradient,
    )

    __all__ = [
        'QuantumLayer',
        'AmplitudeEncoding',
        'AngleEncoding',
        'PyTorchCircuitExecutor',
        'QuantumExecution',
        'ParameterShiftGradient',
        'AdjointGradient',
    ]
else:
    __all__ = []

    def __getattr__(name):
        raise ImportError(
            f"PyTorch is required to use q_store.torch module. "
            f"Install it with: pip install torch"
        )
