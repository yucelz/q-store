"""
PyTorch integration for Q-Store v4.0 + v4.1.

This module provides PyTorch nn.Module compatible quantum layers that integrate
seamlessly with PyTorch's training infrastructure, autograd system, and distributed
training capabilities.

v4.1 Additions:
- Quantum-first layers (QuantumLinear for nn.Linear replacement)
- Async execution support
- SPSA gradient estimation
- Storage integration

Key Components:
    - QuantumLayer: PyTorch nn.Module for parameterized quantum circuits (v4.0 + v4.1)
    - QuantumLinear: Quantum replacement for nn.Linear (v4.1)
    - AmplitudeEncoding: Encode classical data as quantum amplitudes
    - AngleEncoding: Encode classical data as rotation angles
    - PyTorchCircuitExecutor: Execute quantum circuits within PyTorch computation graph
    - QuantumExecution: Custom autograd.Function for gradient computation

Example:
    >>> import torch
    >>> from q_store.torch import QuantumLinear
    >>>
    >>> # v4.1: Quantum-first architecture (70% quantum)
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Flatten(),
    ...     QuantumLinear(n_qubits=7),  # Replaces nn.Linear(in, 128)
    ...     torch.nn.Linear(21, 10),
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
    # v4.0 components
    from .layers import (
        QuantumLayer as QuantumLayerV4,
        AmplitudeEncoding,
        AngleEncoding,
    )
    from .circuit_executor import PyTorchCircuitExecutor
    from .gradients import (
        QuantumExecution,
        ParameterShiftGradient,
        AdjointGradient,
    )

    # v4.1 components (new)
    try:
        from .quantum_layer import QuantumLayer
        from .quantum_linear import QuantumLinear
        from .spsa_gradients import spsa_gradient
        HAS_V4_1 = True
    except ImportError:
        HAS_V4_1 = False
        QuantumLayer = None
        QuantumLinear = None
        spsa_gradient = None

    __all__ = [
        # v4.0
        'QuantumLayerV4',
        'AmplitudeEncoding',
        'AngleEncoding',
        'PyTorchCircuitExecutor',
        'QuantumExecution',
        'ParameterShiftGradient',
        'AdjointGradient',
        # v4.1
        'QuantumLayer',
        'QuantumLinear',
        'spsa_gradient',
    ]
else:
    __all__ = []

    def __getattr__(name):
        raise ImportError(
            f"PyTorch is required to use q_store.torch module. "
            f"Install it with: pip install torch"
        )
