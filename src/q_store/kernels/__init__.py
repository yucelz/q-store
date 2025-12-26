"""
Quantum kernel methods for machine learning.

Quantum kernels provide a way to compute similarity between data points
using quantum feature maps, enabling quantum-enhanced machine learning.
"""

from q_store.kernels.quantum_kernel import (
    QuantumKernel,
    compute_kernel_matrix,
    kernel_alignment,
    kernel_target_alignment
)
from q_store.kernels.fidelity_kernel import (
    FidelityQuantumKernel,
    state_fidelity,
    compute_fidelity_kernel
)
from q_store.kernels.projected_kernel import (
    ProjectedQuantumKernel,
    measurement_kernel,
    compute_projected_kernel
)
from q_store.kernels.trainable_kernel import (
    TrainableQuantumKernel,
    optimize_kernel_parameters,
    kernel_loss
)

__all__ = [
    # Quantum Kernel
    'QuantumKernel',
    'compute_kernel_matrix',
    'kernel_alignment',
    'kernel_target_alignment',
    # Fidelity Kernel
    'FidelityQuantumKernel',
    'state_fidelity',
    'compute_fidelity_kernel',
    # Projected Kernel
    'ProjectedQuantumKernel',
    'measurement_kernel',
    'compute_projected_kernel',
    # Trainable Kernel
    'TrainableQuantumKernel',
    'optimize_kernel_parameters',
    'kernel_loss',
]
