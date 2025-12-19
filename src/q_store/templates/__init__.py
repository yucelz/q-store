"""
Pre-built quantum circuit templates.

This module provides ready-to-use implementations of common quantum algorithms:
- Quantum Fourier Transform (QFT)
- Grover's Algorithm
- Quantum Phase Estimation (QPE)
- Amplitude Amplification
"""

from q_store.templates.qft import qft, inverse_qft, qft_rotations
from q_store.templates.grover import (
    grover_circuit,
    grover_diffusion,
    create_oracle_from_bitstring,
    grover_search
)
from q_store.templates.phase_estimation import (
    phase_estimation,
    iterative_phase_estimation,
    create_phase_estimation_unitary,
    phase_kickback_circuit
)
from q_store.templates.amplitude_amplification import (
    amplitude_amplification,
    fixed_point_amplification,
    oblivious_amplitude_amplification
)

__all__ = [
    # QFT
    'qft',
    'inverse_qft',
    'qft_rotations',
    # Grover
    'grover_circuit',
    'grover_diffusion',
    'create_oracle_from_bitstring',
    'grover_search',
    # Phase Estimation
    'phase_estimation',
    'iterative_phase_estimation',
    'create_phase_estimation_unitary',
    'phase_kickback_circuit',
    # Amplitude Amplification
    'amplitude_amplification',
    'fixed_point_amplification',
    'oblivious_amplitude_amplification',
]
