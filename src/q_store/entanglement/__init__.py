"""
Quantum entanglement measures, witnesses, and verification protocols.

This module provides tools for quantifying and detecting quantum entanglement,
including:
- Entanglement measures (concurrence, negativity, entropy)
- Entanglement witnesses and Bell inequality tests
- Verification protocols for entangled states
"""

from .measures import (
    concurrence,
    negativity,
    entropy_of_entanglement,
    entanglement_of_formation,
    EntanglementMeasure
)

from .witnesses import (
    bell_inequality_test,
    witness_operator,
    ppt_criterion,
    ccnr_criterion,
    EntanglementWitness
)

from .verification import (
    verify_bell_state,
    verify_ghz_state,
    verify_w_state,
    state_fidelity,
    EntanglementVerifier
)

__all__ = [
    # Entanglement measures
    'concurrence',
    'negativity',
    'entropy_of_entanglement',
    'entanglement_of_formation',
    'EntanglementMeasure',

    # Entanglement witnesses
    'bell_inequality_test',
    'witness_operator',
    'ppt_criterion',
    'ccnr_criterion',
    'EntanglementWitness',

    # Verification protocols
    'verify_bell_state',
    'verify_ghz_state',
    'verify_w_state',
    'state_fidelity',
    'EntanglementVerifier'
]
