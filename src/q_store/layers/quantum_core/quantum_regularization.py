"""
Quantum Regularization - v4.1 Enhanced
Regularization techniques for quantum neural networks

Key Innovation: Quantum-specific regularization to prevent overfitting
- Quantum dropout (qubit, basis, gate dropout)
- Entanglement sparsification
- Training vs inference mode switching
- Measurement basis subsampling

Motivation:
- Quantum models can overfit on small datasets
- Over-entanglement leads to barren plateaus
- Too many measurements waste resources

Design:
- Training-only regularization (disabled during inference)
- Probabilistic dropout mechanisms
- Entanglement capacity control
- Compatible with existing quantum layers
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantumCircuit:
    """
    Placeholder for QuantumCircuit class.

    In production, this would be imported from core.
    Here we define minimal interface for regularization.
    """
    n_qubits: int
    gates: List
    measurement_bases: List[str]
    metadata: dict

    def copy(self):
        """Create a copy of the circuit."""
        import copy
        return copy.deepcopy(self)


class QuantumDropout:
    """
    Quantum dropout for regularization.

    Randomly suppresses circuit elements during training to prevent overfitting.

    Three types of dropout:
    1. **Qubit dropout**: Randomly exclude qubits from computation
    2. **Basis dropout**: Randomly skip measurement bases
    3. **Gate dropout**: Randomly skip entangling gates

    Parameters
    ----------
    qubit_dropout_rate : float, default=0.1
        Probability of dropping a qubit (0.0 = no dropout, 1.0 = drop all)
    basis_dropout_rate : float, default=0.2
        Probability of dropping a measurement basis
    gate_dropout_rate : float, default=0.1
        Probability of dropping an entangling gate

    Examples
    --------
    >>> dropout = QuantumDropout(
    ...     qubit_dropout_rate=0.1,
    ...     basis_dropout_rate=0.2,
    ...     gate_dropout_rate=0.1
    ... )
    >>>
    >>> # During training
    >>> regularized_circuit = dropout.apply_dropout(circuit, training=True)
    >>>
    >>> # During inference (no dropout)
    >>> inference_circuit = dropout.apply_dropout(circuit, training=False)
    >>> assert inference_circuit == circuit  # No changes
    """

    def __init__(
        self,
        qubit_dropout_rate: float = 0.1,
        basis_dropout_rate: float = 0.2,
        gate_dropout_rate: float = 0.1
    ):
        if not (0.0 <= qubit_dropout_rate <= 1.0):
            raise ValueError(f"qubit_dropout_rate must be in [0, 1], got {qubit_dropout_rate}")
        if not (0.0 <= basis_dropout_rate <= 1.0):
            raise ValueError(f"basis_dropout_rate must be in [0, 1], got {basis_dropout_rate}")
        if not (0.0 <= gate_dropout_rate <= 1.0):
            raise ValueError(f"gate_dropout_rate must be in [0, 1], got {gate_dropout_rate}")

        self.qubit_dropout_rate = qubit_dropout_rate
        self.basis_dropout_rate = basis_dropout_rate
        self.gate_dropout_rate = gate_dropout_rate

        logger.info(
            f"Initialized QuantumDropout: "
            f"qubit={qubit_dropout_rate}, basis={basis_dropout_rate}, "
            f"gate={gate_dropout_rate}"
        )

    def apply_dropout(
        self,
        circuit,
        training: bool = True
    ):
        """
        Apply quantum dropout to circuit.

        Only active during training! Returns original circuit during inference.

        Parameters
        ----------
        circuit : QuantumCircuit
            Input circuit
        training : bool, default=True
            Whether in training mode

        Returns
        -------
        QuantumCircuit
            Regularized circuit (or original if not training)
        """
        if not training:
            # No dropout during inference
            return circuit

        # Create modified circuit
        dropped_circuit = circuit.copy()

        # 1. Qubit dropout
        active_qubits, dropped_qubits = self._apply_qubit_dropout(circuit.n_qubits)

        # 2. Filter gates (remove gates operating on dropped qubits + gate dropout)
        dropped_circuit.gates = self._apply_gate_dropout(
            circuit.gates,
            dropped_qubits
        )

        # 3. Basis dropout
        dropped_circuit.measurement_bases = self._apply_basis_dropout(
            circuit.measurement_bases
        )

        # Update metadata
        dropped_circuit.metadata = circuit.metadata.copy()
        dropped_circuit.metadata['quantum_dropout_applied'] = True
        dropped_circuit.metadata['qubits_dropped'] = len(dropped_qubits)
        dropped_circuit.metadata['gates_before'] = len(circuit.gates)
        dropped_circuit.metadata['gates_after'] = len(dropped_circuit.gates)

        logger.debug(
            f"Applied quantum dropout: "
            f"qubits: {circuit.n_qubits} → {len(active_qubits)}, "
            f"gates: {len(circuit.gates)} → {len(dropped_circuit.gates)}, "
            f"bases: {len(circuit.measurement_bases)} → {len(dropped_circuit.measurement_bases)}"
        )

        return dropped_circuit

    def _apply_qubit_dropout(self, n_qubits: int) -> tuple:
        """
        Apply qubit dropout.

        Returns
        -------
        tuple
            (active_qubits: set, dropped_qubits: set)
        """
        active_qubits = set(range(n_qubits))
        n_drop = int(n_qubits * self.qubit_dropout_rate)

        if n_drop > 0 and n_drop < n_qubits:  # Keep at least one qubit
            dropped_qubits = set(np.random.choice(
                list(active_qubits),
                size=n_drop,
                replace=False
            ))
            active_qubits -= dropped_qubits
        else:
            dropped_qubits = set()

        return active_qubits, dropped_qubits

    def _apply_gate_dropout(self, gates: List, dropped_qubits: Set[int]) -> List:
        """
        Apply gate dropout.

        Removes:
        1. Gates operating on dropped qubits
        2. Randomly dropped entangling gates
        """
        kept_gates = []

        for gate in gates:
            # Get qubits this gate operates on
            gate_qubits = self._get_gate_qubits(gate)

            # Skip if operates on dropped qubit
            if any(q in dropped_qubits for q in gate_qubits):
                continue

            # Randomly skip entangling gates
            if self._is_two_qubit_gate(gate):
                if np.random.random() < self.gate_dropout_rate:
                    continue  # Drop this gate

            kept_gates.append(gate)

        return kept_gates

    def _apply_basis_dropout(self, measurement_bases: List[str]) -> List[str]:
        """
        Apply basis dropout.

        Randomly removes measurement bases while keeping at least one.
        """
        if len(measurement_bases) <= 1:
            # Keep at least one basis
            return measurement_bases

        n_bases_drop = int(len(measurement_bases) * self.basis_dropout_rate)

        if n_bases_drop > 0 and n_bases_drop < len(measurement_bases):
            # Keep at least one basis
            n_keep = max(1, len(measurement_bases) - n_bases_drop)
            kept_bases = list(np.random.choice(
                measurement_bases,
                size=n_keep,
                replace=False
            ))
            return kept_bases
        else:
            return measurement_bases

    def _get_gate_qubits(self, gate) -> List[int]:
        """Extract qubits from gate."""
        if hasattr(gate, 'qubits'):
            return gate.qubits
        elif hasattr(gate, 'target'):
            return [gate.target]
        elif isinstance(gate, (list, tuple)) and len(gate) >= 2:
            # Assume format: (gate_type, qubits, ...)
            return gate[1] if isinstance(gate[1], (list, tuple)) else [gate[1]]
        else:
            return []

    def _is_two_qubit_gate(self, gate) -> bool:
        """Check if gate is a two-qubit (entangling) gate."""
        gate_qubits = self._get_gate_qubits(gate)
        return len(gate_qubits) >= 2


class QuantumRegularization:
    """
    Comprehensive quantum regularization.

    Combines multiple regularization techniques:
    - Quantum dropout
    - Entanglement sparsification
    - Measurement subsampling

    Parameters
    ----------
    dropout : QuantumDropout, optional
        Dropout configuration (default: 10% qubit, 20% basis, 10% gate)
    entanglement_penalty : float, default=0.0
        Penalty for excessive entanglement (0.0 = no penalty, 0.5 = remove 50% of entangling gates)
    measurement_penalty : float, default=0.0
        Penalty for excessive measurements (unused in v4.1, reserved for v4.2)

    Examples
    --------
    >>> regularizer = QuantumRegularization(
    ...     dropout=QuantumDropout(qubit_dropout_rate=0.1),
    ...     entanglement_penalty=0.2  # Remove 20% of entangling gates
    ... )
    >>>
    >>> # During training
    >>> reg_circuit = regularizer.regularize_circuit(circuit, training=True)
    >>>
    >>> # During inference
    >>> inf_circuit = regularizer.regularize_circuit(circuit, training=False)
    """

    def __init__(
        self,
        dropout: Optional[QuantumDropout] = None,
        entanglement_penalty: float = 0.0,
        measurement_penalty: float = 0.0
    ):
        self.dropout = dropout or QuantumDropout()
        self.entanglement_penalty = entanglement_penalty
        self.measurement_penalty = measurement_penalty

        logger.info(
            f"Initialized QuantumRegularization: "
            f"entanglement_penalty={entanglement_penalty}"
        )

    def regularize_circuit(
        self,
        circuit,
        training: bool = True
    ):
        """
        Apply all regularization techniques.

        Parameters
        ----------
        circuit : QuantumCircuit
            Input circuit
        training : bool, default=True
            Whether in training mode

        Returns
        -------
        QuantumCircuit
            Regularized circuit
        """
        if not training:
            # No regularization during inference
            return circuit

        # 1. Apply dropout
        regularized = self.dropout.apply_dropout(circuit, training=training)

        # 2. Entanglement sparsification
        if self.entanglement_penalty > 0:
            regularized = self._sparsify_entanglement(regularized)

        return regularized

    def _sparsify_entanglement(self, circuit):
        """
        Reduce entangling gates to prevent over-entanglement.

        Over-entanglement can lead to:
        - Barren plateaus (vanishing gradients)
        - Training instability
        - Overfitting to quantum noise

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to sparsify

        Returns
        -------
        QuantumCircuit
            Circuit with reduced entanglement
        """
        # Count current entangling gates
        entangling_gates = []
        single_qubit_gates = []

        for gate in circuit.gates:
            gate_qubits = self._get_gate_qubits(gate)
            if len(gate_qubits) >= 2:
                entangling_gates.append(gate)
            else:
                single_qubit_gates.append(gate)

        n_entangling = len(entangling_gates)

        # Target: reduce by penalty fraction
        target_entangling = int(n_entangling * (1 - self.entanglement_penalty))

        if n_entangling <= target_entangling:
            # Already sparse enough
            return circuit

        # Randomly sample gates to keep
        # (In v4.2, we could use parameter magnitudes for smarter selection)
        kept_indices = np.random.choice(
            n_entangling,
            size=target_entangling,
            replace=False
        )

        kept_entangling = [entangling_gates[i] for i in sorted(kept_indices)]

        # Reconstruct circuit
        sparse_circuit = circuit.copy()
        sparse_circuit.gates = single_qubit_gates + kept_entangling

        # Update metadata
        sparse_circuit.metadata = circuit.metadata.copy()
        sparse_circuit.metadata['entanglement_sparsified'] = True
        sparse_circuit.metadata['entangling_gates_before'] = n_entangling
        sparse_circuit.metadata['entangling_gates_after'] = target_entangling

        logger.debug(
            f"Sparsified entanglement: "
            f"{n_entangling} → {target_entangling} gates "
            f"({self.entanglement_penalty*100:.0f}% reduction)"
        )

        return sparse_circuit

    def _get_gate_qubits(self, gate) -> List[int]:
        """Extract qubits from gate (same as in QuantumDropout)."""
        if hasattr(gate, 'qubits'):
            return gate.qubits
        elif hasattr(gate, 'target'):
            return [gate.target]
        elif isinstance(gate, (list, tuple)) and len(gate) >= 2:
            return gate[1] if isinstance(gate[1], (list, tuple)) else [gate[1]]
        else:
            return []


# Utility functions

def apply_quantum_regularization(
    circuit,
    training: bool = True,
    qubit_dropout: float = 0.1,
    basis_dropout: float = 0.2,
    gate_dropout: float = 0.1,
    entanglement_penalty: float = 0.0
):
    """
    Convenience function to apply quantum regularization.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input circuit
    training : bool, default=True
        Training mode flag
    qubit_dropout : float, default=0.1
        Qubit dropout rate
    basis_dropout : float, default=0.2
        Basis dropout rate
    gate_dropout : float, default=0.1
        Gate dropout rate
    entanglement_penalty : float, default=0.0
        Entanglement sparsification penalty

    Returns
    -------
    QuantumCircuit
        Regularized circuit

    Examples
    --------
    >>> reg_circuit = apply_quantum_regularization(
    ...     circuit,
    ...     training=True,
    ...     qubit_dropout=0.1,
    ...     entanglement_penalty=0.2
    ... )
    """
    dropout = QuantumDropout(
        qubit_dropout_rate=qubit_dropout,
        basis_dropout_rate=basis_dropout,
        gate_dropout_rate=gate_dropout
    )

    regularizer = QuantumRegularization(
        dropout=dropout,
        entanglement_penalty=entanglement_penalty
    )

    return regularizer.regularize_circuit(circuit, training=training)
