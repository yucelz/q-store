"""
Measurement Error Mitigation.

Corrects readout errors using:
1. Calibration matrices from state preparation experiments
2. Matrix inversion or least-squares fitting
3. Tensored or correlated noise models

Reference:
- Nachman et al., "Unfolding quantum computer readout noise"
  npj Quantum Information 6, 84 (2020)
- Nation et al., Qiskit Textbook
"""

from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from scipy.linalg import lstsq, inv

from ..core import UnifiedCircuit, GateType

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Calibration data for measurement error mitigation."""
    confusion_matrix: np.ndarray  # P(measure j | prepared i)
    mitigation_matrix: np.ndarray  # Inverse or pseudo-inverse
    n_qubits: int
    condition_number: float
    tensored: bool  # True if single-qubit, False if full correlated


@dataclass
class MitigationResult:
    """Result from measurement error mitigation."""
    mitigated_probabilities: np.ndarray
    raw_probabilities: np.ndarray
    total_variation_distance: float
    condition_number: float
    metadata: Dict[str, Any]


class MeasurementErrorMitigator:
    """
    Measurement error mitigation via calibration matrices.

    Workflow:
    1. Calibrate: prepare all computational basis states, measure
    2. Build confusion matrix M[i,j] = P(measure j | prepared i)
    3. Mitigate: solve M @ x = measured for ideal distribution x

    Args:
        n_qubits: Number of qubits
        tensored: Use tensor product of single-qubit calibrations

    Example:
        >>> mitigator = MeasurementErrorMitigator(n_qubits=3)
        >>> mitigator.calibrate(backend)
        >>> result = mitigator.mitigate(measured_counts)
    """

    def __init__(
        self,
        n_qubits: int,
        tensored: bool = True,
        shots_per_calibration: int = 10000
    ):
        self.n_qubits = n_qubits
        self.tensored = tensored
        self.shots_per_calibration = shots_per_calibration
        self.calibration_data: Optional[CalibrationData] = None

    def calibrate_single_qubit(
        self,
        qubit_index: int,
        executor: Callable[[UnifiedCircuit], Dict[str, int]]
    ) -> np.ndarray:
        """
        Calibrate single qubit readout.

        Prepares |0⟩ and |1⟩, measures to get confusion matrix:
        M = [[P(0|0), P(1|0)],
             [P(0|1), P(1|1)]]

        Returns:
            2x2 confusion matrix
        """
        confusion = np.zeros((2, 2))

        # Calibrate |0⟩ state (no gates needed)
        circuit_0 = UnifiedCircuit(n_qubits=self.n_qubits)
        counts_0 = executor(circuit_0)

        # Extract counts for this qubit
        total_0 = sum(counts_0.values())
        prob_0_given_0 = sum(
            count for bitstring, count in counts_0.items()
            if len(bitstring) > qubit_index and bitstring[-(qubit_index+1)] == '0'
        ) / total_0
        confusion[0, 0] = prob_0_given_0
        confusion[1, 0] = 1 - prob_0_given_0

        # Calibrate |1⟩ state (X gate)
        circuit_1 = UnifiedCircuit(n_qubits=self.n_qubits)
        circuit_1.add_gate(GateType.X, targets=[qubit_index])
        counts_1 = executor(circuit_1)

        total_1 = sum(counts_1.values())
        prob_0_given_1 = sum(
            count for bitstring, count in counts_1.items()
            if len(bitstring) > qubit_index and bitstring[-(qubit_index+1)] == '0'
        ) / total_1
        confusion[0, 1] = prob_0_given_1
        confusion[1, 1] = 1 - prob_0_given_1

        return confusion

    def calibrate_tensored(
        self,
        executor: Callable[[UnifiedCircuit], Dict[str, int]]
    ) -> CalibrationData:
        """
        Calibrate using tensor product of single-qubit matrices.

        Efficient: only 2n circuits needed vs 2^n for full calibration.
        Assumes uncorrelated readout errors.
        """
        single_qubit_matrices = []

        logger.info(f"Calibrating {self.n_qubits} qubits (tensored mode)")

        for q in range(self.n_qubits):
            confusion = self.calibrate_single_qubit(q, executor)
            single_qubit_matrices.append(confusion)
            logger.debug(f"Qubit {q} confusion matrix:\n{confusion}")

        # Tensor product of single-qubit matrices
        full_confusion = single_qubit_matrices[0]
        for q in range(1, self.n_qubits):
            full_confusion = np.kron(full_confusion, single_qubit_matrices[q])

        # Compute mitigation matrix (pseudo-inverse for stability)
        try:
            mitigation = inv(full_confusion)
            condition_number = np.linalg.cond(full_confusion)
        except np.linalg.LinAlgError:
            logger.warning("Confusion matrix singular, using pseudo-inverse")
            mitigation = np.linalg.pinv(full_confusion)
            condition_number = float('inf')

        if condition_number > 100:
            logger.warning(
                f"High condition number {condition_number:.2f} - "
                "mitigation may amplify statistical noise"
            )

        return CalibrationData(
            confusion_matrix=full_confusion,
            mitigation_matrix=mitigation,
            n_qubits=self.n_qubits,
            condition_number=condition_number,
            tensored=True
        )

    def calibrate_full(
        self,
        executor: Callable[[UnifiedCircuit], Dict[str, int]]
    ) -> CalibrationData:
        """
        Full calibration measuring all 2^n basis states.

        More accurate but exponentially expensive.
        Captures correlated readout errors.
        """
        dim = 2 ** self.n_qubits
        confusion = np.zeros((dim, dim))

        logger.info(f"Calibrating {self.n_qubits} qubits (full mode, {dim} states)")

        for prepared_state in range(dim):
            # Prepare computational basis state
            circuit = UnifiedCircuit(n_qubits=self.n_qubits)

            # Apply X gates for |1⟩ components
            for q in range(self.n_qubits):
                if (prepared_state >> q) & 1:
                    circuit.add_gate(GateType.X, targets=[q])

            # Execute and collect counts
            counts = executor(circuit)
            total = sum(counts.values())

            # Fill confusion matrix column
            for measured_str, count in counts.items():
                measured_state = int(measured_str, 2) if measured_str else 0
                confusion[measured_state, prepared_state] = count / total

        # Compute mitigation matrix
        try:
            mitigation = inv(confusion)
            condition_number = np.linalg.cond(confusion)
        except np.linalg.LinAlgError:
            logger.warning("Confusion matrix singular, using pseudo-inverse")
            mitigation = np.linalg.pinv(confusion)
            condition_number = float('inf')

        if condition_number > 100:
            logger.warning(
                f"High condition number {condition_number:.2f} - "
                "mitigation may amplify statistical noise"
            )

        return CalibrationData(
            confusion_matrix=confusion,
            mitigation_matrix=mitigation,
            n_qubits=self.n_qubits,
            condition_number=condition_number,
            tensored=False
        )

    def calibrate(
        self,
        executor: Callable[[UnifiedCircuit], Dict[str, int]]
    ) -> CalibrationData:
        """
        Run calibration procedure.

        Args:
            executor: Function that executes circuit and returns measurement counts

        Returns:
            Calibration data with confusion and mitigation matrices
        """
        if self.tensored:
            self.calibration_data = self.calibrate_tensored(executor)
        else:
            self.calibration_data = self.calibrate_full(executor)

        return self.calibration_data

    def _counts_to_probabilities(
        self,
        counts: Dict[str, int]
    ) -> np.ndarray:
        """Convert measurement counts to probability vector."""
        dim = 2 ** self.n_qubits
        probs = np.zeros(dim)
        total = sum(counts.values())

        for bitstring, count in counts.items():
            state = int(bitstring, 2) if bitstring else 0
            probs[state] = count / total

        return probs

    def _probabilities_to_counts(
        self,
        probabilities: np.ndarray,
        shots: int
    ) -> Dict[str, int]:
        """Convert probability vector to measurement counts."""
        counts = {}

        for state, prob in enumerate(probabilities):
            if prob > 1e-10:  # Skip negligible probabilities
                bitstring = format(state, f'0{self.n_qubits}b')
                counts[bitstring] = int(prob * shots)

        return counts

    def mitigate_counts(
        self,
        measured_counts: Dict[str, int],
        method: str = 'least_squares'
    ) -> MitigationResult:
        """
        Mitigate measurement errors in counts.

        Args:
            measured_counts: Raw measurement counts from backend
            method: 'inversion' or 'least_squares'

        Returns:
            MitigationResult with corrected probabilities
        """
        if self.calibration_data is None:
            raise ValueError("Must calibrate before mitigation")

        # Convert counts to probabilities
        measured_probs = self._counts_to_probabilities(measured_counts)

        # Apply mitigation
        if method == 'inversion':
            # Direct matrix inversion: x = M^(-1) @ measured
            mitigated_probs = self.calibration_data.mitigation_matrix @ measured_probs
        elif method == 'least_squares':
            # Least squares: minimize ||M @ x - measured||^2
            # More stable for ill-conditioned matrices
            mitigated_probs, _, _, _ = lstsq(
                self.calibration_data.confusion_matrix,
                measured_probs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Enforce physical constraints
        mitigated_probs = np.maximum(mitigated_probs, 0)  # Non-negative
        mitigated_probs /= np.sum(mitigated_probs)  # Normalize

        # Calculate total variation distance
        tvd = 0.5 * np.sum(np.abs(mitigated_probs - measured_probs))

        logger.info(
            f"Mitigation: TVD={tvd:.4f}, "
            f"condition={self.calibration_data.condition_number:.2f}"
        )

        return MitigationResult(
            mitigated_probabilities=mitigated_probs,
            raw_probabilities=measured_probs,
            total_variation_distance=float(tvd),
            condition_number=self.calibration_data.condition_number,
            metadata={
                'method': method,
                'n_qubits': self.n_qubits,
                'tensored': self.calibration_data.tensored
            }
        )

    def mitigate(
        self,
        measured_counts: Dict[str, int],
        method: str = 'least_squares'
    ) -> MitigationResult:
        """
        Apply measurement error mitigation.

        Args:
            measured_counts: Raw measurement counts
            method: Mitigation method ('inversion' or 'least_squares')

        Returns:
            MitigationResult with corrected probabilities
        """
        return self.mitigate_counts(measured_counts, method)

    def get_calibration_circuits(self) -> List[UnifiedCircuit]:
        """
        Get list of circuits needed for calibration.

        Useful for batching calibration jobs.
        """
        circuits = []

        if self.tensored:
            # Single-qubit calibrations: |0⟩ and |1⟩ for each qubit
            for q in range(self.n_qubits):
                # |0⟩ state
                circuit_0 = UnifiedCircuit(n_qubits=self.n_qubits)
                circuits.append(circuit_0)

                # |1⟩ state
                circuit_1 = UnifiedCircuit(n_qubits=self.n_qubits)
                circuit_1.add_gate(GateType.X, targets=[q])
                circuits.append(circuit_1)
        else:
            # Full calibration: all 2^n basis states
            dim = 2 ** self.n_qubits
            for state in range(dim):
                circuit = UnifiedCircuit(n_qubits=self.n_qubits)
                for q in range(self.n_qubits):
                    if (state >> q) & 1:
                        circuit.add_gate(GateType.X, targets=[q])
                circuits.append(circuit)

        return circuits


def create_measurement_mitigator(
    n_qubits: int,
    tensored: bool = True,
    shots_per_calibration: int = 10000
) -> MeasurementErrorMitigator:
    """
    Factory function to create measurement error mitigator.

    Args:
        n_qubits: Number of qubits
        tensored: Use efficient tensor product calibration
        shots_per_calibration: Shots for each calibration circuit

    Returns:
        Configured mitigator

    Example:
        >>> mitigator = create_measurement_mitigator(n_qubits=5, tensored=True)
        >>> mitigator.calibrate(backend.execute)
        >>> result = mitigator.mitigate(counts)
    """
    return MeasurementErrorMitigator(
        n_qubits=n_qubits,
        tensored=tensored,
        shots_per_calibration=shots_per_calibration
    )
