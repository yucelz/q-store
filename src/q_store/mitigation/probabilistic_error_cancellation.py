"""
Probabilistic Error Cancellation (PEC) Error Mitigation.

PEC mitigates errors by:
1. Decomposing noisy operations into quasi-probability distributions over implementable operations
2. Sampling from the quasi-probability distribution
3. Post-processing results with appropriate weights

Reference:
- Temme et al., "Error Mitigation for Short-Depth Quantum Circuits"
  Phys. Rev. Lett. 119, 180509 (2017)
- Endo et al., "Practical Quantum Error Mitigation for Near-Future Applications"
  Phys. Rev. X 8, 031027 (2018)
"""

from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import defaultdict

from ..core import UnifiedCircuit, GateType

logger = logging.getLogger(__name__)


@dataclass
class QuasiProbabilityDecomposition:
    """
    Quasi-probability decomposition of a noisy operation.

    Represents: noisy_op = Σᵢ αᵢ · Cᵢ
    where Cᵢ are implementable operations and αᵢ are quasi-probabilities
    (can be negative, but Σ|αᵢ| gives sampling overhead).
    """
    operations: List[UnifiedCircuit]  # Implementable circuits
    coefficients: List[float]  # Quasi-probabilities
    sampling_overhead: float  # Σ|αᵢ| (shots multiplier)


@dataclass
class PECResult:
    """Result from probabilistic error cancellation."""
    mitigated_value: float
    raw_value: float
    sampling_overhead: float
    n_samples_used: int
    variance: float
    metadata: Dict[str, Any]


class ProbabilisticErrorCanceller:
    """
    Probabilistic Error Cancellation (PEC) for quantum error mitigation.

    PEC works by:
    1. Learning or modeling the noise process
    2. Decomposing noisy gates into quasi-probability distributions
    3. Implementing Monte Carlo sampling with inverse noise
    4. Post-processing with appropriate weights

    Args:
        noise_model: Optional noise characterization
        max_overhead: Maximum sampling overhead to tolerate
        n_samples: Number of Monte Carlo samples

    Example:
        >>> pec = ProbabilisticErrorCanceller(n_samples=1000)
        >>> result = pec.mitigate(circuit, backend)
        >>> print(f"Mitigated: {result.mitigated_value}")
    """

    def __init__(
        self,
        noise_model: Optional[Dict[str, Any]] = None,
        max_overhead: float = 10.0,
        n_samples: int = 1000,
        seed: Optional[int] = None
    ):
        self.noise_model = noise_model or self._default_noise_model()
        self.max_overhead = max_overhead
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def _default_noise_model(self) -> Dict[str, Any]:
        """
        Default depolarizing noise model.

        Assumes each gate has depolarizing noise with rate ε.
        """
        return {
            'type': 'depolarizing',
            'single_qubit_error_rate': 0.001,  # 0.1% error per single-qubit gate
            'two_qubit_error_rate': 0.01,      # 1% error per two-qubit gate
        }

    def decompose_noisy_gate(
        self,
        gate_type: GateType,
        n_qubits: int = 1
    ) -> QuasiProbabilityDecomposition:
        """
        Decompose noisy gate into quasi-probability distribution.

        For depolarizing noise with rate ε:
        - Noisy single-qubit gate: (1-ε)·U + ε·(X·U + Y·U + Z·U)/3
        - Decomposition: α₀·U + α₁·(X·U) + α₂·(Y·U) + α₃·(Z·U)

        Args:
            gate_type: Type of gate to decompose
            n_qubits: Number of qubits gate acts on

        Returns:
            Quasi-probability decomposition
        """
        if n_qubits == 1:
            error_rate = self.noise_model['single_qubit_error_rate']
        else:
            error_rate = self.noise_model['two_qubit_error_rate']

        # For depolarizing noise, we need to implement the inverse channel
        # Ideal operation has coefficient (1 + 3ε)/(1 - ε)
        # Pauli errors have coefficients -ε/(1 - ε) each

        epsilon = error_rate
        if epsilon >= 1.0:
            logger.error("Error rate >= 1.0, PEC not applicable")
            epsilon = 0.99

        # Coefficients for quasi-probability decomposition
        ideal_coeff = (1 + 3 * epsilon) / (1 - epsilon)
        pauli_coeff = -epsilon / (1 - epsilon)

        # Create operations
        operations = []
        coefficients = []

        # Ideal operation
        ideal_circuit = UnifiedCircuit(n_qubits=n_qubits)
        ideal_circuit.add_gate(gate_type, targets=list(range(n_qubits)))
        operations.append(ideal_circuit)
        coefficients.append(ideal_coeff)

        # Pauli errors (for single qubit)
        if n_qubits == 1:
            for pauli_gate in [GateType.X, GateType.Y, GateType.Z]:
                pauli_circuit = UnifiedCircuit(n_qubits=1)
                pauli_circuit.add_gate(pauli_gate, targets=[0])
                pauli_circuit.add_gate(gate_type, targets=[0])
                operations.append(pauli_circuit)
                coefficients.append(pauli_coeff)

        # Sampling overhead = Σ|αᵢ|
        sampling_overhead = sum(abs(c) for c in coefficients)

        if sampling_overhead > self.max_overhead:
            logger.warning(
                f"Sampling overhead {sampling_overhead:.2f} exceeds max {self.max_overhead}. "
                f"Consider using ZNE or reducing error rate."
            )

        return QuasiProbabilityDecomposition(
            operations=operations,
            coefficients=coefficients,
            sampling_overhead=sampling_overhead
        )

    def sample_from_decomposition(
        self,
        decomposition: QuasiProbabilityDecomposition
    ) -> Tuple[UnifiedCircuit, float]:
        """
        Sample one operation from quasi-probability distribution.

        Returns:
            (sampled_circuit, weight_factor)
        """
        # Convert quasi-probabilities to sampling probabilities
        abs_coeffs = np.abs(decomposition.coefficients)
        probs = abs_coeffs / np.sum(abs_coeffs)

        # Sample operation
        idx = self.rng.choice(len(decomposition.operations), p=probs)

        # Calculate weight (includes sign and normalization)
        weight = (
            np.sign(decomposition.coefficients[idx])
            * decomposition.sampling_overhead
        )

        return decomposition.operations[idx], weight

    def mitigate_circuit(
        self,
        circuit: UnifiedCircuit,
        executor: Callable[[UnifiedCircuit], float]
    ) -> PECResult:
        """
        Apply PEC to a circuit.

        Args:
            circuit: Circuit to mitigate
            executor: Function that executes circuit and returns expectation value

        Returns:
            PECResult with mitigated expectation value
        """
        # Decompose each gate
        gate_decompositions = []
        total_overhead = 1.0

        for gate in circuit.gates:
            n_qubits = len(gate.targets)
            decomp = self.decompose_noisy_gate(gate.gate_type, n_qubits)
            gate_decompositions.append(decomp)
            total_overhead *= decomp.sampling_overhead

        logger.info(f"Total sampling overhead: {total_overhead:.2f}x")

        if total_overhead > self.max_overhead:
            logger.warning(
                f"Circuit overhead {total_overhead:.2f} exceeds max {self.max_overhead}. "
                f"Results may have high variance."
            )

        # Monte Carlo sampling
        weighted_values = []

        for _ in range(self.n_samples):
            # Sample a circuit realization
            sampled_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
            total_weight = 1.0

            for gate_idx, gate in enumerate(circuit.gates):
                decomp = gate_decompositions[gate_idx]
                sampled_ops, weight = self.sample_from_decomposition(decomp)

                # Add sampled operations to circuit
                for op_gate in sampled_ops.gates:
                    sampled_circuit.gates.append(op_gate)

                total_weight *= weight

            # Execute sampled circuit
            value = executor(sampled_circuit)
            weighted_values.append(value * total_weight)

        # Estimate mitigated expectation value
        weighted_values = np.array(weighted_values)
        mitigated_value = np.mean(weighted_values)
        variance = np.var(weighted_values)

        # Get raw (unmitigated) value
        raw_value = executor(circuit)

        logger.info(
            f"PEC: raw={raw_value:.4f}, "
            f"mitigated={mitigated_value:.4f}, "
            f"variance={variance:.6f}"
        )

        return PECResult(
            mitigated_value=float(mitigated_value),
            raw_value=float(raw_value),
            sampling_overhead=float(total_overhead),
            n_samples_used=self.n_samples,
            variance=float(variance),
            metadata={
                'n_gates': len(circuit.gates),
                'noise_model': self.noise_model
            }
        )

    def mitigate(
        self,
        circuit: UnifiedCircuit,
        executor: Callable[[UnifiedCircuit], float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> PECResult:
        """
        Apply probabilistic error cancellation.

        Args:
            circuit: Circuit to mitigate
            executor: Function that executes circuit and returns expectation value
            metadata: Optional metadata

        Returns:
            PECResult with mitigated value
        """
        result = self.mitigate_circuit(circuit, executor)
        if metadata:
            result.metadata.update(metadata)
        return result


class AdaptivePEC(ProbabilisticErrorCanceller):
    """
    Adaptive PEC that adjusts sampling based on variance estimates.

    Dynamically allocates more samples to high-variance contributions.
    """

    def __init__(
        self,
        noise_model: Optional[Dict[str, Any]] = None,
        max_overhead: float = 10.0,
        initial_samples: int = 100,
        max_samples: int = 10000,
        target_precision: float = 0.01,
        seed: Optional[int] = None
    ):
        super().__init__(noise_model, max_overhead, initial_samples, seed)
        self.max_samples = max_samples
        self.target_precision = target_precision

    def mitigate_circuit(
        self,
        circuit: UnifiedCircuit,
        executor: Callable[[UnifiedCircuit], float]
    ) -> PECResult:
        """
        Adaptive PEC with dynamic sample allocation.
        """
        # Start with initial samples
        result = super().mitigate_circuit(circuit, executor)

        current_samples = self.n_samples
        current_variance = result.variance

        # Adaptively add more samples if variance is high
        while current_samples < self.max_samples:
            # Estimate standard error
            std_error = np.sqrt(current_variance / current_samples)

            if std_error < self.target_precision:
                logger.info(f"Target precision reached with {current_samples} samples")
                break

            # Add more samples
            additional_samples = min(
                current_samples,  # Double samples
                self.max_samples - current_samples
            )

            if additional_samples == 0:
                break

            self.n_samples = additional_samples
            additional_result = super().mitigate_circuit(circuit, executor)

            # Combine results
            total_weight = current_samples + additional_samples
            result.mitigated_value = (
                (result.mitigated_value * current_samples +
                 additional_result.mitigated_value * additional_samples)
                / total_weight
            )

            current_samples += additional_samples
            current_variance = (
                (current_variance * (current_samples - additional_samples) +
                 additional_result.variance * additional_samples)
                / current_samples
            )

            logger.debug(f"Added {additional_samples} samples, total: {current_samples}")

        result.n_samples_used = current_samples
        result.variance = current_variance

        return result


def create_pec_mitigator(
    noise_model: Optional[Dict[str, Any]] = None,
    max_overhead: float = 10.0,
    n_samples: int = 1000,
    adaptive: bool = False,
    seed: Optional[int] = None
) -> ProbabilisticErrorCanceller:
    """
    Factory function to create PEC mitigator.

    Args:
        noise_model: Noise characterization dict
        max_overhead: Maximum sampling overhead
        n_samples: Number of Monte Carlo samples
        adaptive: Use adaptive sampling
        seed: Random seed

    Returns:
        Configured PEC mitigator

    Example:
        >>> pec = create_pec_mitigator(n_samples=5000, adaptive=True)
        >>> result = pec.mitigate(circuit, executor)
    """
    if adaptive:
        return AdaptivePEC(
            noise_model=noise_model,
            max_overhead=max_overhead,
            initial_samples=min(n_samples, 100),
            max_samples=n_samples,
            seed=seed
        )
    else:
        return ProbabilisticErrorCanceller(
            noise_model=noise_model,
            max_overhead=max_overhead,
            n_samples=n_samples,
            seed=seed
        )
