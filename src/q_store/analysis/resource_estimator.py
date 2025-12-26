"""
Quantum resource estimation tools.

Provides resource estimation for quantum circuits including
execution time, hardware costs, and error budgets.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ..core import UnifiedCircuit
from .complexity import CircuitComplexity


@dataclass
class HardwareModel:
    """
    Model of quantum hardware characteristics.

    Attributes:
        name: Hardware platform name
        single_qubit_gate_time: Time for single-qubit gate (μs)
        two_qubit_gate_time: Time for two-qubit gate (μs)
        measurement_time: Time for measurement (μs)
        single_qubit_error_rate: Error rate for single-qubit gates
        two_qubit_error_rate: Error rate for two-qubit gates
        readout_error_rate: Measurement error rate
        t1: Relaxation time (μs)
        t2: Dephasing time (μs)
        max_qubits: Maximum number of qubits
        connectivity: Connectivity type ('all-to-all', 'linear', 'grid')
    """
    name: str = "Generic"
    single_qubit_gate_time: float = 0.05  # μs
    two_qubit_gate_time: float = 0.3  # μs
    measurement_time: float = 1.0  # μs
    single_qubit_error_rate: float = 0.001
    two_qubit_error_rate: float = 0.01
    readout_error_rate: float = 0.01
    t1: float = 100.0  # μs
    t2: float = 50.0  # μs
    max_qubits: int = 127
    connectivity: str = 'all-to-all'


# Pre-defined hardware models
HARDWARE_MODELS = {
    'generic': HardwareModel(),
    'ibm_quantum': HardwareModel(
        name='IBM Quantum',
        single_qubit_gate_time=0.05,
        two_qubit_gate_time=0.3,
        measurement_time=1.0,
        single_qubit_error_rate=0.0005,
        two_qubit_error_rate=0.01,
        readout_error_rate=0.015,
        t1=100.0,
        t2=80.0,
        max_qubits=127,
        connectivity='grid'
    ),
    'ionq': HardwareModel(
        name='IonQ',
        single_qubit_gate_time=10.0,
        two_qubit_gate_time=200.0,
        measurement_time=100.0,
        single_qubit_error_rate=0.0001,
        two_qubit_error_rate=0.005,
        readout_error_rate=0.001,
        t1=1000000.0,
        t2=500000.0,
        max_qubits=32,
        connectivity='all-to-all'
    ),
    'rigetti': HardwareModel(
        name='Rigetti',
        single_qubit_gate_time=0.05,
        two_qubit_gate_time=0.2,
        measurement_time=2.0,
        single_qubit_error_rate=0.001,
        two_qubit_error_rate=0.02,
        readout_error_rate=0.03,
        t1=50.0,
        t2=30.0,
        max_qubits=80,
        connectivity='linear'
    )
}


class ResourceEstimator:
    """
    Estimate quantum circuit resource requirements.

    Provides estimates for:
    - Execution time
    - Error budget
    - Hardware compatibility
    - Approximate cost
    """

    def __init__(
        self,
        circuit: UnifiedCircuit,
        hardware_model: Optional[HardwareModel] = None
    ):
        """
        Initialize resource estimator.

        Args:
            circuit: Quantum circuit to analyze
            hardware_model: Hardware model for estimation (default: generic)
        """
        self.circuit = circuit
        self.hardware = hardware_model or HARDWARE_MODELS['generic']
        self.complexity = CircuitComplexity(circuit)

    def estimate_execution_time(self) -> float:
        """
        Estimate total execution time in microseconds.

        Returns:
            Estimated execution time (μs)
        """
        # Count gate types
        single_qubit_count = self.complexity.single_qubit_gate_count()
        two_qubit_count = self.complexity.two_qubit_gate_count()

        # Estimate time based on circuit depth and gate times
        depth = self.complexity.depth()
        avg_gate_time = (
            single_qubit_count * self.hardware.single_qubit_gate_time +
            two_qubit_count * self.hardware.two_qubit_gate_time
        ) / max(self.complexity.total_gates(), 1)

        circuit_time = depth * avg_gate_time
        measurement_time = self.hardware.measurement_time

        return circuit_time + measurement_time

    def estimate_error_rate(self) -> float:
        """
        Estimate total error rate for circuit execution.

        Returns:
            Estimated error probability
        """
        single_qubit_count = self.complexity.single_qubit_gate_count()
        two_qubit_count = self.complexity.two_qubit_gate_count()

        # Simple error model: 1 - (1 - p)^n ≈ n*p for small p
        gate_error = (
            single_qubit_count * self.hardware.single_qubit_error_rate +
            two_qubit_count * self.hardware.two_qubit_error_rate
        )

        readout_error = self.circuit.n_qubits * self.hardware.readout_error_rate

        return min(gate_error + readout_error, 1.0)

    def estimate_decoherence_error(self) -> float:
        """
        Estimate error from decoherence during execution.

        Returns:
            Estimated decoherence error probability
        """
        execution_time = self.estimate_execution_time()

        # Exponential decay model
        t1_error = 1 - np.exp(-execution_time / self.hardware.t1)
        t2_error = 1 - np.exp(-execution_time / self.hardware.t2)

        # Combined decoherence error
        return 1 - (1 - t1_error) * (1 - t2_error)

    def check_hardware_compatibility(self) -> Dict:
        """
        Check if circuit is compatible with hardware.

        Returns:
            Dictionary with compatibility information
        """
        compatible = True
        issues = []

        # Check qubit count
        if self.circuit.n_qubits > self.hardware.max_qubits:
            compatible = False
            issues.append(
                f"Circuit requires {self.circuit.n_qubits} qubits, "
                f"hardware supports {self.hardware.max_qubits}"
            )

        # Check gate support (simplified - assume all gates supported)

        # Estimate if error rate is reasonable
        error_rate = self.estimate_error_rate()
        if error_rate > 0.5:
            issues.append(
                f"High estimated error rate: {error_rate:.2%}"
            )

        return {
            'compatible': compatible,
            'issues': issues,
            'hardware': self.hardware.name,
            'qubits_required': self.circuit.n_qubits,
            'estimated_error_rate': error_rate
        }

    def estimate_cost(self, shots: int = 1024) -> Dict:
        """
        Estimate computational cost.

        Args:
            shots: Number of circuit executions

        Returns:
            Dictionary with cost estimates
        """
        time_per_shot = self.estimate_execution_time()
        total_time = time_per_shot * shots / 1e6  # Convert to seconds

        # Arbitrary cost units (could be mapped to real pricing)
        qubit_cost = self.circuit.n_qubits * 0.1
        gate_cost = self.complexity.total_gates() * 0.01
        depth_cost = self.complexity.depth() * 0.05

        base_cost = qubit_cost + gate_cost + depth_cost
        total_cost = base_cost * shots

        return {
            'base_cost_per_shot': base_cost,
            'total_cost': total_cost,
            'execution_time_seconds': total_time,
            'shots': shots
        }

    def summary(self, shots: int = 1024) -> Dict:
        """
        Get comprehensive resource summary.

        Args:
            shots: Number of circuit executions

        Returns:
            Dictionary with all resource estimates
        """
        return {
            'complexity': self.complexity.summary(),
            'execution_time_us': self.estimate_execution_time(),
            'error_rate': self.estimate_error_rate(),
            'decoherence_error': self.estimate_decoherence_error(),
            'hardware_compatibility': self.check_hardware_compatibility(),
            'cost_estimate': self.estimate_cost(shots)
        }


def estimate_resources(
    circuit: UnifiedCircuit,
    hardware: str = 'generic',
    shots: int = 1024
) -> Dict:
    """
    Estimate resources for circuit execution.

    Args:
        circuit: Quantum circuit
        hardware: Hardware model name ('generic', 'ibm_quantum', 'ionq', 'rigetti')
        shots: Number of circuit executions

    Returns:
        Resource estimation summary
    """
    hardware_model = HARDWARE_MODELS.get(hardware, HARDWARE_MODELS['generic'])
    estimator = ResourceEstimator(circuit, hardware_model)
    return estimator.summary(shots)


def estimate_execution_time(
    circuit: UnifiedCircuit,
    hardware: str = 'generic'
) -> float:
    """
    Estimate circuit execution time.

    Args:
        circuit: Quantum circuit
        hardware: Hardware model name

    Returns:
        Estimated execution time (μs)
    """
    hardware_model = HARDWARE_MODELS.get(hardware, HARDWARE_MODELS['generic'])
    estimator = ResourceEstimator(circuit, hardware_model)
    return estimator.estimate_execution_time()


def estimate_hardware_cost(
    circuit: UnifiedCircuit,
    hardware: str = 'generic',
    shots: int = 1024
) -> Dict:
    """
    Estimate hardware cost for circuit.

    Args:
        circuit: Quantum circuit
        hardware: Hardware model name
        shots: Number of executions

    Returns:
        Cost estimate dictionary
    """
    hardware_model = HARDWARE_MODELS.get(hardware, HARDWARE_MODELS['generic'])
    estimator = ResourceEstimator(circuit, hardware_model)
    return estimator.estimate_cost(shots)
