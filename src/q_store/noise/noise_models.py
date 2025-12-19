"""
Realistic Quantum Noise Models.

Implements various noise channels for NISQ device simulation:
- Depolarizing: Random Pauli errors
- Amplitude damping: Energy relaxation (T1)
- Phase damping: Dephasing without energy loss (T2)
- Thermal relaxation: Combined T1/T2 effects with temperature
- Readout errors: Measurement bit-flip errors

Reference:
- Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
- Qiskit Aer noise models
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import numpy as np

from ..core import UnifiedCircuit, GateType

logger = logging.getLogger(__name__)


@dataclass
class NoiseParameters:
    """Parameters for noise models."""
    error_rate: float = 0.001
    t1_time: Optional[float] = None  # T1 relaxation time (µs)
    t2_time: Optional[float] = None  # T2 dephasing time (µs)
    gate_time: Optional[float] = None  # Gate execution time (µs)
    temperature: float = 0.015  # Temperature (K)
    readout_error: Optional[float] = None  # Readout error probability


class NoiseModel(ABC):
    """
    Abstract base class for quantum noise models.
    
    Noise models transform ideal quantum operations into noisy ones
    by adding error channels that represent physical imperfections.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.parameters = parameters
    
    @abstractmethod
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Apply noise model to a circuit.
        
        Args:
            circuit: Ideal quantum circuit
            
        Returns:
            Noisy circuit with error channels inserted
        """
        pass
    
    @abstractmethod
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """
        Get Kraus operators for this noise channel.
        
        Args:
            gate_type: Type of gate to get noise for
            
        Returns:
            List of Kraus operators representing the noise channel
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize noise model to dictionary."""
        return {
            'type': self.__class__.__name__,
            'parameters': {
                'error_rate': self.parameters.error_rate,
                't1_time': self.parameters.t1_time,
                't2_time': self.parameters.t2_time,
                'gate_time': self.parameters.gate_time,
                'temperature': self.parameters.temperature,
                'readout_error': self.parameters.readout_error,
            }
        }


class DepolarizingNoise(NoiseModel):
    """
    Depolarizing noise channel.
    
    With probability p, applies random Pauli error (X, Y, or Z).
    With probability (1-p), applies identity (no error).
    
    For n qubits, depolarizing with probability p applies:
    - Identity with probability (1 - p)
    - Random Pauli on each qubit with probability p/(4^n - 1)
    
    Args:
        parameters: Noise parameters with error_rate = p
        
    Example:
        >>> noise = DepolarizingNoise(NoiseParameters(error_rate=0.01))
        >>> noisy_circuit = noise.apply_to_circuit(circuit)
    """
    
    def __init__(self, parameters: NoiseParameters):
        super().__init__(parameters)
        self.single_qubit_error_rate = parameters.error_rate
        self.two_qubit_error_rate = parameters.error_rate * 10  # 10x worse for 2Q gates
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """
        Get Kraus operators for depolarizing channel.
        
        For single qubit with error rate p:
        K₀ = √(1-p) * I
        K₁ = √(p/3) * X
        K₂ = √(p/3) * Y
        K₃ = √(p/3) * Z
        """
        # Determine if single or two-qubit gate
        is_two_qubit = gate_type in [GateType.CNOT, GateType.CZ, GateType.SWAP]
        p = self.two_qubit_error_rate if is_two_qubit else self.single_qubit_error_rate
        
        if is_two_qubit:
            # For two qubits, we have 4^2 = 16 Pauli operators
            # Simplified: just return identity for now (full impl would be large)
            dim = 4
            kraus_ops = [np.sqrt(1 - p) * np.eye(dim)]
            # Add simplified two-qubit errors
            kraus_ops.append(np.sqrt(p) * np.kron([[0, 1], [1, 0]], np.eye(2)))
            return kraus_ops
        else:
            # Single qubit Pauli operators
            I = np.array([[1, 0], [0, 1]], dtype=complex)
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.array([[1, 0], [0, -1]], dtype=complex)
            
            kraus_ops = [
                np.sqrt(1 - p) * I,
                np.sqrt(p / 3) * X,
                np.sqrt(p / 3) * Y,
                np.sqrt(p / 3) * Z,
            ]
            return kraus_ops
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """
        Apply depolarizing noise after each gate.
        
        Note: This returns the original circuit with metadata.
        Actual noise application happens in the simulator backend.
        """
        noisy_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
        noisy_circuit.gates = circuit.gates.copy()
        noisy_circuit._metadata['noise_model'] = self.to_dict()
        return noisy_circuit


class AmplitudeDampingNoise(NoiseModel):
    """
    Amplitude damping noise (T1 relaxation).
    
    Models energy relaxation from |1⟩ to |0⟩ due to interaction
    with environment at zero temperature.
    
    Kraus operators:
    K₀ = [[1, 0], [0, √(1-γ)]]
    K₁ = [[0, √γ], [0, 0]]
    
    where γ = 1 - exp(-t/T1) is the damping parameter.
    
    Args:
        parameters: NoiseParameters with t1_time and gate_time
        
    Example:
        >>> params = NoiseParameters(t1_time=50, gate_time=0.05)  # 50µs T1, 50ns gate
        >>> noise = AmplitudeDampingNoise(params)
    """
    
    def __init__(self, parameters: NoiseParameters):
        super().__init__(parameters)
        if parameters.t1_time is None or parameters.gate_time is None:
            raise ValueError("AmplitudeDamping requires t1_time and gate_time")
        
        if parameters.t1_time == 0:
            raise ValueError("t1_time cannot be zero")
        
        # Calculate damping parameter: γ = 1 - exp(-t/T1)
        self.gamma = 1 - np.exp(-parameters.gate_time / parameters.t1_time)
        logger.debug(f"Amplitude damping: γ = {self.gamma:.6f} (T1={parameters.t1_time}µs)")
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """Get Kraus operators for amplitude damping."""
        gamma = self.gamma
        
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ], dtype=complex)
        
        return [K0, K1]
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply amplitude damping noise."""
        noisy_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
        noisy_circuit.gates = circuit.gates.copy()
        noisy_circuit._metadata['noise_model'] = self.to_dict()
        return noisy_circuit


class PhaseDampingNoise(NoiseModel):
    """
    Phase damping noise (T2* dephasing).
    
    Models loss of quantum phase information without energy relaxation.
    This is "pure dephasing" - the qubit stays in its computational state
    but loses coherence.
    
    Kraus operators:
    K₀ = [[1, 0], [0, √(1-λ)]]
    K₁ = [[0, 0], [0, √λ]]
    
    where λ = 1 - exp(-t/T2*) is the dephasing parameter.
    
    Args:
        parameters: NoiseParameters with t2_time and gate_time
        
    Example:
        >>> params = NoiseParameters(t2_time=70, gate_time=0.05)  # 70µs T2*, 50ns gate
        >>> noise = PhaseDampingNoise(params)
    """
    
    def __init__(self, parameters: NoiseParameters):
        super().__init__(parameters)
        if parameters.t2_time is None or parameters.gate_time is None:
            raise ValueError("PhaseDamping requires t2_time and gate_time")
        
        if parameters.t2_time == 0:
            raise ValueError("t2_time cannot be zero")
        
        # Calculate dephasing parameter: λ = 1 - exp(-t/T2)
        self.lambda_param = 1 - np.exp(-parameters.gate_time / parameters.t2_time)
        logger.debug(f"Phase damping: λ = {self.lambda_param:.6f} (T2={parameters.t2_time}µs)")
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """Get Kraus operators for phase damping."""
        lam = self.lambda_param
        
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - lam)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, 0],
            [0, np.sqrt(lam)]
        ], dtype=complex)
        
        return [K0, K1]
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply phase damping noise."""
        noisy_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
        noisy_circuit.gates = circuit.gates.copy()
        noisy_circuit._metadata['noise_model'] = self.to_dict()
        return noisy_circuit


class ThermalRelaxationNoise(NoiseModel):
    """
    Thermal relaxation noise (combined T1 and T2 effects).
    
    Models both energy relaxation (T1) and dephasing (T2) in a
    thermal environment. This is the most realistic single-qubit
    noise model for superconducting qubits.
    
    Combines amplitude damping and pure dephasing with the constraint
    that T2 ≤ 2*T1 (physical requirement).
    
    Args:
        parameters: NoiseParameters with t1_time, t2_time, gate_time, temperature
        
    Example:
        >>> params = NoiseParameters(
        ...     t1_time=50,      # 50µs T1
        ...     t2_time=70,      # 70µs T2
        ...     gate_time=0.05,  # 50ns gate
        ...     temperature=0.015 # 15mK
        ... )
        >>> noise = ThermalRelaxationNoise(params)
    """
    
    def __init__(self, parameters: NoiseParameters):
        super().__init__(parameters)
        if parameters.t1_time is None or parameters.t2_time is None or parameters.gate_time is None:
            raise ValueError("ThermalRelaxation requires t1_time, t2_time, and gate_time")
        
        if parameters.t1_time == 0 or parameters.t2_time == 0:
            raise ValueError("t1_time and t2_time cannot be zero")
        
        # Physical constraint: T2 ≤ 2*T1
        if parameters.t2_time > 2 * parameters.t1_time:
            logger.warning(
                f"T2 ({parameters.t2_time}µs) > 2*T1 ({2*parameters.t1_time}µs) "
                "violates physical constraint, clamping T2"
            )
            parameters.t2_time = 2 * parameters.t1_time
        
        # Thermal population of excited state
        # p_excited = 1 / (1 + exp(ℏω/kT))
        # For typical superconducting qubits at 15mK: p_excited ≈ 0
        k_boltzmann = 1.380649e-23  # J/K
        h_bar = 1.054571817e-34     # J·s
        typical_freq = 5e9           # 5 GHz
        
        energy = h_bar * 2 * np.pi * typical_freq
        
        # Handle zero temperature (perfect ground state)
        if parameters.temperature == 0 or parameters.temperature is None:
            self.p_excited = 0.0
        else:
            self.p_excited = 1 / (1 + np.exp(energy / (k_boltzmann * parameters.temperature)))
        
        # Relaxation and dephasing rates
        self.gamma_t1 = parameters.gate_time / parameters.t1_time
        self.gamma_t2 = parameters.gate_time / parameters.t2_time
        
        # Pure dephasing rate: Γ_φ = 1/T2 - 1/(2T1)
        self.gamma_phi = max(0, self.gamma_t2 - self.gamma_t1 / 2)
        
        logger.debug(
            f"Thermal relaxation: T1={parameters.t1_time}µs, T2={parameters.t2_time}µs, "
            f"p_excited={self.p_excited:.6f}"
        )
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """
        Get Kraus operators for thermal relaxation.
        
        This is a combination of:
        1. Amplitude damping from |1⟩ to |0⟩
        2. Excitation from |0⟩ to |1⟩ (thermal)
        3. Pure dephasing
        """
        p_reset = 1 - np.exp(-self.gamma_t1)
        p_excited = self.p_excited
        p_dephase = 1 - np.exp(-self.gamma_phi)
        
        # Simplified Kraus operators (full thermal channel is complex)
        # K0: No error
        K0 = np.array([
            [np.sqrt(1 - p_reset - p_dephase), 0],
            [0, np.sqrt(1 - p_reset - p_dephase)]
        ], dtype=complex)
        
        # K1: Relaxation |1⟩ → |0⟩
        K1 = np.array([
            [0, np.sqrt(p_reset * (1 - p_excited))],
            [0, 0]
        ], dtype=complex)
        
        # K2: Excitation |0⟩ → |1⟩ (thermal)
        K2 = np.array([
            [0, 0],
            [np.sqrt(p_reset * p_excited), 0]
        ], dtype=complex)
        
        # K3: Pure dephasing
        K3 = np.array([
            [0, 0],
            [0, np.sqrt(p_dephase)]
        ], dtype=complex)
        
        return [K0, K1, K2, K3]
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply thermal relaxation noise."""
        noisy_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
        noisy_circuit.gates = circuit.gates.copy()
        noisy_circuit._metadata['noise_model'] = self.to_dict()
        return noisy_circuit


class ReadoutErrorNoise(NoiseModel):
    """
    Readout error noise (measurement bit-flip errors).
    
    Models imperfect measurement where:
    - P(measure 0 | prepared 0) = 1 - p0
    - P(measure 1 | prepared 1) = 1 - p1
    
    Args:
        parameters: NoiseParameters with readout_error
        
    Example:
        >>> params = NoiseParameters(readout_error=0.02)  # 2% readout error
        >>> noise = ReadoutErrorNoise(params)
    """
    
    def __init__(self, parameters: NoiseParameters):
        super().__init__(parameters)
        if parameters.readout_error is None:
            parameters.readout_error = 0.01  # Default 1% error
        
        self.p0 = parameters.readout_error  # P(read 1 | prepared 0)
        self.p1 = parameters.readout_error  # P(read 0 | prepared 1)
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """Readout errors are not represented as Kraus operators."""
        return [np.eye(2, dtype=complex)]
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get readout confusion matrix.
        
        Returns:
            2x2 matrix where M[i,j] = P(measure j | prepared i)
        """
        return np.array([
            [1 - self.p0, self.p0],
            [self.p1, 1 - self.p1]
        ])
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply readout error noise."""
        noisy_circuit = UnifiedCircuit(n_qubits=circuit.n_qubits)
        noisy_circuit.gates = circuit.gates.copy()
        noisy_circuit._metadata['noise_model'] = self.to_dict()
        noisy_circuit._metadata['readout_confusion'] = self.get_confusion_matrix().tolist()
        return noisy_circuit


class CompositeNoiseModel(NoiseModel):
    """
    Composite noise model combining multiple noise channels.
    
    Applies multiple noise models in sequence to create realistic
    NISQ device simulation.
    
    Args:
        noise_models: List of noise models to apply
        
    Example:
        >>> params = NoiseParameters(t1_time=50, t2_time=70, gate_time=0.05)
        >>> composite = CompositeNoiseModel([
        ...     ThermalRelaxationNoise(params),
        ...     DepolarizingNoise(NoiseParameters(error_rate=0.001)),
        ...     ReadoutErrorNoise(NoiseParameters(readout_error=0.02))
        ... ])
    """
    
    def __init__(self, noise_models: List[NoiseModel]):
        if not noise_models:
            raise ValueError("CompositeNoiseModel requires at least one noise model")
        super().__init__(noise_models[0].parameters)
        self.noise_models = noise_models
    
    def get_kraus_operators(self, gate_type: GateType) -> List[np.ndarray]:
        """
        Get combined Kraus operators.
        
        Combines multiple noise channels via Kraus operator composition.
        For channels E1 and E2: (E2 ∘ E1)(ρ) = E2(E1(ρ))
        """
        # Start with first noise model
        kraus_list = self.noise_models[0].get_kraus_operators(gate_type)
        
        # Compose with remaining models
        for noise_model in self.noise_models[1:]:
            new_kraus = []
            next_kraus = noise_model.get_kraus_operators(gate_type)
            
            # Composition: for each pair (K_i, K_j), add K_j @ K_i
            for K_next in next_kraus:
                for K_prev in kraus_list:
                    if K_next.shape == K_prev.shape:
                        new_kraus.append(K_next @ K_prev)
            
            kraus_list = new_kraus if new_kraus else kraus_list
        
        return kraus_list
    
    def apply_to_circuit(self, circuit: UnifiedCircuit) -> UnifiedCircuit:
        """Apply all noise models in sequence."""
        noisy_circuit = circuit
        for noise_model in self.noise_models:
            noisy_circuit = noise_model.apply_to_circuit(noisy_circuit)
        return noisy_circuit
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize composite model."""
        return {
            'type': 'CompositeNoiseModel',
            'noise_models': [nm.to_dict() for nm in self.noise_models]
        }


def create_device_noise_model(
    device_name: str = 'generic',
    custom_params: Optional[NoiseParameters] = None
) -> CompositeNoiseModel:
    """
    Create realistic noise model for common quantum devices.
    
    Provides pre-configured noise models matching real hardware:
    - 'ibm_generic': IBM superconducting qubits
    - 'ionq': IonQ trapped ions
    - 'rigetti': Rigetti superconducting qubits
    - 'google': Google Sycamore
    - 'generic': Generic NISQ device
    
    Args:
        device_name: Name of device to model
        custom_params: Override default parameters
        
    Returns:
        Composite noise model matching device characteristics
        
    Example:
        >>> noise_model = create_device_noise_model('ibm_generic')
        >>> noisy_circuit = noise_model.apply_to_circuit(circuit)
    """
    device_configs = {
        'ibm_generic': NoiseParameters(
            error_rate=0.001,      # 0.1% single-qubit gate error
            t1_time=100,           # 100µs T1
            t2_time=75,            # 75µs T2
            gate_time=0.05,        # 50ns gate time
            temperature=0.015,     # 15mK
            readout_error=0.02,    # 2% readout error
        ),
        'ionq': NoiseParameters(
            error_rate=0.0001,     # 0.01% gate error (better than SC)
            t1_time=10000,         # 10ms T1 (much longer)
            t2_time=1000,          # 1ms T2
            gate_time=10,          # 10µs gate (slower)
            temperature=0.0,       # Effectively 0K
            readout_error=0.005,   # 0.5% readout error
        ),
        'rigetti': NoiseParameters(
            error_rate=0.002,      # 0.2% gate error
            t1_time=30,            # 30µs T1
            t2_time=20,            # 20µs T2
            gate_time=0.04,        # 40ns gate
            temperature=0.015,     # 15mK
            readout_error=0.03,    # 3% readout error
        ),
        'google': NoiseParameters(
            error_rate=0.0015,     # 0.15% gate error
            t1_time=80,            # 80µs T1
            t2_time=60,            # 60µs T2
            gate_time=0.025,       # 25ns gate
            temperature=0.020,     # 20mK
            readout_error=0.015,   # 1.5% readout error
        ),
        'generic': NoiseParameters(
            error_rate=0.005,      # 0.5% gate error
            t1_time=50,            # 50µs T1
            t2_time=40,            # 40µs T2
            gate_time=0.1,         # 100ns gate
            temperature=0.015,     # 15mK
            readout_error=0.05,    # 5% readout error
        ),
    }
    
    params = custom_params or device_configs.get(device_name, device_configs['generic'])
    
    # Build composite model
    noise_models = [
        ThermalRelaxationNoise(params),
        DepolarizingNoise(params),
        ReadoutErrorNoise(params),
    ]
    
    logger.info(f"Created {device_name} noise model: {params}")
    
    return CompositeNoiseModel(noise_models)
