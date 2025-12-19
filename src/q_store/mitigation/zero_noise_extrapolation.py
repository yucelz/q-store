"""
Zero-Noise Extrapolation (ZNE) Error Mitigation.

ZNE mitigates errors by:
1. Running circuits at different noise levels (via gate stretching)
2. Extrapolating to zero-noise limit
3. Supporting multiple extrapolation methods (linear, polynomial, exponential)

Reference: 
- Temme et al., "Error Mitigation for Short-Depth Quantum Circuits"
  Phys. Rev. Lett. 119, 180509 (2017)
"""

from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
import numpy as np
from scipy.optimize import curve_fit

from ..core import UnifiedCircuit, GateType

logger = logging.getLogger(__name__)


class ExtrapolationMethod(str, Enum):
    """Methods for extrapolating to zero noise."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    RICHARDSON = "richardson"


@dataclass
class ZNEResult:
    """Result from zero-noise extrapolation."""
    mitigated_value: float
    raw_values: List[float]
    noise_factors: List[float]
    extrapolation_method: str
    fit_quality: float  # R² or similar metric
    metadata: Dict[str, Any]


class ZeroNoiseExtrapolator:
    """
    Zero-Noise Extrapolation (ZNE) error mitigation.
    
    ZNE works by:
    1. Amplifying noise by stretching gates (inserting identity sequences)
    2. Measuring expectation values at different noise levels
    3. Extrapolating to zero-noise limit
    
    Noise amplification strategies:
    - Unitary folding: Replace U → U†U†U (scales noise by ~3x)
    - Gate stretching: Decompose gates and insert delays
    
    Args:
        noise_factors: List of noise scaling factors (e.g., [1, 2, 3])
        extrapolation_method: Method for extrapolation
        polynomial_degree: Degree for polynomial extrapolation
        
    Example:
        >>> zne = ZeroNoiseExtrapolator(noise_factors=[1, 2, 3])
        >>> result = zne.mitigate(circuit, backend, observable)
        >>> print(f"Mitigated: {result.mitigated_value}")
    """
    
    def __init__(
        self,
        noise_factors: Optional[List[float]] = None,
        extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.LINEAR,
        polynomial_degree: int = 2
    ):
        self.noise_factors = noise_factors or [1.0, 2.0, 3.0]
        self.extrapolation_method = extrapolation_method
        self.polynomial_degree = polynomial_degree
        
        # Validate noise factors
        if not all(f >= 1.0 for f in self.noise_factors):
            raise ValueError("All noise factors must be >= 1.0")
        
        if 1.0 not in self.noise_factors:
            logger.warning("Adding baseline noise factor 1.0")
            self.noise_factors = [1.0] + self.noise_factors
        
        self.noise_factors = sorted(set(self.noise_factors))
    
    def amplify_noise(
        self,
        circuit: UnifiedCircuit,
        noise_factor: float
    ) -> UnifiedCircuit:
        """
        Amplify circuit noise by unitary folding.
        
        For noise_factor = 2n+1, applies folding n times:
        G → G·G†·G (noise scales by ~3x per fold)
        
        Args:
            circuit: Original circuit
            noise_factor: Target noise amplification (must be odd integer)
            
        Returns:
            Circuit with amplified noise
        """
        if noise_factor == 1.0:
            return circuit
        
        # Convert to odd integer
        n_folds = int((noise_factor - 1) / 2)
        if n_folds < 1:
            logger.warning(f"Noise factor {noise_factor} too small, using factor 1.0")
            return circuit
        
        amplified = UnifiedCircuit(n_qubits=circuit.n_qubits)
        
        for gate in circuit.gates:
            # Add original gate
            amplified.gates.append(gate)
            
            # Apply folding (G†G) n times
            for _ in range(n_folds):
                # Add inverse gate
                inverse_gate = self._get_inverse_gate(gate)
                if inverse_gate:
                    amplified.gates.append(inverse_gate)
                    # Add forward gate again
                    amplified.gates.append(gate)
        
        return amplified
    
    def _get_inverse_gate(self, gate) -> Optional[Any]:
        """Get inverse of a gate."""
        # Self-inverse gates
        if gate.gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z, 
                               GateType.CNOT, GateType.CZ, GateType.SWAP]:
            return gate
        
        # Rotation gates: negate angle
        if gate.gate_type in [GateType.RX, GateType.RY, GateType.RZ]:
            from ..core.unified_circuit import Gate
            return Gate(
                gate_type=gate.gate_type,
                targets=gate.targets,
                controls=gate.controls,
                parameters={'angle': -gate.parameters.get('angle', 0.0)} if gate.parameters else None
            )
        
        # For other gates, return None (skip folding)
        logger.debug(f"Cannot fold gate {gate.gate_type}, skipping")
        return None
    
    def extrapolate(
        self,
        noise_factors: List[float],
        measured_values: List[float]
    ) -> Tuple[float, float]:
        """
        Extrapolate to zero noise.
        
        Args:
            noise_factors: Noise scaling factors
            measured_values: Measured expectation values
            
        Returns:
            (extrapolated_value, fit_quality)
        """
        noise_factors = np.array(noise_factors)
        measured_values = np.array(measured_values)
        
        if self.extrapolation_method == ExtrapolationMethod.LINEAR:
            return self._linear_extrapolation(noise_factors, measured_values)
        
        elif self.extrapolation_method == ExtrapolationMethod.POLYNOMIAL:
            return self._polynomial_extrapolation(noise_factors, measured_values)
        
        elif self.extrapolation_method == ExtrapolationMethod.EXPONENTIAL:
            return self._exponential_extrapolation(noise_factors, measured_values)
        
        elif self.extrapolation_method == ExtrapolationMethod.RICHARDSON:
            return self._richardson_extrapolation(noise_factors, measured_values)
        
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolation_method}")
    
    def _linear_extrapolation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Linear extrapolation: y = a + b*x, return y(0)."""
        # Fit line
        coeffs = np.polyfit(x, y, 1)
        y_fit = np.polyval(coeffs, x)
        
        # R² score
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Extrapolate to x=0
        zero_noise_value = coeffs[1]  # Intercept
        
        return float(zero_noise_value), float(r_squared)
    
    def _polynomial_extrapolation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Polynomial extrapolation of specified degree."""
        # Fit polynomial
        coeffs = np.polyfit(x, y, self.polynomial_degree)
        y_fit = np.polyval(coeffs, x)
        
        # R² score
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Extrapolate to x=0
        zero_noise_value = coeffs[-1]  # Constant term
        
        return float(zero_noise_value), float(r_squared)
    
    def _exponential_extrapolation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Exponential extrapolation: y = a * exp(-b*x) + c
        
        This assumes exponential decay with noise.
        """
        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            # Initial guess
            p0 = [y[0] - y[-1], 0.5, y[-1]]
            
            # Fit
            popt, _ = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
            
            # Evaluate fit
            y_fit = exp_func(x, *popt)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Extrapolate to x=0
            a, b, c = popt
            zero_noise_value = a + c
            
            return float(zero_noise_value), float(r_squared)
            
        except Exception as e:
            logger.warning(f"Exponential fit failed: {e}, falling back to linear")
            return self._linear_extrapolation(x, y)
    
    def _richardson_extrapolation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Richardson extrapolation (successive refinement).
        
        Assumes error scales as a polynomial in noise.
        """
        n = len(x)
        if n < 2:
            return float(y[0]), 1.0
        
        # Create Richardson table
        table = np.zeros((n, n))
        table[:, 0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                # Richardson formula
                r = x[i+j] / x[i]
                table[i, j] = (r**j * table[i+1, j-1] - table[i, j-1]) / (r**j - 1)
        
        # Best estimate is top-right corner
        zero_noise_value = table[0, -1]
        
        # Estimate quality from convergence
        if n > 2:
            improvement = abs(table[0, -1] - table[0, -2]) / (abs(table[0, -1]) + 1e-10)
            fit_quality = max(0.0, 1.0 - improvement)
        else:
            fit_quality = 0.8
        
        return float(zero_noise_value), float(fit_quality)
    
    def mitigate(
        self,
        circuit: UnifiedCircuit,
        executor: Callable[[UnifiedCircuit], float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ZNEResult:
        """
        Apply zero-noise extrapolation.
        
        Args:
            circuit: Circuit to mitigate
            executor: Function that executes circuit and returns expectation value
            metadata: Optional metadata
            
        Returns:
            ZNEResult with mitigated value
        """
        raw_values = []
        
        # Execute at each noise level
        for noise_factor in self.noise_factors:
            amplified_circuit = self.amplify_noise(circuit, noise_factor)
            value = executor(amplified_circuit)
            raw_values.append(value)
            logger.debug(f"Noise factor {noise_factor}: value = {value}")
        
        # Extrapolate to zero noise
        mitigated_value, fit_quality = self.extrapolate(
            self.noise_factors,
            raw_values
        )
        
        logger.info(
            f"ZNE: raw={raw_values[0]:.4f}, "
            f"mitigated={mitigated_value:.4f}, "
            f"improvement={abs(mitigated_value - raw_values[0]):.4f}"
        )
        
        return ZNEResult(
            mitigated_value=mitigated_value,
            raw_values=raw_values,
            noise_factors=self.noise_factors,
            extrapolation_method=self.extrapolation_method.value,
            fit_quality=fit_quality,
            metadata=metadata or {}
        )


def create_zne_mitigator(
    noise_factors: Optional[List[float]] = None,
    method: str = "linear",
    polynomial_degree: int = 2
) -> ZeroNoiseExtrapolator:
    """
    Factory function to create ZNE mitigator.
    
    Args:
        noise_factors: Noise amplification factors (default: [1, 2, 3])
        method: Extrapolation method ('linear', 'polynomial', 'exponential', 'richardson')
        polynomial_degree: Degree for polynomial extrapolation
        
    Returns:
        Configured ZeroNoiseExtrapolator
        
    Example:
        >>> zne = create_zne_mitigator(noise_factors=[1, 3, 5], method='exponential')
        >>> result = zne.mitigate(circuit, executor)
    """
    try:
        extrapolation_method = ExtrapolationMethod(method)
    except ValueError:
        logger.warning(f"Unknown method '{method}', using 'linear'")
        extrapolation_method = ExtrapolationMethod.LINEAR
    
    return ZeroNoiseExtrapolator(
        noise_factors=noise_factors,
        extrapolation_method=extrapolation_method,
        polynomial_degree=polynomial_degree
    )
