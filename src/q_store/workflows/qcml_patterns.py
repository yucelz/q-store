"""
Quantum-Classical Machine Learning (QCML) patterns.

Provides reusable patterns for integrating quantum circuits
into classical machine learning workflows.
"""

from typing import Callable, Optional, List, Dict, Any
import numpy as np
from ..core import UnifiedCircuit


class QuantumClassicalHybrid:
    """
    Base class for hybrid quantum-classical models.

    Combines quantum and classical processing stages in a unified workflow.
    """

    def __init__(
        self,
        quantum_component: Callable,
        classical_preprocessor: Optional[Callable] = None,
        classical_postprocessor: Optional[Callable] = None
    ):
        """
        Initialize hybrid model.

        Args:
            quantum_component: Quantum circuit or layer
            classical_preprocessor: Optional preprocessing function
            classical_postprocessor: Optional postprocessing function
        """
        self.quantum_component = quantum_component
        self.classical_preprocessor = classical_preprocessor
        self.classical_postprocessor = classical_postprocessor

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through hybrid model.

        Args:
            x: Input data

        Returns:
            Output after quantum-classical processing
        """
        # Classical preprocessing
        if self.classical_preprocessor is not None:
            x = self.classical_preprocessor(x)

        # Quantum processing
        quantum_output = self.quantum_component(x)

        # Classical postprocessing
        if self.classical_postprocessor is not None:
            quantum_output = self.classical_postprocessor(quantum_output)

        return quantum_output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow model to be called directly."""
        return self.forward(x)


class QuantumPreprocessor:
    """
    Quantum preprocessing layer.

    Encodes classical data into quantum states for processing.
    """

    def __init__(
        self,
        encoding_circuit: Callable[[np.ndarray], UnifiedCircuit],
        n_qubits: int
    ):
        """
        Initialize quantum preprocessor.

        Args:
            encoding_circuit: Function to encode data into quantum circuit
            n_qubits: Number of qubits
        """
        self.encoding_circuit = encoding_circuit
        self.n_qubits = n_qubits

    def encode(self, data: np.ndarray) -> UnifiedCircuit:
        """
        Encode classical data into quantum circuit.

        Args:
            data: Classical input data

        Returns:
            Quantum circuit with encoded data
        """
        return self.encoding_circuit(data)

    def __call__(self, data: np.ndarray) -> UnifiedCircuit:
        """Allow preprocessor to be called directly."""
        return self.encode(data)


class QuantumLayer:
    """
    Quantum layer for hybrid models.

    Represents a parameterized quantum circuit that can be
    integrated into larger hybrid architectures.
    """

    def __init__(
        self,
        circuit_builder: Callable[[np.ndarray, np.ndarray], UnifiedCircuit],
        n_qubits: int,
        n_parameters: int
    ):
        """
        Initialize quantum layer.

        Args:
            circuit_builder: Function that builds circuit from data and parameters
            n_qubits: Number of qubits
            n_parameters: Number of trainable parameters
        """
        self.circuit_builder = circuit_builder
        self.n_qubits = n_qubits
        self.n_parameters = n_parameters
        self.parameters = np.random.randn(n_parameters) * 0.1

    def forward(self, x: np.ndarray) -> UnifiedCircuit:
        """
        Forward pass through quantum layer.

        Args:
            x: Input data

        Returns:
            Quantum circuit
        """
        return self.circuit_builder(x, self.parameters)

    def update_parameters(self, new_parameters: np.ndarray):
        """Update layer parameters."""
        if len(new_parameters) != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"got {len(new_parameters)}"
            )
        self.parameters = new_parameters.copy()

    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        return self.parameters.copy()

    def __call__(self, x: np.ndarray) -> UnifiedCircuit:
        """Allow layer to be called directly."""
        return self.forward(x)


class ClassicalPostprocessor:
    """
    Classical postprocessing layer.

    Processes quantum measurement results with classical operations.
    """

    def __init__(self, processing_function: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize classical postprocessor.

        Args:
            processing_function: Function to process quantum outputs
        """
        self.processing_function = processing_function

    def process(self, quantum_output: np.ndarray) -> np.ndarray:
        """
        Process quantum measurement results.

        Args:
            quantum_output: Output from quantum circuit

        Returns:
            Processed output
        """
        return self.processing_function(quantum_output)

    def __call__(self, quantum_output: np.ndarray) -> np.ndarray:
        """Allow postprocessor to be called directly."""
        return self.process(quantum_output)


def create_hybrid_model(
    quantum_circuit: Callable,
    preprocessing: Optional[Callable] = None,
    postprocessing: Optional[Callable] = None
) -> QuantumClassicalHybrid:
    """
    Create a hybrid quantum-classical model.

    Args:
        quantum_circuit: Quantum circuit function
        preprocessing: Optional preprocessing function
        postprocessing: Optional postprocessing function

    Returns:
        Hybrid model instance
    """
    return QuantumClassicalHybrid(
        quantum_component=quantum_circuit,
        classical_preprocessor=preprocessing,
        classical_postprocessor=postprocessing
    )
