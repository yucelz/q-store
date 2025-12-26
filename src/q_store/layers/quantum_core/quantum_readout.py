"""
Quantum Readout Layer - v4.1

Final layer for classification/regression tasks.
Maps quantum measurements to class predictions using Born rule.

Key Features:
- Computational basis measurements
- Multi-class classification support
- Born rule probability extraction
- Efficient qubit encoding (log2(n_classes) qubits)
- Parameterized final rotations for learning
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional


class QuantumReadout:
    """
    Quantum readout layer for classification/regression.

    Maps quantum measurements to class predictions.
    Uses multi-class measurement strategy.

    For n_classes:
    - Use ceil(log2(n_classes)) qubits
    - Measure in computational basis
    - Probabilities → class scores

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    n_classes : int
        Number of output classes
    readout_type : str, default='computational'
        Type of readout: 'computational' or 'expectation'
    backend : str, default='ionq'
        Quantum backend to use

    Examples
    --------
    >>> layer = QuantumReadout(n_qubits=4, n_classes=10)
    >>> probs = await layer.call_async(inputs)
    >>> print(probs.shape)  # (batch_size, 10)
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        readout_type: str = 'computational',
        backend: str = 'ionq',
        **kwargs
    ):
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.readout_type = readout_type
        self.backend = backend

        # Check if n_qubits sufficient
        max_classes = 2 ** n_qubits
        if n_classes > max_classes:
            min_qubits = int(np.ceil(np.log2(n_classes)))
            raise ValueError(
                f"Need {min_qubits} qubits for {n_classes} classes, got {n_qubits}"
            )

        # Readout parameters (trainable)
        self.readout_params = self._initialize_parameters()

        self.executor = None  # Will be initialized in Phase 2

    def _initialize_parameters(self) -> Dict[str, float]:
        """Initialize readout layer parameters."""
        params = {}
        for qubit in range(self.n_qubits):
            # Final rotation for each qubit
            params[f'readout_theta_{qubit}'] = np.random.randn() * 0.1
        return params

    def set_parameters(self, params: Dict[str, float]):
        """Set readout parameters."""
        self.readout_params = params

    async def call_async(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Quantum readout for classification.

        Returns: (batch_size, n_classes) probabilities

        Parameters
        ----------
        inputs : np.ndarray
            Input features of shape (batch_size, n_features)
        training : bool, default=False
            Whether in training mode

        Returns
        -------
        probabilities : np.ndarray
            Class probabilities of shape (batch_size, n_classes)
        """
        batch_size = inputs.shape[0]

        # For now, simulate quantum readout classically
        # In Phase 2, this will execute on quantum hardware

        if self.readout_type == 'computational':
            return await self._computational_readout(inputs, training)
        elif self.readout_type == 'expectation':
            return await self._expectation_readout(inputs, training)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")

    async def _computational_readout(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Computational basis readout.

        Measures in Z-basis and extracts bitstring probabilities.
        """
        batch_size = inputs.shape[0]
        probabilities = []

        for i in range(batch_size):
            # Simulate measurement in computational basis
            # In reality, this executes on quantum hardware

            # Map input features to quantum state probabilities
            # Use softmax-like transformation
            logits = self._compute_logits(inputs[i])
            probs = self._softmax(logits[:self.n_classes])

            probabilities.append(probs)

        return np.array(probabilities, dtype=np.float32)

    async def _expectation_readout(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Expectation value readout.

        Uses expectation values in different bases as class scores.
        """
        batch_size = inputs.shape[0]
        n_features = min(inputs.shape[1], self.n_classes)

        # Use expectation values directly as class scores
        # Normalize to probabilities
        class_scores = inputs[:, :n_features]

        # Pad if needed
        if n_features < self.n_classes:
            padding = np.zeros((batch_size, self.n_classes - n_features))
            class_scores = np.concatenate([class_scores, padding], axis=1)

        # Convert to probabilities
        probabilities = self._softmax(class_scores)

        return probabilities

    def _compute_logits(self, features: np.ndarray) -> np.ndarray:
        """
        Compute class logits from quantum features.

        Uses readout parameters to map features to class scores.
        """
        # Apply final parametric transformation
        # This simulates the final quantum rotations + measurement

        # Use weighted combination of features
        n_features = len(features)
        max_states = 2 ** self.n_qubits

        logits = np.zeros(max_states)

        for i in range(min(n_features, max_states)):
            # Apply readout transformation
            qubit_idx = i % self.n_qubits
            theta = self.readout_params.get(f'readout_theta_{qubit_idx}', 0.0)

            # Combine feature with parameter
            logits[i] = features[i] * np.cos(theta) + np.sin(theta)

        return logits

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        if x.ndim == 1:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        return {
            'n_qubits': self.n_qubits,
            'n_classes': self.n_classes,
            'readout_type': self.readout_type,
            'backend': self.backend,
        }

    def __call__(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Synchronous forward pass wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot use synchronous call in async context. "
                    "Use 'await layer.call_async(inputs)' instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.call_async(inputs, training))
