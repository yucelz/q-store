"""
Variational Quantum Eigensolver (VQE).

VQE is a hybrid quantum-classical algorithm for finding eigenvalues of Hamiltonians.
"""

from typing import Callable, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from q_store.core import UnifiedCircuit, GateType


@dataclass
class VQEResult:
    """
    Result from VQE optimization.

    Attributes:
        eigenvalue: Optimized eigenvalue (energy)
        optimal_parameters: Optimal circuit parameters
        optimal_circuit: Circuit with optimal parameters
        iteration_history: History of (parameters, energy) per iteration
        n_iterations: Number of optimizer iterations
        success: Whether optimization converged
        message: Optimization status message
    """
    eigenvalue: float
    optimal_parameters: np.ndarray
    optimal_circuit: UnifiedCircuit
    iteration_history: List[Tuple[np.ndarray, float]]
    n_iterations: int
    success: bool
    message: str


class VQE:
    """
    Variational Quantum Eigensolver.

    VQE finds the ground state energy of a Hamiltonian by optimizing
    a parameterized quantum circuit (ansatz).

    Args:
        hamiltonian: Hamiltonian as list of (coeff, pauli_string) tuples
        ansatz_factory: Function that creates ansatz circuit from parameters
        optimizer_method: Scipy optimizer method ('COBYLA', 'BFGS', etc.)
        max_iterations: Maximum optimizer iterations
        tol: Convergence tolerance
    """

    def __init__(
        self,
        hamiltonian: List[Tuple[float, str]],
        ansatz_factory: Callable[[np.ndarray], UnifiedCircuit],
        optimizer_method: str = 'COBYLA',
        max_iterations: int = 1000,
        tol: float = 1e-6
    ):
        self.hamiltonian = hamiltonian
        self.ansatz_factory = ansatz_factory
        self.optimizer_method = optimizer_method
        self.max_iterations = max_iterations
        self.tol = tol

        # Track optimization history
        self.iteration_history: List[Tuple[np.ndarray, float]] = []
        self.n_evals = 0

    def run(
        self,
        initial_parameters: Optional[np.ndarray] = None,
        backend: Optional[Any] = None
    ) -> VQEResult:
        """
        Run VQE optimization.

        Args:
            initial_parameters: Initial parameter values (random if None)
            backend: Quantum backend for circuit execution

        Returns:
            VQEResult with optimized eigenvalue and circuit
        """
        # Create initial circuit to determine parameter count
        if initial_parameters is None:
            # Try to create a test circuit to infer parameter count
            # Start with a single parameter and let the ansatz fail gracefully
            try:
                test_params = np.random.uniform(0, 2*np.pi, 10)  # Try with 10 params
                test_circuit = self.ansatz_factory(test_params)
                n_params = test_circuit.n_parameters
                initial_parameters = np.random.uniform(0, 2*np.pi, n_params)
            except:
                # If that fails, use a reasonable default
                initial_parameters = np.random.uniform(0, 2*np.pi, 4)

        # Reset history
        self.iteration_history = []
        self.n_evals = 0

        # Define objective function
        def objective(params: np.ndarray) -> float:
            """Compute expectation value of Hamiltonian."""
            energy = self._compute_expectation(params, backend)
            self.iteration_history.append((params.copy(), energy))
            self.n_evals += 1
            return energy

        # Run optimization
        result = minimize(
            objective,
            initial_parameters,
            method=self.optimizer_method,
            options={'maxiter': self.max_iterations},
            tol=self.tol
        )

        # Create optimal circuit
        optimal_circuit = self.ansatz_factory(result.x)

        return VQEResult(
            eigenvalue=result.fun,
            optimal_parameters=result.x,
            optimal_circuit=optimal_circuit,
            iteration_history=self.iteration_history,
            n_iterations=self.n_evals,
            success=result.success,
            message=result.message
        )

    def _compute_expectation(
        self,
        parameters: np.ndarray,
        backend: Optional[Any] = None
    ) -> float:
        """
        Compute expectation value of Hamiltonian.

        Args:
            parameters: Circuit parameters
            backend: Quantum backend (uses statevector if None)

        Returns:
            Expectation value <ψ|H|ψ>
        """
        # Create parameterized circuit
        circuit = self.ansatz_factory(parameters)

        # Get statevector
        if backend is not None:
            state = backend.execute(circuit)
        else:
            # Simple statevector simulation
            state = self._simulate_circuit(circuit)

        # Compute expectation for each Hamiltonian term
        expectation = 0.0
        for coeff, pauli_string in self.hamiltonian:
            term_exp = self._pauli_expectation(state, pauli_string)
            expectation += coeff * term_exp

        return expectation

    def _simulate_circuit(self, circuit: UnifiedCircuit) -> np.ndarray:
        """Simple statevector simulation."""
        n_qubits = circuit.n_qubits
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0  # |0...0⟩

        for gate in circuit.gates:
            state = self._apply_gate(state, gate, n_qubits)

        return state

    def _apply_gate(self, state: np.ndarray, gate, n_qubits: int) -> np.ndarray:
        """Apply single gate to statevector."""
        # This is a simplified implementation
        # In production, use proper statevector backend

        if gate.gate_type == GateType.H:
            q = gate.targets[0]
            new_state = np.zeros_like(state)
            for i in range(len(state)):
                if (i >> q) & 1:  # Qubit is |1⟩
                    j = i ^ (1 << q)  # Flip qubit
                    new_state[i] = (state[j] - state[i]) / np.sqrt(2)
                else:  # Qubit is |0⟩
                    j = i ^ (1 << q)
                    new_state[i] = (state[i] + state[j]) / np.sqrt(2)
            return new_state

        elif gate.gate_type == GateType.RZ:
            q = gate.targets[0]
            angle = gate.parameters['angle']
            new_state = state.copy()
            for i in range(len(state)):
                if (i >> q) & 1:
                    new_state[i] *= np.exp(-1j * angle / 2)
                else:
                    new_state[i] *= np.exp(1j * angle / 2)
            return new_state

        elif gate.gate_type == GateType.RY:
            q = gate.targets[0]
            angle = gate.parameters['angle']
            cos = np.cos(angle / 2)
            sin = np.sin(angle / 2)
            new_state = np.zeros_like(state)
            for i in range(len(state)):
                j = i ^ (1 << q)
                if (i >> q) & 1:
                    new_state[i] = -sin * state[j] + cos * state[i]
                else:
                    new_state[i] = cos * state[i] + sin * state[j]
            return new_state

        elif gate.gate_type == GateType.CNOT:
            control, target = gate.targets[0], gate.targets[1]
            new_state = state.copy()
            for i in range(len(state)):
                if (i >> control) & 1:  # Control is |1⟩
                    j = i ^ (1 << target)  # Flip target
                    new_state[j] = state[i]
            return new_state

        else:
            # Gate not implemented in simple simulator
            return state

    def _pauli_expectation(self, state: np.ndarray, pauli_string: str) -> float:
        """
        Compute expectation value of Pauli string.

        Args:
            state: Quantum statevector
            pauli_string: Pauli string like "IXYZ"

        Returns:
            Expectation value <ψ|P|ψ>
        """
        n_qubits = int(np.log2(len(state)))
        expectation = 0.0

        # For each basis state
        for i in range(len(state)):
            if abs(state[i]) < 1e-10:
                continue

            # Apply Pauli string
            sign = 1.0
            j = i

            for q, pauli in enumerate(pauli_string):
                if pauli == 'I':
                    continue
                elif pauli == 'Z':
                    if (i >> q) & 1:
                        sign *= -1
                elif pauli == 'X':
                    j ^= (1 << q)
                elif pauli == 'Y':
                    j ^= (1 << q)
                    if (i >> q) & 1:
                        sign *= -1j
                    else:
                        sign *= 1j

            expectation += sign * np.conj(state[i]) * state[j]

        return np.real(expectation)


def create_hardware_efficient_ansatz(n_qubits: int, n_layers: int) -> Callable:
    """
    Create hardware-efficient ansatz factory.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of ansatz layers

    Returns:
        Function that creates ansatz from parameters
    """
    def ansatz_factory(params: np.ndarray) -> UnifiedCircuit:
        circuit = UnifiedCircuit(n_qubits=n_qubits)

        param_idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(n_qubits):
                circuit.add_gate(GateType.RY, targets=[q], parameters={'angle': params[param_idx]})
                param_idx += 1
                circuit.add_gate(GateType.RZ, targets=[q], parameters={'angle': params[param_idx]})
                param_idx += 1

            # Entangling layer
            for q in range(n_qubits - 1):
                circuit.add_gate(GateType.CNOT, targets=[q, q+1])

        return circuit

    return ansatz_factory


def create_uccsd_ansatz(n_qubits: int, n_electrons: int) -> Callable:
    """
    Create UCCSD ansatz for quantum chemistry.

    Args:
        n_qubits: Number of qubits (spin-orbitals)
        n_electrons: Number of electrons

    Returns:
        Function that creates UCCSD ansatz from parameters
    """
    def ansatz_factory(params: np.ndarray) -> UnifiedCircuit:
        circuit = UnifiedCircuit(n_qubits=n_qubits)

        # Initialize Hartree-Fock state
        for i in range(n_electrons):
            circuit.add_gate(GateType.X, targets=[i])

        # Single excitations
        param_idx = 0
        for i in range(n_electrons):
            for a in range(n_electrons, n_qubits):
                # Apply exp(-i θ (a†_a a_i - a†_i a_a))
                circuit.add_gate(GateType.CNOT, targets=[i, a])
                circuit.add_gate(GateType.RY, targets=[a], parameters={'angle': params[param_idx]})
                circuit.add_gate(GateType.CNOT, targets=[i, a])
                param_idx += 1

        return circuit

    return ansatz_factory
