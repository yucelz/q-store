"""
Quantum Approximate Optimization Algorithm (QAOA).

QAOA is a variational algorithm for combinatorial optimization problems.
"""

from typing import Callable, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from q_store.core import UnifiedCircuit, GateType


@dataclass
class QAOAResult:
    """
    Result from QAOA optimization.

    Attributes:
        optimal_cost: Optimized cost function value
        optimal_parameters: Optimal (beta, gamma) parameters
        optimal_circuit: Circuit with optimal parameters
        optimal_solution: Most probable bitstring solution
        solution_probability: Probability of optimal solution
        iteration_history: History of (parameters, cost) per iteration
        n_iterations: Number of optimizer iterations
        success: Whether optimization converged
        message: Optimization status message
    """
    optimal_cost: float
    optimal_parameters: np.ndarray
    optimal_circuit: UnifiedCircuit
    optimal_solution: str
    solution_probability: float
    iteration_history: List[Tuple[np.ndarray, float]]
    n_iterations: int
    success: bool
    message: str


class QAOA:
    """
    Quantum Approximate Optimization Algorithm.

    QAOA solves combinatorial optimization problems by alternating between
    problem Hamiltonian and mixer Hamiltonian layers.

    Args:
        cost_hamiltonian: Problem cost as list of (coeff, pauli_string) tuples
        n_layers: Number of QAOA layers (p parameter)
        mixer: Mixer Hamiltonian ('X' for standard transverse field)
        optimizer_method: Scipy optimizer method
        max_iterations: Maximum optimizer iterations
        tol: Convergence tolerance
    """

    def __init__(
        self,
        cost_hamiltonian: List[Tuple[float, str]],
        n_layers: int = 1,
        mixer: str = 'X',
        optimizer_method: str = 'COBYLA',
        max_iterations: int = 1000,
        tol: float = 1e-6
    ):
        self.cost_hamiltonian = cost_hamiltonian
        self.n_layers = n_layers
        self.mixer = mixer
        self.optimizer_method = optimizer_method
        self.max_iterations = max_iterations
        self.tol = tol

        # Determine number of qubits from Hamiltonian
        max_qubit = 0
        for _, pauli_string in cost_hamiltonian:
            max_qubit = max(max_qubit, len(pauli_string))
        self.n_qubits = max_qubit

        # Track optimization
        self.iteration_history: List[Tuple[np.ndarray, float]] = []
        self.n_evals = 0

    def run(
        self,
        initial_parameters: Optional[np.ndarray] = None,
        backend: Optional[Any] = None,
        n_shots: int = 1000
    ) -> QAOAResult:
        """
        Run QAOA optimization.

        Args:
            initial_parameters: Initial [beta, gamma] parameters
            backend: Quantum backend for circuit execution
            n_shots: Number of measurement shots

        Returns:
            QAOAResult with optimal solution
        """
        # Initialize parameters
        if initial_parameters is None:
            # Random initialization for beta and gamma
            initial_parameters = np.random.uniform(0, 2*np.pi, 2*self.n_layers)

        # Reset history
        self.iteration_history = []
        self.n_evals = 0

        # Define objective function
        def objective(params: np.ndarray) -> float:
            """Compute expectation value of cost Hamiltonian."""
            cost = self._compute_cost(params, backend, n_shots)
            self.iteration_history.append((params.copy(), cost))
            self.n_evals += 1
            return cost

        # Run optimization
        result = minimize(
            objective,
            initial_parameters,
            method=self.optimizer_method,
            options={'maxiter': self.max_iterations},
            tol=self.tol
        )

        # Create optimal circuit and extract solution
        optimal_circuit = self._create_circuit(result.x)
        solution, prob = self._extract_solution(optimal_circuit, backend, n_shots)

        return QAOAResult(
            optimal_cost=result.fun,
            optimal_parameters=result.x,
            optimal_circuit=optimal_circuit,
            optimal_solution=solution,
            solution_probability=prob,
            iteration_history=self.iteration_history,
            n_iterations=self.n_evals,
            success=result.success,
            message=result.message
        )

    def _create_circuit(self, parameters: np.ndarray) -> UnifiedCircuit:
        """
        Create QAOA circuit with given parameters.

        Args:
            parameters: [beta_1, gamma_1, beta_2, gamma_2, ...]

        Returns:
            QAOA circuit
        """
        circuit = UnifiedCircuit(n_qubits=self.n_qubits)

        # Initial state: uniform superposition
        for q in range(self.n_qubits):
            circuit.add_gate(GateType.H, targets=[q])

        # QAOA layers
        for layer in range(self.n_layers):
            beta = parameters[2*layer]
            gamma = parameters[2*layer + 1]

            # Problem Hamiltonian exp(-i γ H_C)
            self._apply_cost_hamiltonian(circuit, gamma)

            # Mixer Hamiltonian exp(-i β H_M)
            self._apply_mixer(circuit, beta)

        return circuit

    def _apply_cost_hamiltonian(self, circuit: UnifiedCircuit, gamma: float):
        """Apply problem Hamiltonian rotation."""
        for coeff, pauli_string in self.cost_hamiltonian:
            angle = 2 * gamma * coeff
            self._apply_pauli_rotation(circuit, pauli_string, angle)

    def _apply_pauli_rotation(
        self,
        circuit: UnifiedCircuit,
        pauli_string: str,
        angle: float
    ):
        """
        Apply exp(-i angle P/2) where P is Pauli string.

        Args:
            circuit: Circuit to modify
            pauli_string: Pauli string like "ZZ" or "XZIZ"
            angle: Rotation angle
        """
        # Find qubits with non-identity Paulis
        qubits = []
        paulis = []
        for q, pauli in enumerate(pauli_string):
            if pauli != 'I':
                qubits.append(q)
                paulis.append(pauli)

        if not qubits:
            return

        # Convert to Z basis
        for q, pauli in zip(qubits, paulis):
            if pauli == 'X':
                circuit.add_gate(GateType.H, targets=[q])
            elif pauli == 'Y':
                circuit.add_gate(GateType.RX, targets=[q], parameters={'angle': np.pi/2})

        # Apply ZZ...Z rotation using CNOTs
        if len(qubits) == 1:
            circuit.add_gate(GateType.RZ, targets=[qubits[0]], parameters={'angle': angle})
        else:
            # Entangle qubits
            for i in range(len(qubits) - 1):
                circuit.add_gate(GateType.CNOT, targets=[qubits[i], qubits[i+1]])

            # Rotate last qubit
            circuit.add_gate(GateType.RZ, targets=[qubits[-1]], parameters={'angle': angle})

            # Disentangle
            for i in range(len(qubits) - 2, -1, -1):
                circuit.add_gate(GateType.CNOT, targets=[qubits[i], qubits[i+1]])

        # Convert back from Z basis
        for q, pauli in zip(qubits, paulis):
            if pauli == 'X':
                circuit.add_gate(GateType.H, targets=[q])
            elif pauli == 'Y':
                circuit.add_gate(GateType.RX, targets=[q], parameters={'angle': -np.pi/2})

    def _apply_mixer(self, circuit: UnifiedCircuit, beta: float):
        """Apply mixer Hamiltonian."""
        if self.mixer == 'X':
            # Standard X mixer: exp(-i β Σ_i X_i)
            for q in range(self.n_qubits):
                circuit.add_gate(GateType.RX, targets=[q], parameters={'angle': 2*beta})
        else:
            raise ValueError(f"Unknown mixer: {self.mixer}")

    def _compute_cost(
        self,
        parameters: np.ndarray,
        backend: Optional[Any] = None,
        n_shots: int = 1000
    ) -> float:
        """
        Compute expectation value of cost Hamiltonian.

        Args:
            parameters: QAOA parameters
            backend: Quantum backend
            n_shots: Number of measurement shots

        Returns:
            Expected cost
        """
        circuit = self._create_circuit(parameters)

        # Get measurement results
        if backend is not None:
            counts = backend.execute(circuit, n_shots=n_shots)
        else:
            # Simple simulation
            counts = self._simulate_measurements(circuit, n_shots)

        # Compute expectation value
        expectation = 0.0
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            prob = count / total_shots
            cost = self._evaluate_cost(bitstring)
            expectation += prob * cost

        return expectation

    def _simulate_measurements(
        self,
        circuit: UnifiedCircuit,
        n_shots: int
    ) -> Dict[str, int]:
        """Simple measurement simulation."""
        # Get statevector
        state = self._simulate_circuit(circuit)

        # Sample from probability distribution
        probs = np.abs(state) ** 2
        samples = np.random.choice(len(state), size=n_shots, p=probs)

        # Convert to bitstrings
        counts = {}
        for sample in samples:
            bitstring = format(sample, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def _simulate_circuit(self, circuit: UnifiedCircuit) -> np.ndarray:
        """Simple statevector simulation."""
        n_qubits = circuit.n_qubits
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0

        for gate in circuit.gates:
            state = self._apply_gate(state, gate, n_qubits)

        return state

    def _apply_gate(self, state: np.ndarray, gate, n_qubits: int) -> np.ndarray:
        """Apply gate to statevector (simplified)."""
        if gate.gate_type == GateType.H:
            q = gate.targets[0]
            new_state = np.zeros_like(state)
            for i in range(len(state)):
                if (i >> q) & 1:
                    j = i ^ (1 << q)
                    new_state[i] = (state[j] - state[i]) / np.sqrt(2)
                else:
                    j = i ^ (1 << q)
                    new_state[i] = (state[i] + state[j]) / np.sqrt(2)
            return new_state

        elif gate.gate_type in (GateType.RX, GateType.RY, GateType.RZ):
            q = gate.targets[0]
            angle = gate.parameters['angle']

            if gate.gate_type == GateType.RZ:
                new_state = state.copy()
                for i in range(len(state)):
                    if (i >> q) & 1:
                        new_state[i] *= np.exp(-1j * angle / 2)
                    else:
                        new_state[i] *= np.exp(1j * angle / 2)
                return new_state

            elif gate.gate_type == GateType.RX:
                cos = np.cos(angle / 2)
                sin = np.sin(angle / 2)
                new_state = np.zeros_like(state)
                for i in range(len(state)):
                    j = i ^ (1 << q)
                    if (i >> q) & 1:
                        new_state[i] = cos * state[i] - 1j * sin * state[j]
                    else:
                        new_state[i] = cos * state[i] - 1j * sin * state[j]
                return new_state

            elif gate.gate_type == GateType.RY:
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
                if (i >> control) & 1:
                    j = i ^ (1 << target)
                    new_state[j] = state[i]
            return new_state

        return state

    def _evaluate_cost(self, bitstring: str) -> float:
        """Evaluate cost function for bitstring."""
        cost = 0.0

        for coeff, pauli_string in self.cost_hamiltonian:
            # Compute expectation of Pauli string in computational basis
            sign = 1.0
            for q, pauli in enumerate(pauli_string):
                if pauli == 'Z' and bitstring[q] == '1':
                    sign *= -1
                elif pauli in ('X', 'Y'):
                    # X and Y have zero expectation in Z basis
                    sign = 0.0
                    break

            cost += coeff * sign

        return cost

    def _extract_solution(
        self,
        circuit: UnifiedCircuit,
        backend: Optional[Any] = None,
        n_shots: int = 1000
    ) -> Tuple[str, float]:
        """Extract most probable solution."""
        if backend is not None:
            counts = backend.execute(circuit, n_shots=n_shots)
        else:
            counts = self._simulate_measurements(circuit, n_shots)

        # Find most common bitstring
        solution = max(counts, key=counts.get)
        probability = counts[solution] / sum(counts.values())

        return solution, probability


def create_maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> List[Tuple[float, str]]:
    """
    Create MaxCut problem Hamiltonian.

    Args:
        edges: List of graph edges (i, j)

    Returns:
        Hamiltonian as list of (coeff, pauli_string) tuples
    """
    # Determine number of qubits
    n_qubits = max(max(edge) for edge in edges) + 1

    hamiltonian = []
    for i, j in edges:
        # Each edge contributes 0.5 * (1 - Z_i Z_j)
        pauli_string = ['I'] * n_qubits
        pauli_string[i] = 'Z'
        pauli_string[j] = 'Z'
        hamiltonian.append((-0.5, ''.join(pauli_string)))

    # Constant term (can be omitted for optimization)
    # hamiltonian.append((0.5 * len(edges), 'I' * n_qubits))

    return hamiltonian


def create_partition_hamiltonian(values: List[float]) -> List[Tuple[float, str]]:
    """
    Create Number Partition problem Hamiltonian.

    Minimize |Σ_i v_i z_i|² where z_i ∈ {-1, +1}

    Args:
        values: List of values to partition

    Returns:
        Hamiltonian as list of (coeff, pauli_string) tuples
    """
    n_qubits = len(values)
    hamiltonian = []

    # Quadratic terms: Σ_i Σ_j v_i v_j Z_i Z_j
    for i in range(n_qubits):
        for j in range(i, n_qubits):
            coeff = values[i] * values[j]
            if i == j:
                # Diagonal term (constant)
                continue
            else:
                pauli_string = ['I'] * n_qubits
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                hamiltonian.append((coeff, ''.join(pauli_string)))

    return hamiltonian
