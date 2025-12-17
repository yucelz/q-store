"""
Comprehensive Test Suite for Quantum Backends
Tests backend abstraction, adapters, and backend manager
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from q_store.backends.quantum_backend_interface import (
    QuantumBackend,
    QuantumCircuit,
    QuantumGate,
    ExecutionResult,
    BackendCapabilities,
    BackendType,
    GateType,
    CircuitBuilder,
    amplitude_encode_to_circuit,
    create_bell_state_circuit,
    create_ghz_state_circuit,
)
from q_store.backends.backend_manager import (
    BackendManager,
    MockQuantumBackend,
    create_default_backend_manager,
)
from q_store.backends.ionq_backend import IonQQuantumBackend


class TestQuantumBackendInterface:
    """Test quantum backend interface and utilities"""

    def test_quantum_gate_creation(self):
        """Test quantum gate construction"""
        gate = QuantumGate(
            gate_type=GateType.HADAMARD,
            qubits=[0],
            parameters=None
        )

        assert gate.gate_type == GateType.HADAMARD
        assert gate.qubits == [0]
        assert gate.parameters is None

    def test_quantum_circuit_creation(self):
        """Test quantum circuit creation"""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.gates = [
            QuantumGate(GateType.HADAMARD, [0]),
            QuantumGate(GateType.CNOT, [0, 1])
        ]

        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 2
        assert circuit.gates[0].gate_type == GateType.HADAMARD

    def test_execution_result_creation(self):
        """Test execution result structure"""
        result = ExecutionResult(
            probabilities={'00': 0.5, '11': 0.5},
            measured_states=['00', '11', '00', '11'],
            metadata={'shots': 4}
        )

        assert result.probabilities['00'] == 0.5
        assert len(result.measured_states) == 4
        assert result.metadata['shots'] == 4

    def test_backend_capabilities(self):
        """Test backend capabilities structure"""
        caps = BackendCapabilities(
            backend_type=BackendType.SIMULATOR,
            supported_gates=[GateType.HADAMARD, GateType.CNOT],
            max_qubits=10,
            supports_measurements=True
        )

        assert caps.backend_type == BackendType.SIMULATOR
        assert GateType.HADAMARD in caps.supported_gates
        assert caps.max_qubits == 10

    def test_circuit_builder_hadamard(self):
        """Test circuit builder Hadamard gate"""
        builder = CircuitBuilder(num_qubits=2)
        builder.h(0)

        circuit = builder.build()
        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 1
        assert circuit.gates[0].gate_type == GateType.HADAMARD
        assert circuit.gates[0].qubits == [0]

    def test_circuit_builder_cnot(self):
        """Test circuit builder CNOT gate"""
        builder = CircuitBuilder(num_qubits=2)
        builder.cnot(0, 1)

        circuit = builder.build()
        assert len(circuit.gates) == 1
        assert circuit.gates[0].gate_type == GateType.CNOT
        assert circuit.gates[0].qubits == [0, 1]

    def test_circuit_builder_rotation(self):
        """Test circuit builder rotation gates"""
        builder = CircuitBuilder(num_qubits=1)
        builder.rx(0, np.pi/4)
        builder.ry(0, np.pi/3)
        builder.rz(0, np.pi/2)

        circuit = builder.build()
        assert len(circuit.gates) == 3
        assert circuit.gates[0].gate_type == GateType.RX
        assert circuit.gates[1].gate_type == GateType.RY
        assert circuit.gates[2].gate_type == GateType.RZ
        assert abs(circuit.gates[0].parameters[0] - np.pi/4) < 1e-10

    def test_circuit_builder_measure(self):
        """Test circuit builder measurement"""
        builder = CircuitBuilder(num_qubits=2)
        builder.h(0)
        builder.measure([0, 1])

        circuit = builder.build()
        assert circuit.gates[-1].gate_type == GateType.MEASURE

    def test_amplitude_encode_circuit(self):
        """Test amplitude encoding utility"""
        data = np.array([1, 0, 0, 0])  # |00> state
        circuit = amplitude_encode_to_circuit(data)

        assert circuit.num_qubits == 2  # 4 amplitudes = 2 qubits
        assert len(circuit.gates) > 0

    def test_bell_state_circuit(self):
        """Test Bell state creation"""
        circuit = create_bell_state_circuit()

        assert circuit.num_qubits == 2
        # Should have H and CNOT
        gate_types = [g.gate_type for g in circuit.gates]
        assert GateType.HADAMARD in gate_types
        assert GateType.CNOT in gate_types

    def test_ghz_state_circuit(self):
        """Test GHZ state creation"""
        circuit = create_ghz_state_circuit(num_qubits=3)

        assert circuit.num_qubits == 3
        # Should have H on first qubit and CNOTs
        gate_types = [g.gate_type for g in circuit.gates]
        assert GateType.HADAMARD in gate_types
        assert GateType.CNOT in gate_types


class TestMockQuantumBackend:
    """Test mock quantum backend"""

    def test_mock_backend_creation(self):
        """Test mock backend initialization"""
        backend = MockQuantumBackend(max_qubits=10)

        assert backend.capabilities.backend_type == BackendType.SIMULATOR
        assert backend.capabilities.max_qubits == 10

    def test_mock_backend_execute(self):
        """Test mock backend execution"""
        backend = MockQuantumBackend()
        circuit = create_bell_state_circuit()

        result = backend.execute(circuit, shots=100)

        assert isinstance(result, ExecutionResult)
        assert len(result.probabilities) > 0
        assert sum(result.probabilities.values()) > 0.99  # Probabilities sum to ~1

    def test_mock_backend_statevector(self):
        """Test mock backend statevector simulation"""
        backend = MockQuantumBackend()

        # Simple circuit: H|0>
        builder = CircuitBuilder(num_qubits=1)
        builder.h(0)
        circuit = builder.build()

        statevector = backend.get_statevector(circuit)

        assert len(statevector) == 2  # 2^1 states
        # Should be |+> = (|0> + |1>)/sqrt(2)
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        assert np.allclose(np.abs(statevector), np.abs(expected), atol=1e-6)

    def test_mock_backend_expectation(self):
        """Test mock backend expectation value"""
        backend = MockQuantumBackend()
        circuit = create_bell_state_circuit()

        # Z on qubit 0
        observable = np.array([[1, 0], [0, -1]])

        expectation = backend.compute_expectation(circuit, observable, qubit=0)

        assert isinstance(expectation, (int, float, complex))


class TestBackendManager:
    """Test backend manager"""

    def test_backend_manager_creation(self):
        """Test backend manager initialization"""
        manager = BackendManager()

        assert manager is not None
        assert hasattr(manager, 'backends')

    def test_register_backend(self):
        """Test registering a backend"""
        manager = BackendManager()
        mock_backend = MockQuantumBackend()

        manager.register_backend('test_mock', mock_backend)

        assert 'test_mock' in manager.backends
        assert manager.backends['test_mock'] == mock_backend

    def test_get_backend(self):
        """Test retrieving a backend"""
        manager = BackendManager()
        mock_backend = MockQuantumBackend()
        manager.register_backend('test_backend', mock_backend)

        retrieved = manager.get_backend('test_backend')

        assert retrieved == mock_backend

    def test_get_nonexistent_backend(self):
        """Test getting backend that doesn't exist"""
        manager = BackendManager()

        result = manager.get_backend('nonexistent')

        assert result is None

    def test_list_backends(self):
        """Test listing available backends"""
        manager = BackendManager()
        manager.register_backend('backend1', MockQuantumBackend())
        manager.register_backend('backend2', MockQuantumBackend())

        backends = manager.list_backends()

        assert 'backend1' in backends
        assert 'backend2' in backends
        assert len(backends) >= 2

    def test_create_default_backend_manager(self):
        """Test creating default backend manager"""
        manager = create_default_backend_manager()

        assert manager is not None
        # Should have at least a mock backend
        backends = manager.list_backends()
        assert len(backends) > 0


class TestIonQBackend:
    """Test IonQ backend (with mocking)"""

    @patch('q_store.backends.ionq_backend.requests.post')
    def test_ionq_backend_creation(self, mock_post):
        """Test IonQ backend initialization"""
        backend = IonQQuantumBackend(api_key='test_key')

        assert backend.api_key == 'test_key'
        assert backend.capabilities.backend_type == BackendType.HARDWARE

    @patch('q_store.backends.ionq_backend.requests.post')
    def test_ionq_backend_execute_mocked(self, mock_post):
        """Test IonQ backend execution with mocked API"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'job_123',
            'status': 'completed',
            'data': {
                'histogram': {'0': 0.5, '1': 0.5}
            }
        }
        mock_post.return_value = mock_response

        backend = IonQQuantumBackend(api_key='test_key')
        circuit = create_bell_state_circuit()

        with patch.object(backend, '_wait_for_job') as mock_wait:
            mock_wait.return_value = mock_response.json()
            result = backend.execute(circuit, shots=100)

        assert isinstance(result, ExecutionResult)


class TestCirqAdapter:
    """Test Cirq adapter (if available)"""

    @pytest.mark.skipif(
        True,  # Skip by default unless Cirq is installed
        reason="Cirq adapter tests require Cirq installation"
    )
    def test_cirq_adapter_import(self):
        """Test Cirq adapter can be imported"""
        try:
            from q_store.backends.cirq_ionq_adapter import CirqIonQAdapter
            assert CirqIonQAdapter is not None
        except ImportError:
            pytest.skip("Cirq not installed")


class TestQiskitAdapter:
    """Test Qiskit adapter (if available)"""

    @pytest.mark.skipif(
        True,  # Skip by default unless Qiskit is installed
        reason="Qiskit adapter tests require Qiskit installation"
    )
    def test_qiskit_adapter_import(self):
        """Test Qiskit adapter can be imported"""
        try:
            from q_store.backends.qiskit_ionq_adapter import QiskitIonQAdapter
            assert QiskitIonQAdapter is not None
        except ImportError:
            pytest.skip("Qiskit not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
