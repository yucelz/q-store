"""
Tests for IonQ Hardware Backend.

Comprehensive test suite covering:
- Backend initialization
- Native gate compilation
- Circuit execution (mocked)
- Queue management
- Cost estimation
- Error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time

from q_store.core import UnifiedCircuit, GateType
from q_store.backends.ionq_hardware_backend import (
    IonQHardwareBackend,
    create_ionq_backend
)
from q_store.backends.quantum_backend_interface import (
    BackendType,
    ExecutionResult
)


@pytest.fixture
def mock_ionq_service():
    """Mock cirq_ionq.Service for testing without API access."""
    with patch('q_store.backends.ionq_hardware_backend.ionq') as mock_ionq:
        # Mock service
        service = Mock()
        mock_ionq.Service.return_value = service

        # Mock job
        job = Mock()
        job.job_id.return_value = 'test_job_123'
        job.status.return_value = 'completed'

        # Mock results
        result = Mock()
        measurements = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        result.measurements = {'result': measurements}
        job.results.return_value = result

        service.run.return_value = job

        yield service


class TestIonQHardwareBackendInitialization:
    """Test backend initialization."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_backend_creation_simulator(self, mock_ionq):
        """Test creating backend for simulator."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(
            api_key='test_key',
            target='simulator'
        )

        assert backend.api_key == 'test_key'
        assert backend.target == 'simulator'
        assert backend.use_native_gates == True
        assert backend._initialized == True

        # Service should be initialized
        mock_ionq.Service.assert_called_once_with(
            api_key='test_key',
            default_target='simulator'
        )

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_backend_creation_aria(self, mock_ionq):
        """Test creating backend for Aria QPU."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(
            api_key='test_key',
            target='qpu.aria-1',
            timeout=600
        )

        assert backend.target == 'qpu.aria-1'
        assert backend.timeout == 600

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', False)
    def test_backend_creation_without_cirq_ionq(self):
        """Test error when cirq-ionq not installed."""
        with pytest.raises(ImportError, match='cirq-ionq is required'):
            IonQHardwareBackend(api_key='test_key')


class TestIonQBackendCapabilities:
    """Test backend capabilities reporting."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_capabilities_simulator(self, mock_ionq):
        """Test capabilities for simulator target."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='simulator')
        caps = backend.get_capabilities()

        assert caps.backend_type == BackendType.SIMULATOR
        assert caps.max_qubits == 29
        assert GateType.GPI in caps.native_gate_set
        assert GateType.GPI2 in caps.native_gate_set
        assert GateType.MS in caps.native_gate_set

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_capabilities_aria_qpu(self, mock_ionq):
        """Test capabilities for Aria QPU."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='qpu.aria-1')
        caps = backend.get_capabilities()

        assert caps.backend_type == BackendType.QPU
        assert caps.max_qubits == 25

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_capabilities_forte_qpu(self, mock_ionq):
        """Test capabilities for Forte QPU."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='qpu.forte-1')
        caps = backend.get_capabilities()

        assert caps.backend_type == BackendType.QPU
        assert caps.max_qubits == 32

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_supported_gates(self, mock_ionq):
        """Test that all expected gates are supported."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')
        caps = backend.get_capabilities()

        expected_gates = [
            GateType.H, GateType.X, GateType.Y, GateType.Z,
            GateType.RX, GateType.RY, GateType.RZ,
            GateType.CNOT, GateType.CZ,
            GateType.GPI, GateType.GPI2, GateType.MS
        ]

        for gate in expected_gates:
            assert gate in caps.supported_gates


class TestNativeGateCompilation:
    """Test native gate compilation."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_compile_hadamard(self, mock_ionq):
        """Test Hadamard decomposition to native gates."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])

        native = backend._compile_to_native_gates(circuit)

        # H = GPI2(0) · GPI(π)
        assert len(native.gates) == 2
        assert native.gates[0].gate_type == GateType.GPI2
        assert native.gates[1].gate_type == GateType.GPI

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_compile_pauli_x(self, mock_ionq):
        """Test Pauli-X decomposition."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])

        native = backend._compile_to_native_gates(circuit)

        # X = GPI(0)
        assert len(native.gates) == 1
        assert native.gates[0].gate_type == GateType.GPI
        assert native.gates[0].parameters['phi'] == 0.0

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_compile_cnot(self, mock_ionq):
        """Test CNOT decomposition using MS gate."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        native = backend._compile_to_native_gates(circuit)

        # CNOT uses GPI2, MS, GPI2, GPI
        assert len(native.gates) == 4
        assert any(g.gate_type == GateType.MS for g in native.gates)

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_compile_rotation_gates(self, mock_ionq):
        """Test rotation gate decompositions."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        # RX gate
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RX, targets=[0], parameters={'angle': np.pi/4})
        native = backend._compile_to_native_gates(circuit)
        assert len(native.gates) == 3  # GPI2 + GPI + GPI2

        # RY gate
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': np.pi/4})
        native = backend._compile_to_native_gates(circuit)
        assert len(native.gates) == 3  # GPI2 + GPI + GPI2

        # RZ gate
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RZ, targets=[0], parameters={'angle': np.pi/4})
        native = backend._compile_to_native_gates(circuit)
        assert len(native.gates) == 1  # Just GPI

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_native_gates_passthrough(self, mock_ionq):
        """Test that native gates pass through unchanged."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.GPI, targets=[0], parameters={'phi': np.pi/2})
        circuit.add_gate(GateType.GPI2, targets=[1], parameters={'phi': 0.0})
        circuit.add_gate(GateType.MS, targets=[0, 1], parameters={'phi0': 0.0, 'phi1': 0.0, 'theta': np.pi/4})

        native = backend._compile_to_native_gates(circuit)

        # Should be unchanged
        assert len(native.gates) == 3
        assert native.gates[0].gate_type == GateType.GPI
        assert native.gates[1].gate_type == GateType.GPI2
        assert native.gates[2].gate_type == GateType.MS


class TestCircuitExecution:
    """Test circuit execution with mocked IonQ service."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    @patch('q_store.backends.ionq_hardware_backend.cirq')
    def test_execute_simple_circuit(self, mock_cirq, mock_ionq):
        """Test executing a simple circuit."""
        # Setup mocks
        service_mock = Mock()
        job_mock = Mock()
        result_mock = Mock()

        job_mock.job_id.return_value = 'job_123'
        job_mock.status.return_value = 'completed'

        measurements = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        result_mock.measurements = {'result': measurements}
        job_mock.results.return_value = result_mock

        service_mock.run.return_value = job_mock
        mock_ionq.Service.return_value = service_mock

        # Mock Cirq circuit conversion
        cirq_circuit_mock = Mock()
        cirq_circuit_mock.all_qubits.return_value = [Mock(), Mock()]
        mock_cirq.Circuit.return_value = cirq_circuit_mock
        mock_cirq.MeasurementGate = type('MeasurementGate', (), {})

        # Create backend and circuit
        backend = IonQHardwareBackend(api_key='test_key', target='simulator', timeout=10)

        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        # Mock to_cirq conversion
        with patch.object(circuit, 'to_cirq', return_value=cirq_circuit_mock):
            result = backend.execute(circuit, shots=4)

        # Verify result
        assert isinstance(result, ExecutionResult)
        assert result.total_shots == 4
        assert '00' in result.counts
        assert '11' in result.counts
        assert result.counts['00'] == 2
        assert result.counts['11'] == 2
        assert abs(result.probabilities['00'] - 0.5) < 0.01
        assert abs(result.probabilities['11'] - 0.5) < 0.01

        # Verify metadata
        assert result.metadata['backend'] == 'ionq'
        assert result.metadata['job_id'] == 'job_123'

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_execute_with_timeout(self, mock_ionq):
        """Test timeout handling."""
        service_mock = Mock()
        job_mock = Mock()

        job_mock.job_id.return_value = 'job_456'
        job_mock.status.return_value = 'running'  # Never completes

        service_mock.run.return_value = job_mock
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(
            api_key='test_key',
            timeout=1,  # 1 second timeout
            poll_interval=0.1
        )

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])

        with pytest.raises(TimeoutError, match='did not complete within'):
            backend.execute(circuit, shots=100)

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_execute_with_job_failure(self, mock_ionq):
        """Test handling of failed jobs."""
        service_mock = Mock()
        job_mock = Mock()

        job_mock.job_id.return_value = 'job_789'
        job_mock.status.return_value = 'failed'

        service_mock.run.return_value = job_mock
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')

        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])

        with pytest.raises(RuntimeError, match='job_789 failed'):
            backend.execute(circuit, shots=100)


class TestCostEstimation:
    """Test cost estimation."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_simulator_cost(self, mock_ionq):
        """Test that simulator has zero cost."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='simulator')
        cost = backend.estimate_cost(1000)

        assert cost == 0.0

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_aria_cost(self, mock_ionq):
        """Test Aria QPU cost estimation."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='qpu.aria-1')
        cost = backend.estimate_cost(1000)

        # Should be 1000 * 0.0003 = $0.30
        assert cost == 0.30

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_forte_cost(self, mock_ionq):
        """Test Forte QPU cost estimation."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key', target='qpu.forte-1')
        cost = backend.estimate_cost(1000)

        # Should be 1000 * 0.00035 = $0.35
        assert cost == 0.35


class TestBatchExecution:
    """Test batch circuit execution."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    @patch('q_store.backends.ionq_hardware_backend.cirq')
    def test_execute_batch(self, mock_cirq, mock_ionq):
        """Test batch execution (sequential)."""
        # Setup mocks
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        # Mock Cirq
        cirq_circuit_mock = Mock()
        cirq_circuit_mock.all_qubits.return_value = [Mock()]
        mock_cirq.Circuit.return_value = cirq_circuit_mock
        mock_cirq.MeasurementGate = type('MeasurementGate', (), {})

        # Mock job results
        def create_job():
            job = Mock()
            job.job_id.return_value = f'job_{np.random.randint(1000)}'
            job.status.return_value = 'completed'
            result = Mock()
            result.measurements = {'result': np.array([[0], [1]])}
            job.results.return_value = result
            return job

        service_mock.run.side_effect = [create_job() for _ in range(3)]

        backend = IonQHardwareBackend(api_key='test_key')

        # Create batch of circuits
        circuits = [UnifiedCircuit(n_qubits=1) for _ in range(3)]
        for circuit in circuits:
            circuit.add_gate(GateType.H, targets=[0])

        # Mock to_cirq for all circuits
        for circuit in circuits:
            circuit.to_cirq = Mock(return_value=cirq_circuit_mock)

        results = backend.execute_batch(circuits, shots=2)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExecutionResult)
            assert result.total_shots == 2


class TestUtilityMethods:
    """Test utility methods."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    @patch('q_store.backends.ionq_hardware_backend.cirq')
    def test_job_history(self, mock_cirq, mock_ionq):
        """Test job history tracking."""
        service_mock = Mock()
        job_mock = Mock()
        job_mock.job_id.return_value = 'test_job_999'
        job_mock.status.return_value = 'completed'

        result_mock = Mock()
        result_mock.measurements = {'result': np.array([[0]])}
        job_mock.results.return_value = result_mock

        service_mock.run.return_value = job_mock
        mock_ionq.Service.return_value = service_mock
        
        # Mock Cirq
        cirq_circuit_mock = Mock()
        cirq_circuit_mock.all_qubits.return_value = [Mock()]
        mock_cirq.Circuit.return_value = cirq_circuit_mock
        mock_cirq.MeasurementGate = type('MeasurementGate', (), {})

        backend = IonQHardwareBackend(api_key='test_key')

        # Initially empty
        assert len(backend.get_job_history()) == 0

        # Execute circuits (mocked)
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])

        with patch.object(circuit, 'to_cirq', return_value=cirq_circuit_mock):
            backend.execute(circuit, shots=1)

        # Should have one job
        history = backend.get_job_history()
        assert len(history) == 1
        assert 'test_job_999' in history

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_reset(self, mock_ionq):
        """Test backend reset."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')
        backend._job_history = ['job1', 'job2']

        backend.reset()

        assert len(backend.get_job_history()) == 0

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_close(self, mock_ionq):
        """Test backend close."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = IonQHardwareBackend(api_key='test_key')
        assert backend._initialized == True

        backend.close()

        assert backend._initialized == False


class TestFactoryFunction:
    """Test factory function."""

    @patch('q_store.backends.ionq_hardware_backend.HAS_IONQ', True)
    @patch('q_store.backends.ionq_hardware_backend.ionq')
    def test_create_ionq_backend(self, mock_ionq):
        """Test create_ionq_backend factory function."""
        service_mock = Mock()
        mock_ionq.Service.return_value = service_mock

        backend = create_ionq_backend(
            api_key='test_key',
            target='qpu.aria-2',
            use_native_gates=False,
            timeout=500
        )

        assert isinstance(backend, IonQHardwareBackend)
        assert backend.api_key == 'test_key'
        assert backend.target == 'qpu.aria-2'
        assert backend.use_native_gates == False
        assert backend.timeout == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
