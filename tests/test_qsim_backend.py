"""
Tests for qsim high-performance quantum simulator backend.
"""

import pytest
import numpy as np

# Check if qsim is available
try:
    import qsimcirq
    HAS_QSIM = True
except ImportError:
    HAS_QSIM = False

from q_store.core import UnifiedCircuit, GateType
from q_store.backends import HAS_QSIM as BACKEND_HAS_QSIM

if HAS_QSIM:
    from q_store.backends.qsim_backend import QsimBackend, create_qsim_backend


@pytest.mark.skipif(not HAS_QSIM, reason="qsim not installed")
class TestQsimBackend:
    """Test suite for qsim backend."""

    def test_backend_initialization(self):
        """Test qsim backend can be created."""
        backend = QsimBackend(num_threads=2)
        assert backend is not None
        assert backend._initialized

        caps = backend.get_capabilities()
        assert caps.backend_type.value == 'simulator'
        assert caps.supports_state_vector
        assert caps.max_qubits >= 20

    def test_factory_function(self):
        """Test create_qsim_backend factory."""
        backend = create_qsim_backend(num_threads=1)
        assert isinstance(backend, QsimBackend)
        assert backend.num_threads == 1

    def test_simple_circuit_execution(self):
        """Test executing a simple quantum circuit."""
        backend = QsimBackend(num_threads=2)

        # Create simple 2-qubit circuit: H-CNOT (Bell state)
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.h(0)
        circuit.cnot(0, 1)

        result = backend.execute(circuit, shots=1000)

        # Bell state should give |00⟩ or |11⟩ with ~50% probability each
        assert result.samples.shape == (1000, 2)
        assert '00' in result.counts
        assert '11' in result.counts

        # Check probabilities are roughly 50/50
        prob_00 = result.probabilities.get('00', 0)
        prob_11 = result.probabilities.get('11', 0)
        assert 0.4 < prob_00 < 0.6
        assert 0.4 < prob_11 < 0.6

    def test_state_vector(self):
        """Test state vector retrieval."""
        backend = QsimBackend()

        # Create |+⟩ state (Hadamard on |0⟩)
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.h(0)

        state = backend.get_state_vector(circuit)

        # |+⟩ = (|0⟩ + |1⟩) / √2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_allclose(state, expected, atol=1e-6)

    def test_parameterized_circuit(self):
        """Test circuit with parameters."""
        backend = QsimBackend()

        # Create parameterized rotation circuit
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.rx(0, 'theta')

        # Execute with theta = π (should give |1⟩)
        result = backend.execute(circuit, shots=1000, parameters={'theta': np.pi})

        # Should get mostly |1⟩
        prob_1 = result.probabilities.get('1', 0)
        assert prob_1 > 0.95

    def test_batch_execution(self):
        """Test batch circuit execution."""
        backend = QsimBackend(num_threads=2)

        # Create multiple simple circuits
        circuits = []
        for i in range(3):
            circuit = UnifiedCircuit(n_qubits=1)
            if i % 2 == 0:
                circuit.x(0)  # |1⟩
            # else leave as |0⟩
            circuits.append(circuit)

        results = backend.execute_batch(circuits, shots=100)

        assert len(results) == 3
        # First circuit should give |1⟩
        assert results[0].probabilities.get('1', 0) > 0.95
        # Second should give |0⟩
        assert results[1].probabilities.get('0', 0) > 0.95
        # Third should give |1⟩
        assert results[2].probabilities.get('1', 0) > 0.95

    def test_multi_qubit_circuit(self):
        """Test larger multi-qubit circuit."""
        backend = QsimBackend()

        # Create 4-qubit GHZ state
        circuit = UnifiedCircuit(n_qubits=4)
        circuit.h(0)
        for i in range(3):
            circuit.cnot(i, i+1)

        result = backend.execute(circuit, shots=1000)

        # GHZ state should give |0000⟩ or |1111⟩
        assert '0000' in result.counts
        assert '1111' in result.counts

        # Most counts should be in these two states
        total_ghz = result.counts.get('0000', 0) + result.counts.get('1111', 0)
        assert total_ghz > 950  # >95% in GHZ states

    def test_expectation_value(self):
        """Test expectation value computation."""
        backend = QsimBackend()

        # Create |+⟩ state
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.h(0)

        # Compute <Z> for |+⟩ (should be 0)
        expectation = backend.compute_expectation(circuit, 'Z')
        assert abs(expectation) < 0.01

    def test_reset_and_close(self):
        """Test backend reset and cleanup."""
        backend = QsimBackend()

        # Reset should work
        backend.reset()
        assert backend._initialized

        # Close should work
        backend.close()
        assert not backend._initialized

    def test_thread_configuration(self):
        """Test different thread configurations."""
        # Auto-detect threads
        backend_auto = QsimBackend()
        assert backend_auto.num_threads is None

        # Explicit thread count
        backend_4 = QsimBackend(num_threads=4)
        assert backend_4.num_threads == 4

        # Both should work
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.h(0)

        result_auto = backend_auto.execute(circuit, shots=100)
        result_4 = backend_4.execute(circuit, shots=100)

        assert result_auto.samples.shape == (100, 2)
        assert result_4.samples.shape == (100, 2)


@pytest.mark.skipif(not HAS_QSIM, reason="qsim not installed")
class TestQsimPerformance:
    """Performance-focused tests for qsim."""

    def test_moderate_circuit_performance(self):
        """Test qsim performance on moderate-sized circuit."""
        import time

        backend = QsimBackend(num_threads=2)

        # Create 10-qubit circuit with depth 20
        circuit = UnifiedCircuit(n_qubits=10)
        for layer in range(20):
            for q in range(10):
                circuit.ry(q, f'theta_{layer}_{q}')
            for q in range(0, 9, 2):
                circuit.cnot(q, q+1)

        # Bind parameters
        parameters = {f'theta_{layer}_{q}': np.random.rand()
                     for layer in range(20) for q in range(10)}

        # Measure execution time
        start = time.time()
        result = backend.execute(circuit, shots=1000, parameters=parameters)
        elapsed = time.time() - start

        # Should complete reasonably fast (<5 seconds for 10 qubits)
        assert elapsed < 5.0
        assert result.samples.shape == (1000, 10)

        # Log performance
        print(f"\\n10-qubit, depth-20 circuit: {elapsed:.3f}s for 1000 shots")
        print(f"Throughput: {1000/elapsed:.1f} shots/second")


def test_qsim_availability():
    """Test that qsim availability is correctly detected."""
    # This test always runs to verify import detection
    if HAS_QSIM:
        assert BACKEND_HAS_QSIM
        assert QsimBackend is not None
    else:
        # qsim not installed - this is fine, just skip other tests
        pytest.skip("qsim not installed - this is expected")
