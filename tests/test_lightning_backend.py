"""
Tests for PennyLane Lightning GPU-accelerated quantum simulator backend.
"""

import pytest
import numpy as np

# Check if PennyLane is available
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

from q_store.core import UnifiedCircuit, GateType
from q_store.backends import HAS_LIGHTNING

if HAS_PENNYLANE:
    from q_store.backends.lightning_backend import LightningBackend, create_lightning_backend


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestLightningBackend:
    """Test suite for Lightning backend."""
    
    def test_backend_initialization(self):
        """Test Lightning backend can be created."""
        backend = LightningBackend(device='lightning.qubit')
        assert backend is not None
        assert backend._initialized
        
        caps = backend.get_capabilities()
        assert caps.backend_type.value == 'simulator'
        assert caps.supports_state_vector
        assert caps.max_qubits >= 20
    
    def test_factory_function(self):
        """Test create_lightning_backend factory."""
        backend = create_lightning_backend(use_gpu=False)
        assert isinstance(backend, LightningBackend)
        assert backend.device_name == 'lightning.qubit'
    
    def test_gpu_fallback(self):
        """Test automatic GPU fallback to CPU."""
        # Try to create GPU backend (will fallback to CPU if GPU unavailable)
        backend = LightningBackend(device='lightning.gpu')
        assert backend is not None
        
        # Should either be GPU or have fallen back to CPU
        assert backend.device_name in ['lightning.gpu', 'lightning.qubit']
    
    def test_simple_circuit_execution(self):
        """Test executing a simple quantum circuit."""
        backend = LightningBackend(device='lightning.qubit')
        
        # Create simple 2-qubit circuit: H-CNOT (Bell state)
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        result = backend.execute(circuit, shots=1000)
        
        # Bell state should give |00⟩ or |11⟩ with ~50% probability each
        assert result.samples.shape[0] == 1000
        assert '00' in result.counts
        assert '11' in result.counts
        
        # Check probabilities are roughly 50/50
        prob_00 = result.probabilities.get('00', 0)
        prob_11 = result.probabilities.get('11', 0)
        assert 0.3 < prob_00 < 0.7  # Allow more variance than qsim
        assert 0.3 < prob_11 < 0.7
    
    def test_state_vector(self):
        """Test state vector retrieval."""
        backend = LightningBackend(device='lightning.qubit')
        
        # Create |+⟩ state (Hadamard on |0⟩)
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.h(0)
        
        state = backend.get_state_vector(circuit)
        
        # |+⟩ = (|0⟩ + |1⟩) / √2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_allclose(np.abs(state), np.abs(expected), atol=1e-6)
    
    def test_parameterized_circuit(self):
        """Test circuit with parameters."""
        backend = LightningBackend(device='lightning.qubit')
        
        # Create parameterized rotation circuit
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.rx(0, 'theta')
        
        # Execute with theta = π (should give |1⟩)
        result = backend.execute(circuit, shots=1000, parameters={'theta': np.pi})
        
        # Should get mostly |1⟩
        prob_1 = result.probabilities.get('1', 0)
        assert prob_1 > 0.90
    
    def test_batch_execution(self):
        """Test batch circuit execution."""
        backend = LightningBackend(device='lightning.qubit')
        
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
        backend = LightningBackend(device='lightning.qubit')
        
        # Create 4-qubit GHZ state
        circuit = UnifiedCircuit(n_qubits=4)
        circuit.h(0)
        for i in range(3):
            circuit.cnot(i, i+1)
        
        result = backend.execute(circuit, shots=1000)
        
        # GHZ state should give |0000⟩ or |1111⟩
        assert '0000' in result.counts or '1111' in result.counts
        
        # Most counts should be in these two states
        total_ghz = result.counts.get('0000', 0) + result.counts.get('1111', 0)
        assert total_ghz > 900  # >90% in GHZ states
    
    def test_expectation_value(self):
        """Test expectation value computation."""
        backend = LightningBackend(device='lightning.qubit')
        
        # Create |+⟩ state
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.h(0)
        
        # Compute <Z> for |+⟩ (should be 0)
        expectation = backend.compute_expectation(circuit, 'Z')
        assert abs(expectation) < 0.01
    
    def test_reset_and_close(self):
        """Test backend reset and cleanup."""
        backend = LightningBackend(device='lightning.qubit')
        
        # Reset should work
        backend.reset()
        assert backend._initialized
        
        # Close should work
        backend.close()
        assert not backend._initialized
    
    def test_metadata(self):
        """Test that execution metadata is correct."""
        backend = LightningBackend(device='lightning.qubit')
        
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.h(0)
        
        result = backend.execute(circuit, shots=500)
        
        assert result.metadata['backend'] == 'lightning'
        assert result.metadata['device'] == 'lightning.qubit'
        assert result.metadata['shots'] == 500
        assert result.metadata['num_qubits'] == 2
        assert 'gpu_used' in result.metadata


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
class TestLightningPerformance:
    """Performance-focused tests for Lightning."""
    
    def test_moderate_circuit_performance(self):
        """Test Lightning performance on moderate-sized circuit."""
        import time
        
        backend = LightningBackend(device='lightning.qubit')
        
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
        
        # Should complete reasonably fast (<10 seconds for CPU)
        assert elapsed < 10.0
        assert result.samples.shape[0] == 1000
        
        # Log performance
        device_type = "GPU" if backend._gpu_available else "CPU"
        print(f"\\n10-qubit, depth-20 circuit ({device_type}): {elapsed:.3f}s for 1000 shots")
        print(f"Throughput: {1000/elapsed:.1f} shots/second")


def test_lightning_availability():
    """Test that Lightning availability is correctly detected."""
    # This test always runs to verify import detection
    if HAS_PENNYLANE:
        assert HAS_LIGHTNING
        assert LightningBackend is not None
    else:
        # PennyLane not installed - this is fine, just skip other tests
        pytest.skip("PennyLane not installed - this is expected")
