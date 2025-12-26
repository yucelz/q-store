"""
Tests for UnifiedCircuit core component
"""

import pytest
import json
import numpy as np

from q_store.core import (
    UnifiedCircuit,
    Gate,
    GateType,
    Parameter,
    CircuitOptimizer,
    optimize
)


class TestGate:
    """Test Gate class"""

    def test_gate_creation(self):
        """Test basic gate creation"""
        gate = Gate(gate_type=GateType.H, targets=[0])
        assert gate.gate_type == GateType.H
        assert gate.targets == [0]
        assert gate.controls is None
        assert gate.parameters is None

    def test_gate_with_parameters(self):
        """Test parameterized gate"""
        param = Parameter(name='theta', value=np.pi/4)
        gate = Gate(
            gate_type=GateType.RY,
            targets=[0],
            parameters={'angle': param}
        )
        assert gate.parameters['angle'].name == 'theta'
        assert gate.parameters['angle'].value == np.pi/4

    def test_gate_validation(self):
        """Test gate validation"""
        # Should raise for wrong number of targets
        with pytest.raises(ValueError):
            Gate(gate_type=GateType.CNOT, targets=[0])  # CNOT needs 2 targets

    def test_gate_serialization(self):
        """Test gate to/from dict"""
        gate = Gate(
            gate_type=GateType.RX,
            targets=[0],
            parameters={'angle': 0.5}
        )
        gate_dict = gate.to_dict()
        restored = Gate.from_dict(gate_dict)

        assert restored.gate_type == gate.gate_type
        assert restored.targets == gate.targets
        assert restored.parameters['angle'] == 0.5


class TestParameter:
    """Test Parameter class"""

    def test_numeric_parameter(self):
        """Test numeric parameter"""
        param = Parameter(name='theta', value=1.5, is_symbolic=False)
        assert param.value == 1.5
        assert not param.is_symbolic

    def test_symbolic_parameter(self):
        """Test symbolic parameter"""
        param = Parameter(name='theta', is_symbolic=True)
        assert param.is_symbolic
        assert param.value is None

    def test_parameter_serialization(self):
        """Test parameter to/from dict"""
        param = Parameter(name='alpha', value=2.0, is_symbolic=False)
        param_dict = param.to_dict()
        restored = Parameter.from_dict(param_dict)

        assert restored.name == param.name
        assert restored.value == param.value
        assert restored.is_symbolic == param.is_symbolic


class TestUnifiedCircuit:
    """Test UnifiedCircuit class"""

    def test_circuit_creation(self):
        """Test basic circuit creation"""
        circuit = UnifiedCircuit(n_qubits=4)
        assert circuit.n_qubits == 4
        assert len(circuit.gates) == 0
        assert circuit.depth == 0

    def test_add_gate(self):
        """Test adding gates"""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        assert len(circuit.gates) == 2
        assert circuit.gates[0].gate_type == GateType.H
        assert circuit.gates[1].gate_type == GateType.CNOT

    def test_parameterized_layer(self):
        """Test adding parameterized layer"""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_parameterized_layer(GateType.RY, 'theta')

        assert len(circuit.gates) == 3
        assert circuit.n_parameters == 3
        assert 'theta_0' in circuit.parameters
        assert 'theta_1' in circuit.parameters
        assert 'theta_2' in circuit.parameters

    def test_entangling_layer_linear(self):
        """Test linear entangling layer"""
        circuit = UnifiedCircuit(n_qubits=4)
        circuit.add_entangling_layer(GateType.CNOT, pattern='linear')

        # Should have 3 CNOTs (0-1, 1-2, 2-3)
        assert len(circuit.gates) == 3
        assert all(g.gate_type == GateType.CNOT for g in circuit.gates)

    def test_entangling_layer_circular(self):
        """Test circular entangling layer"""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_entangling_layer(GateType.CNOT, pattern='circular')

        # Should have 3 CNOTs (0-1, 1-2, 2-0)
        assert len(circuit.gates) == 3

    def test_entangling_layer_full(self):
        """Test full entangling layer"""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_entangling_layer(GateType.CNOT, pattern='full')

        # Should have n*(n-1)/2 = 3 CNOTs (0-1, 0-2, 1-2)
        assert len(circuit.gates) == 3

    def test_circuit_depth(self):
        """Test circuit depth calculation"""
        circuit = UnifiedCircuit(n_qubits=2)

        # Parallel gates (depth = 1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[1])
        assert circuit.depth == 1

        # Sequential gate (depth = 2)
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        assert circuit.depth == 2

    def test_bind_parameters(self):
        """Test parameter binding"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': 'theta'})

        # Bind parameter
        bound_circuit = circuit.bind_parameters({'theta': np.pi/2})

        assert 'theta' in bound_circuit.parameters
        assert bound_circuit.parameters['theta'].value == np.pi/2

    def test_circuit_copy(self):
        """Test circuit copying"""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        copied = circuit.copy()

        assert len(copied.gates) == len(circuit.gates)
        assert copied is not circuit
        assert copied.gates is not circuit.gates

    def test_circuit_serialization_json(self):
        """Test circuit JSON serialization"""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])
        circuit.add_parameterized_layer(GateType.RY, 'theta')

        # To JSON
        json_str = circuit.to_json()
        assert isinstance(json_str, str)

        # From JSON
        restored = UnifiedCircuit.from_json(json_str)
        assert restored.n_qubits == circuit.n_qubits
        assert len(restored.gates) == len(circuit.gates)
        assert restored.n_parameters == circuit.n_parameters

    def test_circuit_serialization_dict(self):
        """Test circuit dict serialization"""
        circuit = UnifiedCircuit(n_qubits=3)
        circuit.add_gate(GateType.H, targets=[0])

        # To dict
        circuit_dict = circuit.to_dict()
        assert circuit_dict['n_qubits'] == 3
        assert len(circuit_dict['gates']) == 1

        # From dict
        restored = UnifiedCircuit.from_dict(circuit_dict)
        assert restored.n_qubits == circuit.n_qubits
        assert len(restored.gates) == len(circuit.gates)

    def test_method_chaining(self):
        """Test method chaining"""
        circuit = (UnifiedCircuit(n_qubits=2)
                  .add_gate(GateType.H, targets=[0])
                  .add_gate(GateType.CNOT, targets=[0, 1])
                  .add_parameterized_layer(GateType.RY, 'theta'))

        assert len(circuit.gates) == 4  # H + CNOT + 2 RY gates

    def test_repr_and_str(self):
        """Test string representations"""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])

        repr_str = repr(circuit)
        assert 'UnifiedCircuit' in repr_str
        assert 'n_qubits=2' in repr_str

        str_str = str(circuit)
        assert 'Qubits: 2' in str_str


class TestCircuitOptimizer:
    """Test CircuitOptimizer class"""

    def test_basic_optimization(self):
        """Test basic gate cancellation"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])  # Should cancel
        circuit.add_gate(GateType.X, targets=[0])

        optimizer = CircuitOptimizer(strategy='basic')
        optimized = optimizer.optimize(circuit)

        # Two H gates should cancel
        assert len(optimized.gates) == 1
        assert optimized.gates[0].gate_type == GateType.X

    def test_rotation_merging(self):
        """Test rotation gate merging"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': 0.5})
        circuit.add_gate(GateType.RY, targets=[0], parameters={'angle': 0.3})

        optimizer = CircuitOptimizer(strategy='basic')
        optimized = optimizer.optimize(circuit)

        # Should merge into single RY with angle 0.8
        assert len(optimized.gates) == 1
        assert abs(optimized.gates[0].parameters['angle'] - 0.8) < 1e-10

    def test_no_optimization(self):
        """Test 'none' strategy"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])

        optimizer = CircuitOptimizer(strategy='none')
        optimized = optimizer.optimize(circuit)

        # Should keep both gates
        assert len(optimized.gates) == 2

    def test_optimization_metrics(self):
        """Test optimization metrics"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])

        optimizer = CircuitOptimizer(strategy='basic')
        optimized = optimizer.optimize(circuit)
        metrics = optimizer.get_metrics()

        assert metrics is not None
        assert metrics.original_gate_count == 3
        assert metrics.optimized_gate_count == 1
        assert metrics.gates_removed == 2

    def test_optimize_convenience_function(self):
        """Test optimize() convenience function"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.H, targets=[0])

        optimized, metrics = optimize(circuit, strategy='basic')

        assert len(optimized.gates) == 0
        assert metrics.gate_reduction_percent == 100.0

    def test_circuit_optimize_method(self):
        """Test circuit.optimize() method"""
        circuit = UnifiedCircuit(n_qubits=1)
        circuit.add_gate(GateType.X, targets=[0])
        circuit.add_gate(GateType.X, targets=[0])

        optimized = circuit.optimize(strategy='basic')

        # X-X should cancel
        assert len(optimized.gates) == 0


class TestComplexCircuits:
    """Test complex circuit scenarios"""

    def test_variational_circuit(self):
        """Test creating a variational quantum circuit"""
        n_qubits = 4
        depth = 3

        circuit = UnifiedCircuit(n_qubits=n_qubits)

        for layer in range(depth):
            # Rotation layer
            circuit.add_parameterized_layer(GateType.RY, f'theta_{layer}')
            circuit.add_parameterized_layer(GateType.RZ, f'phi_{layer}')

            # Entangling layer
            circuit.add_entangling_layer(GateType.CNOT, pattern='linear')

        assert len(circuit.gates) == depth * (2 * n_qubits + (n_qubits - 1))
        assert circuit.n_parameters == depth * 2 * n_qubits

    def test_bell_state_circuit(self):
        """Test creating Bell state circuit"""
        circuit = UnifiedCircuit(n_qubits=2)
        circuit.add_gate(GateType.H, targets=[0])
        circuit.add_gate(GateType.CNOT, targets=[0, 1])

        assert len(circuit.gates) == 2
        assert circuit.depth == 2

    def test_ghz_state_circuit(self):
        """Test creating GHZ state circuit"""
        n_qubits = 5
        circuit = UnifiedCircuit(n_qubits=n_qubits)

        # H on first qubit
        circuit.add_gate(GateType.H, targets=[0])

        # CNOTs in chain
        for i in range(n_qubits - 1):
            circuit.add_gate(GateType.CNOT, targets=[i, i + 1])

        assert len(circuit.gates) == n_qubits
        assert circuit.depth == n_qubits


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
