"""
Quantum Machine Learning examples using Q-Store.

Demonstrates QML features including feature maps, kernels, and training.
"""

import numpy as np
from q_store.core import UnifiedCircuit, GateType
from q_store.ml import QuantumFeatureMap, QuantumKernel, QuantumModel
from q_store.visualization import visualize_circuit


def example_feature_map():
    """Demonstrate quantum feature maps."""
    print("=" * 60)
    print("Example 1: Quantum Feature Maps")
    print("=" * 60)

    # Create feature map
    feature_map = QuantumFeatureMap(n_qubits=3, depth=2)

    # Sample data
    x = np.array([0.5, 0.3, 0.8])

    # Create circuit
    circuit = feature_map.create_circuit(x)

    print("\nFeature map circuit for x=[0.5, 0.3, 0.8]:")
    print(visualize_circuit(circuit))
    print(f"\nCircuit depth: {circuit.depth}")
    print(f"Total gates: {len(circuit.gates)}")


def example_quantum_kernel():
    """Demonstrate quantum kernel computation."""
    print("\n" + "=" * 60)
    print("Example 2: Quantum Kernels")
    print("=" * 60)

    # Create kernel
    kernel = QuantumKernel(n_qubits=2, feature_map_depth=1)

    # Sample data points
    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])
    x3 = np.array([0.9, 0.8])

    # Compute kernel values
    k11 = kernel.compute(x1, x1)
    k12 = kernel.compute(x1, x2)
    k13 = kernel.compute(x1, x3)

    print(f"\nKernel values:")
    print(f"  K(x1, x1) = {k11:.4f}  (should be ~1.0)")
    print(f"  K(x1, x2) = {k12:.4f}")
    print(f"  K(x1, x3) = {k13:.4f}")

    # Compute kernel matrix
    X = np.array([x1, x2, x3])
    K = kernel.compute_matrix(X)

    print(f"\nKernel matrix:")
    print(K)


def example_quantum_model():
    """Demonstrate quantum model creation and inference."""
    print("\n" + "=" * 60)
    print("Example 3: Quantum Model")
    print("=" * 60)

    # Create model
    model = QuantumModel(n_qubits=2, n_layers=2)

    # Initialize parameters
    n_params = model.get_n_parameters()
    print(f"Model has {n_params} parameters")

    # Random initial parameters
    params = np.random.randn(n_params) * 0.1
    model.set_parameters(params)

    # Sample input
    x = np.array([0.5, 0.3])

    # Forward pass
    output = model.forward(x)

    print(f"\nInput: {x}")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")

    # Get model circuit
    circuit = model.get_circuit(x)
    print(f"\nModel circuit:")
    print(visualize_circuit(circuit))


def example_variational_training():
    """Demonstrate variational circuit training."""
    print("\n" + "=" * 60)
    print("Example 4: Variational Training")
    print("=" * 60)

    # Create a simple variational circuit
    def create_ansatz(params):
        circuit = UnifiedCircuit(2)
        circuit.add_gate(GateType.RY, [0], parameters={'angle': params[0]})
        circuit.add_gate(GateType.RY, [1], parameters={'angle': params[1]})
        circuit.add_gate(GateType.CNOT, [0, 1])
        circuit.add_gate(GateType.RY, [0], parameters={'angle': params[2]})
        circuit.add_gate(GateType.RY, [1], parameters={'angle': params[3]})
        return circuit

    # Initial parameters
    params = np.array([0.1, 0.2, 0.3, 0.4])

    print("Initial ansatz:")
    circuit = create_ansatz(params)
    print(visualize_circuit(circuit))

    print(f"\nParameters: {params}")
    print(f"Circuit depth: {circuit.depth}")

    # Simulate parameter update
    params_updated = params - 0.1 * np.random.randn(4)

    print(f"\nUpdated parameters: {params_updated}")
    print("(In real training, these would be optimized via gradient descent)")


def example_data_encoding():
    """Demonstrate different data encoding strategies."""
    print("\n" + "=" * 60)
    print("Example 5: Data Encoding Strategies")
    print("=" * 60)

    # Sample data
    x = np.array([0.5, 0.3])

    print(f"Input data: {x}")

    # Angle encoding
    print("\n1. Angle Encoding (RY rotations):")
    circuit_angle = UnifiedCircuit(2)
    circuit_angle.add_gate(GateType.RY, [0], parameters={'angle': x[0] * np.pi})
    circuit_angle.add_gate(GateType.RY, [1], parameters={'angle': x[1] * np.pi})
    print(visualize_circuit(circuit_angle))

    # IQP encoding
    print("\n2. IQP-style Encoding:")
    circuit_iqp = UnifiedCircuit(2)
    circuit_iqp.add_gate(GateType.H, [0])
    circuit_iqp.add_gate(GateType.H, [1])
    circuit_iqp.add_gate(GateType.RZ, [0], parameters={'angle': x[0] * 2 * np.pi})
    circuit_iqp.add_gate(GateType.RZ, [1], parameters={'angle': x[1] * 2 * np.pi})
    circuit_iqp.add_gate(GateType.CNOT, [0, 1])
    print(visualize_circuit(circuit_iqp))


def example_qml_workflow():
    """Demonstrate complete QML workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Complete QML Workflow")
    print("=" * 60)

    # 1. Create dataset
    np.random.seed(42)
    X_train = np.random.randn(5, 2) * 0.5
    y_train = np.array([0, 1, 0, 1, 0])

    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # 2. Create quantum model
    model = QuantumModel(n_qubits=2, n_layers=2)
    n_params = model.get_n_parameters()

    # 3. Initialize parameters
    params = np.random.randn(n_params) * 0.1
    model.set_parameters(params)

    print(f"Model initialized with {n_params} parameters")

    # 4. Forward pass on training data
    print("\nPredictions on training data:")
    for i, x in enumerate(X_train):
        output = model.forward(x)
        pred_class = 1 if output[0] > 0.5 else 0
        print(f"  Sample {i}: true={y_train[i]}, pred={pred_class}, output={output[0]:.3f}")

    print("\n(In practice, you would optimize parameters using a training loop)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q-STORE QUANTUM ML EXAMPLES")
    print("=" * 60)

    example_feature_map()
    example_quantum_kernel()
    example_quantum_model()
    example_variational_training()
    example_data_encoding()
    example_qml_workflow()

    print("\n" + "=" * 60)
    print("QML examples completed!")
    print("=" * 60)
