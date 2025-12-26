"""
Gradient Computation Validation for Q-Store v4.1.

This script validates that gradients are computed correctly for both
TensorFlow and PyTorch implementations using numerical gradient checking.
"""

import numpy as np


def numerical_gradient(func, x, eps=1e-5):
    """Compute numerical gradient using finite differences.

    Args:
        func: Function to compute gradient for
        x: Point at which to compute gradient
        eps: Small perturbation for finite difference

    Returns:
        Numerical gradient array
    """
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps

        x_minus = x.copy()
        x_minus[i] -= eps

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)

    return grad


def validate_tensorflow_gradients():
    """Validate TensorFlow gradient computation."""
    print("=" * 60)
    print("TensorFlow Gradient Validation")
    print("=" * 60)

    try:
        import tensorflow as tf
        from q_store.tensorflow import QuantumLayer

        # Create a simple quantum layer
        n_qubits = 2
        depth = 1

        layer = QuantumLayer(n_qubits=n_qubits, depth=depth)        # Test input
        x = tf.constant([[0.5, 0.3]], dtype=tf.float32)

        # Compute gradient using TensorFlow
        with tf.GradientTape() as tape:
            tape.watch(layer.trainable_variables)
            output = layer(x)
            loss = tf.reduce_sum(output ** 2)

        tf_gradients = tape.gradient(loss, layer.trainable_variables)

        print(f"\nQuantum layer output shape: {output.shape}")
        print(f"Number of trainable parameters: {len(layer.trainable_variables)}")

        # Numerical gradient check
        def loss_func(params_flat):
            # Reshape and set parameters
            idx = 0
            temp_weights = []
            for var in layer.trainable_variables:
                var_size = np.prod(var.shape)
                var_data = params_flat[idx:idx+var_size].reshape(var.shape)
                temp_weights.append(var_data)
                idx += var_size

            # Manual forward pass
            for i, var in enumerate(layer.trainable_variables):
                var.assign(temp_weights[i])

            out = layer(x)
            return float(tf.reduce_sum(out ** 2).numpy())

        # Get current parameters
        current_params = np.concatenate([var.numpy().flatten()
                                        for var in layer.trainable_variables])

        # Compute numerical gradient
        num_grad = numerical_gradient(loss_func, current_params, eps=1e-4)

        # Get analytical gradient
        analytical_grad = np.concatenate([g.numpy().flatten()
                                         for g in tf_gradients])

        # Compare gradients
        grad_diff = np.abs(num_grad - analytical_grad)
        max_diff = np.max(grad_diff)
        mean_diff = np.mean(grad_diff)

        print(f"\nGradient Comparison:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")

        if max_diff < 1e-3:
            print("  ✓ TensorFlow gradients are CORRECT!")
            return True
        else:
            print("  ✗ TensorFlow gradients may have issues")
            return False

    except ImportError as e:
        print(f"Skipping TensorFlow validation: {e}")
        return None


def validate_pytorch_gradients():
    """Validate PyTorch gradient computation."""
    print("\n" + "=" * 60)
    print("PyTorch Gradient Validation")
    print("=" * 60)

    try:
        import torch
        from q_store.torch import QuantumLayer

        # Create a simple quantum layer
        n_qubits = 2
        depth = 1

        layer = QuantumLayer(n_qubits=n_qubits, depth=depth)        # Test input
        x = torch.tensor([[0.5, 0.3]], dtype=torch.float32, requires_grad=False)

        # Forward pass with gradient tracking
        output = layer(x)
        loss = (output ** 2).sum()
        loss.backward()

        # Get gradients
        pytorch_gradients = [p.grad.clone() for p in layer.parameters() if p.grad is not None]

        print(f"\nQuantum layer output shape: {output.shape}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in layer.parameters())}")

        # Numerical gradient check
        def loss_func(params_flat):
            # Set parameters
            idx = 0
            with torch.no_grad():
                for param in layer.parameters():
                    param_size = param.numel()
                    param.copy_(torch.from_numpy(
                        params_flat[idx:idx+param_size].reshape(param.shape)
                    ))
                    idx += param_size

            # Forward pass
            out = layer(x)
            return float((out ** 2).sum().item())

        # Get current parameters
        current_params = np.concatenate([p.detach().numpy().flatten()
                                        for p in layer.parameters()])

        # Compute numerical gradient
        num_grad = numerical_gradient(loss_func, current_params, eps=1e-4)

        # Get analytical gradient
        analytical_grad = np.concatenate([g.numpy().flatten()
                                         for g in pytorch_gradients])

        # Compare gradients
        grad_diff = np.abs(num_grad - analytical_grad)
        max_diff = np.max(grad_diff)
        mean_diff = np.mean(grad_diff)

        print(f"\nGradient Comparison:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")

        if max_diff < 1e-3:
            print("  ✓ PyTorch gradients are CORRECT!")
            return True
        else:
            print("  ✗ PyTorch gradients may have issues")
            return False

    except ImportError as e:
        print(f"Skipping PyTorch validation: {e}")
        return None


def main():
    """Run all gradient validation tests."""
    print("\n" + "=" * 60)
    print("Q-Store v4.1 Gradient Validation Suite")
    print("=" * 60)

    results = {}

    # Validate TensorFlow
    tf_result = validate_tensorflow_gradients()
    if tf_result is not None:
        results['TensorFlow'] = tf_result

    # Validate PyTorch
    pytorch_result = validate_pytorch_gradients()
    if pytorch_result is not None:
        results['PyTorch'] = pytorch_result

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for framework, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{framework}: {status}")

    if all(results.values()):
        print("\nAll gradient tests PASSED! 🎉")
    else:
        print("\nSome tests failed - please investigate")

    return results


if __name__ == '__main__':
    main()
