"""
Unit Tests for Q-Store v4.1 Quantum-First Layers

Tests the core quantum layers implemented in Phase 1:
- QuantumFeatureExtractor
- QuantumNonlinearity
- QuantumPooling
- QuantumReadout
- EncodingLayer
- DecodingLayer
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch

from q_store.layers import (
    QuantumFeatureExtractor,
    QuantumNonlinearity,
    QuantumPooling,
    QuantumReadout,
    EncodingLayer,
    DecodingLayer,
)


class TestQuantumFeatureExtractor:
    """Test QuantumFeatureExtractor layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumFeatureExtractor(n_qubits=8, depth=3)

        assert layer.n_qubits == 8
        assert layer.depth == 3
        assert layer.output_dim == 24  # 8 qubits * 3 bases
        assert layer.n_parameters == 48  # 2 * 8 * 3

    def test_parameter_initialization(self):
        """Test parameter initialization."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        assert len(layer.parameters) == 16  # 2 * 4 * 2
        assert all(isinstance(v, (float, np.floating)) for v in layer.parameters.values())

    def test_circuit_building(self):
        """Test PQC construction."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2, entanglement='linear')

        circuit = layer.pqc
        assert circuit.n_qubits == 4
        assert len(circuit.gates) > 0
        assert circuit.measurement_bases == ['Z', 'X', 'Y']

    @pytest.mark.asyncio
    async def test_forward_pass(self):
        """Test async forward pass."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        inputs = np.random.randn(8, 64).astype(np.float32)
        outputs = await layer.call_async(inputs)

        assert outputs.shape == (8, 12)  # 4 qubits * 3 bases
        assert outputs.dtype == np.float32

    def test_encoding(self):
        """Test amplitude encoding."""
        layer = QuantumFeatureExtractor(n_qubits=4, depth=2)

        inputs = np.random.randn(4, 32).astype(np.float32)
        encoded = layer._encode_batch(inputs)

        assert encoded.shape == (4, 16)  # 2^4 = 16
        # Check normalization
        norms = np.linalg.norm(encoded, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        for pattern in ['linear', 'full', 'circular']:
            layer = QuantumFeatureExtractor(n_qubits=4, depth=2, entanglement=pattern)
            assert layer.entanglement == pattern

            # Count CNOT gates
            cnot_gates = [g for g in layer.pqc.gates if g['type'] == 'CNOT']

            if pattern == 'linear':
                assert len(cnot_gates) == 3 * 2  # (n-1) * depth
            elif pattern == 'circular':
                assert len(cnot_gates) == 4 * 2  # n * depth
            elif pattern == 'full':
                assert len(cnot_gates) == 6 * 2  # n*(n-1)/2 * depth


class TestQuantumNonlinearity:
    """Test QuantumNonlinearity layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumNonlinearity(n_qubits=8, nonlinearity_type='amplitude_damping')

        assert layer.n_qubits == 8
        assert layer.nonlinearity_type == 'amplitude_damping'
        assert layer.strength == 0.1

    @pytest.mark.asyncio
    async def test_amplitude_damping(self):
        """Test amplitude damping nonlinearity."""
        layer = QuantumNonlinearity(n_qubits=4, nonlinearity_type='amplitude_damping', strength=0.2)

        inputs = np.random.randn(8, 16).astype(np.float32)
        outputs = await layer.call_async(inputs)

        assert outputs.shape == inputs.shape
        assert np.all(np.isfinite(outputs))

    @pytest.mark.asyncio
    async def test_phase_damping(self):
        """Test phase damping nonlinearity."""
        layer = QuantumNonlinearity(n_qubits=4, nonlinearity_type='phase_damping', strength=0.1)

        inputs = np.random.randn(8, 16).astype(np.float32)
        outputs = await layer.call_async(inputs, training=True)

        assert outputs.shape == inputs.shape

    @pytest.mark.asyncio
    async def test_parametric_evolution(self):
        """Test parametric evolution nonlinearity."""
        layer = QuantumNonlinearity(n_qubits=4, nonlinearity_type='parametric', strength=1.0)

        inputs = np.random.randn(8, 16).astype(np.float32)
        outputs = await layer.call_async(inputs)

        assert outputs.shape == inputs.shape
        assert np.all(np.abs(outputs) <= 1.0)  # tanh output


class TestQuantumPooling:
    """Test QuantumPooling layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumPooling(n_qubits=8, pool_size=2)

        assert layer.n_qubits == 8
        assert layer.pool_size == 2
        assert layer.output_qubits == 4

    def test_invalid_pool_size(self):
        """Test invalid pool size raises error."""
        with pytest.raises(ValueError):
            QuantumPooling(n_qubits=7, pool_size=2)  # Not divisible

    @pytest.mark.asyncio
    async def test_measurement_pooling(self):
        """Test measurement-based pooling."""
        layer = QuantumPooling(n_qubits=8, pool_size=2, pooling_type='measurement')

        inputs = np.random.randn(4, 24).astype(np.float32)  # 8 qubits * 3 bases
        outputs = await layer.call_async(inputs)

        assert outputs.shape == (4, 12)  # Reduced by pool_size

    @pytest.mark.asyncio
    async def test_aggregation_methods(self):
        """Test different aggregation methods."""
        inputs = np.random.randn(4, 24).astype(np.float32)

        for agg in ['mean', 'max', 'sum']:
            layer = QuantumPooling(n_qubits=8, pool_size=2, aggregation=agg)
            outputs = await layer.call_async(inputs)

            assert outputs.shape == (4, 12)
            assert np.all(np.isfinite(outputs))


class TestQuantumReadout:
    """Test QuantumReadout layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = QuantumReadout(n_qubits=4, n_classes=10)

        assert layer.n_qubits == 4
        assert layer.n_classes == 10

    def test_insufficient_qubits(self):
        """Test error when insufficient qubits."""
        with pytest.raises(ValueError):
            QuantumReadout(n_qubits=2, n_classes=10)  # Need 4 qubits for 10 classes

    @pytest.mark.asyncio
    async def test_computational_readout(self):
        """Test computational basis readout."""
        layer = QuantumReadout(n_qubits=4, n_classes=10, readout_type='computational')

        inputs = np.random.randn(8, 12).astype(np.float32)
        outputs = await layer.call_async(inputs)

        assert outputs.shape == (8, 10)
        # Check probabilities sum to 1
        sums = np.sum(outputs, axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
        # Check probabilities in [0, 1]
        assert np.all(outputs >= 0.0)
        assert np.all(outputs <= 1.0)

    @pytest.mark.asyncio
    async def test_expectation_readout(self):
        """Test expectation value readout."""
        layer = QuantumReadout(n_qubits=4, n_classes=10, readout_type='expectation')

        inputs = np.random.randn(8, 12).astype(np.float32)
        outputs = await layer.call_async(inputs)

        assert outputs.shape == (8, 10)
        # Check probabilities
        assert np.all(outputs >= 0.0)
        assert np.all(outputs <= 1.0)


class TestEncodingLayer:
    """Test EncodingLayer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = EncodingLayer(target_dim=256)

        assert layer.target_dim == 256
        assert layer.normalization == 'l2'

    def test_l2_normalization(self):
        """Test L2 normalization."""
        layer = EncodingLayer(target_dim=64, normalization='l2')

        inputs = np.random.randn(4, 64).astype(np.float32)
        outputs = layer(inputs)

        # Check normalization
        norms = np.linalg.norm(outputs, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_padding(self):
        """Test dimension padding."""
        layer = EncodingLayer(target_dim=128)

        inputs = np.random.randn(4, 64).astype(np.float32)
        outputs = layer(inputs)

        assert outputs.shape == (4, 128)

    def test_truncation(self):
        """Test dimension truncation."""
        layer = EncodingLayer(target_dim=32)

        inputs = np.random.randn(4, 64).astype(np.float32)
        outputs = layer(inputs)

        assert outputs.shape == (4, 32)


class TestDecodingLayer:
    """Test DecodingLayer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = DecodingLayer(output_dim=10)

        assert layer.output_dim == 10
        assert layer.scaling == 'expectation'

    def test_expectation_scaling(self):
        """Test expectation value scaling."""
        layer = DecodingLayer(scaling='expectation')

        inputs = np.random.randn(4, 24) * 2 - 1  # [-1, 1] range
        outputs = layer(inputs)

        # Check scaled to [0, 1]
        assert np.all(outputs >= 0.0)
        assert np.all(outputs <= 1.0)

    def test_projection(self):
        """Test linear projection."""
        layer = DecodingLayer(output_dim=10)

        inputs = np.random.randn(4, 24).astype(np.float32)
        outputs = layer(inputs)

        assert outputs.shape == (4, 10)

    def test_no_projection(self):
        """Test without projection."""
        layer = DecodingLayer(output_dim=None)

        inputs = np.random.randn(4, 24).astype(np.float32)
        outputs = layer(inputs)

        assert outputs.shape == (4, 24)


class TestIntegration:
    """Integration tests for layer combinations."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete quantum-first pipeline."""
        # Build pipeline
        encoder = EncodingLayer(target_dim=256)
        quantum1 = QuantumFeatureExtractor(n_qubits=8, depth=3)
        nonlinearity = QuantumNonlinearity(n_qubits=8)
        pooling = QuantumPooling(n_qubits=8, pool_size=2)
        quantum2 = QuantumFeatureExtractor(n_qubits=4, depth=2)
        readout = QuantumReadout(n_qubits=4, n_classes=10)
        decoder = DecodingLayer(output_dim=10)

        # Sample input
        inputs = np.random.randn(8, 784).astype(np.float32)

        # Forward pass
        x = encoder(inputs)
        x = await quantum1.call_async(x)
        x = await nonlinearity.call_async(x)
        x = await pooling.call_async(x)
        x = await quantum2.call_async(x)
        x = await readout.call_async(x)
        x = decoder(x)

        # Check output
        assert x.shape == (8, 10)
        assert np.all(np.isfinite(x))

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel layer execution."""
        layers = [
            QuantumFeatureExtractor(n_qubits=4, depth=2)
            for _ in range(4)
        ]

        inputs = [
            np.random.randn(4, 64).astype(np.float32)
            for _ in range(4)
        ]

        # Execute in parallel
        tasks = [
            layer.call_async(inp)
            for layer, inp in zip(layers, inputs)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(r.shape == (4, 12) for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
