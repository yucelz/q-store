"""
Quantum Machine Learning Training Components

This module provides complete ML training capabilities for quantum neural networks
with hardware abstraction.

v3.3 Enhancements:
- SPSA gradient estimation (48x faster)
- Circuit batching and caching
- Hardware-efficient quantum layers
- Adaptive gradient optimization
- Performance tracking and monitoring
"""

from .quantum_layer import (
    QuantumLayer,
    QuantumConvolutionalLayer,
    QuantumPoolingLayer,
    LayerConfig
)
from .gradient_computer import (
    QuantumGradientComputer,
    FiniteDifferenceGradient,
    NaturalGradientComputer,
    GradientResult
)
from .data_encoder import (
    QuantumDataEncoder,
    QuantumFeatureMap
)
from .quantum_trainer import (
    QuantumTrainer,
    QuantumModel,
    TrainingConfig,
    TrainingMetrics
)

# v3.3 NEW: Performance optimizations
from .spsa_gradient_estimator import (
    SPSAGradientEstimator,
    SPSAOptimizerWithAdaptiveGains
)
from .circuit_batch_manager import CircuitBatchManager
from .circuit_cache import (
    QuantumCircuitCache,
    AdaptiveCircuitCache
)
from .quantum_layer_v2 import (
    HardwareEfficientQuantumLayer,
    HardwareEfficientLayerConfig
)
from .adaptive_optimizer import (
    AdaptiveGradientOptimizer,
    GradientMethodScheduler
)
from .performance_tracker import (
    PerformanceTracker,
    BatchMetrics,
    EpochMetrics
)

__all__ = [
    # Quantum Layers
    "QuantumLayer",
    "QuantumConvolutionalLayer",
    "QuantumPoolingLayer",
    "LayerConfig",
    # Gradient Computation
    "QuantumGradientComputer",
    "FiniteDifferenceGradient",
    "NaturalGradientComputer",
    "GradientResult",
    # Data Encoding
    "QuantumDataEncoder",
    "QuantumFeatureMap",
    # Training
    "QuantumTrainer",
    "QuantumModel",
    "TrainingConfig",
    "TrainingMetrics",
    # v3.3: SPSA Gradient Estimation
    "SPSAGradientEstimator",
    "SPSAOptimizerWithAdaptiveGains",
    # v3.3: Circuit Optimization
    "CircuitBatchManager",
    "QuantumCircuitCache",
    "AdaptiveCircuitCache",
    # v3.3: Hardware-Efficient Layers
    "HardwareEfficientQuantumLayer",
    "HardwareEfficientLayerConfig",
    # v3.3: Adaptive Optimization
    "AdaptiveGradientOptimizer",
    "GradientMethodScheduler",
    # v3.3: Performance Monitoring
    "PerformanceTracker",
    "BatchMetrics",
    "EpochMetrics",
]
