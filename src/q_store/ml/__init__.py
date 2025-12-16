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

v3.3.1 Critical Fix:
- True batch gradient computation with ParallelSPSAEstimator
- Gradient subsampling for 5-10x speedup
- Corrected SPSA implementation

v3.4 Performance Revolution (8-10x speedup):
- IonQBatchClient: True parallel batch submission (12x faster)
- IonQNativeGateCompiler: Native gate compilation (30% faster execution)
- SmartCircuitCache: Template-based caching (10x faster preparation)
- CircuitBatchManagerV34: Orchestrates all optimizations
- Adaptive batch sizing and connection pooling
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
# v3.3.1 NEW: Corrected batch gradient computation
from .parallel_spsa_estimator import (
    ParallelSPSAEstimator,
    SubsampledSPSAEstimator,
    SPSABatchGradientEstimator
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

# v3.4 NEW: Performance revolution (8-10x speedup)
try:
    from .circuit_batch_manager_v3_4 import CircuitBatchManagerV34
    from .ionq_batch_client import IonQBatchClient, BatchJobResult, JobStatus
    from .ionq_native_gate_compiler import IonQNativeGateCompiler, NativeGateType
    from .smart_circuit_cache import SmartCircuitCache, CircuitTemplate
    V3_4_AVAILABLE = True
except ImportError:
    V3_4_AVAILABLE = False
    CircuitBatchManagerV34 = None
    IonQBatchClient = None
    IonQNativeGateCompiler = None
    SmartCircuitCache = None

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
    # v3.3.1: Corrected Batch SPSA
    "ParallelSPSAEstimator",
    "SubsampledSPSAEstimator",
    "SPSABatchGradientEstimator",
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
    # v3.4: Performance Revolution (if available)
    "CircuitBatchManagerV34",
    "IonQBatchClient",
    "BatchJobResult",
    "JobStatus",
    "IonQNativeGateCompiler",
    "NativeGateType",
    "SmartCircuitCache",
    "CircuitTemplate",
    "V3_4_AVAILABLE",
]
