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

v3.4 Performance Revolution (2x speedup):
- IonQConcurrentClient: Concurrent circuit submission (~1.6x faster)
- IonQNativeGateCompiler: Native gate compilation (30% faster execution)
- SmartCircuitCache: Template-based caching (3-4x faster preparation)
- CircuitBatchManagerV34: Orchestrates all optimizations
- Adaptive batch sizing and connection pooling

v3.5 Advanced Optimizations (2-3x additional speedup):
- MultiBackendOrchestrator: Distribute across multiple backends (2-3x throughput)
- AdaptiveCircuitOptimizer: Dynamic circuit simplification (30-40% faster)
- AdaptiveShotAllocator: Smart shot allocation (20-30% time savings)
- NaturalGradientEstimator: Natural gradient descent (2-3x fewer iterations)
"""

from .adaptive_optimizer import AdaptiveGradientOptimizer, GradientMethodScheduler
from .circuit_batch_manager import CircuitBatchManager
from .circuit_cache import AdaptiveCircuitCache, QuantumCircuitCache
from .data_encoder import QuantumDataEncoder, QuantumFeatureMap
from .gradient_computer import (
    FiniteDifferenceGradient,
    GradientResult,
    NaturalGradientComputer,
    QuantumGradientComputer,
)

# v3.3.1 NEW: Corrected batch gradient computation
from .parallel_spsa_estimator import (
    ParallelSPSAEstimator,
    SPSABatchGradientEstimator,
    SubsampledSPSAEstimator,
)
from .performance_tracker import BatchMetrics, EpochMetrics, PerformanceTracker
from .quantum_layer import LayerConfig, QuantumConvolutionalLayer, QuantumLayer, QuantumPoolingLayer
from .quantum_layer_v2 import HardwareEfficientLayerConfig, HardwareEfficientQuantumLayer
from .quantum_trainer import QuantumModel, QuantumTrainer, TrainingConfig, TrainingMetrics

# v3.3 NEW: Performance optimizations
from .spsa_gradient_estimator import SPSAGradientEstimator, SPSAOptimizerWithAdaptiveGains

# v3.4 NEW: Performance revolution (2x speedup)
try:
    from .circuit_batch_manager_v3_4 import CircuitBatchManagerV34
    from .ionq_concurrent_client import BatchJobResult, IonQConcurrentClient, JobStatus
    from .ionq_native_gate_compiler import IonQNativeGateCompiler, NativeGateType
    from .smart_circuit_cache import CircuitTemplate, SmartCircuitCache

    V3_4_AVAILABLE = True
except ImportError:
    V3_4_AVAILABLE = False
    CircuitBatchManagerV34 = None
    IonQConcurrentClient = None
    IonQNativeGateCompiler = None
    SmartCircuitCache = None

# v3.5 NEW: Advanced optimizations (2-3x additional speedup)
try:
    from .multi_backend_orchestrator import MultiBackendOrchestrator, BackendConfig, BackendStats
    from .adaptive_circuit_optimizer import AdaptiveCircuitOptimizer, CircuitOptimizationResult
    from .adaptive_shot_allocator import AdaptiveShotAllocator
    from .natural_gradient_estimator import NaturalGradientEstimator, QFIMResult

    V3_5_AVAILABLE = True
except ImportError:
    V3_5_AVAILABLE = False
    MultiBackendOrchestrator = None
    BackendConfig = None
    AdaptiveCircuitOptimizer = None
    AdaptiveShotAllocator = None
    NaturalGradientEstimator = None

# v4.1 Enhanced: Gradient noise tracking for training stability
from .gradient_noise_tracker import GradientNoiseTracker, GradientStatistics

# v4.1 Enhanced: Adaptive training controller
from .adaptive_training_controller import (
    AdaptiveTrainingController,
    TrainingOrchestrator,
    TrainingPhase,
    AdaptationEvent
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
    "IonQConcurrentClient",
    "BatchJobResult",
    "JobStatus",
    "IonQNativeGateCompiler",
    "NativeGateType",
    "SmartCircuitCache",
    "CircuitTemplate",
    "V3_4_AVAILABLE",
    # v3.5: Advanced Optimizations (if available)
    "MultiBackendOrchestrator",
    "BackendConfig",
    "BackendStats",
    "AdaptiveCircuitOptimizer",
    "CircuitOptimizationResult",
    "AdaptiveShotAllocator",
    "NaturalGradientEstimator",
    "QFIMResult",
    "V3_5_AVAILABLE",
    # v4.1 Enhanced
    "GradientNoiseTracker",
    "GradientStatistics",
    "AdaptiveTrainingController",
    "TrainingOrchestrator",
    "TrainingPhase",
    "AdaptationEvent",
]
