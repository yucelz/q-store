"""
Advanced circuit optimization techniques.

This module provides sophisticated optimization strategies:
- Commutation analysis and gate reordering
- Gate fusion for reduced circuit depth
- Parallelization detection
- Template matching and substitution

v4.1 Additions:
- Adaptive batch scheduling
- Multi-level caching (L1/L2/L3)
- IonQ native compilation
"""

# v4.0 optimizations
from q_store.optimization.commutation import (
    CommutationAnalyzer,
    commute_gates,
    can_commute,
    reorder_commuting_gates
)
from q_store.optimization.gate_fusion import (
    GateFuser,
    fuse_single_qubit_gates,
    fuse_rotation_gates,
    identify_fusion_opportunities
)
from q_store.optimization.parallelization import (
    ParallelizationAnalyzer,
    find_parallel_layers,
    compute_circuit_depth,
    optimize_for_parallelism
)
from q_store.optimization.template_matching import (
    TemplateOptimizer,
    match_and_replace_templates,
    create_optimization_template,
    standard_templates
)

# v4.1 optimizations (new)
try:
    from q_store.optimization.adaptive_scheduler import (
        AdaptiveBatchScheduler,
        CircuitComplexityEstimator,
    )
    from q_store.optimization.multi_level_cache import MultiLevelCache, LRUCache
    from q_store.optimization.ionq_compiler import IonQNativeCompiler
    HAS_V4_1_OPT = True
except ImportError:
    HAS_V4_1_OPT = False
    AdaptiveBatchScheduler = None
    CircuitComplexityEstimator = None
    MultiLevelCache = None
    LRUCache = None
    IonQNativeCompiler = None

__all__ = [
    # v4.0 Commutation
    'CommutationAnalyzer',
    'commute_gates',
    'can_commute',
    'reorder_commuting_gates',
    # v4.0 Gate Fusion
    'GateFuser',
    'fuse_single_qubit_gates',
    'fuse_rotation_gates',
    'identify_fusion_opportunities',
    # v4.0 Parallelization
    'ParallelizationAnalyzer',
    'find_parallel_layers',
    'compute_circuit_depth',
    'optimize_for_parallelism',
    # v4.0 Template Matching
    'TemplateOptimizer',
    'match_and_replace_templates',
    'create_optimization_template',
    'standard_templates',
    # v4.1 Adaptive Scheduling
    'AdaptiveBatchScheduler',
    'CircuitComplexityEstimator',
    # v4.1 Multi-Level Cache
    'MultiLevelCache',
    'LRUCache',
    # v4.1 IonQ Compiler
    'IonQNativeCompiler',
]
