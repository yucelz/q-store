"""
Advanced circuit optimization techniques.

This module provides sophisticated optimization strategies:
- Commutation analysis and gate reordering
- Gate fusion for reduced circuit depth
- Parallelization detection
- Template matching and substitution
"""

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

__all__ = [
    # Commutation
    'CommutationAnalyzer',
    'commute_gates',
    'can_commute',
    'reorder_commuting_gates',
    # Gate Fusion
    'GateFuser',
    'fuse_single_qubit_gates',
    'fuse_rotation_gates',
    'identify_fusion_opportunities',
    # Parallelization
    'ParallelizationAnalyzer',
    'find_parallel_layers',
    'compute_circuit_depth',
    'optimize_for_parallelism',
    # Template Matching
    'TemplateOptimizer',
    'match_and_replace_templates',
    'create_optimization_template',
    'standard_templates',
]
