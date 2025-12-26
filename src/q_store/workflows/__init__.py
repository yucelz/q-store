"""
Hybrid quantum-classical workflows for Q-Store.

This module provides patterns and tools for integrating quantum and classical
computation including:
- Quantum-Classical Machine Learning (QCML) patterns
- Data pipelines for hybrid workflows
- Parameter optimization loops
- Quantum subroutine integration
"""

from .qcml_patterns import (
    QuantumClassicalHybrid,
    QuantumPreprocessor,
    QuantumLayer as WorkflowQuantumLayer,
    ClassicalPostprocessor,
    create_hybrid_model
)

from .data_pipeline import (
    DataPipeline,
    QuantumDataEncoder,
    BatchProcessor,
    ResultAggregator,
    create_data_pipeline
)

from .optimization_loop import (
    OptimizationLoop,
    ParameterUpdate,
    ConvergenceChecker,
    run_optimization_loop
)

__all__ = [
    # QCML patterns
    'QuantumClassicalHybrid',
    'QuantumPreprocessor',
    'WorkflowQuantumLayer',
    'ClassicalPostprocessor',
    'create_hybrid_model',

    # Data pipelines
    'DataPipeline',
    'QuantumDataEncoder',
    'BatchProcessor',
    'ResultAggregator',
    'create_data_pipeline',

    # Optimization loops
    'OptimizationLoop',
    'ParameterUpdate',
    'ConvergenceChecker',
    'run_optimization_loop'
]
