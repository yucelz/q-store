"""
Hyperparameter Tuning for Quantum ML (v4.1.1).

This module provides comprehensive hyperparameter optimization methods:
- Grid search for exhaustive search
- Random search for efficient exploration
- Bayesian optimization for smart search
- Optuna integration for advanced optimization

Example:
    >>> from q_store.ml.tuning import GridSearch, RandomSearch, OptunaTuner
    >>>
    >>> # Define parameter space
    >>> param_space = {
    ...     'learning_rate': [0.001, 0.01, 0.1],
    ...     'n_qubits': [4, 6, 8],
    ...     'circuit_depth': [2, 3, 4]
    ... }
    >>>
    >>> # Grid search
    >>> grid_search = GridSearch(param_space)
    >>> best_params = grid_search.search(objective_fn, n_trials=10)
"""

from .grid_search import GridSearch
from .random_search import RandomSearch
from .bayesian_optimizer import BayesianOptimizer
from .optuna_integration import OptunaTuner, OptunaConfig

__all__ = [
    'GridSearch',
    'RandomSearch',
    'BayesianOptimizer',
    'OptunaTuner',
    'OptunaConfig',
]
