"""
Grid Search for Hyperparameter Tuning.

Exhaustive search over a specified parameter grid.

Example:
    >>> from q_store.ml.tuning import GridSearch
    >>>
    >>> param_grid = {
    ...     'learning_rate': [0.001, 0.01, 0.1],
    ...     'batch_size': [16, 32, 64],
    ...     'n_qubits': [4, 6, 8]
    ... }
    >>>
    >>> def objective(params):
    ...     # Train model with params and return metric
    ...     return train_and_evaluate(params)
    >>>
    >>> grid_search = GridSearch(param_grid)
    >>> best_params, best_score = grid_search.search(objective)
    >>> print(f"Best params: {best_params}, Score: {best_score}")
"""

import logging
from itertools import product
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GridSearch:
    """
    Grid search for hyperparameter optimization.

    Performs exhaustive search over all combinations of parameters.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        scoring: 'min' or 'max' - whether to minimize or maximize objective (default: 'min')
        verbose: Print progress (default: True)

    Example:
        >>> param_grid = {
        ...     'lr': [0.001, 0.01],
        ...     'depth': [2, 3, 4]
        ... }
        >>> grid = GridSearch(param_grid, scoring='min')
        >>> best_params, best_score = grid.search(objective_fn)
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        scoring: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize grid search.

        Args:
            param_grid: Parameter grid
            scoring: Scoring mode ('min' or 'max')
            verbose: Verbosity
        """
        if scoring not in ['min', 'max']:
            raise ValueError(f"scoring must be 'min' or 'max', got: {scoring}")

        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

        # Calculate total combinations
        self.n_combinations = 1
        for values in param_grid.values():
            self.n_combinations *= len(values)

        logger.info(
            f"GridSearch initialized: {len(param_grid)} parameters, "
            f"{self.n_combinations} combinations"
        )

    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform grid search.

        Args:
            objective_fn: Function that takes params dict and returns score
            n_trials: Maximum number of trials (None = all combinations)

        Returns:
            Tuple of (best_params, best_score)

        Example:
            >>> def objective(params):
            ...     return train_model(**params)
            >>> best_params, best_score = grid.search(objective)
        """
        if self.verbose:
            logger.info(f"Starting grid search over {self.n_combinations} combinations")

        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        # Initialize best score
        if self.scoring == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

        # Iterate over all combinations
        trial_count = 0
        for combination in product(*param_values):
            # Create params dict
            params = dict(zip(param_names, combination))

            # Evaluate objective
            try:
                score = objective_fn(params)

                # Store result
                self.results.append({
                    'params': params,
                    'score': score,
                    'trial': trial_count
                })

                # Update best
                is_better = (
                    (self.scoring == 'min' and score < self.best_score) or
                    (self.scoring == 'max' and score > self.best_score)
                )

                if is_better:
                    self.best_score = score
                    self.best_params = params.copy()

                    if self.verbose:
                        logger.info(
                            f"Trial {trial_count}/{self.n_combinations}: "
                            f"New best score = {score:.6f}"
                        )

                trial_count += 1

                # Check trial limit
                if n_trials and trial_count >= n_trials:
                    break

            except Exception as e:
                logger.error(f"Error evaluating params {params}: {e}")
                continue

        if self.verbose:
            logger.info(
                f"Grid search complete. Best score: {self.best_score:.6f}, "
                f"Best params: {self.best_params}"
            )

        return self.best_params, self.best_score

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all search results.

        Returns:
            List of result dictionaries
        """
        return self.results

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Get best score found."""
        return self.best_score

    def get_top_k(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top k results.

        Args:
            k: Number of top results

        Returns:
            List of top k results
        """
        if not self.results:
            return []

        # Sort by score
        if self.scoring == 'min':
            sorted_results = sorted(self.results, key=lambda x: x['score'])
        else:
            sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)

        return sorted_results[:k]
