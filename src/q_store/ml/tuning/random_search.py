"""
Random Search for Hyperparameter Tuning.

Random sampling from parameter distributions for efficient exploration.

Example:
    >>> from q_store.ml.tuning import RandomSearch
    >>>
    >>> param_distributions = {
    ...     'learning_rate': ('log_uniform', 1e-4, 1e-1),
    ...     'batch_size': ('choice', [16, 32, 64, 128]),
    ...     'n_qubits': ('int_uniform', 4, 12)
    ... }
    >>>
    >>> random_search = RandomSearch(param_distributions)
    >>> best_params, best_score = random_search.search(objective, n_trials=50)
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class RandomSearch:
    """
    Random search for hyperparameter optimization.

    Samples parameters randomly from specified distributions.

    Args:
        param_distributions: Dictionary mapping parameter names to distributions
                            Format: {name: (dist_type, *args)}
                            Distributions: 'uniform', 'log_uniform', 'int_uniform',
                                         'normal', 'choice'
        scoring: 'min' or 'max' (default: 'min')
        random_seed: Random seed for reproducibility (default: None)
        verbose: Print progress (default: True)

    Example:
        >>> param_dist = {
        ...     'lr': ('log_uniform', 1e-4, 1e-1),
        ...     'dropout': ('uniform', 0.1, 0.5),
        ...     'n_layers': ('int_uniform', 2, 5),
        ...     'activation': ('choice', ['relu', 'tanh', 'sigmoid'])
        ... }
        >>> search = RandomSearch(param_dist, random_seed=42)
        >>> best_params, best_score = search.search(objective, n_trials=100)
    """

    def __init__(
        self,
        param_distributions: Dict[str, Tuple[str, ...]],
        scoring: str = 'min',
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize random search.

        Args:
            param_distributions: Parameter distributions
            scoring: Scoring mode
            random_seed: Random seed
            verbose: Verbosity
        """
        if scoring not in ['min', 'max']:
            raise ValueError(f"scoring must be 'min' or 'max', got: {scoring}")

        self.param_distributions = param_distributions
        self.scoring = scoring
        self.verbose = verbose

        # Set random seed
        self.rng = np.random.RandomState(random_seed)

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

        logger.info(
            f"RandomSearch initialized: {len(param_distributions)} parameters, "
            f"seed={random_seed}"
        )

    def _sample_parameter(self, distribution: Tuple[str, ...]) -> Any:
        """
        Sample a value from a parameter distribution.

        Args:
            distribution: Distribution specification

        Returns:
            Sampled value
        """
        dist_type = distribution[0]

        if dist_type == 'uniform':
            # Uniform distribution: (low, high)
            low, high = distribution[1], distribution[2]
            return self.rng.uniform(low, high)

        elif dist_type == 'log_uniform':
            # Log-uniform distribution: (low, high)
            low, high = distribution[1], distribution[2]
            log_low = np.log(low)
            log_high = np.log(high)
            return np.exp(self.rng.uniform(log_low, log_high))

        elif dist_type == 'int_uniform':
            # Integer uniform: (low, high)
            low, high = distribution[1], distribution[2]
            return self.rng.randint(low, high + 1)

        elif dist_type == 'normal':
            # Normal distribution: (mean, std)
            mean, std = distribution[1], distribution[2]
            return self.rng.normal(mean, std)

        elif dist_type == 'choice':
            # Categorical choice: (choices,)
            choices = distribution[1]
            return self.rng.choice(choices)

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample a complete parameter configuration.

        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        for name, distribution in self.param_distributions.items():
            params[name] = self._sample_parameter(distribution)
        return params

    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform random search.

        Args:
            objective_fn: Function that takes params and returns score
            n_trials: Number of trials to run

        Returns:
            Tuple of (best_params, best_score)

        Example:
            >>> def objective(params):
            ...     return train_model(**params)
            >>> best_params, best_score = search.search(objective, n_trials=50)
        """
        if self.verbose:
            logger.info(f"Starting random search with {n_trials} trials")

        # Initialize best score
        if self.scoring == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

        # Run trials
        for trial in range(n_trials):
            # Sample parameters
            params = self._sample_params()

            # Evaluate objective
            try:
                score = objective_fn(params)

                # Store result
                self.results.append({
                    'params': params,
                    'score': score,
                    'trial': trial
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
                            f"Trial {trial}/{n_trials}: "
                            f"New best score = {score:.6f}"
                        )

            except Exception as e:
                logger.error(f"Error in trial {trial} with params {params}: {e}")
                continue

        if self.verbose:
            logger.info(
                f"Random search complete. Best score: {self.best_score:.6f}, "
                f"Best params: {self.best_params}"
            )

        return self.best_params, self.best_score

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all search results."""
        return self.results

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Get best score."""
        return self.best_score

    def get_top_k(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top k results.

        Args:
            k: Number of top results

        Returns:
            Top k results
        """
        if not self.results:
            return []

        if self.scoring == 'min':
            sorted_results = sorted(self.results, key=lambda x: x['score'])
        else:
            sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)

        return sorted_results[:k]
