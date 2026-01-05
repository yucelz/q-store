"""
Bayesian Optimization for Hyperparameter Tuning.

Uses Gaussian process-based Bayesian optimization for efficient search.

Example:
    >>> from q_store.ml.tuning import BayesianOptimizer
    >>>
    >>> param_bounds = {
    ...     'learning_rate': (1e-4, 1e-1),
    ...     'n_qubits': (4, 12),
    ...     'circuit_depth': (2, 6)
    ... }
    >>>
    >>> optimizer = BayesianOptimizer(param_bounds)
    >>> best_params, best_score = optimizer.optimize(objective, n_trials=30)
"""

import logging
from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.

    Uses Gaussian process-based optimization for smart exploration.
    Requires: pip install bayesian-optimization

    Args:
        param_bounds: Dictionary mapping parameter names to (min, max) bounds
        scoring: 'min' or 'max' (default: 'max' for compatibility with bayesian-optimization)
        random_seed: Random seed (default: None)
        verbose: Verbosity level (0-2) (default: 1)
        n_init_points: Number of random initialization points (default: 5)

    Example:
        >>> bounds = {
        ...     'lr': (1e-4, 1e-1),
        ...     'batch_size': (16, 128),
        ...     'n_layers': (2, 8)
        ... }
        >>> optimizer = BayesianOptimizer(bounds, random_seed=42)
        >>> best_params, best_score = optimizer.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        scoring: str = 'max',
        random_seed: Optional[int] = None,
        verbose: int = 1,
        n_init_points: int = 5
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            param_bounds: Parameter bounds
            scoring: Scoring mode
            random_seed: Random seed
            verbose: Verbosity
            n_init_points: Initial random points
        """
        if scoring not in ['min', 'max']:
            raise ValueError(f"scoring must be 'min' or 'max', got: {scoring}")

        self.param_bounds = param_bounds
        self.scoring = scoring
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_init_points = n_init_points

        # Try to import bayesian-optimization
        try:
            from bayes_opt import BayesianOptimization
            self.BayesianOptimization = BayesianOptimization
            self.bayes_opt_available = True
        except ImportError:
            logger.warning(
                "bayesian-optimization not available. "
                "Install with: pip install bayesian-optimization\n"
                "Falling back to random search."
            )
            self.bayes_opt_available = False

        # Results storage
        self.results = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

        logger.info(
            f"BayesianOptimizer initialized: {len(param_bounds)} parameters"
        )

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 50
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform Bayesian optimization.

        Args:
            objective_fn: Objective function (params -> score)
            n_trials: Number of optimization trials

        Returns:
            Tuple of (best_params, best_score)

        Example:
            >>> def objective(params):
            ...     return -train_model(**params)  # Negative for minimization
            >>> best_params, best_score = optimizer.optimize(objective, n_trials=30)
        """
        if not self.bayes_opt_available:
            # Fallback to random search
            return self._random_search_fallback(objective_fn, n_trials)

        if self.verbose > 0:
            logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        # Wrapper for objective function
        def wrapped_objective(**kwargs):
            params = kwargs
            score = objective_fn(params)

            # Store result
            self.results.append({
                'params': params.copy(),
                'score': score
            })

            # Bayesian optimization maximizes, so negate if minimizing
            if self.scoring == 'min':
                return -score
            else:
                return score

        # Create optimizer
        optimizer = self.BayesianOptimization(
            f=wrapped_objective,
            pbounds=self.param_bounds,
            random_state=self.random_seed,
            verbose=max(0, self.verbose - 1)
        )

        # Run optimization
        optimizer.maximize(
            init_points=self.n_init_points,
            n_iter=n_trials - self.n_init_points
        )

        # Extract best results
        self.best_params = optimizer.max['params']
        self.best_score = optimizer.max['target']

        # Correct score if minimizing
        if self.scoring == 'min':
            self.best_score = -self.best_score

        if self.verbose > 0:
            logger.info(
                f"Optimization complete. Best score: {self.best_score:.6f}, "
                f"Best params: {self.best_params}"
            )

        return self.best_params, self.best_score

    def _random_search_fallback(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int
    ) -> Tuple[Dict[str, Any], float]:
        """
        Fallback to random search if bayesian-optimization not available.

        Args:
            objective_fn: Objective function
            n_trials: Number of trials

        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info("Using random search fallback")

        rng = np.random.RandomState(self.random_seed)

        # Initialize best
        if self.scoring == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

        # Random search
        for trial in range(n_trials):
            # Sample parameters
            params = {}
            for name, (low, high) in self.param_bounds.items():
                params[name] = rng.uniform(low, high)

            # Evaluate
            try:
                score = objective_fn(params)

                self.results.append({
                    'params': params,
                    'score': score
                })

                # Update best
                is_better = (
                    (self.scoring == 'min' and score < self.best_score) or
                    (self.scoring == 'max' and score > self.best_score)
                )

                if is_better:
                    self.best_score = score
                    self.best_params = params.copy()

                    if self.verbose > 0:
                        logger.info(f"Trial {trial}: New best = {score:.6f}")

            except Exception as e:
                logger.error(f"Error in trial {trial}: {e}")

        return self.best_params, self.best_score

    def get_results(self):
        """Get all optimization results."""
        return self.results

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Get best score."""
        return self.best_score
