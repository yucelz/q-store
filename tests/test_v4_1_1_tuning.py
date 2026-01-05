"""
Unit tests for Q-Store v4.1.1 Hyperparameter Tuning.

Tests cover:
- GridSearch
- RandomSearch
- BayesianOptimizer
- OptunaTuner
- OptunaConfig
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from q_store.ml.tuning import (
    GridSearch,
    RandomSearch,
    BayesianOptimizer,
    OptunaTuner,
    OptunaConfig,
)


class TestGridSearch:
    """Test GridSearch hyperparameter optimization."""

    def test_grid_search_basic(self):
        """Test basic grid search."""
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32]
        }

        grid_search = GridSearch(param_grid)

        # Count total combinations
        assert grid_search.get_total_combinations() == 6  # 3 * 2

    def test_grid_search_iteration(self):
        """Test iterating through parameter combinations."""
        param_grid = {
            'a': [1, 2],
            'b': [3, 4]
        }

        grid_search = GridSearch(param_grid)

        combinations = list(grid_search.get_param_combinations())

        assert len(combinations) == 4
        assert {'a': 1, 'b': 3} in combinations
        assert {'a': 2, 'b': 4} in combinations

    def test_grid_search_with_objective(self):
        """Test grid search with objective function."""
        param_grid = {
            'x': [1, 2, 3],
            'y': [4, 5]
        }

        # Simple objective: minimize x + y
        def objective(params):
            return params['x'] + params['y']

        grid_search = GridSearch(param_grid)
        best_params, best_score = grid_search.search(
            objective_fn=objective,
            minimize=True
        )

        assert best_params == {'x': 1, 'y': 4}
        assert best_score == 5

    def test_grid_search_maximize(self):
        """Test grid search with maximization."""
        param_grid = {
            'x': [1, 2, 3]
        }

        def objective(params):
            return params['x'] ** 2

        grid_search = GridSearch(param_grid)
        best_params, best_score = grid_search.search(
            objective_fn=objective,
            minimize=False
        )

        assert best_params == {'x': 3}
        assert best_score == 9

    def test_grid_search_single_param(self):
        """Test grid search with single parameter."""
        param_grid = {'learning_rate': [0.001, 0.01, 0.1]}

        grid_search = GridSearch(param_grid)

        combinations = list(grid_search.get_param_combinations())
        assert len(combinations) == 3

    def test_grid_search_empty_grid(self):
        """Test grid search with empty parameter grid."""
        param_grid = {}

        with pytest.raises(ValueError):
            grid_search = GridSearch(param_grid)


class TestRandomSearch:
    """Test RandomSearch hyperparameter optimization."""

    def test_random_search_basic(self):
        """Test basic random search."""
        param_distributions = {
            'learning_rate': (0.001, 0.1, 'log-uniform'),
            'batch_size': ([16, 32, 64], 'choice')
        }

        random_search = RandomSearch(
            param_distributions,
            n_iter=10,
            random_seed=42
        )

        assert random_search.n_iter == 10

    def test_random_search_uniform(self):
        """Test random search with uniform distribution."""
        param_distributions = {
            'x': (0.0, 1.0, 'uniform')
        }

        random_search = RandomSearch(param_distributions, n_iter=100)

        def objective(params):
            return (params['x'] - 0.5) ** 2

        best_params, best_score = random_search.search(
            objective_fn=objective,
            minimize=True
        )

        # Best should be close to 0.5
        assert 0.3 < best_params['x'] < 0.7

    def test_random_search_log_uniform(self):
        """Test random search with log-uniform distribution."""
        param_distributions = {
            'lr': (1e-4, 1e-1, 'log-uniform')
        }

        random_search = RandomSearch(param_distributions, n_iter=50)

        samples = []
        for params in random_search.get_param_samples():
            samples.append(params['lr'])

        # Check that samples span the range
        assert min(samples) >= 1e-4
        assert max(samples) <= 1e-1

    def test_random_search_choice(self):
        """Test random search with categorical choice."""
        param_distributions = {
            'optimizer': (['adam', 'sgd', 'rmsprop'], 'choice')
        }

        random_search = RandomSearch(param_distributions, n_iter=30)

        choices = []
        for params in random_search.get_param_samples():
            choices.append(params['optimizer'])

        # All choices should be from the list
        assert all(c in ['adam', 'sgd', 'rmsprop'] for c in choices)

    def test_random_search_int_uniform(self):
        """Test random search with integer uniform distribution."""
        param_distributions = {
            'n_layers': (1, 10, 'int-uniform')
        }

        random_search = RandomSearch(param_distributions, n_iter=20)

        samples = []
        for params in random_search.get_param_samples():
            samples.append(params['n_layers'])

        # All should be integers
        assert all(isinstance(s, int) for s in samples)
        assert all(1 <= s <= 10 for s in samples)

    def test_random_search_reproducibility(self):
        """Test random search reproducibility with seed."""
        param_distributions = {
            'x': (0.0, 1.0, 'uniform')
        }

        random_search1 = RandomSearch(param_distributions, n_iter=10, random_seed=42)
        random_search2 = RandomSearch(param_distributions, n_iter=10, random_seed=42)

        samples1 = list(random_search1.get_param_samples())
        samples2 = list(random_search2.get_param_samples())

        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert s1 == s2


class TestBayesianOptimizer:
    """Test BayesianOptimizer."""

    def test_bayesian_optimizer_basic(self):
        """Test basic Bayesian optimization."""
        param_bounds = {
            'x': (0.0, 1.0),
            'y': (0.0, 1.0)
        }

        optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            n_iter=20,
            random_seed=42
        )

        def objective(params):
            return -(params['x'] - 0.5) ** 2 - (params['y'] - 0.5) ** 2

        best_params, best_score = optimizer.search(
            objective_fn=objective,
            minimize=False
        )

        # Should find optimum near (0.5, 0.5)
        assert abs(best_params['x'] - 0.5) < 0.2
        assert abs(best_params['y'] - 0.5) < 0.2

    def test_bayesian_optimizer_with_init_points(self):
        """Test Bayesian optimization with initial random points."""
        param_bounds = {'x': (-5.0, 5.0)}

        optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            n_iter=15,
            n_init_points=5
        )

        def objective(params):
            return -params['x'] ** 2

        best_params, best_score = optimizer.search(objective_fn=objective)

        # Should find optimum near x=0
        assert abs(best_params['x']) < 1.0

    def test_bayesian_optimizer_acquisition_function(self):
        """Test different acquisition functions."""
        param_bounds = {'x': (0.0, 10.0)}

        for acq_func in ['ei', 'ucb', 'poi']:
            optimizer = BayesianOptimizer(
                param_bounds=param_bounds,
                n_iter=10,
                acquisition_function=acq_func
            )

            def objective(params):
                return params['x'] ** 2

            best_params, best_score = optimizer.search(
                objective_fn=objective,
                minimize=True
            )

            # Should find minimum near x=0
            assert best_params['x'] < 2.0


class TestOptunaTuner:
    """Test OptunaTuner integration."""

    @patch('q_store.ml.tuning.optuna_integration.optuna')
    def test_optuna_tuner_basic(self, mock_optuna):
        """Test basic Optuna tuner."""
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        mock_study.best_params = {'x': 0.5}
        mock_study.best_value = 0.25

        config = OptunaConfig(
            study_name='test_study',
            direction='minimize',
            n_trials=20
        )

        tuner = OptunaTuner(config)

        def objective(trial):
            x = trial.suggest_float('x', 0.0, 1.0)
            return x ** 2

        best_params, best_value = tuner.optimize(objective)

        assert mock_optuna.create_study.called
        assert mock_study.optimize.called

    @patch('q_store.ml.tuning.optuna_integration.optuna')
    def test_optuna_tuner_with_pruning(self, mock_optuna):
        """Test Optuna tuner with pruning."""
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        mock_study.best_params = {'x': 1.0}
        mock_study.best_value = 1.0

        config = OptunaConfig(
            study_name='test_study',
            direction='minimize',
            n_trials=50,
            pruner='median'
        )

        tuner = OptunaTuner(config)

        def objective(trial):
            x = trial.suggest_int('x', 1, 10)

            # Simulate training loop with intermediate values
            for step in range(10):
                intermediate_value = x * step
                trial.report(intermediate_value, step)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return x

        tuner.optimize(objective)

        assert mock_study.optimize.called

    @patch('q_store.ml.tuning.optuna_integration.optuna')
    def test_optuna_tuner_param_types(self, mock_optuna):
        """Test different parameter types in Optuna."""
        mock_study = Mock()
        mock_trial = Mock()
        mock_optuna.create_study.return_value = mock_study

        config = OptunaConfig(study_name='test', n_trials=10)
        tuner = OptunaTuner(config)

        def objective(trial):
            # Test different suggest methods
            x_float = trial.suggest_float('x_float', 0.0, 1.0)
            x_int = trial.suggest_int('x_int', 1, 10)
            x_cat = trial.suggest_categorical('x_cat', ['a', 'b', 'c'])
            x_log = trial.suggest_float('x_log', 1e-5, 1e-1, log=True)

            return x_float + x_int

        tuner.optimize(objective)

    @patch('q_store.ml.tuning.optuna_integration.optuna')
    def test_optuna_tuner_with_storage(self, mock_optuna):
        """Test Optuna with storage backend."""
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        mock_study.best_params = {}
        mock_study.best_value = 0.0

        config = OptunaConfig(
            study_name='test_study',
            storage='sqlite:///optuna.db',
            n_trials=10
        )

        tuner = OptunaTuner(config)

        def objective(trial):
            return 0.0

        tuner.optimize(objective)

        # Should create study with storage
        mock_optuna.create_study.assert_called()


class TestTuningIntegration:
    """Integration tests for hyperparameter tuning."""

    def test_compare_tuning_methods(self):
        """Test comparing different tuning methods."""
        # Simple quadratic function
        def objective(params):
            return (params['x'] - 5.0) ** 2 + (params['y'] - 3.0) ** 2

        # Grid search
        grid_params = {
            'x': [3.0, 4.0, 5.0, 6.0, 7.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        grid_search = GridSearch(grid_params)
        grid_best, grid_score = grid_search.search(objective, minimize=True)

        # Random search
        random_params = {
            'x': (0.0, 10.0, 'uniform'),
            'y': (0.0, 10.0, 'uniform')
        }
        random_search = RandomSearch(random_params, n_iter=25)
        random_best, random_score = random_search.search(objective, minimize=True)

        # All should find reasonable solutions
        assert grid_score < 10.0
        assert random_score < 10.0

    def test_tuning_with_early_stopping(self):
        """Test tuning with early stopping callback."""
        param_grid = {
            'x': [1, 2, 3, 4, 5]
        }

        def objective(params):
            # Simulate expensive computation
            if params['x'] > 3:
                return float('inf')  # Bad parameter
            return params['x'] ** 2

        grid_search = GridSearch(param_grid)
        best_params, best_score = grid_search.search(
            objective_fn=objective,
            minimize=True
        )

        # Should handle inf values
        assert best_params['x'] <= 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_grid_search_single_value_per_param(self):
        """Test grid search with single value per parameter."""
        param_grid = {
            'a': [1],
            'b': [2]
        }

        grid_search = GridSearch(param_grid)

        def objective(params):
            return params['a'] + params['b']

        best_params, best_score = grid_search.search(objective)

        assert best_params == {'a': 1, 'b': 2}
        assert best_score == 3

    def test_random_search_zero_iterations(self):
        """Test random search with zero iterations."""
        param_distributions = {
            'x': (0.0, 1.0, 'uniform')
        }

        with pytest.raises(ValueError):
            random_search = RandomSearch(param_distributions, n_iter=0)

    def test_bayesian_optimizer_invalid_bounds(self):
        """Test Bayesian optimizer with invalid bounds."""
        param_bounds = {
            'x': (10.0, 0.0)  # Invalid: min > max
        }

        with pytest.raises(ValueError):
            optimizer = BayesianOptimizer(param_bounds)

    def test_objective_function_exception(self):
        """Test handling of exceptions in objective function."""
        param_grid = {'x': [1, 2, 3]}

        def bad_objective(params):
            if params['x'] == 2:
                raise RuntimeError("Computation failed")
            return params['x']

        grid_search = GridSearch(param_grid)

        # Should handle exception gracefully
        try:
            best_params, best_score = grid_search.search(
                objective_fn=bad_objective,
                minimize=True
            )
        except RuntimeError:
            pass  # Expected if not caught internally

    def test_nan_objective_values(self):
        """Test handling of NaN objective values."""
        param_grid = {'x': [1, 2, 3]}

        def nan_objective(params):
            if params['x'] == 2:
                return float('nan')
            return params['x']

        grid_search = GridSearch(param_grid)
        best_params, best_score = grid_search.search(
            objective_fn=nan_objective,
            minimize=True
        )

        # Should skip NaN values
        assert best_params['x'] in [1, 3]


class TestOptunaConfig:
    """Test OptunaConfig dataclass."""

    def test_optuna_config_creation(self):
        """Test creating Optuna configuration."""
        config = OptunaConfig(
            study_name='test_study',
            direction='minimize',
            n_trials=100
        )

        assert config.study_name == 'test_study'
        assert config.direction == 'minimize'
        assert config.n_trials == 100

    def test_optuna_config_with_sampler(self):
        """Test Optuna config with custom sampler."""
        config = OptunaConfig(
            study_name='test',
            sampler='tpe'
        )

        assert config.sampler == 'tpe'

    def test_optuna_config_with_pruner(self):
        """Test Optuna config with pruner."""
        config = OptunaConfig(
            study_name='test',
            pruner='median'
        )

        assert config.pruner == 'median'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
