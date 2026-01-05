"""
Hyperparameter Tuning with Q-Store v4.1.1

Demonstrates all tuning methods:
- Grid Search
- Random Search
- Bayesian Optimization
- Optuna Integration

Prerequisites:
    pip install q-store[ml,tuning]
"""

import asyncio
import numpy as np
from q_store.backends import BackendManager
from q_store.ml import (
    QuantumModel,
    QuantumTrainer,
    TrainingConfig,
)

# v4.1.1: Tuning imports
from q_store.ml import (
    GridSearch,
    RandomSearch,
    BayesianOptimizer,
    OptunaTuner,
    OptunaConfig,
)


# Mock training function for demonstration
async def train_model_with_params(params):
    """
    Simplified training function for hyperparameter tuning.

    In real use, this would train a full quantum model.
    For demo purposes, we simulate with a simple function.
    """
    # Simulate training with loss function
    lr = params['learning_rate']
    n_qubits = params['n_qubits']
    depth = params['circuit_depth']

    # Simulate: loss decreases with better parameters
    # (This is just for demo - replace with actual training)
    loss = 0.5 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth
    loss += np.random.normal(0, 0.01)  # Add noise

    return loss


def objective_sync(params):
    """Synchronous wrapper for async training."""
    return asyncio.run(train_model_with_params(params))


async def main():
    """Main tuning demonstration."""

    print("=" * 80)
    print("Q-Store v4.1.1: Hyperparameter Tuning Example")
    print("=" * 80)

    # =========================================================================
    # 1. Grid Search
    # =========================================================================
    print("\n[1/4] Grid Search")
    print("-" * 80)

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_qubits': [4, 6, 8],
        'circuit_depth': [2, 3, 4]
    }

    grid_search = GridSearch(
        param_grid=param_grid,
        scoring='min',
        verbose=True
    )

    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {grid_search.n_combinations}")
    print("\nRunning grid search...")

    best_params, best_score = grid_search.search(
        objective_fn=objective_sync,
        n_trials=10  # Limit for demo
    )

    print("\n✓ Grid Search Results:")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_score:.6f}")

    # Show top 3 results
    print("\n  Top 3 configurations:")
    top_3 = grid_search.get_top_k(3)
    for i, result in enumerate(top_3, 1):
        print(f"    {i}. Score: {result['score']:.6f}, Params: {result['params']}")

    # =========================================================================
    # 2. Random Search
    # =========================================================================
    print("\n\n[2/4] Random Search")
    print("-" * 80)

    param_distributions = {
        'learning_rate': ('log_uniform', 1e-4, 1e-1),
        'n_qubits': ('int_uniform', 4, 12),
        'circuit_depth': ('int_uniform', 2, 6)
    }

    random_search = RandomSearch(
        param_distributions=param_distributions,
        scoring='min',
        random_seed=42,
        verbose=True
    )

    print("Parameter distributions:")
    for name, dist in param_distributions.items():
        print(f"  {name}: {dist[0]} {dist[1:]}")
    print("\nRunning random search with 20 trials...")

    best_params, best_score = random_search.search(
        objective_fn=objective_sync,
        n_trials=20
    )

    print("\n✓ Random Search Results:")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_score:.6f}")

    # Show top 3 results
    print("\n  Top 3 configurations:")
    top_3 = random_search.get_top_k(3)
    for i, result in enumerate(top_3, 1):
        print(f"    {i}. Score: {result['score']:.6f}")
        for k, v in result['params'].items():
            print(f"       {k}: {v if isinstance(v, int) else f'{v:.6f}'}")

    # =========================================================================
    # 3. Bayesian Optimization
    # =========================================================================
    print("\n\n[3/4] Bayesian Optimization")
    print("-" * 80)

    param_bounds = {
        'learning_rate': (1e-4, 1e-1),
        'n_qubits': (4, 12),
        'circuit_depth': (2, 6)
    }

    bayes_optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        scoring='min',
        random_seed=42,
        verbose=1,
        n_init_points=5
    )

    print("Parameter bounds:")
    for name, bounds in param_bounds.items():
        print(f"  {name}: [{bounds[0]}, {bounds[1]}]")
    print("\nRunning Bayesian optimization with 15 trials...")

    # Note: Bayesian optimization expects to maximize, so we negate
    def objective_for_bayes(params):
        loss = objective_sync(params)
        return -loss  # Negate for maximization

    best_params, best_score = bayes_optimizer.optimize(
        objective_fn=objective_for_bayes,
        n_trials=15
    )

    # Correct the score back
    best_score = -best_score

    print("\n✓ Bayesian Optimization Results:")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_score:.6f}")

    # =========================================================================
    # 4. Optuna Integration
    # =========================================================================
    print("\n\n[4/4] Optuna Integration")
    print("-" * 80)

    optuna_config = OptunaConfig(
        study_name='quantum_ml_tuning',
        direction='minimize',
        n_trials=20,
        sampler='TPE',
        pruner='MedianPruner'
    )

    optuna_tuner = OptunaTuner(optuna_config)

    if not optuna_tuner.optuna_available:
        print("⚠ Optuna not available. Install with: pip install optuna")
        print("  Skipping Optuna example...")
    else:
        print("Optuna configuration:")
        print(f"  Study name: {optuna_config.study_name}")
        print(f"  Direction: {optuna_config.direction}")
        print(f"  Sampler: {optuna_config.sampler}")
        print(f"  Pruner: {optuna_config.pruner}")

        # Define Optuna objective
        def optuna_objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'n_qubits': trial.suggest_int('n_qubits', 4, 12),
                'circuit_depth': trial.suggest_int('circuit_depth', 2, 6)
            }
            return objective_sync(params)

        print("\nRunning Optuna optimization with 20 trials...")
        best_params = optuna_tuner.optimize(
            objective=optuna_objective,
            n_trials=20
        )

        best_value = optuna_tuner.get_best_value()

        print("\n✓ Optuna Results:")
        print(f"  Best params: {best_params}")
        print(f"  Best value: {best_value:.6f}")

        # Show study statistics
        trials_df = optuna_tuner.get_trials_dataframe()
        if trials_df is not None:
            print(f"\n  Total trials: {len(trials_df)}")
            print(f"  Complete trials: {len(trials_df[trials_df['state'] == 'COMPLETE'])}")

    # =========================================================================
    # Summary and Comparison
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("Tuning Methods Comparison")
    print("=" * 80)

    results = [
        ("Grid Search", grid_search.get_best_score(), grid_search.get_best_params()),
        ("Random Search", random_search.get_best_score(), random_search.get_best_params()),
        ("Bayesian Opt", best_score, best_params),
    ]

    if optuna_tuner.optuna_available:
        results.append(
            ("Optuna", optuna_tuner.get_best_value(), optuna_tuner.get_best_params())
        )

    # Sort by score (lower is better for loss)
    results.sort(key=lambda x: x[1])

    print("\nRanked Results (best to worst):")
    for i, (method, score, params) in enumerate(results, 1):
        print(f"\n{i}. {method}")
        print(f"   Score: {score:.6f}")
        print(f"   Params: {params}")

    print("\n" + "=" * 80)
    print("Tuning Method Recommendations:")
    print("=" * 80)
    print("""
Grid Search:
  ✓ Exhaustive search over discrete parameter space
  ✓ Best when: Few parameters, discrete values, need guaranteed coverage
  ✗ Slow with many parameters (combinatorial explosion)

Random Search:
  ✓ Efficient exploration of large parameter spaces
  ✓ Best when: Many parameters, continuous values, limited budget
  ✓ Often finds good solutions faster than grid search

Bayesian Optimization:
  ✓ Smart search using Gaussian processes
  ✓ Best when: Expensive evaluations, need quick convergence
  ✓ Balances exploration and exploitation

Optuna (Recommended):
  ✓ State-of-the-art algorithms (TPE, CMA-ES)
  ✓ Automatic pruning of unpromising trials
  ✓ Best when: Complex search spaces, need best performance
  ✓ Parallel optimization, visualization, persistence
    """)
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
