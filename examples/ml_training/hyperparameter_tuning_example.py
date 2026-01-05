"""
Hyperparameter Tuning Example - Demonstrates various tuning strategies.

This example shows:
- Grid search
- Random search
- Bayesian optimization
- Optuna integration
- Complete tuning workflow
"""

import numpy as np
from q_store.ml.tuning import (
    GridSearch,
    RandomSearch,
    BayesianOptimizer,
    OptunaTuner,
    OptunaConfig
)


def example_grid_search():
    """Grid search for hyperparameters."""
    print("\n" + "="*70)
    print("Example 1: Grid Search")
    print("="*70)

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_qubits': [4, 6, 8],
        'circuit_depth': [2, 3, 4]
    }

    # Define objective function
    def objective(params):
        """Simulate training and return loss."""
        lr = params['learning_rate']
        n_qubits = params['n_qubits']
        depth = params['circuit_depth']

        # Simulate loss (lower is better)
        loss = 1.0 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth
        loss += np.random.randn() * 0.01  # Add noise

        return loss

    # Run grid search
    print("\nRunning grid search:")
    print(f"  Parameter combinations: {3 * 3 * 3} (27)")

    grid_search = GridSearch(
        param_grid=param_grid,
        scoring='min',
        verbose=True
    )

    best_params, best_score = grid_search.search(objective)

    print(f"\n✓ Grid search completed")
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {best_score:.6f}")
    print(f"  Total evaluations: {len(grid_search.results)}")

    return grid_search


def example_random_search():
    """Random search for hyperparameters."""
    print("\n" + "="*70)
    print("Example 2: Random Search")
    print("="*70)

    # Define parameter distributions
    param_distributions = {
        'learning_rate': ('log_uniform', 1e-4, 1e-1),
        'n_qubits': ('int_uniform', 4, 12),
        'circuit_depth': ('int_uniform', 2, 6),
        'batch_size': ('choice', [16, 32, 64, 128])
    }

    # Define objective
    def objective(params):
        lr = params['learning_rate']
        n_qubits = params['n_qubits']
        depth = params['circuit_depth']
        batch_size = params['batch_size']

        # Simulate training
        loss = 1.0 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth + 0.001 * batch_size
        loss += np.random.randn() * 0.01

        return loss

    # Run random search
    print("\nRunning random search:")
    random_search = RandomSearch(
        param_distributions=param_distributions,
        scoring='min',
        random_seed=42,
        verbose=True
    )

    best_params, best_score = random_search.search(objective, n_trials=50)

    print(f"\n✓ Random search completed")
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {best_score:.6f}")
    print(f"  Trials: {len(random_search.results)}")

    return random_search


def example_bayesian_optimization():
    """Bayesian optimization for hyperparameters."""
    print("\n" + "="*70)
    print("Example 3: Bayesian Optimization")
    print("="*70)

    # Define parameter bounds
    param_bounds = {
        'learning_rate': (1e-4, 1e-1),
        'n_qubits': (4, 12),
        'circuit_depth': (2, 6)
    }

    # Define objective
    def objective(params_or_lr=None, n_qubits=None, circuit_depth=None):
        # Handle both dict (fallback) and separate args (bayesian-optimization)
        if isinstance(params_or_lr, dict):
            # Fallback mode passes params as dict
            params = params_or_lr
            learning_rate = params['learning_rate']
            n_qubits = params['n_qubits']
            circuit_depth = params['circuit_depth']
        else:
            # Bayesian-optimization mode passes as separate args
            learning_rate = params_or_lr
        
        # Simulate training
        loss = 1.0 / (learning_rate * 10) + 0.1 / n_qubits + 0.05 * circuit_depth
        loss += np.random.randn() * 0.01

        # Return negative for maximization (bayes_opt maximizes)
        return -loss

    print("\nRunning Bayesian optimization:")

    try:
        optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            scoring='max',  # Maximizing objective
            random_seed=42,
            verbose=1,
            n_init_points=5
        )

        best_params, best_score = optimizer.optimize(
            objective_fn=objective,
            n_trials=30
        )

        print(f"\n✓ Bayesian optimization completed")
        print(f"  Best parameters: {best_params}")
        print(f"  Best score: {-best_score:.6f}")  # Negate back
        print(f"  Trials: 30")

        return optimizer

    except ImportError:
        print("⚠ bayesian-optimization not available")
        print("Install with: pip install bayesian-optimization")
        return None


def example_optuna_basic():
    """Basic Optuna tuning."""
    print("\n" + "="*70)
    print("Example 4: Optuna Basic Tuning")
    print("="*70)

    config = OptunaConfig(
        study_name='quantum_ml_basic',
        direction='minimize',
        n_trials=30,
        sampler='TPE',
        pruner='MedianPruner'
    )

    try:
        tuner = OptunaTuner(config)

        # Define objective
        def objective(trial):
            # Suggest parameters
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            n_qubits = trial.suggest_int('n_qubits', 4, 12)
            depth = trial.suggest_int('circuit_depth', 2, 6)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

            # Simulate training
            loss = 1.0 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth + 0.001 * batch_size
            loss += np.random.randn() * 0.01

            # Report intermediate values for pruning
            for epoch in range(10):
                intermediate_loss = loss * (1.0 - epoch * 0.05)
                trial.report(intermediate_loss, epoch)

                if trial.should_prune():
                    raise Exception("Trial pruned")

            return loss

        print("\nRunning Optuna optimization:")
        best_params = tuner.optimize(objective)

        print(f"\n✓ Optuna optimization completed")
        print(f"  Best parameters: {best_params}")
        if hasattr(tuner, 'study') and tuner.study:
            print(f"  Best value: {tuner.study.best_value:.6f}")
            print(f"  Trials: {len(tuner.study.trials)}")
            pruned = [t for t in tuner.study.trials if t.state.name == 'PRUNED']
            print(f"  Pruned trials: {len(pruned)}")
        else:
            print(f"  (Optuna not available - no study created)")

        return tuner

    except ImportError:
        print("⚠ Optuna not available")
        print("Install with: pip install optuna")
        return None


def example_optuna_advanced():
    """Advanced Optuna with visualization."""
    print("\n" + "="*70)
    print("Example 5: Optuna Advanced Features")
    print("="*70)

    config = OptunaConfig(
        study_name='quantum_ml_advanced',
        direction='minimize',
        n_trials=50,
        n_jobs=2,  # Parallel optimization
        sampler='TPE',
        pruner='MedianPruner',
        storage=None  # In-memory
    )

    try:
        tuner = OptunaTuner(config)

        def objective(trial):
            # Multi-dimensional parameter space
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            n_qubits = trial.suggest_int('n_qubits', 4, 12)
            depth = trial.suggest_int('circuit_depth', 2, 6)
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
            dropout = trial.suggest_float('dropout', 0.0, 0.5)

            # Simulate training
            loss = 1.0 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth
            loss += {'adam': 0.0, 'sgd': 0.1, 'rmsprop': 0.05}[optimizer_name]
            loss += dropout * 0.2
            loss += np.random.randn() * 0.01

            return loss

        print("\nRunning advanced Optuna optimization:")
        best_params = tuner.optimize(objective)

        print(f"\n✓ Advanced Optuna optimization completed")
        print(f"  Best parameters: {best_params}")
        if hasattr(tuner, 'study') and tuner.study:
            print(f"  Best value: {tuner.study.best_value:.6f}")
            
            # Get best trials
            best_trials = tuner.get_best_trials(n_trials=3)
            print(f"\nTop 3 trials:")
            for i, trial in enumerate(best_trials, 1):
                print(f"  {i}. Value: {trial.value:.6f}, Params: {trial.params}")
        else:
            print(f"  (Optuna not available - no study created)")

        # Generate visualizations
        print("\nGenerating visualizations...")
        try:
            tuner.plot_optimization_history('optuna_history.png')
            print("  Saved: optuna_history.png")

            tuner.plot_param_importances('optuna_importance.png')
            print("  Saved: optuna_importance.png")

            tuner.plot_parallel_coordinate('optuna_parallel.png')
            print("  Saved: optuna_parallel.png")
        except Exception as e:
            print(f"  ⚠ Visualization failed: {e}")

        return tuner

    except ImportError:
        print("⚠ Optuna not available")
        print("Install with: pip install optuna")
        return None


def example_complete_tuning_workflow():
    """Complete hyperparameter tuning workflow."""
    print("\n" + "="*70)
    print("Example 6: Complete Tuning Workflow")
    print("="*70)

    # Step 1: Quick grid search
    print("\nStep 1: Coarse grid search")
    coarse_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_qubits': [4, 8]
    }

    def objective(params):
        lr = params.get('learning_rate', 0.01)
        n_qubits = params.get('n_qubits', 4)
        depth = params.get('circuit_depth', 3)

        loss = 1.0 / (lr * 10) + 0.1 / n_qubits + 0.05 * depth
        return loss

    grid = GridSearch(coarse_grid, scoring='min', verbose=False)
    best_coarse, _ = grid.search(objective)
    print(f"  Best from coarse search: {best_coarse}")

    # Step 2: Refined random search
    print("\nStep 2: Refined random search around best parameters")
    best_lr = best_coarse['learning_rate']
    refined_distributions = {
        'learning_rate': ('uniform', best_lr * 0.5, best_lr * 1.5),
        'n_qubits': ('int_uniform', 6, 10),
        'circuit_depth': ('int_uniform', 2, 5)
    }

    random_search = RandomSearch(
        refined_distributions,
        scoring='min',
        verbose=False
    )
    best_refined, _ = random_search.search(objective, n_trials=20)
    print(f"  Best from refined search: {best_refined}")

    # Step 3: Final Bayesian optimization
    print("\nStep 3: Fine-tuning with Bayesian optimization")
    try:
        best_lr = best_refined['learning_rate']
        bounds = {
            'learning_rate': (best_lr * 0.8, best_lr * 1.2),
            'n_qubits': (7, 10),
            'circuit_depth': (2, 4)
        }

        def bayes_objective(learning_rate, n_qubits, circuit_depth):
            loss = 1.0 / (learning_rate * 10) + 0.1 / n_qubits + 0.05 * circuit_depth
            return -loss  # Maximize

        optimizer = BayesianOptimizer(bounds, scoring='max', verbose=0)
        best_final, _ = optimizer.optimize(bayes_objective, n_trials=15)
        print(f"  Best from Bayesian opt: {best_final}")

    except ImportError:
        print("  ⚠ Skipping Bayesian optimization (not installed)")
        best_final = best_refined

    print(f"\n✓ Complete tuning workflow finished")
    print(f"  Final best parameters: {best_final}")
    print(f"\nWorkflow steps:")
    print(f"  1. Coarse grid search: {len(coarse_grid['learning_rate']) * len(coarse_grid['n_qubits'])} trials")
    print(f"  2. Refined random search: 20 trials")
    print(f"  3. Bayesian optimization: 15 trials")
    print(f"  Total: ~47 trials (vs {3**3 * 2} (54) for exhaustive grid)")

    return best_final


def example_comparison():
    """Compare different tuning methods."""
    print("\n" + "="*70)
    print("Example 7: Method Comparison")
    print("="*70)

    # Common objective
    def objective_dict(params):
        lr = params.get('learning_rate', 0.01)
        n_qubits = params.get('n_qubits', 4)
        loss = 1.0 / (lr * 10) + 0.1 / n_qubits
        return loss

    results = {}

    # Grid search
    print("\n1. Grid Search (27 trials)")
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_qubits': [4, 6, 8]
    }
    grid = GridSearch(param_grid, scoring='min', verbose=False)
    best_params, best_score = grid.search(objective_dict)
    results['grid'] = {'params': best_params, 'score': best_score, 'trials': 27}
    print(f"   Best score: {best_score:.6f}")

    # Random search
    print("\n2. Random Search (27 trials)")
    param_dist = {
        'learning_rate': ('log_uniform', 1e-4, 1e-1),
        'n_qubits': ('int_uniform', 4, 12)
    }
    random = RandomSearch(param_dist, scoring='min', verbose=False)
    best_params, best_score = random.search(objective_dict, n_trials=27)
    results['random'] = {'params': best_params, 'score': best_score, 'trials': 27}
    print(f"   Best score: {best_score:.6f}")

    # Summary
    print("\n" + "="*50)
    print("Comparison Summary:")
    print("="*50)
    for method, data in results.items():
        print(f"{method.capitalize():12s}: score={data['score']:.6f}, trials={data['trials']}")

    print("\n✓ Method comparison completed")

    return results


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Q-Store Hyperparameter Tuning Examples")
    print("="*70)

    # Example 1: Grid search
    try:
        example_grid_search()
    except Exception as e:
        print(f"⚠ Grid search failed: {e}")

    # Example 2: Random search
    try:
        example_random_search()
    except Exception as e:
        print(f"⚠ Random search failed: {e}")

    # Example 3: Bayesian optimization
    example_bayesian_optimization()

    # Example 4: Optuna basic
    example_optuna_basic()

    # Example 5: Optuna advanced
    example_optuna_advanced()

    # Example 6: Complete workflow
    try:
        example_complete_tuning_workflow()
    except Exception as e:
        print(f"⚠ Complete workflow failed: {e}")

    # Example 7: Comparison
    try:
        example_comparison()
    except Exception as e:
        print(f"⚠ Comparison failed: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
