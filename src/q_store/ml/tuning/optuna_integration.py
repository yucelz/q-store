"""
Optuna Integration for Advanced Hyperparameter Optimization.

Provides interface to Optuna for state-of-the-art hyperparameter tuning.

Example:
    >>> from q_store.ml.tuning import OptunaTuner, OptunaConfig
    >>>
    >>> config = OptunaConfig(
    ...     study_name='quantum_ml_study',
    ...     direction='minimize',
    ...     n_trials=100
    ... )
    >>>
    >>> def objective(trial):
    ...     lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    ...     n_qubits = trial.suggest_int('n_qubits', 4, 12)
    ...     depth = trial.suggest_int('circuit_depth', 2, 6)
    ...     return train_model(lr=lr, n_qubits=n_qubits, depth=depth)
    >>>
    >>> tuner = OptunaTuner(config)
    >>> best_params = tuner.optimize(objective)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    """
    Configuration for Optuna tuner.

    Args:
        study_name: Name of the Optuna study (default: 'quantum_ml_study')
        direction: 'minimize' or 'maximize' (default: 'minimize')
        n_trials: Number of trials to run (default: 100)
        timeout: Time limit in seconds (default: None)
        n_jobs: Number of parallel jobs (default: 1)
        sampler: Optuna sampler name (default: 'TPE')
        pruner: Optuna pruner name (default: 'MedianPruner')
        load_if_exists: Load existing study if available (default: False)
        storage: Database URL for persistent storage (default: None, in-memory)

    Example:
        >>> config = OptunaConfig(
        ...     study_name='my_experiment',
        ...     direction='maximize',
        ...     n_trials=50,
        ...     n_jobs=4
        ... )
    """
    study_name: str = 'quantum_ml_study'
    direction: str = 'minimize'
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    sampler: str = 'TPE'
    pruner: str = 'MedianPruner'
    load_if_exists: bool = False
    storage: Optional[str] = None
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    pruner_kwargs: Dict[str, Any] = field(default_factory=dict)


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for quantum ML.

    Provides advanced optimization with features:
    - Multiple sampling algorithms (TPE, CMA-ES, Grid, Random)
    - Pruning for early stopping of unpromising trials
    - Parallel optimization
    - Visualization and analysis tools

    Requires: pip install optuna

    Args:
        config: OptunaConfig instance or None for defaults

    Example:
        >>> config = OptunaConfig(n_trials=50, n_jobs=2)
        >>> tuner = OptunaTuner(config)
        >>>
        >>> def objective(trial):
        ...     params = {
        ...         'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        ...         'depth': trial.suggest_int('depth', 2, 6)
        ...     }
        ...     return train_and_evaluate(**params)
        >>>
        >>> best_params = tuner.optimize(objective)
        >>> print(f"Best params: {best_params}")
    """

    def __init__(self, config: Optional[OptunaConfig] = None):
        """
        Initialize Optuna tuner.

        Args:
            config: OptunaConfig instance
        """
        self.config = config or OptunaConfig()

        # Try to import Optuna
        try:
            import optuna
            self.optuna = optuna
            self.optuna_available = True

            # Suppress Optuna logging if needed
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        except ImportError:
            logger.warning(
                "Optuna not available. Install with: pip install optuna\n"
                "Tuner will not function without Optuna."
            )
            self.optuna = None
            self.optuna_available = False
            return

        # Initialize study
        self.study = None
        self._create_study()

        logger.info(
            f"OptunaTuner initialized: study='{self.config.study_name}', "
            f"direction={self.config.direction}"
        )

    def _create_study(self):
        """Create or load Optuna study."""
        if not self.optuna_available:
            return

        try:
            # Create sampler
            sampler = self._create_sampler()

            # Create pruner
            pruner = self._create_pruner()

            # Create study
            self.study = self.optuna.create_study(
                study_name=self.config.study_name,
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
                storage=self.config.storage,
                load_if_exists=self.config.load_if_exists
            )

            logger.info(f"Created Optuna study: {self.config.study_name}")

        except Exception as e:
            logger.error(f"Failed to create study: {e}")
            raise

    def _create_sampler(self):
        """Create Optuna sampler."""
        sampler_name = self.config.sampler
        kwargs = self.config.sampler_kwargs

        if sampler_name == 'TPE':
            return self.optuna.samplers.TPESampler(**kwargs)
        elif sampler_name == 'CMA-ES':
            return self.optuna.samplers.CmaEsSampler(**kwargs)
        elif sampler_name == 'Grid':
            return self.optuna.samplers.GridSampler(**kwargs)
        elif sampler_name == 'Random':
            return self.optuna.samplers.RandomSampler(**kwargs)
        else:
            logger.warning(f"Unknown sampler '{sampler_name}', using TPE")
            return self.optuna.samplers.TPESampler()

    def _create_pruner(self):
        """Create Optuna pruner."""
        pruner_name = self.config.pruner
        kwargs = self.config.pruner_kwargs

        if pruner_name == 'MedianPruner':
            return self.optuna.pruners.MedianPruner(**kwargs)
        elif pruner_name == 'PercentilePruner':
            return self.optuna.pruners.PercentilePruner(**kwargs)
        elif pruner_name == 'SuccessiveHalvingPruner':
            return self.optuna.pruners.SuccessiveHalvingPruner(**kwargs)
        elif pruner_name == 'HyperbandPruner':
            return self.optuna.pruners.HyperbandPruner(**kwargs)
        elif pruner_name == 'NopPruner':
            return self.optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner '{pruner_name}', using MedianPruner")
            return self.optuna.pruners.MedianPruner()

    def optimize(
        self,
        objective: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            objective: Objective function that takes trial and returns score
            n_trials: Number of trials (overrides config if provided)
            timeout: Timeout in seconds (overrides config if provided)
            callbacks: List of callback functions

        Returns:
            Best parameters found

        Example:
            >>> def objective(trial):
            ...     lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            ...     return train_model(lr=lr)
            >>> best_params = tuner.optimize(objective, n_trials=50)
        """
        if not self.optuna_available or self.study is None:
            logger.error("Optuna not available or study not created")
            return {}

        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout

        logger.info(
            f"Starting optimization: n_trials={n_trials}, "
            f"timeout={timeout}, n_jobs={self.config.n_jobs}"
        )

        try:
            # Run optimization
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.config.n_jobs,
                callbacks=callbacks,
                show_progress_bar=True
            )

            # Get best results
            best_trial = self.study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value

            logger.info(
                f"Optimization complete. Best value: {best_value:.6f}, "
                f"Best params: {best_params}"
            )

            return best_params

        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best parameters from study.

        Returns:
            Best parameters dictionary
        """
        if self.study is None:
            return {}
        return self.study.best_params

    def get_best_value(self) -> Optional[float]:
        """
        Get best objective value.

        Returns:
            Best value
        """
        if self.study is None:
            return None
        return self.study.best_value

    def get_best_trial(self):
        """
        Get best trial object.

        Returns:
            Best trial
        """
        if self.study is None:
            return None
        return self.study.best_trial

    def get_trials(self) -> List:
        """
        Get all trials.

        Returns:
            List of trial objects
        """
        if self.study is None:
            return []
        return self.study.trials

    def get_trials_dataframe(self):
        """
        Get trials as pandas DataFrame.

        Returns:
            DataFrame with trial results

        Example:
            >>> df = tuner.get_trials_dataframe()
            >>> print(df[['number', 'value', 'params_lr', 'params_depth']])
        """
        if self.study is None:
            return None
        return self.study.trials_dataframe()

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save plot (default: None, show plot)

        Example:
            >>> tuner.plot_optimization_history('optimization_history.png')
        """
        if not self.optuna_available or self.study is None:
            logger.warning("Cannot plot: Optuna not available or no study")
            return

        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)

            if save_path:
                fig.write_image(save_path)
                logger.info(f"Saved optimization history to {save_path}")
            else:
                fig.show()

        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances.

        Args:
            save_path: Path to save plot

        Example:
            >>> tuner.plot_param_importances('param_importances.png')
        """
        if not self.optuna_available or self.study is None:
            logger.warning("Cannot plot: Optuna not available or no study")
            return

        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_image(save_path)
                logger.info(f"Saved parameter importances to {save_path}")
            else:
                fig.show()

        except Exception as e:
            logger.error(f"Error plotting parameter importances: {e}")

    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """
        Plot parallel coordinate plot.

        Args:
            save_path: Path to save plot

        Example:
            >>> tuner.plot_parallel_coordinate('parallel_coordinate.png')
        """
        if not self.optuna_available or self.study is None:
            logger.warning("Cannot plot: Optuna not available or no study")
            return

        try:
            from optuna.visualization import plot_parallel_coordinate

            fig = plot_parallel_coordinate(self.study)

            if save_path:
                fig.write_image(save_path)
                logger.info(f"Saved parallel coordinate plot to {save_path}")
            else:
                fig.show()

        except Exception as e:
            logger.error(f"Error plotting parallel coordinate: {e}")

    def __repr__(self) -> str:
        """String representation."""
        if self.study:
            n_trials = len(self.study.trials)
            return f"OptunaTuner(study='{self.config.study_name}', n_trials={n_trials})"
        else:
            return f"OptunaTuner(study='{self.config.study_name}', not initialized)"
