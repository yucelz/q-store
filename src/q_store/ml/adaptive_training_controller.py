"""
Adaptive Training Controller - v4.1 Enhanced
Self-optimizing training loop based on quantum metrics

Key Innovation: Metrics-driven adaptation for optimal training
- Automatic circuit depth adjustment based on expressibility
- Dynamic measurement policy updates based on gradient variance
- Loss plateau detection and response
- Training phase management (exploration → convergence → refinement)
- Comprehensive adaptation logging

Use Cases:
- Hands-free quantum ML training
- Automatic hyperparameter tuning
- Cost optimization during training
- Training stability improvement

Design:
- Integrates with AdaptiveMeasurementPolicy
- Uses QuantumMetrics for decision making
- Logs all adaptations for analysis
- Non-intrusive (can be disabled)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase enum."""
    EXPLORATION = "exploration"  # Early training, explore broadly
    CONVERGENCE = "convergence"  # Mid training, optimize
    REFINEMENT = "refinement"    # Late training, fine-tune


@dataclass
class AdaptationEvent:
    """
    Record of a training adaptation.

    Attributes
    ----------
    step : int
        Training step when adaptation occurred
    phase : str
        Training phase
    action : str
        Action taken (e.g., 'increase_depth', 'reduce_shots')
    reason : str
        Reason for adaptation
    old_value : any
        Previous value
    new_value : any
        New value
    metadata : dict, optional
        Additional information
    """
    step: int
    phase: str
    action: str
    reason: str
    old_value: Any
    new_value: Any
    metadata: Optional[Dict[str, Any]] = None


class AdaptiveTrainingController:
    """
    Adaptive training controller based on quantum metrics.

    Automatically adjusts training hyperparameters based on observed
    metrics to optimize training efficiency and cost.

    **Adaptations:**
    1. Circuit depth (based on expressibility)
    2. Measurement policy (based on gradient variance)
    3. Training phase transitions
    4. Loss plateau detection

    Parameters
    ----------
    initial_depth : int, default=3
        Initial circuit depth
    max_depth : int, default=8
        Maximum circuit depth
    min_depth : int, default=2
        Minimum circuit depth
    measurement_policy : AdaptiveMeasurementPolicy, optional
        Measurement policy instance
    expressibility_threshold : float, default=0.3
        Threshold for increasing depth
    plateau_window : int, default=20
        Window for plateau detection
    plateau_threshold : float, default=0.01
        Loss improvement threshold

    Examples
    --------
    >>> from q_store.storage import AdaptiveMeasurementPolicy, QuantumMetrics
    >>>
    >>> controller = AdaptiveTrainingController(
    ...     initial_depth=3,
    ...     max_depth=8,
    ...     measurement_policy=AdaptiveMeasurementPolicy()
    ... )
    >>>
    >>> # During training loop
    >>> for step in range(1000):
    ...     # ... compute metrics ...
    ...     metrics = QuantumMetrics(...)
    ...
    ...     # Adapt based on metrics
    ...     changes = controller.adapt(metrics, model)
    ...
    ...     if changes:
    ...         print(f"Adaptations made: {changes}")
    """

    def __init__(
        self,
        initial_depth: int = 3,
        max_depth: int = 8,
        min_depth: int = 2,
        measurement_policy: Optional['AdaptiveMeasurementPolicy'] = None,
        expressibility_threshold: float = 0.3,
        plateau_window: int = 20,
        plateau_threshold: float = 0.01
    ):
        self.current_depth = initial_depth
        self.max_depth = max_depth
        self.min_depth = min_depth

        # Import here to avoid circular dependency
        if measurement_policy is None:
            from q_store.storage.adaptive_measurement import AdaptiveMeasurementPolicy
            self.measurement_policy = AdaptiveMeasurementPolicy()
        else:
            self.measurement_policy = measurement_policy

        self.expressibility_threshold = expressibility_threshold
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

        # State tracking
        self.current_phase = TrainingPhase.EXPLORATION
        self.metrics_history = []
        self.adaptation_log: List[AdaptationEvent] = []

        # Adaptation tracking
        self.last_depth_increase_step = 0
        self.depth_increase_cooldown = 50  # Don't increase depth too frequently

        logger.info(
            f"Initialized AdaptiveTrainingController: "
            f"depth={initial_depth}-{max_depth}, "
            f"expressibility_threshold={expressibility_threshold}"
        )

    def adapt(
        self,
        metrics: 'QuantumMetrics',
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Adapt training based on metrics.

        Analyzes current metrics and makes adaptations to:
        - Circuit depth
        - Measurement policy
        - Training phase

        Parameters
        ----------
        metrics : QuantumMetrics
            Current training metrics
        model : optional
            Model instance (for setting circuit depth)

        Returns
        -------
        dict
            Dictionary of changes made
        """
        self.metrics_history.append(metrics)
        changes = {}

        # 1. Update training phase
        phase_changed = self._update_training_phase(metrics)
        if phase_changed:
            changes['phase'] = self.current_phase.value

        # 2. Adapt circuit depth
        depth_change = self._adapt_circuit_depth(metrics, model)
        if depth_change:
            changes['circuit_depth'] = self.current_depth

        # 3. Adapt measurement policy
        if hasattr(metrics, 'gradient_variance'):
            self.measurement_policy.update_policy(metrics.gradient_variance)

        # 4. Detect plateau
        plateau_detected = self._detect_plateau()
        if plateau_detected:
            changes['plateau_detected'] = True
            self._handle_plateau(metrics, model)

        # Log changes
        if changes:
            logger.info(
                f"Step {metrics.step}: Adaptations made: {changes}"
            )

        return changes

    def _update_training_phase(self, metrics: 'QuantumMetrics') -> bool:
        """
        Update training phase based on metrics.

        Phase transitions:
        - EXPLORATION → CONVERGENCE: After 20% of training
        - CONVERGENCE → REFINEMENT: When gradient variance is low and stable

        Returns
        -------
        bool
            True if phase changed
        """
        old_phase = self.current_phase

        # Heuristic phase transitions
        if self.current_phase == TrainingPhase.EXPLORATION:
            # Transition to convergence after initial exploration
            if metrics.step > 100:  # After ~100 steps
                self.current_phase = TrainingPhase.CONVERGENCE

                self.adaptation_log.append(AdaptationEvent(
                    step=metrics.step,
                    phase=old_phase.value,
                    action='phase_transition',
                    reason='completed exploration phase',
                    old_value=old_phase.value,
                    new_value=self.current_phase.value
                ))

        elif self.current_phase == TrainingPhase.CONVERGENCE:
            # Transition to refinement when gradients are stable
            if len(self.metrics_history) >= 50:
                recent_variance = [
                    m.gradient_variance for m in self.metrics_history[-20:]
                    if hasattr(m, 'gradient_variance')
                ]

                if recent_variance and np.mean(recent_variance) < 0.05:
                    self.current_phase = TrainingPhase.REFINEMENT

                    self.adaptation_log.append(AdaptationEvent(
                        step=metrics.step,
                        phase=old_phase.value,
                        action='phase_transition',
                        reason='gradients stabilized',
                        old_value=old_phase.value,
                        new_value=self.current_phase.value,
                        metadata={'mean_variance': np.mean(recent_variance)}
                    ))

        return self.current_phase != old_phase

    def _adapt_circuit_depth(
        self,
        metrics: 'QuantumMetrics',
        model: Optional[Any]
    ) -> bool:
        """
        Adapt circuit depth based on expressibility.

        Increases depth if:
        - Expressibility is low (< threshold)
        - Not at max depth
        - Enough steps since last increase (cooldown)

        Returns
        -------
        bool
            True if depth changed
        """
        # Check if we should adapt depth
        if not hasattr(metrics, 'expressibility_score'):
            return False

        if metrics.expressibility_score is None:
            return False

        if self.current_depth >= self.max_depth:
            return False

        # Cooldown check
        if metrics.step - self.last_depth_increase_step < self.depth_increase_cooldown:
            return False

        # Low expressibility → increase depth
        if metrics.expressibility_score < self.expressibility_threshold:
            old_depth = self.current_depth
            self.current_depth += 1

            # Update model if provided
            if model is not None and hasattr(model, 'set_circuit_depth'):
                model.set_circuit_depth(self.current_depth)

            self.last_depth_increase_step = metrics.step

            self.adaptation_log.append(AdaptationEvent(
                step=metrics.step,
                phase=self.current_phase.value,
                action='increase_depth',
                reason=f'low expressibility ({metrics.expressibility_score:.3f})',
                old_value=old_depth,
                new_value=self.current_depth,
                metadata={'expressibility': metrics.expressibility_score}
            ))

            logger.info(
                f"Step {metrics.step}: Increased circuit depth: "
                f"{old_depth} → {self.current_depth} "
                f"(expressibility={metrics.expressibility_score:.3f})"
            )

            return True

        return False

    def _detect_plateau(self) -> bool:
        """
        Detect if training has plateaued.

        Plateau criteria:
        - Loss improvement over window < threshold

        Returns
        -------
        bool
            True if plateau detected
        """
        if len(self.metrics_history) < self.plateau_window:
            return False

        # Get recent losses
        recent_losses = [
            m.train_loss for m in self.metrics_history[-self.plateau_window:]
        ]

        # Compute improvement
        early_loss = np.mean(recent_losses[:5])
        late_loss = np.mean(recent_losses[-5:])
        improvement = early_loss - late_loss

        plateau = abs(improvement) < self.plateau_threshold

        if plateau:
            logger.warning(
                f"Training plateau detected: improvement={improvement:.6f} "
                f"over {self.plateau_window} steps"
            )

        return plateau

    def _handle_plateau(
        self,
        metrics: 'QuantumMetrics',
        model: Optional[Any]
    ):
        """
        Handle plateau detection.

        Actions:
        - Log plateau event
        - Could trigger learning rate adjustment (v4.2)
        - Could trigger architecture changes (v4.2)
        """
        self.adaptation_log.append(AdaptationEvent(
            step=metrics.step,
            phase=self.current_phase.value,
            action='plateau_detected',
            reason='loss stagnation',
            old_value=None,
            new_value=None,
            metadata={
                'window': self.plateau_window,
                'threshold': self.plateau_threshold
            }
        ))

        # In v4.1, just log. More sophisticated responses in v4.2
        logger.warning(
            f"Step {metrics.step}: Plateau detected. "
            f"Consider adjusting learning rate or circuit architecture."
        )

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all adaptations made.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.adaptation_log:
            return {'total_adaptations': 0}

        # Count by action type
        action_counts = {}
        for event in self.adaptation_log:
            action_counts[event.action] = action_counts.get(event.action, 0) + 1

        # Recent adaptations
        recent_adaptations = [
            {
                'step': e.step,
                'action': e.action,
                'reason': e.reason
            }
            for e in self.adaptation_log[-10:]
        ]

        return {
            'total_adaptations': len(self.adaptation_log),
            'action_counts': action_counts,
            'current_depth': self.current_depth,
            'current_phase': self.current_phase.value,
            'recent_adaptations': recent_adaptations
        }

    def get_training_phase(self) -> str:
        """Get current training phase."""
        return self.current_phase.value

    def get_measurement_config(self) -> Dict[str, Any]:
        """Get current measurement configuration."""
        return self.measurement_policy.get_measurement_config(
            self.current_phase.value
        )

    def reset(self):
        """Reset controller state."""
        self.current_phase = TrainingPhase.EXPLORATION
        self.metrics_history.clear()
        self.adaptation_log.clear()
        self.last_depth_increase_step = 0
        self.measurement_policy.reset()

        logger.info("Reset AdaptiveTrainingController")


class TrainingOrchestrator:
    """
    High-level training orchestrator combining all adaptive components.

    Combines:
    - AdaptiveTrainingController
    - GradientNoiseTracker
    - QuantumMetricsComputer

    This is a convenience wrapper for fully adaptive training.

    Examples
    --------
    >>> orchestrator = TrainingOrchestrator(
    ...     initial_depth=3,
    ...     max_depth=8
    ... )
    >>>
    >>> # During training
    >>> for step in range(1000):
    ...     # ... forward pass, compute loss ...
    ...
    ...     # Compute gradients
    ...     gradient = ...
    ...
    ...     # Update orchestrator
    ...     stats = orchestrator.update(
    ...         step=step,
    ...         epoch=epoch,
    ...         loss=loss,
    ...         gradient=gradient,
    ...         circuit=circuit
    ...     )
    ...
    ...     # Get adaptive config for next step
    ...     config = orchestrator.get_config()
    ...     print(f"Phase: {config['phase']}, Depth: {config['depth']}")
    """

    def __init__(
        self,
        initial_depth: int = 3,
        max_depth: int = 8,
        **controller_kwargs
    ):
        # Import here to avoid circular dependencies
        from q_store.ml.gradient_noise_tracker import GradientNoiseTracker
        from q_store.analysis.quantum_metrics_computer import QuantumMetricsComputer

        self.controller = AdaptiveTrainingController(
            initial_depth=initial_depth,
            max_depth=max_depth,
            **controller_kwargs
        )

        self.gradient_tracker = GradientNoiseTracker()
        self.metrics_computer = QuantumMetricsComputer()

        logger.info("Initialized TrainingOrchestrator")

    def update(
        self,
        step: int,
        epoch: int,
        loss: float,
        gradient: np.ndarray,
        circuit: Optional[Any] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update all components and compute adaptations.

        Parameters
        ----------
        step : int
            Global training step
        epoch : int
            Training epoch
        loss : float
            Current loss
        gradient : np.ndarray
            Current gradient
        circuit : optional
            Quantum circuit
        model : optional
            Model instance
        **kwargs
            Additional metrics

        Returns
        -------
        dict
            Combined statistics and adaptations
        """
        import time

        # Update gradient tracker
        grad_stats = self.gradient_tracker.update(gradient, step)

        # Compute quantum metrics if circuit provided
        quantum_metrics = {}
        if circuit is not None:
            quantum_metrics = {
                'expressibility_score': self.metrics_computer.compute_expressibility(circuit),
                'circuit_depth': self.metrics_computer._get_circuit_depth(circuit),
                'entangling_gates': self.metrics_computer._count_entangling_gates(circuit)
            }

        # Create QuantumMetrics
        from q_store.storage.metrics_schema import QuantumMetrics

        metrics = QuantumMetrics(
            epoch=epoch,
            step=step,
            timestamp=time.time(),
            train_loss=loss,
            grad_norm=grad_stats.gradient_norm,
            gradient_variance=grad_stats.gradient_variance,
            gradient_snr=grad_stats.gradient_snr,
            circuit_depth=quantum_metrics.get('circuit_depth', 0),
            entangling_gates=quantum_metrics.get('entangling_gates', 0),
            expressibility_score=quantum_metrics.get('expressibility_score'),
            **kwargs
        )

        # Adapt
        adaptations = self.controller.adapt(metrics, model)

        return {
            'gradient_stats': grad_stats,
            'quantum_metrics': quantum_metrics,
            'adaptations': adaptations,
            'phase': self.controller.get_training_phase()
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current training configuration."""
        return {
            'phase': self.controller.get_training_phase(),
            'depth': self.controller.current_depth,
            'measurement': self.controller.get_measurement_config()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'controller': self.controller.get_adaptation_summary(),
            'gradients': self.gradient_tracker.get_summary()
        }

    def reset(self):
        """Reset all components."""
        self.controller.reset()
        self.gradient_tracker.reset()
        logger.info("Reset TrainingOrchestrator")
