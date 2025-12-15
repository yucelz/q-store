"""
Quantum Trainer
Orchestrates quantum ML training with hardware abstraction
"""

import asyncio
import time
import json
import os
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..backends.backend_manager import BackendManager
from .gradient_computer import QuantumGradientComputer, GradientResult
from .quantum_layer import QuantumLayer
from .data_encoder import QuantumDataEncoder

# v3.3 imports
from .spsa_gradient_estimator import SPSAGradientEstimator
from .circuit_batch_manager import CircuitBatchManager
from .circuit_cache import QuantumCircuitCache
from .quantum_layer_v2 import HardwareEfficientQuantumLayer
from .adaptive_optimizer import AdaptiveGradientOptimizer
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for quantum training"""
    # Database config
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "quantum-ml-training"

    # Quantum backend
    quantum_sdk: str = "mock"  # 'cirq', 'qiskit', 'mock'
    quantum_api_key: Optional[str] = None
    quantum_target: str = "simulator"

    # Training hyperparameters
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100

    # Model architecture
    n_qubits: int = 10
    circuit_depth: int = 4
    entanglement: str = 'linear'

    # Optimization
    optimizer: str = 'adam'  # 'adam', 'sgd', 'natural_gradient'
    gradient_method: str = 'parameter_shift'  # or 'finite_diff', 'spsa', 'adaptive'
    momentum: float = 0.9
    weight_decay: float = 0.0

    # Training options
    shots_per_circuit: int = 1000
    max_concurrent_circuits: int = 5
    use_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_directory: str = './checkpoints'

    # Monitoring
    log_interval: int = 10
    track_gradients: bool = True

    # v3.3 NEW: Performance optimizations
    enable_circuit_cache: bool = True
    enable_batch_execution: bool = True
    cache_size: int = 1000
    batch_timeout: float = 60.0
    hardware_efficient_ansatz: bool = True

    # v3.3 NEW: SPSA parameters
    spsa_c_initial: float = 0.1
    spsa_a_initial: float = 0.01

    # v3.3 NEW: Performance tracking
    enable_performance_tracking: bool = True
    performance_log_dir: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    circuit_execution_time_ms: float = 0.0
    epoch_time_ms: float = 0.0
    n_circuit_executions: int = 0


class QuantumModel:
    """
    Base class for quantum ML models
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        output_dim: int,
        backend,
        depth: int = 4,
        hardware_efficient: bool = False
    ):
        """
        Initialize quantum model

        Args:
            input_dim: Input feature dimension
            n_qubits: Number of qubits
            output_dim: Output dimension
            backend: Quantum backend
            depth: Circuit depth
            hardware_efficient: Use v3.3 hardware-efficient layer
        """
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.output_dim = output_dim
        self.backend = backend
        self.depth = depth
        self.hardware_efficient = hardware_efficient

        # Quantum layer (v3.3: hardware-efficient option)
        if hardware_efficient:
            self.quantum_layer = HardwareEfficientQuantumLayer(
                n_qubits=n_qubits,
                depth=depth,
                backend=backend
            )
            logger.info(f"Using hardware-efficient layer with {self.quantum_layer.n_parameters} parameters")
        else:
            self.quantum_layer = QuantumLayer(
                n_qubits=n_qubits,
                depth=depth,
                backend=backend
            )

        # Parameters are managed by quantum layer
        self.parameters = self.quantum_layer.parameters

    async def forward(self, x: np.ndarray, shots: int = 1000) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input data
            shots: Number of measurement shots

        Returns:
            Model output
        """
        # Encode and process through quantum layer
        output = await self.quantum_layer.forward(x, shots)

        # Project to output dimension if needed
        if len(output) != self.output_dim:
            # Simple projection: take first output_dim elements
            output = output[:self.output_dim]

        return output

    def save_state(self) -> Dict[str, Any]:
        """Save model state"""
        return {
            'config': {
                'input_dim': self.input_dim,
                'n_qubits': self.n_qubits,
                'output_dim': self.output_dim,
                'depth': self.depth
            },
            'quantum_layer': self.quantum_layer.save_state()
        }

    def load_state(self, state: Dict[str, Any]):
        """Load model state"""
        self.quantum_layer.load_state(state['quantum_layer'])
        self.parameters = self.quantum_layer.parameters


class QuantumTrainer:
    """
    Quantum ML trainer with hardware abstraction
    """

    def __init__(
        self,
        config: TrainingConfig,
        backend_manager: BackendManager
    ):
        """
        Initialize trainer

        Args:
            config: Training configuration
            backend_manager: Backend manager for quantum execution
        """
        self.config = config
        self.backend_manager = backend_manager

        # Get quantum backend
        self.backend = backend_manager.get_backend()

        # Initialize components (v3.3 enhanced)
        if config.gradient_method == 'adaptive':
            self.gradient_computer = AdaptiveGradientOptimizer(
                self.backend,
                enable_adaptation=True
            )
            logger.info("Using adaptive gradient optimizer (v3.3)")
        elif config.gradient_method == 'spsa':
            self.gradient_computer = SPSAGradientEstimator(
                self.backend,
                c_initial=config.spsa_c_initial,
                a_initial=config.spsa_a_initial
            )
            logger.info("Using SPSA gradient estimator (v3.3)")
        else:
            # Default to parameter shift
            self.gradient_computer = QuantumGradientComputer(self.backend)
            logger.info("Using parameter shift gradient computation")

        # v3.3 NEW: Circuit optimization infrastructure
        self.circuit_cache = QuantumCircuitCache(
            max_compiled_circuits=config.cache_size,
            max_results=config.cache_size * 5
        ) if config.enable_circuit_cache else None

        self.batch_manager = CircuitBatchManager(
            self.backend,
            timeout=config.batch_timeout
        ) if config.enable_batch_execution else None

        # v3.3 NEW: Performance monitoring
        self.performance_tracker = PerformanceTracker(
            log_dir=config.performance_log_dir,
            save_interval=config.log_interval
        ) if config.enable_performance_tracking else None

        self.data_encoder = QuantumDataEncoder()

        # Optimizer state
        self.optimizer_state = self._initialize_optimizer()

        # Training state
        self.current_epoch = 0
        self.training_history: List[TrainingMetrics] = []

        # Loss function
        self.loss_function = self._default_loss_function

        # Pinecone connection (optional for checkpointing/storage)
        self._pinecone_client = None
        self._pinecone_index = None
        self._pinecone_initialized = False

        # Training data storage tracking
        self._training_vectors_stored = 0
        self._enable_vector_storage = False

    async def _init_pinecone(self):
        """Initialize Pinecone connection for model checkpointing"""
        if self._pinecone_initialized:
            return

        try:
            from pinecone.grpc import PineconeGRPC as Pinecone
            from pinecone import ServerlessSpec

            if not self.config.pinecone_api_key or self.config.pinecone_api_key == "mock-key":
                logger.info("Skipping Pinecone initialization (mock mode or no API key)")
                return

            pc = Pinecone(api_key=self.config.pinecone_api_key)

            # Create index if it doesn't exist
            existing_indexes = [index.name for index in pc.list_indexes()]
            if self.config.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.pinecone_index_name}")
                pc.create_index(
                    name=self.config.pinecone_index_name,
                    dimension=768,  # Default dimension for embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.config.pinecone_environment
                    )
                )
                logger.info(f"Pinecone index '{self.config.pinecone_index_name}' created successfully")
            else:
                logger.info(f"Using existing Pinecone index: {self.config.pinecone_index_name}")

            self._pinecone_index = pc.Index(self.config.pinecone_index_name)
            self._pinecone_client = pc
            self._pinecone_initialized = True
            self._enable_vector_storage = True
            logger.info("Pinecone connection established for model checkpointing and vector storage")

        except ImportError:
            logger.warning("Pinecone package not installed. Checkpointing to Pinecone disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone: {e}. Continuing without Pinecone.")

    def _initialize_optimizer(self) -> Dict[str, Any]:
        """Initialize optimizer state"""
        if self.config.optimizer == 'adam':
            return {
                'm': None,  # First moment
                'v': None,  # Second moment
                't': 0,     # Time step
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8
            }
        elif self.config.optimizer == 'sgd':
            return {
                'velocity': None,
                'momentum': self.config.momentum
            }
        else:
            return {}

    def _default_loss_function(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Default MSE loss function"""
        return np.mean((predictions - targets) ** 2)

    def set_loss_function(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float]
    ):
        """Set custom loss function"""
        self.loss_function = loss_fn

    async def train_epoch(
        self,
        model: QuantumModel,
        data_loader,
        epoch: int
    ) -> TrainingMetrics:
        """
        Train for one epoch

        Args:
            model: Quantum model to train
            data_loader: Data loader providing batches
            epoch: Current epoch number

        Returns:
            Training metrics for this epoch
        """
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        total_gradient_norm = 0.0
        total_circuit_executions = 0
        total_circuit_time = 0.0

        async for batch_x, batch_y in data_loader:
            # Train on batch
            batch_metrics = await self.train_batch(model, batch_x, batch_y)

            epoch_loss += batch_metrics['loss']
            total_gradient_norm += batch_metrics['gradient_norm']
            total_circuit_executions += batch_metrics['n_circuits']
            total_circuit_time += batch_metrics['circuit_time_ms']
            batch_count += 1

            # Store training vectors to Pinecone if enabled
            if self._enable_vector_storage and self._pinecone_index is not None:
                await self._store_training_batch_to_pinecone(batch_x, batch_y, epoch, batch_count)

            # Log progress
            if batch_count % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_count}: "
                    f"Loss={batch_metrics['loss']:.4f}, "
                    f"Grad Norm={batch_metrics['gradient_norm']:.4f}"
                )

        # Compute epoch metrics
        epoch_time = (time.time() - epoch_start) * 1000

        metrics = TrainingMetrics(
            epoch=epoch,
            loss=epoch_loss / batch_count if batch_count > 0 else 0.0,
            gradient_norm=total_gradient_norm / batch_count if batch_count > 0 else 0.0,
            learning_rate=self.config.learning_rate,
            circuit_execution_time_ms=total_circuit_time / batch_count if batch_count > 0 else 0.0,
            epoch_time_ms=epoch_time,
            n_circuit_executions=total_circuit_executions
        )

        self.training_history.append(metrics)

        return metrics

    async def train_batch(
        self,
        model: QuantumModel,
        batch_x: np.ndarray,
        batch_y: np.ndarray
    ) -> Dict[str, float]:
        """
        Train on a single batch (v3.3 optimized)

        Args:
            model: Model to train
            batch_x: Input batch
            batch_y: Target batch

        Returns:
            Batch metrics
        """
        batch_start = time.time()

        # v3.3 OPTIMIZATION: Batch gradient computation
        # Instead of computing gradients for each sample separately,
        # compute them for the entire batch at once

        # Average over batch
        batch_loss = 0.0
        batch_gradients = None
        n_circuits = 0

        for x, y in zip(batch_x, batch_y):
            # Compute gradients
            def circuit_builder(params):
                # Update model parameters
                temp_params = model.quantum_layer.parameters.copy()
                model.quantum_layer.parameters = params
                circuit = model.quantum_layer.build_circuit(x)
                model.quantum_layer.parameters = temp_params
                return circuit

            def loss_from_result(result):
                # Extract output from execution result
                output = model.quantum_layer._process_measurements(result)
                if len(output) != model.output_dim:
                    output = output[:model.output_dim]
                return self.loss_function(output, y)

            # v3.3: Use appropriate gradient method
            if hasattr(self.gradient_computer, 'estimate_gradient'):
                # SPSA or adaptive optimizer
                grad_result = await self.gradient_computer.estimate_gradient(
                    circuit_builder=circuit_builder,
                    loss_function=loss_from_result,
                    parameters=model.quantum_layer.parameters,
                    frozen_indices=model.quantum_layer._frozen_params,
                    shots=self.config.shots_per_circuit
                )
            else:
                # Standard parameter shift
                grad_result = await self.gradient_computer.compute_gradients(
                    circuit_builder=circuit_builder,
                    loss_function=loss_from_result,
                    parameters=model.quantum_layer.parameters,
                    frozen_indices=list(model.quantum_layer._frozen_params)
                )

            batch_loss += grad_result.function_value
            n_circuits += grad_result.n_circuit_executions

            # Accumulate gradients
            if batch_gradients is None:
                batch_gradients = grad_result.gradients
            else:
                batch_gradients += grad_result.gradients

        # Average gradients
        batch_gradients /= len(batch_x)

        # Gradient clipping
        if self.config.use_gradient_clipping:
            grad_norm = np.linalg.norm(batch_gradients)
            if grad_norm > self.config.gradient_clip_value:
                batch_gradients = batch_gradients * (self.config.gradient_clip_value / grad_norm)

        # Update parameters
        new_params = self._optimizer_step(
            model.quantum_layer.parameters,
            batch_gradients
        )
        model.quantum_layer.update_parameters(new_params)

        batch_time = (time.time() - batch_start) * 1000

        # v3.3 NEW: Track performance
        if self.performance_tracker:
            cache_stats = self.circuit_cache.get_stats() if self.circuit_cache else None
            method_used = getattr(grad_result, 'method', None)

            self.performance_tracker.log_batch(
                batch_idx=self.current_epoch,
                epoch=self.current_epoch,
                loss=batch_loss / len(batch_x),
                gradient_norm=np.linalg.norm(batch_gradients),
                n_circuits=n_circuits,
                time_ms=batch_time,
                learning_rate=self.config.learning_rate,
                cache_stats=cache_stats,
                method_used=method_used
            )

        return {
            'loss': batch_loss / len(batch_x),
            'gradient_norm': np.linalg.norm(batch_gradients),
            'n_circuits': n_circuits,
            'circuit_time_ms': batch_time
        }

    def _optimizer_step(
        self,
        parameters: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        """
        Perform optimizer step

        Args:
            parameters: Current parameters
            gradients: Computed gradients

        Returns:
            Updated parameters
        """
        if self.config.optimizer == 'adam':
            return self._adam_step(parameters, gradients)
        elif self.config.optimizer == 'sgd':
            return self._sgd_step(parameters, gradients)
        else:
            # Simple gradient descent
            return parameters - self.config.learning_rate * gradients

    def _adam_step(
        self,
        parameters: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        """Adam optimizer step"""
        state = self.optimizer_state

        # Initialize moments if needed
        if state['m'] is None:
            state['m'] = np.zeros_like(parameters)
            state['v'] = np.zeros_like(parameters)

        state['t'] += 1

        # Update biased first moment
        state['m'] = state['beta1'] * state['m'] + (1 - state['beta1']) * gradients

        # Update biased second moment
        state['v'] = state['beta2'] * state['v'] + (1 - state['beta2']) * (gradients ** 2)

        # Bias correction
        m_hat = state['m'] / (1 - state['beta1'] ** state['t'])
        v_hat = state['v'] / (1 - state['beta2'] ** state['t'])

        # Update parameters
        update = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + state['epsilon'])

        return parameters - update

    def _sgd_step(
        self,
        parameters: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        """SGD with momentum"""
        state = self.optimizer_state

        if state['velocity'] is None:
            state['velocity'] = np.zeros_like(parameters)

        # Update velocity
        state['velocity'] = (
            state['momentum'] * state['velocity'] +
            self.config.learning_rate * gradients
        )

        # Update parameters
        return parameters - state['velocity']

    async def train(
        self,
        model: QuantumModel,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None
    ):
        """
        Train the model

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs (uses config if None)
        """
        epochs = epochs or self.config.epochs

        # Initialize Pinecone if configured
        await self._init_pinecone()

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Model: {model.n_qubits} qubits, depth {model.depth}")
        logger.info(f"Backend: {self.backend.get_backend_info()}")

        for epoch in range(epochs):
            # Train epoch
            metrics = await self.train_epoch(model, train_loader, epoch)

            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"Loss={metrics.loss:.4f}, "
                f"Grad Norm={metrics.gradient_norm:.4f}, "
                f"Time={metrics.epoch_time_ms/1000:.2f}s"
            )

            # Validation
            if val_loader is not None:
                val_metrics = await self.validate(model, val_loader)
                logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")

            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                await self.save_checkpoint(model, epoch, metrics)

            self.current_epoch = epoch

        logger.info("Training complete!")

    async def validate(
        self,
        model: QuantumModel,
        val_loader
    ) -> Dict[str, float]:
        """
        Validate the model

        Args:
            model: Model to validate
            val_loader: Validation data loader

        Returns:
            Validation metrics
        """
        total_loss = 0.0
        count = 0

        async for batch_x, batch_y in val_loader:
            for x, y in zip(batch_x, batch_y):
                prediction = await model.forward(x, shots=self.config.shots_per_circuit)
                loss = self.loss_function(prediction, y)
                total_loss += loss
                count += 1

        return {
            'loss': total_loss / count if count > 0 else 0.0,
            'count': count
        }

    async def save_checkpoint(
        self,
        model: QuantumModel,
        epoch: int,
        metrics: TrainingMetrics
    ):
        """
        Save training checkpoint

        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint_dir = Path(self.config.checkpoint_directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"

        checkpoint = {
            'epoch': epoch,
            'model_state': model.save_state(),
            'optimizer_state': self.optimizer_state,
            'metrics': {
                'loss': metrics.loss,
                'gradient_norm': metrics.gradient_norm,
                'learning_rate': metrics.learning_rate
            },
            'config': {
                'learning_rate': self.config.learning_rate,
                'optimizer': self.config.optimizer,
                'batch_size': self.config.batch_size
            }
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            return obj

        checkpoint = convert_arrays(checkpoint)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    async def load_checkpoint(
        self,
        checkpoint_path: str,
        model: QuantumModel
    ):
        """
        Load training checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Load model state
        model.load_state(checkpoint['model_state'])

        # Load optimizer state
        def convert_to_arrays(obj):
            if isinstance(obj, dict):
                return {k: convert_to_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return np.array(obj)
            return obj

        self.optimizer_state = convert_to_arrays(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    async def _store_training_batch_to_pinecone(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        epoch: int,
        batch_num: int
    ):
        """
        Store training batch vectors to Pinecone

        Args:
            batch_x: Input batch
            batch_y: Target batch
            epoch: Current epoch
            batch_num: Current batch number
        """
        try:
            vectors_to_upsert = []

            for idx, (x, y) in enumerate(zip(batch_x, batch_y)):
                # Create unique ID for this training sample
                vector_id = f"train_e{epoch}_b{batch_num}_s{idx}"

                # Pad or truncate to match index dimension (768)
                vector = x.flatten()
                if len(vector) < 768:
                    vector = np.pad(vector, (0, 768 - len(vector)), 'constant')
                elif len(vector) > 768:
                    vector = vector[:768]

                # Prepare metadata
                metadata = {
                    'epoch': epoch,
                    'batch': batch_num,
                    'sample_idx': idx,
                    'label': str(y.tolist() if isinstance(y, np.ndarray) else y),
                    'type': 'training_data'
                }

                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': vector.tolist(),
                    'metadata': metadata
                })

            # Upsert to Pinecone
            if vectors_to_upsert:
                self._pinecone_index.upsert(vectors=vectors_to_upsert)
                self._training_vectors_stored += len(vectors_to_upsert)
                logger.debug(f"Stored {len(vectors_to_upsert)} training vectors to Pinecone")

        except Exception as e:
            logger.warning(f"Failed to store training vectors to Pinecone: {e}")

    def get_training_history(self) -> List[TrainingMetrics]:
        """Get training history"""
        return self.training_history

    def get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone storage statistics"""
        return {
            'enabled': self._enable_vector_storage,
            'initialized': self._pinecone_initialized,
            'vectors_stored': self._training_vectors_stored,
            'index_name': self.config.pinecone_index_name if self._pinecone_initialized else None
        }
