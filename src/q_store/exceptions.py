"""
Custom exceptions for Q-Store quantum database.

This module defines domain-specific exceptions for better error handling
and debugging throughout the quantum database system.
"""


class QStoreError(Exception):
    """Base exception for all Q-Store errors."""

    pass


# Backend Errors
class QuantumBackendError(QStoreError):
    """Raised when quantum backend operations fail."""

    pass


class BackendInitializationError(QuantumBackendError):
    """Raised when backend initialization fails."""

    pass


class BackendConnectionError(QuantumBackendError):
    """Raised when backend connection fails or is lost."""

    pass


# Circuit Errors
class CircuitError(QStoreError):
    """Base class for circuit-related errors."""

    pass


class CircuitExecutionError(CircuitError):
    """Raised when circuit execution fails."""

    pass


class CircuitBuildError(CircuitError):
    """Raised when circuit construction fails."""

    pass


class CircuitCompilationError(CircuitError):
    """Raised when circuit compilation fails."""

    pass


# Configuration Errors
class ConfigurationError(QStoreError):
    """Raised when configuration is invalid or incomplete."""

    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter value is invalid."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


# Training Errors
class TrainingError(QStoreError):
    """Base class for training-related errors."""

    pass


class GradientComputationError(TrainingError):
    """Raised when gradient computation fails."""

    pass


class OptimizationError(TrainingError):
    """Raised when optimization step fails."""

    pass


class ConvergenceError(TrainingError):
    """Raised when training fails to converge."""

    pass


# Database Errors
class DatabaseError(QStoreError):
    """Base class for database operations errors."""

    pass


class VectorStoreError(DatabaseError):
    """Raised when vector store operations fail."""

    pass


class IndexError(DatabaseError):
    """Raised when index operations fail."""

    pass


class QueryError(DatabaseError):
    """Raised when query execution fails."""

    pass


# State Management Errors
class StateError(QStoreError):
    """Base class for quantum state errors."""

    pass


class DecoherenceError(StateError):
    """Raised when decoherence occurs or is detected."""

    pass


class StatePreparationError(StateError):
    """Raised when quantum state preparation fails."""

    pass


class EntanglementError(StateError):
    """Raised when entanglement operations fail."""

    pass


# Resource Errors
class ResourceError(QStoreError):
    """Base class for resource-related errors."""

    pass


class QubitLimitExceededError(ResourceError):
    """Raised when requested qubits exceed backend capacity."""

    pass


class TimeoutError(ResourceError):
    """Raised when operations exceed timeout limits."""

    pass


class RateLimitError(ResourceError):
    """Raised when API rate limits are exceeded."""

    pass
