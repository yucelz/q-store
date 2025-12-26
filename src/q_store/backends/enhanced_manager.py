"""
Enhanced Backend Manager with Registry, Health Checking, and Lifecycle Management.

This module extends the base BackendManager with advanced features:
- Backend registry with discovery
- Health monitoring and status tracking
- Dynamic backend loading
- Lifecycle management (initialization, cleanup)
- Resource pooling and connection management
"""

from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from datetime import datetime, timedelta
import importlib
import inspect

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    """Backend health status."""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class BackendHealth:
    """Health information for a backend."""
    status: BackendStatus
    last_check: datetime
    response_time: float  # seconds
    success_rate: float  # 0-1
    error_count: int
    total_requests: int
    message: str = ""

    def is_available(self) -> bool:
        """Check if backend is available for use."""
        return self.status in [BackendStatus.HEALTHY, BackendStatus.DEGRADED]


@dataclass
class BackendRegistry:
    """Registry entry for a backend."""
    name: str
    backend_class: Optional[Type] = None
    backend_instance: Optional[Any] = None
    module_path: Optional[str] = None
    factory_function: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health: Optional[BackendHealth] = None
    auto_load: bool = True
    priority: int = 100  # Lower is higher priority

    def is_loaded(self) -> bool:
        """Check if backend is loaded."""
        return self.backend_instance is not None


class EnhancedBackendManager:
    """
    Enhanced backend manager with registry, health checking, and lifecycle management.

    Features:
    - Backend registry with discovery and dynamic loading
    - Health monitoring with periodic checks
    - Automatic failover to healthy backends
    - Resource lifecycle management
    - Connection pooling and reuse
    - Priority-based backend selection

    Example:
        >>> manager = EnhancedBackendManager()
        >>> manager.register_backend_class(
        ...     'qsim',
        ...     QsimBackend,
        ...     config={'num_threads': 4},
        ...     auto_load=True
        ... )
        >>> manager.start_health_monitoring(interval=60)
        >>> backend = manager.get_healthy_backend('qsim')
    """

    def __init__(self, health_check_interval: int = 60):
        """
        Initialize enhanced backend manager.

        Args:
            health_check_interval: Seconds between health checks (0 = disabled)
        """
        self.registry: Dict[str, BackendRegistry] = {}
        self.health_check_interval = health_check_interval
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.Lock()

        logger.info("EnhancedBackendManager initialized")

    def register_backend_class(
        self,
        name: str,
        backend_class: Type,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_load: bool = True,
        priority: int = 100
    ):
        """
        Register a backend class for later instantiation.

        Args:
            name: Unique backend identifier
            backend_class: Backend class to register
            config: Configuration for backend instantiation
            metadata: Additional metadata
            auto_load: Whether to auto-load on first access
            priority: Priority for selection (lower = higher priority)
        """
        with self._lock:
            entry = BackendRegistry(
                name=name,
                backend_class=backend_class,
                config=config or {},
                metadata=metadata or {},
                auto_load=auto_load,
                priority=priority,
                health=BackendHealth(
                    status=BackendStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time=0.0,
                    success_rate=1.0,
                    error_count=0,
                    total_requests=0
                )
            )

            self.registry[name] = entry
            logger.info(f"Registered backend class: {name} (priority={priority})")

    def register_backend_instance(
        self,
        name: str,
        backend_instance: Any,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 100
    ):
        """
        Register an already-instantiated backend.

        Args:
            name: Unique backend identifier
            backend_instance: Backend instance
            metadata: Additional metadata
            priority: Priority for selection
        """
        with self._lock:
            entry = BackendRegistry(
                name=name,
                backend_instance=backend_instance,
                metadata=metadata or {},
                priority=priority,
                health=BackendHealth(
                    status=BackendStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time=0.0,
                    success_rate=1.0,
                    error_count=0,
                    total_requests=0
                )
            )

            self.registry[name] = entry
            logger.info(f"Registered backend instance: {name}")

            # Perform initial health check
            self._check_backend_health(name)

    def register_backend_module(
        self,
        name: str,
        module_path: str,
        factory_function: str = "create_backend",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_load: bool = True,
        priority: int = 100
    ):
        """
        Register a backend from a module path with factory function.

        Args:
            name: Unique backend identifier
            module_path: Python module path (e.g., 'q_store.backends.qsim_backend')
            factory_function: Name of factory function in module
            config: Configuration to pass to factory
            metadata: Additional metadata
            auto_load: Whether to auto-load on first access
            priority: Priority for selection
        """
        with self._lock:
            entry = BackendRegistry(
                name=name,
                module_path=module_path,
                factory_function=lambda: self._load_from_module(module_path, factory_function, config),
                config=config or {},
                metadata=metadata or {},
                auto_load=auto_load,
                priority=priority,
                health=BackendHealth(
                    status=BackendStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time=0.0,
                    success_rate=1.0,
                    error_count=0,
                    total_requests=0
                )
            )

            self.registry[name] = entry
            logger.info(f"Registered backend module: {name} from {module_path}")

    def _load_from_module(
        self,
        module_path: str,
        factory_name: str,
        config: Optional[Dict[str, Any]]
    ):
        """Load backend from module using factory function."""
        try:
            module = importlib.import_module(module_path)
            factory = getattr(module, factory_name)

            # Call factory with config
            if config:
                return factory(**config)
            else:
                return factory()
        except Exception as e:
            logger.error(f"Failed to load backend from {module_path}.{factory_name}: {e}")
            raise

    def load_backend(self, name: str) -> Any:
        """
        Load/instantiate a backend.

        Args:
            name: Backend name

        Returns:
            Backend instance

        Raises:
            ValueError: If backend not found
            RuntimeError: If loading fails
        """
        with self._lock:
            if name not in self.registry:
                raise ValueError(f"Backend '{name}' not registered")

            entry = self.registry[name]

            # Already loaded
            if entry.is_loaded():
                return entry.backend_instance

            # Load backend
            try:
                entry.health.status = BackendStatus.INITIALIZING
                logger.info(f"Loading backend: {name}")

                if entry.backend_class:
                    # Instantiate from class
                    entry.backend_instance = entry.backend_class(**entry.config)
                elif entry.factory_function:
                    # Use factory function
                    entry.backend_instance = entry.factory_function()
                else:
                    raise RuntimeError(f"No backend_class or factory_function for {name}")

                # Perform health check
                self._check_backend_health(name)

                logger.info(f"Successfully loaded backend: {name}")
                return entry.backend_instance

            except Exception as e:
                entry.health.status = BackendStatus.OFFLINE
                entry.health.message = str(e)
                logger.error(f"Failed to load backend {name}: {e}")
                raise RuntimeError(f"Failed to load backend {name}: {e}")

    def unload_backend(self, name: str):
        """
        Unload a backend and free resources.

        Args:
            name: Backend name
        """
        with self._lock:
            if name not in self.registry:
                return

            entry = self.registry[name]

            if entry.backend_instance:
                # Try to call close/cleanup if available
                if hasattr(entry.backend_instance, 'close'):
                    try:
                        entry.backend_instance.close()
                    except Exception as e:
                        logger.warning(f"Error closing backend {name}: {e}")

                entry.backend_instance = None
                entry.health.status = BackendStatus.OFFLINE
                logger.info(f"Unloaded backend: {name}")

    def get_backend(self, name: str, auto_load: bool = True) -> Any:
        """
        Get backend by name, optionally auto-loading.

        Args:
            name: Backend name
            auto_load: Whether to auto-load if not loaded

        Returns:
            Backend instance

        Raises:
            ValueError: If backend not found
            RuntimeError: If backend not loaded and auto_load=False
        """
        if name not in self.registry:
            raise ValueError(f"Backend '{name}' not registered")

        entry = self.registry[name]

        if not entry.is_loaded():
            if auto_load:
                return self.load_backend(name)
            else:
                raise RuntimeError(f"Backend '{name}' not loaded")

        return entry.backend_instance

    def get_healthy_backend(
        self,
        name: Optional[str] = None,
        min_success_rate: float = 0.8
    ) -> Any:
        """
        Get a healthy backend, optionally by name or best available.

        Args:
            name: Specific backend name (None = best available)
            min_success_rate: Minimum success rate threshold

        Returns:
            Healthy backend instance

        Raises:
            RuntimeError: If no healthy backend available
        """
        if name:
            # Get specific backend
            backend = self.get_backend(name)
            entry = self.registry[name]

            if not entry.health.is_available():
                raise RuntimeError(f"Backend '{name}' is not healthy (status={entry.health.status.value})")

            if entry.health.success_rate < min_success_rate:
                raise RuntimeError(f"Backend '{name}' success rate too low: {entry.health.success_rate:.2f}")

            return backend
        else:
            # Find best available backend
            candidates = []

            for name, entry in self.registry.items():
                if entry.health and entry.health.is_available():
                    if entry.health.success_rate >= min_success_rate:
                        candidates.append((name, entry))

            if not candidates:
                raise RuntimeError("No healthy backends available")

            # Sort by priority, then success rate
            candidates.sort(key=lambda x: (x[1].priority, -x[1].health.success_rate))

            best_name, best_entry = candidates[0]
            logger.info(f"Selected healthy backend: {best_name} (priority={best_entry.priority}, "
                       f"success_rate={best_entry.health.success_rate:.2f})")

            return self.get_backend(best_name)

    def _check_backend_health(self, name: str):
        """
        Check health of a specific backend.

        Args:
            name: Backend name
        """
        if name not in self.registry:
            return

        entry = self.registry[name]

        if not entry.is_loaded():
            entry.health.status = BackendStatus.OFFLINE
            return

        backend = entry.backend_instance
        start_time = time.time()

        try:
            # Check if backend has health check method
            if hasattr(backend, 'health_check'):
                healthy = backend.health_check()
                response_time = time.time() - start_time

                if healthy:
                    entry.health.status = BackendStatus.HEALTHY
                    entry.health.response_time = response_time
                    entry.health.message = "Health check passed"
                else:
                    entry.health.status = BackendStatus.UNHEALTHY
                    entry.health.message = "Health check failed"

            # Fallback: check is_available if exists
            elif hasattr(backend, 'is_available'):
                available = backend.is_available()
                response_time = time.time() - start_time

                if available:
                    entry.health.status = BackendStatus.HEALTHY
                    entry.health.response_time = response_time
                else:
                    entry.health.status = BackendStatus.OFFLINE

            # Fallback: check get_capabilities
            elif hasattr(backend, 'get_capabilities'):
                backend.get_capabilities()
                response_time = time.time() - start_time
                entry.health.status = BackendStatus.HEALTHY
                entry.health.response_time = response_time

            else:
                # No health check available, assume healthy if loaded
                entry.health.status = BackendStatus.HEALTHY
                entry.health.response_time = 0.0

        except Exception as e:
            entry.health.status = BackendStatus.UNHEALTHY
            entry.health.message = str(e)
            logger.warning(f"Health check failed for {name}: {e}")

        entry.health.last_check = datetime.now()

    def check_all_health(self):
        """Check health of all loaded backends."""
        for name in list(self.registry.keys()):
            entry = self.registry[name]
            if entry.is_loaded():
                self._check_backend_health(name)

    def start_health_monitoring(self, interval: Optional[int] = None):
        """
        Start background health monitoring.

        Args:
            interval: Check interval in seconds (None = use default)
        """
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return

        check_interval = interval or self.health_check_interval

        if check_interval <= 0:
            logger.info("Health monitoring disabled (interval <= 0)")
            return

        def monitor_loop():
            logger.info(f"Health monitoring started (interval={check_interval}s)")
            while self._monitoring_active:
                try:
                    self.check_all_health()
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")

                time.sleep(check_interval)

            logger.info("Health monitoring stopped")

        self._monitoring_active = True
        self._health_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._health_monitor_thread.start()

    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._monitoring_active:
            self._monitoring_active = False
            if self._health_monitor_thread:
                self._health_monitor_thread.join(timeout=5)
            logger.info("Health monitoring stopped")

    def update_backend_stats(
        self,
        name: str,
        success: bool,
        execution_time: Optional[float] = None
    ):
        """
        Update backend statistics after execution.

        Args:
            name: Backend name
            success: Whether execution succeeded
            execution_time: Execution time in seconds
        """
        if name not in self.registry:
            return

        entry = self.registry[name]
        health = entry.health

        health.total_requests += 1
        if not success:
            health.error_count += 1

        # Update success rate (exponential moving average)
        alpha = 0.1  # Weight for new data
        new_success = 1.0 if success else 0.0
        health.success_rate = (1 - alpha) * health.success_rate + alpha * new_success

        # Update response time if provided
        if execution_time is not None:
            health.response_time = (
                (1 - alpha) * health.response_time + alpha * execution_time
            )

        # Update status based on success rate
        if health.success_rate < 0.5:
            health.status = BackendStatus.UNHEALTHY
        elif health.success_rate < 0.8:
            health.status = BackendStatus.DEGRADED

    def list_backends(self) -> List[Dict[str, Any]]:
        """
        List all registered backends with status.

        Returns:
            List of backend information dictionaries
        """
        backends = []

        for name, entry in self.registry.items():
            info = {
                'name': name,
                'loaded': entry.is_loaded(),
                'priority': entry.priority,
                'auto_load': entry.auto_load,
                'metadata': entry.metadata,
            }

            if entry.health:
                info['health'] = {
                    'status': entry.health.status.value,
                    'last_check': entry.health.last_check.isoformat(),
                    'response_time': entry.health.response_time,
                    'success_rate': entry.health.success_rate,
                    'error_count': entry.health.error_count,
                    'total_requests': entry.health.total_requests,
                    'message': entry.health.message
                }

            # Get capabilities if loaded
            if entry.is_loaded() and hasattr(entry.backend_instance, 'get_capabilities'):
                try:
                    caps = entry.backend_instance.get_capabilities()
                    info['max_qubits'] = caps.max_qubits
                    info['backend_type'] = caps.backend_type.value if hasattr(caps.backend_type, 'value') else str(caps.backend_type)
                except:
                    pass

            backends.append(info)

        return backends

    def unregister_backend(self, name: str):
        """
        Unregister a backend completely.

        Args:
            name: Backend name
        """
        self.unload_backend(name)

        with self._lock:
            if name in self.registry:
                del self.registry[name]
                logger.info(f"Unregistered backend: {name}")

    def shutdown(self):
        """Shutdown manager and clean up all backends."""
        logger.info("Shutting down EnhancedBackendManager")

        # Stop monitoring
        self.stop_health_monitoring()

        # Unload all backends
        for name in list(self.registry.keys()):
            self.unload_backend(name)


def create_enhanced_manager(
    backends: Optional[Dict[str, Any]] = None,
    health_check_interval: int = 60
) -> EnhancedBackendManager:
    """
    Factory function to create an enhanced backend manager.

    Args:
        backends: Dict of {name: backend_instance or backend_class}
        health_check_interval: Health check interval in seconds

    Returns:
        Configured EnhancedBackendManager

    Example:
        >>> from q_store.backends import QsimBackend
        >>> manager = create_enhanced_manager({
        ...     'qsim': QsimBackend(num_threads=4)
        ... })
    """
    manager = EnhancedBackendManager(health_check_interval=health_check_interval)

    if backends:
        for name, backend in backends.items():
            if inspect.isclass(backend):
                # Backend class
                manager.register_backend_class(name, backend)
            else:
                # Backend instance
                manager.register_backend_instance(name, backend)

    return manager
