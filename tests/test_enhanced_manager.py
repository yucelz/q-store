"""
Tests for Enhanced Backend Manager.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from q_store.backends.enhanced_manager import (
    EnhancedBackendManager,
    BackendStatus,
    BackendHealth,
    BackendRegistry,
    create_enhanced_manager
)


class SimpleBackend:
    """Simple backend for testing."""
    
    def __init__(self, name="test", fail_health=False):
        self.name = name
        self.fail_health = fail_health
        self.initialized = False
        self.closed = False
        
    def get_capabilities(self):
        return Mock(max_qubits=10, backend_type=Mock(value='simulator'))
    
    def is_available(self):
        return not self.fail_health
    
    def health_check(self):
        if self.fail_health:
            raise RuntimeError("Health check failed")
        return True
    
    def close(self):
        self.closed = True


class TestEnhancedBackendManager:
    """Test enhanced backend manager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = EnhancedBackendManager(health_check_interval=30)
        
        assert manager.health_check_interval == 30
        assert len(manager.registry) == 0
        assert not manager._monitoring_active
    
    def test_register_backend_class(self):
        """Test registering a backend class."""
        manager = EnhancedBackendManager()
        
        manager.register_backend_class(
            'test',
            SimpleBackend,
            config={'name': 'test_backend'},
            priority=50
        )
        
        assert 'test' in manager.registry
        entry = manager.registry['test']
        assert entry.backend_class == SimpleBackend
        assert entry.config == {'name': 'test_backend'}
        assert entry.priority == 50
        assert not entry.is_loaded()
    
    def test_register_backend_instance(self):
        """Test registering a backend instance."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend('test_backend')
        
        manager.register_backend_instance('test', backend, priority=75)
        
        assert 'test' in manager.registry
        entry = manager.registry['test']
        assert entry.backend_instance == backend
        assert entry.is_loaded()
        assert entry.priority == 75
        assert entry.health.status in [BackendStatus.HEALTHY, BackendStatus.UNKNOWN]
    
    def test_load_backend(self):
        """Test loading a backend."""
        manager = EnhancedBackendManager()
        
        manager.register_backend_class(
            'test',
            SimpleBackend,
            config={'name': 'loaded_backend'}
        )
        
        # Backend not loaded initially
        entry = manager.registry['test']
        assert not entry.is_loaded()
        
        # Load backend
        backend = manager.load_backend('test')
        
        assert backend is not None
        assert isinstance(backend, SimpleBackend)
        assert backend.name == 'loaded_backend'
        assert entry.is_loaded()
        assert entry.health.status == BackendStatus.HEALTHY
    
    def test_load_nonexistent_backend(self):
        """Test loading non-existent backend raises error."""
        manager = EnhancedBackendManager()
        
        with pytest.raises(ValueError, match="not registered"):
            manager.load_backend('nonexistent')
    
    def test_unload_backend(self):
        """Test unloading a backend."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        assert manager.registry['test'].is_loaded()
        
        manager.unload_backend('test')
        
        assert not manager.registry['test'].is_loaded()
        assert backend.closed
        assert manager.registry['test'].health.status == BackendStatus.OFFLINE
    
    def test_get_backend_auto_load(self):
        """Test getting backend with auto-load."""
        manager = EnhancedBackendManager()
        
        manager.register_backend_class('test', SimpleBackend)
        
        # Should auto-load
        backend = manager.get_backend('test', auto_load=True)
        
        assert backend is not None
        assert manager.registry['test'].is_loaded()
    
    def test_get_backend_no_auto_load(self):
        """Test getting unloaded backend without auto-load fails."""
        manager = EnhancedBackendManager()
        
        manager.register_backend_class('test', SimpleBackend)
        
        with pytest.raises(RuntimeError, match="not loaded"):
            manager.get_backend('test', auto_load=False)
    
    def test_get_healthy_backend_specific(self):
        """Test getting specific healthy backend."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        
        healthy = manager.get_healthy_backend('test')
        
        assert healthy == backend
    
    def test_get_healthy_backend_unhealthy_fails(self):
        """Test getting unhealthy backend fails."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend(fail_health=True)
        
        manager.register_backend_instance('test', backend)
        
        with pytest.raises(RuntimeError, match="not healthy"):
            manager.get_healthy_backend('test')
    
    def test_get_healthy_backend_best_available(self):
        """Test getting best available healthy backend."""
        manager = EnhancedBackendManager()
        
        # Register multiple backends with different priorities
        backend1 = SimpleBackend('backend1')
        backend2 = SimpleBackend('backend2')
        backend3 = SimpleBackend('backend3', fail_health=True)
        
        manager.register_backend_instance('low_priority', backend1, priority=100)
        manager.register_backend_instance('high_priority', backend2, priority=10)
        manager.register_backend_instance('unhealthy', backend3, priority=1)
        
        # Should select high_priority (lowest priority value, healthy)
        best = manager.get_healthy_backend()
        
        assert best == backend2
    
    def test_get_healthy_backend_none_available(self):
        """Test error when no healthy backends available."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend(fail_health=True)
        
        manager.register_backend_instance('test', backend)
        
        with pytest.raises(RuntimeError, match="No healthy backends"):
            manager.get_healthy_backend()
    
    def test_check_backend_health(self):
        """Test health checking."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        
        # Initial check should pass
        assert manager.registry['test'].health.status == BackendStatus.HEALTHY
        
        # Make backend unhealthy
        backend.fail_health = True
        manager._check_backend_health('test')
        
        assert manager.registry['test'].health.status == BackendStatus.UNHEALTHY
    
    def test_check_all_health(self):
        """Test checking health of all backends."""
        manager = EnhancedBackendManager()
        
        backend1 = SimpleBackend('b1')
        backend2 = SimpleBackend('b2')
        
        manager.register_backend_instance('b1', backend1)
        manager.register_backend_instance('b2', backend2)
        
        # Both should be healthy initially
        manager.check_all_health()
        
        assert manager.registry['b1'].health.status == BackendStatus.HEALTHY
        assert manager.registry['b2'].health.status == BackendStatus.HEALTHY
        
        # Make one unhealthy
        backend1.fail_health = True
        manager.check_all_health()
        
        assert manager.registry['b1'].health.status == BackendStatus.UNHEALTHY
        assert manager.registry['b2'].health.status == BackendStatus.HEALTHY
    
    def test_update_backend_stats(self):
        """Test updating backend statistics."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        entry = manager.registry['test']
        
        # Initial state
        assert entry.health.total_requests == 0
        assert entry.health.error_count == 0
        assert entry.health.success_rate == 1.0
        
        # Record successful execution
        manager.update_backend_stats('test', success=True, execution_time=0.5)
        
        assert entry.health.total_requests == 1
        assert entry.health.error_count == 0
        assert entry.health.success_rate > 0.9
        
        # Record failed executions
        for _ in range(10):
            manager.update_backend_stats('test', success=False)
        
        assert entry.health.total_requests == 11
        assert entry.health.error_count == 10
        assert entry.health.success_rate < 0.5
        assert entry.health.status == BackendStatus.UNHEALTHY
    
    def test_list_backends(self):
        """Test listing backends."""
        manager = EnhancedBackendManager()
        
        backend1 = SimpleBackend('b1')
        manager.register_backend_class('b2', SimpleBackend, priority=50)
        manager.register_backend_instance('b1', backend1, priority=10)
        
        backends = manager.list_backends()
        
        assert len(backends) == 2
        
        # Check b1 info
        b1_info = next(b for b in backends if b['name'] == 'b1')
        assert b1_info['loaded'] == True
        assert b1_info['priority'] == 10
        assert 'health' in b1_info
        assert b1_info['health']['status'] == BackendStatus.HEALTHY.value
        
        # Check b2 info
        b2_info = next(b for b in backends if b['name'] == 'b2')
        assert b2_info['loaded'] == False
        assert b2_info['priority'] == 50
    
    def test_unregister_backend(self):
        """Test unregistering a backend."""
        manager = EnhancedBackendManager()
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        assert 'test' in manager.registry
        
        manager.unregister_backend('test')
        
        assert 'test' not in manager.registry
        assert backend.closed
    
    def test_shutdown(self):
        """Test manager shutdown."""
        manager = EnhancedBackendManager()
        
        backend1 = SimpleBackend('b1')
        backend2 = SimpleBackend('b2')
        
        manager.register_backend_instance('b1', backend1)
        manager.register_backend_instance('b2', backend2)
        
        manager.shutdown()
        
        # All backends should be unloaded
        assert not manager.registry['b1'].is_loaded()
        assert not manager.registry['b2'].is_loaded()
        assert backend1.closed
        assert backend2.closed
        assert not manager._monitoring_active
    
    def test_health_monitoring_start_stop(self):
        """Test starting and stopping health monitoring."""
        manager = EnhancedBackendManager(health_check_interval=1)
        backend = SimpleBackend()
        
        manager.register_backend_instance('test', backend)
        
        # Start monitoring
        manager.start_health_monitoring()
        assert manager._monitoring_active
        
        # Wait a bit for monitoring to run
        time.sleep(1.5)
        
        # Stop monitoring
        manager.stop_health_monitoring()
        assert not manager._monitoring_active
        
        # Health should have been checked
        assert manager.registry['test'].health.last_check is not None


class TestCreateEnhancedManager:
    """Test enhanced manager factory function."""
    
    def test_create_empty(self):
        """Test creating empty manager."""
        manager = create_enhanced_manager()
        
        assert isinstance(manager, EnhancedBackendManager)
        assert len(manager.registry) == 0
    
    def test_create_with_instances(self):
        """Test creating with backend instances."""
        backend1 = SimpleBackend('b1')
        backend2 = SimpleBackend('b2')
        
        manager = create_enhanced_manager({
            'b1': backend1,
            'b2': backend2
        })
        
        assert len(manager.registry) == 2
        assert manager.registry['b1'].backend_instance == backend1
        assert manager.registry['b2'].backend_instance == backend2
    
    def test_create_with_classes(self):
        """Test creating with backend classes."""
        manager = create_enhanced_manager({
            'test': SimpleBackend
        })
        
        assert len(manager.registry) == 1
        assert manager.registry['test'].backend_class == SimpleBackend
        assert not manager.registry['test'].is_loaded()
    
    def test_create_with_health_interval(self):
        """Test creating with custom health check interval."""
        manager = create_enhanced_manager(health_check_interval=120)
        
        assert manager.health_check_interval == 120


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
