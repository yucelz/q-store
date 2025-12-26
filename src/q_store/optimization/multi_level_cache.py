"""
Multi-Level Cache System

Three-level caching hierarchy:
- L1: Parameter cache (hot parameters)
- L2: Compiled circuit cache (avoid recompilation)
- L3: Result cache (measurement results)

Design:
- L1: Small, fast, parameter-based lookup
- L2: Medium, compiled circuits, avoid expensive compilation
- L3: Large, measurement results, avoid quantum execution

Eviction: LRU (Least Recently Used)

Example:
    >>> cache = MultiLevelCache(
    ...     l1_size=100,
    ...     l2_size=1000,
    ...     l3_size=10000,
    ... )
    >>> 
    >>> # Check L3 first
    >>> result = cache.get_result(circuit_hash)
    >>> if result is None:
    ...     result = execute_circuit(circuit)
    ...     cache.put_result(circuit_hash, result)
"""

from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import hashlib
import pickle
import time


class LRUCache:
    """
    LRU Cache implementation.
    
    Thread-safe ordered dictionary with size limit.
    
    Parameters
    ----------
    maxsize : int
        Maximum cache size
    name : str, optional
        Cache name for statistics
    """
    
    def __init__(self, maxsize: int, name: str = 'cache'):
        self.maxsize = maxsize
        self.name = name
        self.cache = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Parameters
        ----------
        key : str
            Cache key
        
        Returns
        -------
        value : Any or None
            Cached value, or None if not found
        """
        if key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]['value']
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """
        Put item in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
            }
        else:
            # Add new
            if len(self.cache) >= self.maxsize:
                # Evict oldest
                self.cache.popitem(last=False)
                self.evictions += 1
            
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
            }
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'name': self.name,
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.maxsize,
        }


class MultiLevelCache:
    """
    Three-level cache system.
    
    L1: Hot parameters (small, fast)
    L2: Compiled circuits (medium)
    L3: Measurement results (large)
    
    Parameters
    ----------
    l1_size : int, default=100
        L1 cache size (parameters)
    l2_size : int, default=1000
        L2 cache size (circuits)
    l3_size : int, default=10000
        L3 cache size (results)
    
    Examples
    --------
    >>> cache = MultiLevelCache()
    >>> 
    >>> # L1: Parameter cache
    >>> params_key = cache.hash_params(params)
    >>> cached_params = cache.get_l1(params_key)
    >>> 
    >>> # L2: Circuit cache
    >>> circuit_key = cache.hash_circuit(circuit)
    >>> cached_circuit = cache.get_l2(circuit_key)
    >>> 
    >>> # L3: Result cache
    >>> result_key = cache.hash_result(circuit, params)
    >>> cached_result = cache.get_l3(result_key)
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l3_size: int = 10000,
    ):
        self.l1 = LRUCache(maxsize=l1_size, name='L1_params')
        self.l2 = LRUCache(maxsize=l2_size, name='L2_circuits')
        self.l3 = LRUCache(maxsize=l3_size, name='L3_results')
    
    # ========================================================================
    # L1: Parameter Cache
    # ========================================================================
    
    def hash_params(self, params: Any) -> str:
        """
        Hash parameters for L1 cache.
        
        Parameters
        ----------
        params : Any
            Parameters (array, list, dict)
        
        Returns
        -------
        hash : str
            Parameter hash
        """
        try:
            # Convert to bytes
            if hasattr(params, 'tobytes'):
                data = params.tobytes()
            else:
                data = pickle.dumps(params)
            
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return str(hash(str(params)))[:16]
    
    def get_l1(self, key: str) -> Optional[Any]:
        """Get from L1 cache."""
        return self.l1.get(key)
    
    def put_l1(self, key: str, value: Any):
        """Put in L1 cache."""
        self.l1.put(key, value)
    
    # ========================================================================
    # L2: Circuit Cache
    # ========================================================================
    
    def hash_circuit(self, circuit: Any, params: Optional[Any] = None) -> str:
        """
        Hash circuit for L2 cache.
        
        Parameters
        ----------
        circuit : Any
            Quantum circuit
        params : Any, optional
            Circuit parameters
        
        Returns
        -------
        hash : str
            Circuit hash
        """
        try:
            # Use circuit string representation + params
            circuit_str = str(circuit)
            if params is not None:
                params_str = str(params)
                circuit_str += params_str
            
            return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]
        except Exception:
            return str(hash(str(circuit)))[:16]
    
    def get_l2(self, key: str) -> Optional[Any]:
        """Get from L2 cache."""
        return self.l2.get(key)
    
    def put_l2(self, key: str, value: Any):
        """Put in L2 cache."""
        self.l2.put(key, value)
    
    # ========================================================================
    # L3: Result Cache
    # ========================================================================
    
    def hash_result(
        self,
        circuit: Any,
        params: Optional[Any] = None,
        shots: Optional[int] = None,
    ) -> str:
        """
        Hash for L3 result cache.
        
        Parameters
        ----------
        circuit : Any
            Quantum circuit
        params : Any, optional
            Parameters
        shots : int, optional
            Number of shots
        
        Returns
        -------
        hash : str
            Result hash
        """
        try:
            # Combine circuit, params, shots
            circuit_str = str(circuit)
            if params is not None:
                circuit_str += str(params)
            if shots is not None:
                circuit_str += str(shots)
            
            return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]
        except Exception:
            return str(hash((str(circuit), str(params), shots)))[:16]
    
    def get_l3(self, key: str) -> Optional[Any]:
        """Get from L3 cache."""
        return self.l3.get(key)
    
    def put_l3(self, key: str, value: Any):
        """Put in L3 cache."""
        self.l3.put(key, value)
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def clear_all(self):
        """Clear all caches."""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
    
    def stats(self) -> Dict:
        """
        Get all cache statistics.
        
        Returns
        -------
        stats : dict
            Statistics for all cache levels
        """
        return {
            'l1': self.l1.stats(),
            'l2': self.l2.stats(),
            'l3': self.l3.stats(),
            'total_hits': self.l1.hits + self.l2.hits + self.l3.hits,
            'total_misses': self.l1.misses + self.l2.misses + self.l3.misses,
            'overall_hit_rate': (
                (self.l1.hits + self.l2.hits + self.l3.hits) /
                (self.l1.hits + self.l2.hits + self.l3.hits + 
                 self.l1.misses + self.l2.misses + self.l3.misses)
                if (self.l1.hits + self.l2.hits + self.l3.hits + 
                    self.l1.misses + self.l2.misses + self.l3.misses) > 0
                else 0
            ),
        }
