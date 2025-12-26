"""
Result Cache - LRU Cache for Quantum Measurement Results

Caches quantum circuit execution results to avoid redundant quantum execution.
Uses LRU (Least Recently Used) eviction policy.

Features:
- Fast O(1) lookup and insertion
- Automatic eviction when full
- Thread-safe operations
- Cache statistics (hits/misses)
"""

from collections import OrderedDict
from typing import Any, Optional, Dict
import threading
import hashlib
import json


class ResultCache:
    """
    LRU cache for quantum circuit results.
    
    Thread-safe cache with automatic eviction.
    
    Parameters
    ----------
    max_size : int, default=1000
        Maximum number of cached results
    
    Examples
    --------
    >>> cache = ResultCache(max_size=100)
    >>> cache.put('key1', result)
    >>> result = cache.get('key1')
    >>> stats = cache.stats()
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached result.
        
        Parameters
        ----------
        key : str
            Cache key
        
        Returns
        -------
        result : Any or None
            Cached result if found, None otherwise
        """
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """
        Put result in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Result to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Evict least recently used
                    self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        with self.lock:
            return key in self.cache
    
    def __len__(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)
    
    def clear(self):
        """Clear all cached results."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        stats : dict
            Cache statistics including hit rate, size, etc.
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': total,
                'hit_rate': hit_rate,
            }
    
    @staticmethod
    def generate_key(circuit: Any, parameters: Optional[Dict] = None) -> str:
        """
        Generate cache key from circuit and parameters.
        
        Parameters
        ----------
        circuit : Any
            Quantum circuit object
        parameters : dict, optional
            Circuit parameters
        
        Returns
        -------
        key : str
            Cache key (hash)
        """
        # Create deterministic string representation
        circuit_str = str(circuit)
        if parameters:
            params_str = json.dumps(parameters, sort_keys=True)
            circuit_str += params_str
        
        # Generate hash
        return hashlib.sha256(circuit_str.encode()).hexdigest()[:16]
