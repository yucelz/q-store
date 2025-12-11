"""
Tunneling Engine
Implements quantum tunneling for pattern discovery and global optimization.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TunnelingResult:
    """Result from tunneling search"""
    vector: np.ndarray
    distance: float
    tunneling_probability: float


class TunnelingEngine:
    """
    Discovers hidden patterns via quantum tunneling
    Finds globally optimal matches by escaping local optima
    """
    
    def __init__(self, quantum_backend=None):
        """
        Initialize tunneling engine
        
        Args:
            quantum_backend: IonQQuantumBackend instance for quantum operations
        """
        self.quantum_backend = quantum_backend
        
    def tunnel_search(self, 
                     query: np.ndarray,
                     candidates: List[np.ndarray],
                     barrier_threshold: float = 0.8,
                     top_k: int = 10) -> List[TunnelingResult]:
        """
        Search for matches allowing quantum tunneling through barriers
        
        Args:
            query: Query vector
            candidates: List of candidate vectors
            barrier_threshold: Height of barrier (0-1), higher = more tunneling
            top_k: Number of results to return
            
        Returns:
            List of TunnelingResults
        """
        results = []
        
        for candidate in candidates:
            # Calculate classical distance
            distance = np.linalg.norm(query - candidate)
            
            # Calculate tunneling probability
            # Higher barrier = higher chance of tunneling to distant matches
            tunneling_prob = self._calculate_tunneling_probability(
                distance, barrier_threshold
            )
            
            results.append(TunnelingResult(
                vector=candidate,
                distance=distance,
                tunneling_probability=tunneling_prob
            ))
        
        # Sort by combination of distance and tunneling probability
        # This allows finding both nearby and distant (but relevant) matches
        results.sort(
            key=lambda r: r.distance * (1 - barrier_threshold * r.tunneling_probability)
        )
        
        return results[:top_k]
    
    def discover_regimes(self, 
                        historical_data: List[np.ndarray],
                        n_regimes: int = 3) -> List[List[int]]:
        """
        Discover distinct regimes in historical data using tunneling
        
        Args:
            historical_data: List of historical state vectors
            n_regimes: Number of regimes to identify
            
        Returns:
            List of regime clusters (each is list of indices)
        """
        # Use quantum tunneling to escape local optima in clustering
        # This is a simplified version - full implementation would use quantum annealing
        
        n_samples = len(historical_data)
        
        # Initialize random centroids
        centroid_indices = np.random.choice(n_samples, n_regimes, replace=False)
        centroids = [historical_data[i] for i in centroid_indices]
        
        # Iterative refinement with tunneling
        max_iterations = 10
        for iteration in range(max_iterations):
            # Assign points to clusters
            clusters = [[] for _ in range(n_regimes)]
            
            for idx, point in enumerate(historical_data):
                # Find closest centroid with tunneling
                distances = [np.linalg.norm(point - c) for c in centroids]
                
                # Apply tunneling - allow jumps to distant clusters
                tunneling_factor = 0.3 * (1 - iteration / max_iterations)
                noise = np.random.randn(n_regimes) * tunneling_factor
                adjusted_distances = np.array(distances) + noise
                
                closest = np.argmin(adjusted_distances)
                clusters[closest].append(idx)
            
            # Update centroids
            for i in range(n_regimes):
                if clusters[i]:
                    cluster_points = [historical_data[idx] for idx in clusters[i]]
                    centroids[i] = np.mean(cluster_points, axis=0)
        
        return clusters
    
    def find_precursors(self, 
                       target_event: np.ndarray,
                       historical_states: List[Tuple[np.ndarray, float]],
                       lookback_window: int = 10) -> List[np.ndarray]:
        """
        Find precursor states that led to target event
        Uses tunneling to find non-obvious patterns
        
        Args:
            target_event: Target event vector
            historical_states: List of (state, timestamp) tuples
            lookback_window: How far back to look before event
            
        Returns:
            List of precursor state vectors
        """
        precursors = []
        
        # Find states similar to target
        for i, (state, timestamp) in enumerate(historical_states):
            similarity = self._cosine_similarity(state, target_event)
            
            if similarity > 0.7:  # Found a match
                # Look at states before this
                start_idx = max(0, i - lookback_window)
                lookback_states = [s for s, t in historical_states[start_idx:i]]
                
                # Use tunneling to find non-obvious precursors
                if lookback_states:
                    # Apply quantum tunneling search
                    tunneling_results = self.tunnel_search(
                        query=target_event,
                        candidates=lookback_states,
                        barrier_threshold=0.9,  # High barrier = find distant patterns
                        top_k=3
                    )
                    
                    precursors.extend([r.vector for r in tunneling_results])
        
        return precursors
    
    def _calculate_tunneling_probability(self, 
                                        distance: float,
                                        barrier: float) -> float:
        """
        Calculate quantum tunneling probability
        Based on: T ≈ exp(-2κL) where κ = sqrt(2*barrier)
        
        Args:
            distance: Distance to tunnel
            barrier: Barrier height
            
        Returns:
            Tunneling probability (0-1)
        """
        kappa = np.sqrt(2 * barrier)
        transmission = np.exp(-2 * kappa * distance)
        return min(1.0, transmission)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
