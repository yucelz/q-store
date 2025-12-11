"""
State Manager for Quantum Database
Handles superposition, coherence, and measurement of quantum states.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class QuantumState:
    """Represents a quantum state with metadata"""
    state_id: str
    vector: np.ndarray
    contexts: List[Tuple[str, float]]  # (context_name, probability)
    coherence_time: float  # milliseconds
    creation_time: float  # timestamp
    
    def is_coherent(self, current_time: float) -> bool:
        """Check if state is still coherent"""
        elapsed_ms = (current_time - self.creation_time) * 1000
        return elapsed_ms < self.coherence_time


class StateManager:
    """
    Manages quantum states with superposition and coherence
    """
    
    def __init__(self):
        self.states: Dict[str, QuantumState] = {}
        
    def create_superposition(self, 
                           state_id: str,
                           vectors: List[np.ndarray], 
                           contexts: List[str],
                           coherence_time: float = 1000.0) -> QuantumState:
        """
        Create a quantum state in superposition of multiple contexts
        
        Args:
            state_id: Unique identifier for this state
            vectors: List of vectors (one per context)
            contexts: Context labels
            coherence_time: How long state remains coherent (ms)
            
        Returns:
            QuantumState in superposition
        """
        if len(vectors) != len(contexts):
            raise ValueError("Number of vectors must match number of contexts")
        
        # Equal superposition probabilities
        n = len(contexts)
        probabilities = [1.0 / n] * n
        
        # Create combined context list
        context_probs = list(zip(contexts, probabilities))
        
        # Use first vector as representative (measurement will select context)
        state = QuantumState(
            state_id=state_id,
            vector=vectors[0],  # Representative vector
            contexts=context_probs,
            coherence_time=coherence_time,
            creation_time=time.time()
        )
        
        self.states[state_id] = state
        return state
    
    def measure_with_context(self, 
                            state_id: str, 
                            query_context: str) -> Optional[np.ndarray]:
        """
        Measure quantum state, collapsing to context most relevant to query
        
        Args:
            state_id: ID of state to measure
            query_context: Context of the query
            
        Returns:
            Collapsed vector, or None if state not found or decoherent
        """
        state = self.states.get(state_id)
        
        if state is None:
            return None
        
        # Check coherence
        if not state.is_coherent(time.time()):
            # State has decohered, remove it
            del self.states[state_id]
            return None
        
        # Find matching context
        for context, prob in state.contexts:
            if context == query_context:
                # Measurement collapses to this context
                # Return the vector (in real implementation, different vectors per context)
                return state.vector
        
        # No exact match, return default (highest probability context)
        return state.vector
    
    def apply_decoherence(self) -> List[str]:
        """
        Remove decohered states (adaptive TTL)
        
        Returns:
            List of removed state IDs
        """
        current_time = time.time()
        removed = []
        
        for state_id, state in list(self.states.items()):
            if not state.is_coherent(current_time):
                del self.states[state_id]
                removed.append(state_id)
        
        return removed
    
    def get_coherent_states(self) -> List[QuantumState]:
        """
        Get all currently coherent states
        
        Returns:
            List of coherent QuantumStates
        """
        current_time = time.time()
        return [
            state for state in self.states.values()
            if state.is_coherent(current_time)
        ]
    
    def update_state(self, state_id: str, new_vector: np.ndarray):
        """
        Update a quantum state with new vector
        
        Args:
            state_id: ID of state to update
            new_vector: New vector data
        """
        if state_id in self.states:
            self.states[state_id].vector = new_vector
            # Reset creation time on update
            self.states[state_id].creation_time = time.time()
    
    def get_state(self, state_id: str) -> Optional[QuantumState]:
        """
        Retrieve a quantum state by ID
        
        Args:
            state_id: State identifier
            
        Returns:
            QuantumState or None if not found
        """
        state = self.states.get(state_id)
        
        if state and state.is_coherent(time.time()):
            return state
        elif state:
            # Remove decoherent state
            del self.states[state_id]
            
        return None
    
    def get_state_count(self) -> int:
        """Get number of active coherent states"""
        return len(self.get_coherent_states())
