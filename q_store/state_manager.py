"""
State Manager for Quantum Database
Handles superposition, coherence, and measurement of quantum states.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class StateStatus(Enum):
    """Quantum state lifecycle status"""
    CREATED = "created"
    ACTIVE = "active"
    MEASURED = "measured"
    DECOHERED = "decohered"
    ARCHIVED = "archived"


@dataclass
class QuantumState:
    """Represents a quantum state with metadata"""
    state_id: str
    vector: np.ndarray
    contexts: List[Tuple[str, float]]  # (context_name, probability)
    coherence_time: float  # milliseconds
    creation_time: float  # timestamp
    status: StateStatus = StateStatus.CREATED
    measurement_count: int = 0
    last_measured: Optional[float] = None
    
    def is_coherent(self, current_time: float) -> bool:
        """Check if state is still coherent"""
        elapsed_ms = (current_time - self.creation_time) * 1000
        return elapsed_ms < self.coherence_time and self.status != StateStatus.DECOHERED


class StateManager:
    """
    Manages quantum states with superposition and coherence
    """
    
    def __init__(self, config=None):
        self.config = config
        self.states: Dict[str, QuantumState] = {}
        self._lock = asyncio.Lock()
        self._decoherence_task = None
        
    async def start(self):
        """Start background tasks"""
        self._decoherence_task = asyncio.create_task(self._decoherence_loop())
        logger.info("State manager started")
    
    async def stop(self):
        """Stop background tasks"""
        if self._decoherence_task:
            self._decoherence_task.cancel()
            try:
                await self._decoherence_task
            except asyncio.CancelledError:
                pass
        logger.info("State manager stopped")
    
    async def create_superposition(
        self,
        state_id: str,
        vectors: List[np.ndarray],
        contexts: List[str],
        coherence_time: Optional[float] = None
    ) -> QuantumState:
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
        async with self._lock:
            # Check capacity
            if self.config and len(self.states) >= self.config.max_quantum_states:
                await self._evict_oldest_state()
            
            # Equal probability distribution
            n = len(contexts)
            probabilities = [1.0 / n] * n
            context_probs = list(zip(contexts, probabilities))
            
            coherence = coherence_time or (self.config.default_coherence_time if self.config else 1000.0)
            
            state = QuantumState(
                state_id=state_id,
                vector=vectors[0],  # Representative vector
                contexts=context_probs,
                coherence_time=coherence,
                creation_time=time.time(),
                status=StateStatus.ACTIVE
            )
            
            self.states[state_id] = state
            
            logger.debug(f"Created superposition state: {state_id}")
            return state
    
    async def measure_with_context(
        self,
        state_id: str,
        query_context: str
    ) -> Optional[np.ndarray]:
        """
        Measure quantum state with context collapse
        
        Args:
            state_id: ID of state to measure
            query_context: Context of the query
            
        Returns:
            Collapsed vector, or None if state not found or decoherent
        """
        async with self._lock:
            state = self.states.get(state_id)
            
            if state is None:
                return None
            
            current_time = time.time()
            
            # Check coherence
            if not state.is_coherent(current_time):
                state.status = StateStatus.DECOHERED
                del self.states[state_id]
                return None
            
            # Find matching context
            for context, prob in state.contexts:
                if context == query_context:
                    # Measurement collapses to this context
                    state.status = StateStatus.MEASURED
                    state.measurement_count += 1
                    state.last_measured = current_time
                    return state.vector
            
            # Default to highest probability context
            state.status = StateStatus.MEASURED
            state.measurement_count += 1
            state.last_measured = current_time
            return state.vector
    
    async def _decoherence_loop(self):
        """Background task to clean up decohered states"""
        while True:
            try:
                interval = self.config.decoherence_check_interval if self.config else 60
                await asyncio.sleep(interval)
                await self.apply_decoherence()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decoherence loop: {e}")
    
    async def apply_decoherence(self) -> List[str]:
        """
        Remove decohered states (adaptive TTL)
        
        Returns:
            List of removed state IDs
        """
        async with self._lock:
            current_time = time.time()
            removed = []
            
            for state_id, state in list(self.states.items()):
                if not state.is_coherent(current_time):
                    state.status = StateStatus.DECOHERED
                    del self.states[state_id]
                    removed.append(state_id)
            
            return removed
    
    async def _evict_oldest_state(self):
        """Evict oldest state when capacity is reached"""
        if not self.states:
            return
        
        oldest_id = min(self.states.keys(), 
                       key=lambda k: self.states[k].creation_time)
        del self.states[oldest_id]
        logger.debug(f"Evicted state: {oldest_id}")
    
    def get_active_count(self) -> int:
        """Get count of active coherent states"""
        current_time = time.time()
        return sum(1 for s in self.states.values() 
                  if s.is_coherent(current_time))
    
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
        return self.get_active_count()
