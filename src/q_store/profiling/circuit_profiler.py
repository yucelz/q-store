"""
Circuit profiling for performance analysis.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from q_store.core import UnifiedCircuit, GateType


@dataclass
class GateProfile:
    """Profile data for a single gate."""
    gate_type: GateType
    targets: List[int]
    execution_time: float
    position: int
    
    
@dataclass
class CircuitProfile:
    """Profile data for an entire circuit."""
    n_qubits: int
    n_gates: int
    depth: int
    total_time: float
    gate_profiles: List[GateProfile] = field(default_factory=list)
    gate_counts: Dict[GateType, int] = field(default_factory=dict)
    avg_gate_time: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.gate_profiles:
            self.avg_gate_time = self.total_time / len(self.gate_profiles)


class CircuitProfiler:
    """
    Profiler for quantum circuits.
    
    Measures execution time, gate statistics, and performance metrics.
    """
    
    def __init__(self):
        """Initialize circuit profiler."""
        self.profiles: List[CircuitProfile] = []
        
    def profile_circuit(self, circuit: UnifiedCircuit,
                       execute_fn: Optional[Callable] = None) -> CircuitProfile:
        """
        Profile a quantum circuit.
        
        Args:
            circuit: Circuit to profile
            execute_fn: Optional function to execute circuit (for timing)
            
        Returns:
            CircuitProfile with performance data
        """
        start_time = time.perf_counter()
        
        # Collect gate profiles
        gate_profiles = []
        gate_counts = {}
        
        for i, gate in enumerate(circuit.gates):
            gate_start = time.perf_counter()
            
            # Simulate gate execution (or use custom function)
            if execute_fn:
                execute_fn(gate)
            else:
                # Simple delay simulation
                time.sleep(1e-6)
            
            gate_time = time.perf_counter() - gate_start
            
            gate_profile = GateProfile(
                gate_type=gate.gate_type,
                targets=gate.targets,
                execution_time=gate_time,
                position=i
            )
            gate_profiles.append(gate_profile)
            
            # Update counts
            gate_counts[gate.gate_type] = gate_counts.get(gate.gate_type, 0) + 1
        
        total_time = time.perf_counter() - start_time
        
        profile = CircuitProfile(
            n_qubits=circuit.n_qubits,
            n_gates=len(circuit.gates),
            depth=circuit.depth,
            total_time=total_time,
            gate_profiles=gate_profiles,
            gate_counts=gate_counts
        )
        
        self.profiles.append(profile)
        return profile
    
    def get_gate_time_distribution(self, profile: CircuitProfile) -> Dict[GateType, float]:
        """
        Get time distribution by gate type.
        
        Args:
            profile: Circuit profile to analyze
            
        Returns:
            Dictionary mapping gate types to total execution time
        """
        distribution = {}
        for gate_prof in profile.gate_profiles:
            gt = gate_prof.gate_type
            distribution[gt] = distribution.get(gt, 0.0) + gate_prof.execution_time
        return distribution
    
    def get_bottlenecks(self, profile: CircuitProfile, 
                       threshold: float = 0.1) -> List[GateProfile]:
        """
        Identify performance bottlenecks.
        
        Args:
            profile: Circuit profile to analyze
            threshold: Fraction of total time to consider bottleneck
            
        Returns:
            List of gate profiles that are bottlenecks
        """
        cutoff = profile.total_time * threshold
        bottlenecks = [
            gp for gp in profile.gate_profiles
            if gp.execution_time > cutoff
        ]
        return sorted(bottlenecks, key=lambda x: x.execution_time, reverse=True)
    
    def compare_profiles(self, profile1: CircuitProfile,
                        profile2: CircuitProfile) -> Dict[str, Any]:
        """
        Compare two circuit profiles.
        
        Args:
            profile1: First profile
            profile2: Second profile
            
        Returns:
            Comparison results
        """
        return {
            'time_diff': profile2.total_time - profile1.total_time,
            'time_ratio': profile2.total_time / profile1.total_time if profile1.total_time > 0 else float('inf'),
            'gate_count_diff': profile2.n_gates - profile1.n_gates,
            'depth_diff': profile2.depth - profile1.depth,
            'speedup': profile1.total_time / profile2.total_time if profile2.total_time > 0 else float('inf')
        }
    
    def get_summary(self, profile: CircuitProfile) -> Dict[str, Any]:
        """
        Get summary statistics for a profile.
        
        Args:
            profile: Circuit profile
            
        Returns:
            Summary dictionary
        """
        gate_times = [gp.execution_time for gp in profile.gate_profiles]
        
        return {
            'n_qubits': profile.n_qubits,
            'n_gates': profile.n_gates,
            'depth': profile.depth,
            'total_time': profile.total_time,
            'avg_gate_time': profile.avg_gate_time,
            'min_gate_time': min(gate_times) if gate_times else 0.0,
            'max_gate_time': max(gate_times) if gate_times else 0.0,
            'std_gate_time': np.std(gate_times) if gate_times else 0.0,
            'gate_counts': profile.gate_counts,
            'time_per_qubit': profile.total_time / profile.n_qubits if profile.n_qubits > 0 else 0.0,
            'time_per_gate': profile.total_time / profile.n_gates if profile.n_gates > 0 else 0.0
        }


def profile_circuit(circuit: UnifiedCircuit,
                   execute_fn: Optional[Callable] = None) -> CircuitProfile:
    """
    Convenience function to profile a circuit.
    
    Args:
        circuit: Circuit to profile
        execute_fn: Optional execution function
        
    Returns:
        CircuitProfile
    """
    profiler = CircuitProfiler()
    return profiler.profile_circuit(circuit, execute_fn)
