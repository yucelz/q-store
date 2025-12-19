"""
Optimization profiler for comparing circuit transformations.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from q_store.core import UnifiedCircuit


@dataclass
class OptimizationResult:
    """Results from an optimization profiling run."""
    original_gates: int
    optimized_gates: int
    original_depth: int
    optimized_depth: int
    optimization_time: float
    gate_reduction: int
    depth_reduction: int
    gate_reduction_pct: float
    depth_reduction_pct: float
    preserves_functionality: bool
    
    
class OptimizationProfiler:
    """
    Profiler for optimization techniques.
    
    Compares circuits before and after optimization.
    """
    
    def __init__(self):
        """Initialize optimization profiler."""
        self.results: List[OptimizationResult] = []
    
    def profile_optimization(self, 
                           original: UnifiedCircuit,
                           optimized: UnifiedCircuit,
                           verify_fn: Optional[Callable] = None) -> OptimizationResult:
        """
        Profile an optimization transformation.
        
        Args:
            original: Original circuit
            optimized: Optimized circuit
            verify_fn: Optional function to verify equivalence
            
        Returns:
            OptimizationResult with metrics
        """
        # Calculate reductions
        gate_reduction = len(original.gates) - len(optimized.gates)
        depth_reduction = original.depth - optimized.depth
        
        gate_reduction_pct = 100 * gate_reduction / len(original.gates) if len(original.gates) > 0 else 0.0
        depth_reduction_pct = 100 * depth_reduction / original.depth if original.depth > 0 else 0.0
        
        # Verify functionality preservation
        preserves = True
        if verify_fn:
            try:
                preserves = verify_fn(original, optimized)
            except Exception:
                preserves = False
        
        result = OptimizationResult(
            original_gates=len(original.gates),
            optimized_gates=len(optimized.gates),
            original_depth=original.depth,
            optimized_depth=optimized.depth,
            optimization_time=0.0,  # Can be measured separately
            gate_reduction=gate_reduction,
            depth_reduction=depth_reduction,
            gate_reduction_pct=gate_reduction_pct,
            depth_reduction_pct=depth_reduction_pct,
            preserves_functionality=preserves
        )
        
        self.results.append(result)
        return result
    
    def profile_optimization_with_timing(self,
                                        original: UnifiedCircuit,
                                        optimization_fn: Callable[[UnifiedCircuit], UnifiedCircuit],
                                        verify_fn: Optional[Callable] = None) -> OptimizationResult:
        """
        Profile an optimization with execution timing.
        
        Args:
            original: Original circuit
            optimization_fn: Function that returns optimized circuit
            verify_fn: Optional verification function
            
        Returns:
            OptimizationResult with timing
        """
        start_time = time.perf_counter()
        optimized = optimization_fn(original)
        optimization_time = time.perf_counter() - start_time
        
        result = self.profile_optimization(original, optimized, verify_fn)
        result.optimization_time = optimization_time
        
        return result
    
    def compare_optimizations(self,
                            original: UnifiedCircuit,
                            optimizations: Dict[str, UnifiedCircuit]) -> Dict[str, Any]:
        """
        Compare multiple optimization strategies.
        
        Args:
            original: Original circuit
            optimizations: Dictionary mapping names to optimized circuits
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, optimized in optimizations.items():
            result = self.profile_optimization(original, optimized)
            results[name] = {
                'gate_reduction': result.gate_reduction,
                'gate_reduction_pct': result.gate_reduction_pct,
                'depth_reduction': result.depth_reduction,
                'depth_reduction_pct': result.depth_reduction_pct,
                'final_gates': result.optimized_gates,
                'final_depth': result.optimized_depth,
                'preserves_functionality': result.preserves_functionality
            }
        
        # Find best optimization
        valid_opts = {k: v for k, v in results.items() if v['preserves_functionality']}
        if valid_opts:
            best_gate_reduction = max(valid_opts.items(), key=lambda x: x[1]['gate_reduction_pct'])
            best_depth_reduction = max(valid_opts.items(), key=lambda x: x[1]['depth_reduction_pct'])
            
            results['best_gate_reduction'] = best_gate_reduction[0]
            results['best_depth_reduction'] = best_depth_reduction[0]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics across all profiled optimizations.
        
        Returns:
            Statistics dictionary
        """
        if not self.results:
            return {}
        
        gate_reductions = [r.gate_reduction_pct for r in self.results]
        depth_reductions = [r.depth_reduction_pct for r in self.results]
        times = [r.optimization_time for r in self.results]
        
        return {
            'n_optimizations': len(self.results),
            'avg_gate_reduction_pct': np.mean(gate_reductions),
            'avg_depth_reduction_pct': np.mean(depth_reductions),
            'avg_optimization_time': np.mean(times),
            'max_gate_reduction_pct': max(gate_reductions),
            'max_depth_reduction_pct': max(depth_reductions),
            'min_optimization_time': min(times) if times else 0.0,
            'max_optimization_time': max(times) if times else 0.0,
            'successful_optimizations': sum(1 for r in self.results if r.preserves_functionality),
            'failed_optimizations': sum(1 for r in self.results if not r.preserves_functionality)
        }
    
    def benchmark_optimization(self,
                             circuits: List[UnifiedCircuit],
                             optimization_fn: Callable[[UnifiedCircuit], UnifiedCircuit],
                             verify_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Benchmark an optimization across multiple circuits.
        
        Args:
            circuits: List of circuits to optimize
            optimization_fn: Optimization function
            verify_fn: Optional verification function
            
        Returns:
            Benchmark results
        """
        results = []
        
        for circuit in circuits:
            result = self.profile_optimization_with_timing(
                circuit, optimization_fn, verify_fn
            )
            results.append(result)
        
        gate_reductions = [r.gate_reduction_pct for r in results]
        depth_reductions = [r.depth_reduction_pct for r in results]
        times = [r.optimization_time for r in results]
        
        return {
            'n_circuits': len(circuits),
            'avg_gate_reduction_pct': np.mean(gate_reductions),
            'std_gate_reduction_pct': np.std(gate_reductions),
            'avg_depth_reduction_pct': np.mean(depth_reductions),
            'std_depth_reduction_pct': np.std(depth_reductions),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'total_time': sum(times),
            'success_rate': sum(1 for r in results if r.preserves_functionality) / len(results) if results else 0.0,
            'results': results
        }
    
    def generate_report(self, result: OptimizationResult) -> str:
        """
        Generate a human-readable report for an optimization.
        
        Args:
            result: Optimization result
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("OPTIMIZATION PROFILE REPORT")
        report.append("=" * 60)
        report.append(f"Original Gates: {result.original_gates}")
        report.append(f"Optimized Gates: {result.optimized_gates}")
        report.append(f"Gate Reduction: {result.gate_reduction} ({result.gate_reduction_pct:.2f}%)")
        report.append("")
        report.append(f"Original Depth: {result.original_depth}")
        report.append(f"Optimized Depth: {result.optimized_depth}")
        report.append(f"Depth Reduction: {result.depth_reduction} ({result.depth_reduction_pct:.2f}%)")
        report.append("")
        report.append(f"Optimization Time: {result.optimization_time:.6f} seconds")
        report.append(f"Preserves Functionality: {'✓ Yes' if result.preserves_functionality else '✗ No'}")
        report.append("=" * 60)
        
        return "\n".join(report)


def profile_optimization(original: UnifiedCircuit,
                        optimized: UnifiedCircuit,
                        verify_fn: Optional[Callable] = None) -> OptimizationResult:
    """
    Convenience function to profile an optimization.
    
    Args:
        original: Original circuit
        optimized: Optimized circuit
        verify_fn: Optional verification function
        
    Returns:
        OptimizationResult
    """
    profiler = OptimizationProfiler()
    return profiler.profile_optimization(original, optimized, verify_fn)
