"""
Advanced Quantum Circuit Compilation.

Provides:
- Topology mapping for hardware constraints
- SWAP insertion for limited connectivity
- Gate decomposition to native gate sets
- Routing optimization
"""

from .topology import (
    DeviceTopology,
    create_topology,
)

from .gate_decomposition import (
    GateDecomposer,
    decompose_to_native_gates,
)

from .circuit_compiler import (
    CircuitCompiler,
    CompilationResult,
    compile_circuit,
)

__all__ = [
    'DeviceTopology',
    'create_topology',
    'GateDecomposer',
    'decompose_to_native_gates',
    'CircuitCompiler',
    'CompilationResult',
    'compile_circuit',
]
