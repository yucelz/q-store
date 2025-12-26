"""
Device Topology for Hardware-Aware Compilation.

Defines qubit connectivity constraints and routing for real quantum hardware.
"""

from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class DeviceTopology:
    """
    Represents the connectivity topology of a quantum device.

    Args:
        n_qubits: Number of qubits
        edges: List of (qubit1, qubit2) tuples representing two-qubit gate connectivity
        name: Optional device name

    Example:
        >>> # Linear topology: 0-1-2-3
        >>> topology = DeviceTopology(
        ...     n_qubits=4,
        ...     edges=[(0,1), (1,2), (2,3)],
        ...     name='linear_4'
        ... )
    """
    n_qubits: int
    edges: List[Tuple[int, int]]
    name: Optional[str] = None

    def __post_init__(self):
        """Build connectivity graph."""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_qubits))
        self.graph.add_edges_from(self.edges)

        # Make edges bidirectional
        self._bidirectional_edges = set()
        for q1, q2 in self.edges:
            self._bidirectional_edges.add((q1, q2))
            self._bidirectional_edges.add((q2, q1))

    def is_connected(self, qubit1: int, qubit2: int) -> bool:
        """Check if two qubits are directly connected."""
        return (qubit1, qubit2) in self._bidirectional_edges

    def get_neighbors(self, qubit: int) -> List[int]:
        """Get all qubits directly connected to given qubit."""
        return list(self.graph.neighbors(qubit))

    def shortest_path(self, qubit1: int, qubit2: int) -> List[int]:
        """
        Find shortest path between two qubits.

        Returns:
            List of qubits forming shortest path from qubit1 to qubit2
        """
        try:
            return nx.shortest_path(self.graph, qubit1, qubit2)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between qubits {qubit1} and {qubit2}")

    def distance(self, qubit1: int, qubit2: int) -> int:
        """
        Get distance (number of hops) between two qubits.

        Returns:
            Number of edges in shortest path
        """
        try:
            return nx.shortest_path_length(self.graph, qubit1, qubit2)
        except nx.NetworkXNoPath:
            return float('inf')

    def is_fully_connected(self) -> bool:
        """Check if all qubits can reach all other qubits."""
        return nx.is_connected(self.graph)

    def diameter(self) -> int:
        """
        Get diameter of topology (maximum distance between any two qubits).

        Returns:
            Maximum shortest path length, or inf if not connected
        """
        if not self.is_fully_connected():
            return float('inf')
        return nx.diameter(self.graph)

    def degree(self, qubit: int) -> int:
        """Get degree (number of connections) for a qubit."""
        return self.graph.degree(qubit)

    def to_dict(self) -> Dict:
        """Serialize topology to dictionary."""
        return {
            'n_qubits': self.n_qubits,
            'edges': self.edges,
            'name': self.name,
        }


def create_topology(
    topology_type: str,
    n_qubits: Optional[int] = None,
    **kwargs
) -> DeviceTopology:
    """
    Create standard device topologies.

    Supported topologies:
    - 'linear': Linear chain (1D)
    - 'ring': Circular ring
    - 'grid': 2D grid/lattice
    - 'heavy_hex': IBM heavy-hex lattice
    - 'all_to_all': Fully connected (IonQ, neutral atoms)
    - 'custom': Custom edges

    Args:
        topology_type: Type of topology
        n_qubits: Number of qubits (required for most topologies)
        **kwargs: Additional topology-specific parameters

    Returns:
        DeviceTopology instance

    Example:
        >>> # Create 5-qubit linear chain
        >>> topo = create_topology('linear', n_qubits=5)
        >>> # Create 3x3 grid
        >>> topo = create_topology('grid', rows=3, cols=3)
        >>> # Create all-to-all
        >>> topo = create_topology('all_to_all', n_qubits=10)
    """
    if topology_type == 'linear':
        if n_qubits is None:
            raise ValueError("linear topology requires n_qubits")
        edges = [(i, i+1) for i in range(n_qubits - 1)]
        return DeviceTopology(n_qubits=n_qubits, edges=edges, name=f'linear_{n_qubits}')

    elif topology_type == 'ring':
        if n_qubits is None:
            raise ValueError("ring topology requires n_qubits")
        edges = [(i, (i+1) % n_qubits) for i in range(n_qubits)]
        return DeviceTopology(n_qubits=n_qubits, edges=edges, name=f'ring_{n_qubits}')

    elif topology_type == 'grid':
        rows = kwargs.get('rows')
        cols = kwargs.get('cols')
        if rows is None or cols is None:
            raise ValueError("grid topology requires rows and cols")

        n_qubits = rows * cols
        edges = []

        # Horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                q1 = r * cols + c
                q2 = r * cols + (c + 1)
                edges.append((q1, q2))

        # Vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                q1 = r * cols + c
                q2 = (r + 1) * cols + c
                edges.append((q1, q2))

        return DeviceTopology(n_qubits=n_qubits, edges=edges, name=f'grid_{rows}x{cols}')

    elif topology_type == 'heavy_hex':
        # IBM heavy-hex: hexagonal lattice with alternating connectivity
        # Simplified version for demonstration
        if n_qubits is None:
            raise ValueError("heavy_hex topology requires n_qubits")

        # Create a simplified heavy-hex pattern
        edges = []
        for i in range(0, n_qubits - 1, 2):
            edges.append((i, i+1))
            if i + 2 < n_qubits:
                edges.append((i+1, i+2))

        return DeviceTopology(n_qubits=n_qubits, edges=edges, name=f'heavy_hex_{n_qubits}')

    elif topology_type == 'all_to_all':
        if n_qubits is None:
            raise ValueError("all_to_all topology requires n_qubits")

        # Fully connected graph
        edges = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
        return DeviceTopology(n_qubits=n_qubits, edges=edges, name=f'all_to_all_{n_qubits}')

    elif topology_type == 'custom':
        edges = kwargs.get('edges')
        if edges is None or n_qubits is None:
            raise ValueError("custom topology requires n_qubits and edges")

        return DeviceTopology(n_qubits=n_qubits, edges=edges, name='custom')

    else:
        raise ValueError(f"Unknown topology type: {topology_type}")


# Pre-defined device topologies
IBM_BROOKLYN = DeviceTopology(
    n_qubits=65,
    edges=[
        # Heavy-hex topology for IBM Eagle
        # (simplified representation)
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
        # ... (full topology would have ~200 edges)
    ],
    name='ibm_brooklyn'
)

GOOGLE_SYCAMORE = DeviceTopology(
    n_qubits=54,
    edges=[
        # 2D grid with some missing qubits
        # (simplified representation)
        (0, 1), (1, 2), (2, 3),
        (0, 5), (1, 6), (2, 7), (3, 8),
        # ... (full topology would have ~80 edges)
    ],
    name='google_sycamore'
)

RIGETTI_ASPEN = DeviceTopology(
    n_qubits=32,
    edges=[
        # Octagonal topology
        # (simplified representation)
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 7), (4, 5), (5, 6), (6, 7),
        # ... (full topology would have ~40 edges)
    ],
    name='rigetti_aspen'
)
