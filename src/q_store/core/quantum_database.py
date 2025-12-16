"""
Quantum Database Implementation v3.1
Production-ready with Hardware Abstraction Layer

Key improvements:
- Hardware-agnostic design via backend abstraction
- Support for multiple SDKs (Cirq, Qiskit, etc.)
- Plugin architecture for easy backend addition
- Improved testability with mock backends
- Enhanced error handling and fallback mechanisms
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..backends.backend_manager import BackendManager, MockQuantumBackend

# Import abstraction layer
from ..backends.quantum_backend_interface import (
    CircuitBuilder,
    ExecutionResult,
    QuantumCircuit,
    amplitude_encode_to_circuit,
)
from .entanglement_registry import EntanglementRegistry
from .state_manager import QuantumState, StateManager, StateStatus
from .tunneling_engine import TunnelingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration and Types
# ============================================================================


class QueryMode(Enum):
    """Query execution modes"""

    PRECISE = "precise"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


@dataclass
class DatabaseConfig:
    """Comprehensive database configuration"""

    # Pinecone configuration
    pinecone_api_key: str
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "quantum-vectors"
    pinecone_dimension: int = 768
    pinecone_metric: str = "cosine"

    # Quantum backend configuration (hardware-agnostic)
    quantum_backend_name: Optional[str] = None  # Use default if None
    quantum_api_key: Optional[str] = None
    quantum_target: str = "simulator"
    quantum_sdk: str = "mock"  # 'cirq', 'qiskit', or 'mock'

    # Legacy IonQ config (for backward compatibility)
    ionq_api_key: Optional[str] = None
    ionq_target: str = "simulator"  # or qpu.aria, qpu.forte

    # Connection pooling
    max_connections: int = 50
    min_connections: int = 10
    connection_timeout: int = 30
    idle_timeout: int = 300

    # Quantum settings
    enable_quantum: bool = True
    enable_superposition: bool = True
    enable_entanglement: bool = True
    enable_tunneling: bool = True
    max_quantum_states: int = 1000

    # Performance settings
    classical_candidate_pool: int = 1000
    quantum_batch_size: int = 50
    circuit_cache_size: int = 500
    result_cache_ttl: int = 300

    # Coherence settings
    default_coherence_time: float = 1000.0  # ms
    decoherence_check_interval: int = 60  # seconds

    # Retry and timeout
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    quantum_job_timeout: int = 120  # seconds

    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True


@dataclass
class QueryResult:
    """Enhanced query result with metadata"""

    id: str
    vector: np.ndarray
    score: float
    metadata: Dict = field(default_factory=dict)
    quantum_enhanced: bool = False
    execution_time_ms: float = 0.0
    source: str = "classical"  # classical, quantum, hybrid


@dataclass
class Metrics:
    """Performance and operational metrics"""

    total_queries: int = 0
    quantum_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    active_quantum_states: int = 0
    decoherence_events: int = 0
    error_count: int = 0


# ============================================================================
# Connection Pool Manager
# ============================================================================


class ConnectionPool:
    """Manages connections to classical and quantum backends"""

    def __init__(self, config: DatabaseConfig, backend_manager: BackendManager):
        self.config = config
        self.backend_manager = backend_manager
        self._pinecone_client = None
        self._lock = asyncio.Lock()
        self._connection_count = 0

    async def initialize(self):
        """Initialize all backend connections"""
        try:
            await self._init_pinecone()
            if self.config.enable_quantum:
                await self._init_quantum_backends()
            logger.info("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    async def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            from pinecone import ServerlessSpec
            from pinecone.grpc import PineconeGRPC as Pinecone

            # Get environment from config or environment variable
            environment = self.config.pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")

            if not self.config.pinecone_api_key:
                raise ValueError(
                    "PINECONE_API_KEY is required. Please set it in your .env file or DatabaseConfig."
                )

            if not environment:
                raise ValueError(
                    "PINECONE_ENVIRONMENT is required. Please set it in your .env file or DatabaseConfig."
                )

            # Initialize Pinecone (new API)
            pc = Pinecone(api_key=self.config.pinecone_api_key)

            logger.info(f"Pinecone initialized with environment: {environment}")

            # Create index if it doesn't exist
            existing_indexes = [index.name for index in pc.list_indexes()]
            if self.config.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.pinecone_index_name}")
                pc.create_index(
                    name=self.config.pinecone_index_name,
                    dimension=self.config.pinecone_dimension,
                    metric=self.config.pinecone_metric,
                    spec=ServerlessSpec(cloud="aws", region=environment),
                )
                logger.info(
                    f"Pinecone index '{self.config.pinecone_index_name}' created successfully"
                )
            else:
                logger.info(f"Using existing Pinecone index: {self.config.pinecone_index_name}")

            self._pinecone_client = pc.Index(self.config.pinecone_index_name)
            logger.info("Pinecone connection established")

        except ImportError:
            raise ImportError("Pinecone package is required. Install it with: pip install pinecone")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def _init_quantum_backends(self):
        """Initialize quantum backends via manager"""
        try:
            # Handle backward compatibility with ionq_api_key
            api_key = self.config.quantum_api_key or self.config.ionq_api_key
            target = (
                self.config.quantum_target
                if self.config.quantum_target != "simulator"
                else self.config.ionq_target
            )
            sdk = self.config.quantum_sdk

            if sdk == "mock" or not api_key:
                # Use mock backend (default for testing)
                await self._init_mock_backend()
                logger.info("Initialized mock quantum backend")
            elif sdk == "cirq" and api_key:
                # Use Cirq adapter
                try:
                    from ..backends.cirq_ionq_adapter import CirqIonQBackend

                    cirq_backend = CirqIonQBackend(api_key=api_key, target=target)
                    self.backend_manager.register_backend(
                        "ionq_cirq",
                        cirq_backend,
                        set_as_default=True,
                        metadata={"description": f"IonQ {target} via Cirq"},
                    )
                    await cirq_backend.initialize()
                    logger.info(f"Initialized Cirq-IonQ backend: {target}")
                except ImportError:
                    logger.warning("cirq-ionq not installed, falling back to mock")
                    await self._init_mock_backend()
            elif sdk == "qiskit" and api_key:
                # Use Qiskit adapter
                try:
                    from ..backends.qiskit_ionq_adapter import QiskitIonQBackend

                    qiskit_backend = QiskitIonQBackend(api_key=api_key, target=target)
                    self.backend_manager.register_backend(
                        "ionq_qiskit",
                        qiskit_backend,
                        set_as_default=True,
                        metadata={"description": f"IonQ {target} via Qiskit"},
                    )
                    await qiskit_backend.initialize()
                    logger.info(f"Initialized Qiskit-IonQ backend: {target}")
                except ImportError:
                    logger.warning("qiskit-ionq not installed, falling back to mock")
                    await self._init_mock_backend()
            else:
                await self._init_mock_backend()

        except Exception as e:
            logger.error(f"Failed to initialize quantum backends: {e}")
            # Fall back to mock
            await self._init_mock_backend()

    async def _init_mock_backend(self):
        """Initialize mock backend as fallback"""
        mock_backend = MockQuantumBackend(name="mock_simulator", max_qubits=20, noise_level=0.0)
        self.backend_manager.register_backend(
            "mock_simulator",
            mock_backend,
            set_as_default=True,
            metadata={"description": "Mock simulator for testing"},
        )
        await mock_backend.initialize()

    def get_pinecone_client(self):
        """Get Pinecone client"""
        if self._pinecone_client is None:
            raise RuntimeError("Pinecone client not initialized")
        return self._pinecone_client

    def get_quantum_backend(self, name: Optional[str] = None):
        """Get quantum backend from manager"""
        return self.backend_manager.get_backend(name)

    async def close(self):
        """Close all connections"""
        logger.info("Closing connection pool")
        await self.backend_manager.close_all()
        self._pinecone_client = None


# ============================================================================
# Main Database Class
# ============================================================================


class QuantumDatabase:
    """Production-ready quantum database with hardware abstraction"""

    def __init__(self, config: DatabaseConfig, backend_manager: Optional[BackendManager] = None):
        self.config = config
        self.backend_manager = backend_manager or BackendManager()
        self.pool = ConnectionPool(config, self.backend_manager)
        self.state_manager = StateManager(config)
        self.entanglement_registry = EntanglementRegistry()
        self.tunneling_engine = None  # Will be initialized with quantum backend
        self.metrics = Metrics()
        self._initialized = False
        self._cache: Dict[str, Tuple[Any, float]] = {}

    async def initialize(self):
        """Initialize database and all components"""
        if self._initialized:
            return

        try:
            await self.pool.initialize()
            await self.state_manager.start()

            # Initialize tunneling engine if quantum backend available
            quantum_backend = self.pool.get_quantum_backend()
            if quantum_backend:
                self.tunneling_engine = TunnelingEngine(quantum_backend)

            self._initialized = True
            logger.info("Quantum database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close database and cleanup resources"""
        if not self._initialized:
            return

        try:
            await self.state_manager.stop()
            await self.pool.close()
            self._initialized = False
            logger.info("Quantum database closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    @asynccontextmanager
    async def connect(self):
        """Context manager for database connection"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    async def insert(
        self,
        id: str,
        vector: np.ndarray,
        contexts: Optional[List[Tuple[str, float]]] = None,
        coherence_time: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Insert vector with optional quantum features"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Insert into classical store (Pinecone)
            pinecone_client = self.pool.get_pinecone_client()

            vector_dict = {"id": id, "values": vector.tolist(), "metadata": metadata or {}}

            pinecone_client.upsert(vectors=[vector_dict])

            # Create quantum superposition if contexts provided
            if contexts and self.config.enable_superposition:
                coherence = coherence_time or self.config.default_coherence_time

                # Create vectors for each context
                context_vectors = [vector] * len(contexts)
                context_names = [c[0] for c in contexts]

                await self.state_manager.create_superposition(
                    state_id=id,
                    vectors=context_vectors,
                    contexts=context_names,
                    coherence_time=coherence,
                )

            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Inserted {id} in {duration_ms:.2f}ms")

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Failed to insert {id}: {e}")
            raise

    async def insert_batch(self, vectors: List[Dict[str, Any]]):
        """Batch insert for efficiency"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            pinecone_client = self.pool.get_pinecone_client()

            # Prepare for Pinecone
            pinecone_vectors = []
            quantum_states = []

            for vec_data in vectors:
                vector_id = vec_data["id"]
                vector = vec_data["vector"]
                metadata = vec_data.get("metadata", {})
                contexts = vec_data.get("contexts")

                pinecone_vectors.append(
                    {"id": vector_id, "values": vector.tolist(), "metadata": metadata}
                )

                if contexts and self.config.enable_superposition:
                    quantum_states.append((vector_id, vector, contexts))

            # Batch upsert to Pinecone
            pinecone_client.upsert(vectors=pinecone_vectors)

            # Create quantum states
            for state_id, vector, contexts in quantum_states:
                context_names = [c[0] for c in contexts]
                await self.state_manager.create_superposition(
                    state_id=state_id, vectors=[vector] * len(contexts), contexts=context_names
                )

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Batch inserted {len(vectors)} vectors in {duration_ms:.2f}ms")

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Batch insert failed: {e}")
            raise

    async def query(
        self,
        vector: np.ndarray,
        context: Optional[str] = None,
        mode: QueryMode = QueryMode.BALANCED,
        enable_tunneling: Optional[bool] = None,
        top_k: int = 10,
        timeout_ms: Optional[int] = None,
    ) -> List[QueryResult]:
        """Query with quantum enhancements"""
        if not self._initialized:
            await self.initialize()

        query_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            self.metrics.total_queries += 1

            # Check cache
            cache_key = self._get_cache_key(vector.tobytes(), context, mode, top_k)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.metrics.cache_hits += 1
                return cached_result

            self.metrics.cache_misses += 1

            # Classical similarity search from Pinecone
            classical_results = await self._classical_query(vector, top_k * 2)

            # Apply quantum enhancements if enabled
            if context and self.config.enable_superposition:
                self.metrics.quantum_queries += 1
                results = await self._quantum_refined_query(
                    vector, classical_results, context, mode, top_k
                )
            else:
                results = self._rank_results(classical_results, vector, top_k)

            # Post-process
            results = self._post_process_results(results, vector)

            # Cache results
            self._add_to_cache(cache_key, results)

            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self._update_latency_metrics(duration_ms)

            return results

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Query failed: {e}")
            raise

    async def _classical_query(self, vector: np.ndarray, top_k: int) -> List[Dict]:
        """Execute classical similarity search"""
        pinecone_client = self.pool.get_pinecone_client()

        results = pinecone_client.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)

        return results.get("matches", [])

    async def _quantum_refined_query(
        self,
        query_vector: np.ndarray,
        candidates: List[Dict],
        context: Optional[str],
        mode: QueryMode,
        top_k: int,
    ) -> List[QueryResult]:
        """Apply quantum refinement to candidates"""
        results = []

        for candidate in candidates[:top_k]:
            candidate_id = candidate["id"]
            candidate_vector = np.array(candidate.get("values", []))

            # Try quantum measurement
            quantum_vector = await self.state_manager.measure_with_context(candidate_id, context)

            if quantum_vector is not None:
                # Use quantum-enhanced vector
                score = self._calculate_similarity(query_vector, quantum_vector)
                results.append(
                    QueryResult(
                        id=candidate_id,
                        vector=quantum_vector,
                        score=score,
                        metadata=candidate.get("metadata", {}),
                        quantum_enhanced=True,
                        source="quantum",
                    )
                )
            else:
                # Fall back to classical
                score = candidate.get("score", 0.0)
                results.append(
                    QueryResult(
                        id=candidate_id,
                        vector=candidate_vector,
                        score=score,
                        metadata=candidate.get("metadata", {}),
                        quantum_enhanced=False,
                        source="classical",
                    )
                )

        return results

    def _rank_results(
        self, candidates: List[Dict], query_vector: np.ndarray, top_k: int
    ) -> List[QueryResult]:
        """Rank and return top-k results"""
        results = []

        for candidate in candidates:
            results.append(
                QueryResult(
                    id=candidate["id"],
                    vector=np.array(candidate.get("values", [])),
                    score=candidate.get("score", 0.0),
                    metadata=candidate.get("metadata", {}),
                    quantum_enhanced=False,
                    source="classical",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _post_process_results(
        self, results: List[QueryResult], query_vector: np.ndarray
    ) -> List[QueryResult]:
        """Post-process and deduplicate results"""
        # Remove duplicates by ID
        seen_ids = set()
        unique_results = []

        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        return unique_results

    def _calculate_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot_product / norm_product) if norm_product > 0 else 0.0

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from query parameters"""
        return str(hash(str(args)))

    def _get_from_cache(self, key: str) -> Optional[List[QueryResult]]:
        """Retrieve from cache if valid"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.config.result_cache_ttl:
                return result
            else:
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, result: List[QueryResult]):
        """Add result to cache"""
        self._cache[key] = (result, time.time())

        # Simple cache eviction
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def _update_latency_metrics(self, duration_ms: float):
        """Update latency tracking metrics"""
        # Simplified exponential moving average
        alpha = 0.1
        self.metrics.avg_latency_ms = (
            alpha * duration_ms + (1 - alpha) * self.metrics.avg_latency_ms
        )

    def get_metrics(self) -> Metrics:
        """Get current metrics"""
        self.metrics.active_quantum_states = self.state_manager.get_active_count()
        return self.metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        return {
            "total_vectors": "N/A",  # Would query Pinecone
            "quantum_states": self.state_manager.get_active_count(),
            "metrics": {
                "total_queries": self.metrics.total_queries,
                "quantum_queries": self.metrics.quantum_queries,
                "cache_hit_rate": (self.metrics.cache_hits / max(1, self.metrics.total_queries)),
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "error_count": self.metrics.error_count,
            },
            "config": {
                "quantum_enabled": self.config.enable_quantum,
                "superposition_enabled": self.config.enable_superposition,
                "tunneling_enabled": self.config.enable_tunneling,
            },
        }

    def create_entangled_group(
        self, group_id: str, entity_ids: List[str], correlation_strength: float = 0.85
    ):
        """
        Create an entangled group of entities

        Args:
            group_id: Unique identifier for the group
            entity_ids: List of entity IDs to entangle
            correlation_strength: Strength of correlation (0-1)
        """
        return self.entanglement_registry.create_entangled_group(
            group_id=group_id, entity_ids=entity_ids, correlation_strength=correlation_strength
        )

    def list_backends(self) -> List[Dict[str, Any]]:
        """List all available quantum backends"""
        return self.backend_manager.list_backends()

    def switch_backend(self, backend_name: str):
        """Switch to a different quantum backend"""
        self.backend_manager.set_default_backend(backend_name)
        # Reinitialize tunneling engine with new backend
        quantum_backend = self.pool.get_quantum_backend()
        if quantum_backend:
            self.tunneling_engine = TunnelingEngine(quantum_backend)
        logger.info(f"Switched to backend: {backend_name}")


# ============================================================================
# Mock Backend for Testing
# ============================================================================


class MockPineconeIndex:
    """Mock Pinecone index for testing without real backend"""

    def __init__(self):
        self.vectors = {}

    def upsert(self, vectors: List[Dict]):
        """Store vectors"""
        for v in vectors:
            self.vectors[v["id"]] = v

    def query(self, vector: List[float], top_k: int, include_metadata: bool = True) -> Dict:
        """Mock query"""
        # Return random matches for testing
        matches = [
            {
                "id": f"vec_{i}",
                "score": 0.9 - i * 0.05,
                "values": [0.1] * len(vector),
                "metadata": {},
            }
            for i in range(min(top_k, max(5, len(self.vectors))))
        ]
        return {"matches": matches}
