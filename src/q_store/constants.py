"""
Constants and default values for Q-Store quantum database.

This module centralizes magic numbers and configuration defaults
to improve maintainability and reduce duplication.
"""

# ============================================================================
# Quantum Backend Constants
# ============================================================================

# Default coherence time in milliseconds
DEFAULT_COHERENCE_TIME_MS = 1000.0

# Maximum quantum states to maintain simultaneously
DEFAULT_MAX_QUANTUM_STATES = 1000

# Default number of measurement shots
DEFAULT_SHOTS = 1000

# Circuit execution timeout in seconds
DEFAULT_CIRCUIT_TIMEOUT_SECONDS = 120


# ============================================================================
# Connection Pool Constants
# ============================================================================

# Connection pool sizes
DEFAULT_MAX_CONNECTIONS = 50
DEFAULT_MIN_CONNECTIONS = 10

# Timeouts in seconds
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30
DEFAULT_IDLE_TIMEOUT_SECONDS = 300

# Decoherence check interval in seconds
DEFAULT_DECOHERENCE_CHECK_INTERVAL_SECONDS = 60


# ============================================================================
# Retry and Resilience Constants
# ============================================================================

# Maximum retry attempts for failed operations
DEFAULT_MAX_RETRIES = 3

# Base backoff time for exponential backoff (seconds)
DEFAULT_RETRY_BACKOFF_BASE_SECONDS = 1.0

# Cache time-to-live in seconds
DEFAULT_RESULT_CACHE_TTL_SECONDS = 300


# ============================================================================
# Training Constants
# ============================================================================

# Default training hyperparameters
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# Default model architecture
DEFAULT_N_QUBITS = 10
DEFAULT_CIRCUIT_DEPTH = 4
DEFAULT_ENTANGLEMENT = "linear"

# Optimizer defaults
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0

# Gradient computation
DEFAULT_GRADIENT_CLIP_VALUE = 1.0

# Checkpointing
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_LOG_INTERVAL = 10


# ============================================================================
# Performance and Caching Constants
# ============================================================================

# Classical candidate pool size
DEFAULT_CLASSICAL_CANDIDATE_POOL = 1000

# Quantum batch size
DEFAULT_QUANTUM_BATCH_SIZE = 50

# Circuit cache size
DEFAULT_CIRCUIT_CACHE_SIZE = 500

# Cache sizes for v3.3+ features
DEFAULT_CACHE_SIZE = 1000
DEFAULT_MAX_TEMPLATES = 100
DEFAULT_MAX_BOUND_CIRCUITS = 500

# Parallel execution limits
DEFAULT_MAX_CONCURRENT_CIRCUITS = 5
DEFAULT_MAX_PARALLEL_CIRCUITS = 50

# Batch timeout in seconds
DEFAULT_BATCH_TIMEOUT_SECONDS = 60.0


# ============================================================================
# SPSA Optimizer Constants (v3.3+)
# ============================================================================

# SPSA perturbation parameter
DEFAULT_SPSA_C_INITIAL = 0.1

# SPSA step size parameter
DEFAULT_SPSA_A_INITIAL = 0.01

# Gradient subsampling size
DEFAULT_GRADIENT_SUBSAMPLE_SIZE = 5


# ============================================================================
# Database Configuration Constants
# ============================================================================

# Pinecone defaults
DEFAULT_PINECONE_INDEX_NAME = "quantum-vectors"
DEFAULT_PINECONE_DIMENSION = 768
DEFAULT_PINECONE_METRIC = "cosine"
DEFAULT_PINECONE_ENVIRONMENT = "us-east-1"

# Quantum backend defaults
DEFAULT_QUANTUM_SDK = "mock"
DEFAULT_QUANTUM_TARGET = "simulator"


# ============================================================================
# IonQ Backend Constants
# ============================================================================

# IonQ target defaults
IONQ_SIMULATOR_TARGET = "simulator"
IONQ_QPU_ARIA_TARGET = "qpu.aria-1"
IONQ_QPU_FORTE_TARGET = "qpu.forte-1"

# IonQ qubit limits
IONQ_SIMULATOR_MAX_QUBITS = 29
IONQ_ARIA_MAX_QUBITS = 25
IONQ_FORTE_MAX_QUBITS = 32
IONQ_DEFAULT_MAX_QUBITS = 11

# IonQ shot limits
IONQ_MAX_SHOTS = 10000


# ============================================================================
# Tunneling Constants
# ============================================================================

# Default barrier threshold for quantum tunneling
DEFAULT_BARRIER_THRESHOLD = 0.8

# Default top-k results for tunneling search
DEFAULT_TUNNELING_TOP_K = 10


# ============================================================================
# Logging and Monitoring Constants
# ============================================================================

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Performance percentiles for monitoring
PERFORMANCE_P95 = 0.95
PERFORMANCE_P99 = 0.99


# ============================================================================
# File and Path Constants
# ============================================================================

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

# Default log directory
DEFAULT_LOG_DIR = "./logs"
