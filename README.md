# Q-Store: Quantum-Native Database v2.0

A database architecture that leverages quantum mechanical properties—superposition, entanglement, decoherence, and tunneling—as **core features** for exponential performance advantages in vector similarity search, relationship management, and pattern discovery.

<a href="http://www.q-store.tech" target="_blank">
  <strong>Q-STORE website </strong>
</a>

## 🆕 What's New in v2.0

- **Production-Ready Architecture**: Async/await, connection pooling, robust error handling
- **Enhanced Configuration**: Comprehensive DatabaseConfig with fine-grained control
- **Performance Monitoring**: Built-in metrics, caching, and performance tracking
- **Pinecone Integration**: Native support for Pinecone vector database
- **Lifecycle Management**: Automatic state management with decoherence loops
- **Type Safety**: Full type hints and dataclass-based configuration
- **Testing Suite**: Comprehensive unit, integration, and performance tests

## Overview

Q-Store provides a hybrid classical-quantum database architecture that:
- **Stores data in quantum superposition** for context-aware retrieval
- **Uses entanglement** for automatic relationship synchronization
- **Applies decoherence** as adaptive time-to-live (TTL)
- **Leverages quantum tunneling** for global pattern discovery
- **Integrates with IonQ quantum hardware** via Cirq
- **Scales with Pinecone** for classical vector storage

## Key Features

### 🌌 Quantum Superposition
Store vectors in superposition of multiple contexts simultaneously. Measurement collapses to the most relevant context for your query.

```python
await db.insert(
    id='doc_1',
    vector=embedding,
    contexts=[
        ('technical_query', 0.6),
        ('general_query', 0.3),
        ('historical_query', 0.1)
    ],
    coherence_time=5000.0  # ms
)
```

### 🔗 Quantum Entanglement
Create entangled groups where updates propagate automatically via quantum correlation. No cache invalidation needed.

```python
db.create_entangled_group(
    group_id='related_docs',
    entity_ids=['doc_1', 'doc_2', 'doc_3'],
    correlation_strength=0.85
)
```

### ⏱️ Adaptive Decoherence
Physics-based relevance decay. Old data naturally fades without explicit TTL management.

### ⏱️ Adaptive Decoherence
Physics-based relevance decay. Old data naturally fades without explicit TTL management.

```python
await db.insert(
    id='hot_data',
    vector=embedding,
    coherence_time=1000  # ms - stays relevant
)
```

### 🌀 Quantum Tunneling
Escape local optima to find globally optimal patterns that classical methods miss.

```python
results = await db.query(
    vector=query_embedding,
    enable_tunneling=True,  # Find distant patterns
    mode=QueryMode.EXPLORATORY,
    top_k=10
)
```

## Installation

### Quick Start (5 minutes)

**New users:** See [QUICKSTART.md](QUICKSTART.md) for a step-by-step beginner guide.

### Prerequisites
- Python 3.11+
- Conda package manager
- [Pinecone API key](https://www.pinecone.io/)
- [IonQ API key](https://cloud.ionq.com/settings/keys) (optional for quantum features)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yucelz/q-store.git
cd q-store
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate q-store
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install required libraries:
```bash
# Install the new Pinecone SDK (not pinecone-client)
pip install pinecone

# Verify installation
python -c "import pinecone; print('Pinecone installed successfully')"
```

5. Configure your API keys in `.env` file:

Create a `.env` file in the project root:
```bash
# Required: Pinecone for vector storage
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1

# Optional: IonQ for quantum features
IONQ_API_KEY=your_ionq_api_key
```

Get your API keys:
- **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io/) and get your API key from the dashboard
- **IonQ** (Optional): Get your API key from [cloud.ionq.com/settings/keys](https://cloud.ionq.com/settings/keys)

6. **First Test - Run the Quickstart Example:**
```bash
# Verify installation
python verify_installation.py

# Run the full quickstart demo
python examples/quantum_db_quickstart.py
```

Expected output from verification:
```
============================================================
Q-Store Installation Verification
============================================================

Checking imports...
  ✓ NumPy
  ✓ SciPy
  ✓ Cirq
  ✓ Pinecone
  ✓ Q-Store

Checking .env file...
  ✓ .env file exists
  ✓ PINECONE_API_KEY set
  ✓ PINECONE_ENVIRONMENT set

Testing basic functionality...
  ✓ DatabaseConfig created
  ✓ QuantumDatabase instantiated

============================================================
✓ All checks passed!
============================================================
```

Expected output from quickstart:
```
============================================================
QUANTUM DATABASE - INTERACTIVE DEMO
============================================================

=== Quantum Database Setup ===

Configuration:
  - Pinecone Index: quantum-demo
  - Pinecone Environment: us-east-1
  - Dimension: 768
  - Quantum Enabled: True
  - Superposition: True
  - IonQ Target: simulator

Initializing database...
INFO:q_store.quantum_database:Pinecone initialized with environment: us-east-1
INFO:q_store.quantum_database:Creating Pinecone index: quantum-demo
INFO:q_store.quantum_database:Pinecone index 'quantum-demo' created successfully
✓ Database initialized successfully

=== Example 1: Basic Operations ===
...
```

**Note:** The first run will create Pinecone indexes (`quantum-demo` and `production-index`). Subsequent runs will use existing indexes.

## Quick Start

### Using .env File (Recommended)

1. Create a `.env` file in your project root:
```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_api_key  # Optional
```

2. Run the quickstart example:
```bash
python examples/quantum_db_quickstart.py
```

The example automatically loads credentials from `.env` using `python-dotenv`.

### Basic Usage with Async/Await

```python
import asyncio
import numpy as np
from dotenv import load_dotenv
from q_store import QuantumDatabase, DatabaseConfig, QueryMode

# Load environment variables
load_dotenv()

async def main():
    # Configure database (reads from .env automatically)
    config = DatabaseConfig(
        # Pinecone settings
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
        pinecone_index_name='my-index',
        pinecone_dimension=768,
        
        # Quantum features (optional)
        ionq_api_key=os.getenv('IONQ_API_KEY'),
        ionq_target='simulator',
        enable_quantum=True,
        enable_superposition=True
    )
    
    # Initialize database with context manager
    db = QuantumDatabase(config)
    
    async with db.connect():
        # Insert vector with quantum superposition
        embedding = np.random.randn(768)
        await db.insert(
            id='item_1',
            vector=embedding,
            contexts=[('context_a', 0.7), ('context_b', 0.3)],
            metadata={'category': 'example'}
        )
        
        # Query with context-aware collapse
        results = await db.query(
            vector=embedding,
            context='context_a',
            mode=QueryMode.BALANCED,
            top_k=5
        )
        
        # Display results
        for result in results:
            print(f"ID: {result.id}, Score: {result.score:.4f}")
            print(f"Quantum Enhanced: {result.quantum_enhanced}")

# Run
asyncio.run(main())
```

### Batch Operations

```python
async with db.connect():
    # Prepare batch
    batch = [
        {
            'id': f'doc_{i}',
            'vector': np.random.rand(768),
            'contexts': [('general', 1.0)],
            'metadata': {'index': i}
        }
        for i in range(100)
    ]
    
    # Batch insert (efficient)
    await db.insert_batch(batch)
```

### Monitoring and Metrics

```python
# Get performance metrics
metrics = db.get_metrics()
print(f"Total Queries: {metrics.total_queries}")
print(f"Cache Hit Rate: {metrics.cache_hits / max(1, metrics.total_queries):.2%}")
print(f"Avg Latency: {metrics.avg_latency_ms:.2f}ms")
print(f"Active Quantum States: {metrics.active_quantum_states}")

# Get comprehensive stats
stats = db.get_stats()
print(stats)
```

## Examples

### Quickstart Guide
```bash
python examples/quantum_db_quickstart.py
```

Comprehensive guide covering:
- Basic vector operations
- Context-aware retrieval
- Batch operations
- Query modes
- Monitoring and metrics
- Production patterns

### Quickstart Guide
```bash
python examples/quantum_db_quickstart.py
```

Comprehensive guide covering:
- Basic vector operations
- Context-aware retrieval
- Batch operations
- Query modes
- Monitoring and metrics
- Production patterns

### Original Examples

#### Basic Example
```bash
python examples/basic_example.py
```

Demonstrates core quantum database features.

### Financial Services
```bash
python examples/financial_example.py
```

Portfolio correlation management and crisis pattern detection.

### ML Training
```bash
python examples/ml_training_example.py
```

Training data selection, hyperparameter optimization, and active learning.

### TinyLlama React Fine-Tuning
```bash
python examples/tinyllama_react_training.py
```

Advanced example demonstrating quantum-enhanced LLM fine-tuning:
- Intelligent training data selection with quantum superposition
- Curriculum learning (easy → hard examples)
- Hard negative mining using quantum tunneling
- Context-aware batch sampling
- Multi-context storage for training samples

See [TINYLLAMA_TRAINING_README.md](examples/TINYLLAMA_TRAINING_README.md) for detailed documentation.

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio psutil

# Run unit and integration tests
pytest tests/ -v

# Run with integration tests (requires API keys)
pytest tests/ -v --run-integration

# Run specific test categories
pytest tests/ -v -k "test_state"
pytest tests/ -v -k "test_performance"
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'q_store'**
```bash
# Solution: Install the package in development mode
pip install -e .
```

**2. ImportError: Pinecone package is required**
```bash
# Solution: Install the new Pinecone SDK (not pinecone-client)
pip uninstall -y pinecone-client
pip install pinecone
```

**3. PINECONE_API_KEY not found**
```bash
# Solution: Create a .env file in the project root
cat > .env << EOF
PINECONE_API_KEY=your_actual_api_key
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_ionq_key
EOF
```

**4. Pinecone index creation fails**
- Ensure your Pinecone account has available index quota
- Check that the environment (e.g., `us-east-1`) is valid
- Verify your API key has the necessary permissions

**5. IonQ quantum features not working**
- IonQ API key is optional - the system works without it
- Quantum features will be disabled if `IONQ_API_KEY` is not set
- Verify your IonQ API key at [cloud.ionq.com](https://cloud.ionq.com/settings/keys)

**6. Package version conflicts**
```bash
# Solution: Recreate the conda environment
conda deactivate
conda env remove -n q-store
conda env create -f environment.yml
conda activate q-store
pip install -e .
pip install pinecone
```

### Getting Help

- Check the [examples](examples/) directory for working code
- Review the [design document](quantum_db_design_v2.md) for architecture details
- Submit issues on [GitHub](https://github.com/yucelz/q-store/issues)
- Contact: yucelz@gmail.com

## Common Commands

```bash
# Installation and setup
conda activate q-store              # Activate environment
python verify_installation.py       # Verify installation
pip install -e .                    # Install package in dev mode

# Running examples
python examples/quantum_db_quickstart.py  # Run quickstart demo
python examples/basic_example.py          # Run basic example
python examples/financial_example.py      # Run financial example
python examples/ml_training_example.py    # Run ML training example
python examples/tinyllama_react_training.py  # Run TinyLlama fine-tuning

# Testing
pytest tests/ -v                    # Run all tests
pytest tests/ -v -k "test_state"    # Run specific tests

# Maintenance
conda env update -f environment.yml # Update dependencies
conda deactivate                    # Deactivate environment
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Application Layer                     │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Quantum Database API (v2.0)             │
│  • Async Operations  • Connection Pooling       │
│  • Metrics & Monitoring  • Type Safety          │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐   ┌─────▼──────────┐
│  Classical   │   │    Quantum     │
│   Backend    │◄──►   Processor    │
│              │   │                │
│  • Pinecone  │   │  • IonQ        │
│  • Vector DB │   │  • Cirq        │
│  • Caching   │   │  • State Mgr   │
└──────────────┘   └────────────────┘
```

## Configuration

### DatabaseConfig Options

```python
from q_store import DatabaseConfig

config = DatabaseConfig(
    # Pinecone configuration
    pinecone_api_key='your_key',
    pinecone_environment='us-east-1',
    pinecone_index_name='my-index',
    pinecone_dimension=768,
    pinecone_metric='cosine',
    
    # IonQ quantum backend (optional)
    ionq_api_key='your_ionq_key',
    ionq_target='simulator',  # or 'qpu.aria', 'qpu.forte'
    
    # Feature flags
    enable_quantum=True,
    enable_superposition=True,
    enable_entanglement=True,
    enable_tunneling=True,
    
    # Performance tuning
    max_quantum_states=1000,
    classical_candidate_pool=1000,
    result_cache_ttl=300,  # seconds
    
    # Connection pooling
    max_connections=50,
    connection_timeout=30,
    
    # Coherence settings
    default_coherence_time=1000.0,  # ms
    decoherence_check_interval=60,  # seconds
    
    # Monitoring
    enable_metrics=True,
    enable_tracing=True
)
```

## Project Structure

```
q-store/
├── q_store/                    # Main package (v2.0)
│   ├── __init__.py             # Public API exports
│   ├── quantum_database.py     # Main database class
│   ├── state_manager.py        # State lifecycle management
│   ├── ionq_backend.py         # IonQ quantum backend
│   ├── entanglement_registry.py # Entanglement management
│   └── tunneling_engine.py     # Quantum tunneling
├── examples/                   # Example scripts
│   ├── quantum_db_quickstart.py  # Comprehensive guide (NEW)
│   ├── basic_example.py
│   ├── financial_example.py
│   └── ml_training_example.py
├── tests/                      # Test suite (NEW)
│   └── test_quantum_database.py
├── .env                        # Environment variables (you create this)
├── environment.yml             # Conda dependencies
├── setup.py                    # Package setup
├── verify_installation.py      # Installation verification script (NEW)
├── QUICKSTART.md               # Quick start guide (NEW)
├── quantum_db_design_v2.md     # Architecture documentation v2.0
└── README.md
```

## API Reference v2.0

### QuantumDatabase

**`async def initialize()`**
Initialize database and start background tasks.

**`async def close()`**
Close database and cleanup resources.

**`async def connect()`**
Context manager for database lifecycle.

**`async def insert(id, vector, contexts=None, coherence_time=None, metadata=None)`**
Insert vector with optional quantum superposition.

**`async def insert_batch(vectors: List[Dict])`**
Batch insert for efficiency.

**`async def query(vector, context=None, mode=QueryMode.BALANCED, enable_tunneling=None, top_k=10)`**
Query database with quantum enhancements.

**`get_metrics() -> Metrics`**
Get performance metrics.

**`get_stats() -> Dict`**
Get comprehensive database statistics.

### QueryMode Enum

- `PRECISE`: High precision, narrow results
- `BALANCED`: Balanced precision and coverage  
- `EXPLORATORY`: Broad exploration, diverse results

### StateStatus Enum

- `CREATED`: Newly created state
- `ACTIVE`: Active coherent state
- `MEASURED`: State has been measured
- `DECOHERED`: State has lost coherence
- `ARCHIVED`: Archived state

## Quantum Backend

Q-Store integrates with IonQ quantum hardware using the official Cirq and cirq-ionq SDK.

**Supported Targets:**
- `simulator` - Free simulator (unlimited use)
- `qpu.aria` - 25 qubits, #AQ 25 (production)
- `qpu.forte` - 36 qubits, #AQ 36 (advanced)
- `qpu.forte.1` - 36 qubits, enterprise

**IonQ Advantages:**
- All-to-all qubit connectivity (no SWAP gates)
- High-fidelity native gates (>99.5% single-qubit, >97% two-qubit)
- Native gate set: RX, RY, RZ, XX (Mølmer-Sørensen)

## Performance

| Operation | Classical | Quantum | Speedup |
|-----------|-----------|---------|---------|
| Vector Search | O(N) | O(√N) | Quadratic |
| Pattern Discovery | O(N·M) | O(√(N·M)) | Quadratic |
| Correlation Updates | O(K²) | O(1) | K² (entanglement) |
| Storage Compression | N vectors | log₂(N) qubits | Exponential |

## Use Cases

### Financial Services
- Portfolio correlation management
- Crisis pattern detection
- Time-series prediction
- Risk analysis

### ML Model Training
- Context-aware training data selection
- Hyperparameter optimization
- Multi-task learning
- Active learning

### Recommendation Systems
- User preference modeling
- Item similarity
- Cold start problem
- Session-based recommendations

### Scientific Computing
- Molecular similarity search
- Protein structure comparison
- Drug discovery
- Materials science

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENCE) file for details.

## References

- [Quantum-Native Database Design Document v2.0](quantum_db_design_v2.md)
- [IonQ Documentation](https://docs.ionq.com/)
- [IonQ Getting Started](https://github.com/ionq-samples/getting-started)
- [Cirq Documentation](https://quantumai.google/cirq)
- [Pinecone Documentation](https://docs.pinecone.io/)

## Support

For support, submit issues in this repository or contact yucelz@gmail.com.

## Citation

If you use Q-Store in your research, please cite:

```bibtex
@software{qstore2025,
  title={Q-Store: Quantum-Native Database Architecture v2.0},
  author={Yucel Zengin},
  year={2025},
  url={https://github.com/yucelz/q-store}
}
```

## Changelog

### v2.0.0 (2025-01-11)
- **Breaking Changes**: Full async/await API
- **New**: Production-ready architecture with connection pooling
- **New**: Pinecone integration for classical vector storage
- **New**: Comprehensive monitoring and metrics
- **New**: Enhanced configuration system (DatabaseConfig)
- **New**: Type-safe API with full type hints
- **New**: Lifecycle management with context managers
- **New**: Result caching for improved performance
- **New**: Comprehensive test suite
- **Improved**: State management with background decoherence loops
- **Improved**: Error handling and retry logic
- **Improved**: Documentation and examples

### v1.0.0 (2025-01-08)
- Initial release
- Basic quantum database features
- IonQ integration
- Simple examples

---

**Note:** Q-Store v2.0 is production-ready for research and development use. For mission-critical applications, additional validation and optimization are recommended.
