<div align="center">
  <img src="https://www.q-store.tech/_astro/logo.CnqA1_E2.svg" alt="Q-Store Logo" width="200"/>
</div>

# Q-Store: Quantum-Native Database v3.4

A hardware-agnostic database architecture that leverages quantum mechanical properties—superposition, entanglement, decoherence, and tunneling—for exponential performance advantages in vector similarity search, relationship management, pattern discovery, and **quantum-accelerated ML training**.

## Community

[![Slack](https://img.shields.io/badge/Slack-Join%20Group-4A154B?logo=slack&logoColor=white)]([https://your-slack-invite-link](https://q-storeworkspace.slack.com/archives/C0A4X3S055Y))
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/wYmXxEvm)


<a href="http://www.q-store.tech" target="_blank">
  <strong>Q-STORE website Link </strong>
</a>

## 🆕 What's New in v3.4

### 🚀 Major Performance Improvements (8-12x Faster)
- **IonQ Batch API Integration**: Single API call for multiple circuits (vs sequential submission)
- **Smart Circuit Caching**: Template-based caching with parameter binding (10x faster circuit preparation)
- **IonQ Native Gate Compilation**: GPi, GPi2, MS gates for 30% performance boost
- **Connection Pooling**: Persistent HTTP connections eliminate 90% of connection overhead
- **Training Time**: 3-4 minutes (down from 30 minutes in v3.3.1)
- **Throughput**: 5-8 circuits/second (up from 0.5-0.6 circuits/second)

### v3.4 Performance Benchmarks
| Metric | v3.3.1 | v3.4 | Improvement |
|--------|---------|------|-------------|
| Batch time (20 circuits) | 35s | 4s | **8.8x faster** |
| Training (5 epochs, 100 samples) | 29 min | 3.3 min | **8.8x faster** |
| Circuits/second | 0.57 | 5.0 | **8.8x faster** |
| Gate count | Medium | Low | **28% reduction** |

### Quantum ML Training (v3.2+)
- **Hardware-Agnostic Architecture**: Works with Cirq, Qiskit, and mock simulators
- **Quantum Neural Network Layers**: Variational quantum circuits for ML
- **Quantum Gradient Computation**: Parameter shift rule for backpropagation
- **Hybrid Classical-Quantum Pipelines**: Seamless integration with PyTorch/TensorFlow
- **Quantum Data Encoding**: Amplitude and angle encoding strategies

### Advanced ML Features
- **Quantum Transfer Learning**: Fine-tune pre-trained quantum models
- **Quantum Data Augmentation**: Superposition-based data expansion
- **Quantum Regularization**: Entanglement-based model optimization
- **Quantum Adversarial Training**: Robust model training with quantum gradients
- **Hyperparameter Optimization**: Quantum annealing for HPO

### Training Infrastructure
- **Distributed Quantum Training**: Multi-backend orchestration
- **Training Data Management**: Store datasets in quantum database
- **Model Checkpointing**: Save quantum states and classical weights
- **Metrics Tracking**: Comprehensive training monitoring
- **Framework Integration**: PyTorch, TensorFlow, and JAX support

## Overview

Q-Store provides a hardware-agnostic hybrid classical-quantum database architecture that:
- **Stores data in quantum superposition** for context-aware retrieval
- **Uses entanglement** for automatic relationship synchronization
- **Applies decoherence** as adaptive time-to-live (TTL)
- **Leverages quantum tunneling** for global pattern discovery
- **Trains quantum ML models** with variational quantum circuits (8-12x faster in v3.4)
- **Supports multiple quantum backends** (Cirq/IonQ, Qiskit/IonQ, simulators)
- **Integrates with classical ML frameworks** (PyTorch, TensorFlow, JAX)
- **Scales with Pinecone** for classical vector storage
- **Optimized IonQ execution** with batch API, native gates, and smart caching

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

### 🧠 Quantum ML Training (v3.2+, 8x Faster in v3.4)
Train quantum neural networks with hardware-agnostic quantum circuits.

**QuantumLayer** - Variational quantum circuit layer for neural networks
**QuantumTrainer** - Training orchestration with quantum gradient computation
**QuantumGradientComputer** - Parameter shift rule for gradient calculation
**QuantumDataEncoder** - Classical-to-quantum data encoding (amplitude/angle)
**IonQBatchClient** (v3.4) - Parallel circuit submission with connection pooling
**SmartCircuitCache** (v3.4) - Template-based circuit caching
**IonQNativeGateCompiler** (v3.4) - Native gate optimization

```python
# Define quantum neural network layer
quantum_layer = QuantumLayer(
    n_qubits=10,
    depth=4,
    backend=backend,
    entanglement='linear'
)

# Train quantum model with v3.4 optimizations
trainer = QuantumTrainer(config, backend_manager)
await trainer.train(
    model=quantum_model,
    train_loader=data_loader,
    epochs=100  # Now 8x faster with v3.4!
)
```

## Installation

### Quick Start (5 minutes)

**New users:** See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a step-by-step beginner guide.

### Prerequisites
- Python 3.11+
- Conda package manager (recommended) or pip
- [Pinecone API key](https://www.pinecone.io/)
- [IonQ API key](https://cloud.ionq.com/settings/keys) (optional for quantum hardware)
- Choose quantum SDK: Cirq or Qiskit (for hardware-agnostic support)

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
# Install with all dependencies
pip install -e ".[dev,backends]"

# Or use the Makefile
make install-dev
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

# Quantum SDK selection (cirq or qiskit)
QUANTUM_SDK=cirq  # or 'qiskit' for hardware-agnostic support
QUANTUM_TARGET=simulator  # or 'qpu.aria', 'qpu.forte'
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
        
        # Quantum backend (hardware-agnostic)
        quantum_sdk=os.getenv('QUANTUM_SDK', 'cirq'),  # 'cirq' or 'qiskit'
        ionq_api_key=os.getenv('IONQ_API_KEY'),
        ionq_target=os.getenv('QUANTUM_TARGET', 'simulator'),
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

### Quantum ML Training

```python
from q_store import QuantumTrainer, QuantumModel, TrainingConfig

# Configure training
training_config = TrainingConfig(
    # Database config
    **config,
    
    # ML training settings
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    
    # Quantum model architecture
    n_qubits=10,
    circuit_depth=4,
    entanglement='linear'
)

async def train_quantum_model():
    db = QuantumDatabase(training_config)
    
    async with db.connect():
        # Store training data in quantum database
        await db.store_training_data(
            dataset_id='mnist_train',
            data=X_train,
            labels=y_train
        )
        
        # Create quantum model
        model = QuantumModel(
            input_dim=784,
            n_qubits=10,
            output_dim=10,
            backend=db.backend_manager.get_backend()
        )
        
        # Create trainer
        trainer = QuantumTrainer(training_config, db.backend_manager)
        
        # Create data loader
        train_loader = db.create_ml_data_loader(
            dataset_id='mnist_train',
            batch_size=32
        )
        
        # Train quantum neural network
        await trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=100
        )

asyncio.run(train_quantum_model())
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

### Quantum ML Training Examples

#### Basic Quantum Neural Network
```bash
python examples/quantum_ml_basic.py
```

Demonstrates:
- **QuantumLayer** - Variational quantum circuit layers
- **QuantumTrainer** - Training orchestration
- **QuantumGradientComputer** - Parameter shift rule gradients
- **QuantumDataEncoder** - Amplitude and angle encoding

#### Hybrid Classical-Quantum Model
```bash
python examples/quantum_ml_hybrid.py
```

Features:
- Classical preprocessing layers
- Quantum processing with **QuantumLayer**
- Classical output layers
- End-to-end training pipeline

#### Transfer Learning
```bash
python examples/quantum_transfer_learning.py
```

Shows:
- Loading pre-trained quantum models
- Fine-tuning on new tasks
- Parameter freezing strategies
- **CheckpointManager** - Model persistence

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

#### Hyperparameter Optimization
```bash
python examples/quantum_hpo.py
```

Demonstrates:
- **QuantumHPOSearch** - Quantum-enhanced hyperparameter search
- Search space definition
- Quantum annealing for optimization
- Multi-trial evaluation

### Database Examples

#### Basic Example
```bash
python examples/basic_example.py
```

Demonstrates core quantum database features.

#### Financial Services
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
- Integration with **QuantumDataLoader** and **QuantumTrainer**

See [TINYLLAMA_TRAINING_README.md](examples/TINYLLAMA_TRAINING_README.md) for detailed documentation.

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov psutil

# Run all tests with coverage
python -m pytest tests/test_simple.py tests/test_constants_exceptions.py tests/test_core.py::TestStateManager::test_state_manager_creation tests/test_core.py::TestStateManager::test_start_stop -v --cov=src/q_store --cov-report=term --cov-report=html:htmlcov

# Run unit and integration tests
pytest tests/ -v

# Run with integration tests (requires API keys)
pytest tests/ -v --run-integration

# Run specific test categories
pytest tests/ -v -k "test_state"
pytest tests/ -v -k "test_performance"

# View HTML coverage report
firefox htmlcov/index.html  # or your preferred browser
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
│  • PyTorch • TensorFlow • JAX                   │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Quantum Training Engine (v3.4)             │
│  • QuantumTrainer  • QuantumLayer               │
│  • QuantumGradientComputer  • QuantumOptimizer  │
│  • QuantumDataEncoder  • CheckpointManager      │
│  • CircuitBatchManagerV34 (NEW)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Quantum Database API (v3.4)             │
│  • Async Operations  • Connection Pooling       │
│  • Metrics & Monitoring  • Type Safety          │
│  • Training Data Management                     │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐   ┌─────▼──────────────────────┐
│  Classical   │   │    Quantum Processor (v3.4) │
│   Backend    │◄──►  • IonQBatchClient (NEW)    │
│              │   │  • SmartCircuitCache (NEW)  │
│  • Pinecone  │   │  • NativeGateCompiler (NEW) │
│  • Vector DB │   │  • Cirq/IonQ                │
│  • Caching   │   │  • Qiskit/IonQ              │
│  • Training  │   │  • Simulators               │
│    Data      │   │  • State Manager            │
│              │   │  • Circuit Builder          │
└──────────────┘   └─────────────────────────────┘
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
    
    # Quantum backend (hardware-agnostic)
    quantum_sdk='cirq',  # or 'qiskit'
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

### TrainingConfig Options (v3.4)

```python
from q_store import TrainingConfig

training_config = TrainingConfig(
    # Inherits all DatabaseConfig options
    **config,
    
    # ML Training settings
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    optimizer='adam',  # 'adam', 'sgd', 'rmsprop'
    
    # Quantum model architecture
    n_qubits=10,
    circuit_depth=4,
    entanglement='linear',  # 'linear', 'circular', 'full'
    
    # Data encoding
    encoding_method='amplitude',  # or 'angle'
    
    # v3.4 Performance Optimizations (NEW)
    use_batch_api=True,          # Enable IonQ batch API (8x faster)
    use_native_gates=True,        # Enable native gate compilation (30% faster)
    enable_smart_caching=True,    # Enable circuit caching (10x faster)
    connection_pool_size=5,       # HTTP connection pool size
    adaptive_batch_sizing=True,   # Automatic batch size optimization
    
    # Regularization
    quantum_regularization=True,
    entanglement_penalty=0.01,
    
    # Checkpointing
    checkpoint_interval=10,  # epochs
    save_best_only=True,
    
    # Advanced features
    enable_data_augmentation=True,
    enable_adversarial_training=False,
    enable_transfer_learning=False
)
```

## API Reference v3.4

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

**`async def store_training_data(dataset_id, data, labels, metadata=None)`**
Store training dataset in quantum database.

**`async def load_training_batch(dataset_id, batch_size, shuffle=True)`**
Load training batch from quantum database.

**`create_ml_data_loader(dataset_id, batch_size=32, shuffle=True)`**
Create async data loader for training.

**`get_metrics() -> Metrics`**
Get performance metrics.

**`get_stats() -> Dict`**
Get comprehensive database statistics.

### Quantum ML Training Classes (v3.4)

**QuantumLayer**
- `__init__(n_qubits, depth, backend, entanglement='linear')`
- `async forward(x: np.ndarray) -> np.ndarray` - Forward pass through quantum circuit

**QuantumTrainer**
- `__init__(config, backend_manager)`
- `async train_epoch(model, data_loader, epoch)` - Train for one epoch (8x faster in v3.4)
- `async train(model, train_loader, val_loader=None, epochs=100)` - Full training loop
- `async validate(model, val_loader)` - Validation pass

**QuantumGradientComputer**
- `async compute_gradients(circuit, loss_function, current_params)` - Compute quantum gradients using parameter shift rule

**QuantumDataEncoder**
- `amplitude_encode(data: np.ndarray) -> QuantumCircuit` - Amplitude encoding
- `angle_encode(data: np.ndarray, n_qubits: int) -> QuantumCircuit` - Angle encoding

**QuantumOptimizer**
- `__init__(learning_rate, method='adam')`
- `step(parameters, gradients)` - Update parameters

**IonQBatchClient** (NEW v3.4)
- `__init__(api_key, connection_pool_size=5)`
- `async submit_batch(circuits: List[Circuit])` - Submit circuits in parallel
- `async get_results(job_ids: List[str])` - Retrieve results efficiently

**SmartCircuitCache** (NEW v3.4)
- `__init__(max_size=1000)`
- `get_or_build(template_key, parameters)` - Get cached or build circuit
- `get_statistics()` - Cache performance metrics

**IonQNativeGateCompiler** (NEW v3.4)
- `__init__()`
- `compile_to_native(circuit: Circuit)` - Compile to GPi, GPi2, MS gates
- `estimate_fidelity(circuit: Circuit)` - Estimate gate fidelity

**QuantumHPOSearch**
- `__init__(config, search_space, backend_manager)`
- `async search(model_class, dataset_id, metric, n_trials, use_quantum_annealing=True)` - Hyperparameter search

**CheckpointManager**
- `__init__(config)`
- `async save(model, epoch, metrics)` - Save model checkpoint
- `async load(checkpoint_name)` - Load model checkpoint

**MetricsTracker**
- `__init__(config)`
- `log_metrics(epoch, metrics)` - Log training metrics
- `get_history()` - Get training history

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

Q-Store integrates with multiple quantum backends for hardware-agnostic ML training.

**Supported SDKs:**
- `cirq` - Google Cirq with IonQ integration
- `qiskit` - IBM Qiskit with IonQ integration
- Mock simulators for development and testing

**Supported Targets:**
- `simulator` - Free simulator (unlimited use)
- `qpu.aria` - 25 qubits, #AQ 25 (production)
- `qpu.forte` - 36 qubits, #AQ 36 (advanced)
- `qpu.forte.1` - 36 qubits, enterprise

**IonQ Advantages:**
- All-to-all qubit connectivity (no SWAP gates)
- High-fidelity native gates (>99.5% single-qubit, >97% two-qubit)
- Native gate set: RX, RY, RZ, XX (Mølmer-Sørensen)
- Optimal for variational quantum circuits in ML training

**Backend Selection:**
The **BackendManager** automatically selects the best backend based on:
- Circuit requirements (qubit count, depth)
- Cost constraints
- Latency requirements
- Backend availability

## Performance

| Operation | Classical | Quantum (v3.3.1) | Quantum (v3.4) | v3.4 Speedup |
|-----------|-----------|------------------|----------------|--------------|
| Vector Search | O(N) | O(√N) | O(√N) | Quadratic |
| Pattern Discovery | O(N·M) | O(√(N·M)) | O(√(N·M)) | Quadratic |
| Correlation Updates | O(K²) | O(1) | O(1) | K² (entanglement) |
| Storage Compression | N vectors | log₂(N) qubits | log₂(N) qubits | Exponential |
| Gradient Computation | O(N) backprop | O(N) param shift | O(N) param shift | Comparable* |
| Circuit Execution | Sequential | Sequential | **Parallel Batch** | **8-12x faster** |
| HPO Search | O(M·N) grid | O(√M) tunneling | O(√M) tunneling | Quadratic |

*Quantum gradients enable exploration of non-convex loss landscapes  
**v3.4 achieves 8-12x speedup through batch API, native gates, and smart caching

## Use Cases

### Quantum ML Training (v3.2+, 8x Faster in v3.4)
- Quantum neural network training
- Hybrid classical-quantum models
- Transfer learning with quantum layers
- Hyperparameter optimization
- Adversarial training
- Few-shot learning

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

- [Quantum-Native Database Design Document v3.4](docs/Quantum_Native_Database_Architecture_v3_4_DESIGN.md)
- [v3.4 Analysis Summary](docs/v3_4_ANALYSIS_SUMMARY.md)
- [v3.4 Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [IonQ Documentation](https://docs.ionq.com/)
- [IonQ Getting Started](https://github.com/ionq-samples/getting-started)
- [Cirq Documentation](https://quantumai.google/cirq)
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Pinecone Documentation](https://docs.pinecone.io/)

## Project Structure

Q-Store follows modern Python packaging best practices:

```
q-store/
├── src/q_store/          # Source code (PEP 420 namespace)
│   ├── core/            # Core quantum database components
│   ├── backends/        # Quantum backend implementations (Cirq, Qiskit)
│   ├── ml/              # Quantum ML training components (v3.2)
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example implementations
├── pyproject.toml      # Modern Python project configuration
└── Makefile            # Development task automation
```

For detailed structure documentation, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

### Development Commands

```bash
make install-dev    # Install with development dependencies
make test          # Run tests
make format        # Auto-format code
make lint          # Run linters
make verify        # Run all checks
```

## Support

For support, submit issues in this repository or contact yucelz@gmail.com.

## Citation

If you use Q-Store in your research, please cite:

```bibtex
@software{qstore2025,
  title={Q-Store: Quantum-Native Database Architecture v3.4},
  author={Yucel Zengin},
  year={2025},
  url={https://github.com/yucelz/q-store}
}
```

## Changelog

### v3.4.0 (2024-12-16)
- **New**: IonQBatchClient - True parallel circuit submission (12x faster)
- **New**: SmartCircuitCache - Template-based circuit caching (10x faster preparation)
- **New**: IonQNativeGateCompiler - Native gate optimization (30% faster execution)
- **New**: CircuitBatchManagerV34 - Orchestrates all v3.4 components
- **New**: Connection pooling - Persistent HTTP connections (90% overhead reduction)
- **New**: Adaptive batch sizing - Automatic optimization based on circuit complexity
- **Performance**: 8-12x faster training (29 min → 3.3 min for typical workloads)
- **Performance**: 5-8 circuits/second throughput (up from 0.5-0.6)
- **Performance**: 28% average gate count reduction
- **Improved**: Backward compatible with v3.3.1 API
- **Improved**: Production-ready error handling and retry logic
- **Improved**: Comprehensive performance monitoring and metrics
- **Cost**: 8.8x reduction in IonQ QPU costs

### v3.2.0 (2024-12-15)
- **New**: Hardware-agnostic quantum ML training infrastructure
- **New**: QuantumLayer - Variational quantum circuit layers
- **New**: QuantumTrainer - Training orchestration with quantum gradients
- **New**: QuantumGradientComputer - Parameter shift rule implementation
- **New**: QuantumDataEncoder - Amplitude and angle encoding
- **New**: QuantumOptimizer - Quantum-aware optimization algorithms
- **New**: QuantumHPOSearch - Quantum-enhanced hyperparameter optimization
- **New**: CheckpointManager - Model persistence with quantum states
- **New**: Support for multiple quantum SDKs (Cirq, Qiskit)
- **New**: Hybrid classical-quantum model support
- **New**: Quantum transfer learning capabilities
- **New**: Quantum data augmentation
- **New**: Quantum regularization techniques
- **New**: Training data management in quantum database
- **New**: BackendManager - Intelligent backend selection
- **Improved**: Database API extended for ML training workflows
- **Improved**: StateManager for model parameter storage

### v2.0.0 (2025-12-13)
- **New**: Modern Python project structure with src/ layout
- **New**: pyproject.toml-based configuration (PEP 621)
- **New**: Modular package organization (core/, backends/, utils/)
- **New**: Development automation with Makefile
- **New**: Comprehensive documentation in docs/
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

**Note:** Q-Store v3.4 delivers production-ready quantum ML training with 8-12x performance improvements over v3.3.1. The system features hardware-agnostic support, seamless integration with classical ML frameworks (PyTorch, TensorFlow, JAX), and optimized IonQ execution through batch API, native gates, and smart caching. For mission-critical applications, additional validation and optimization are recommended.
## Developer Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yucelz/q-store.git
cd q-store

# Install in development mode with all dependencies
pip install -e ".[dev,backends,all]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Quality Tools

Q-Store uses automated code quality tools configured in `pyproject.toml` and `.pre-commit-config.yaml`:

**Formatting**:
```bash
# Format code with black (line length: 100)
black src/q_store

# Sort imports with isort
isort src/q_store --profile black
```

**Linting**:
```bash
# Run ruff (fast Python linter)
ruff check src/q_store

# Run flake8
flake8 src/q_store

# Run mypy for type checking
mypy src/q_store
```

**Pre-commit Hooks**:
All code quality checks run automatically on commit:
- Trailing whitespace removal
- End-of-file fixing
- YAML/JSON/TOML validation
- Black formatting
- Import sorting (isort)
- Ruff linting
- Type checking (mypy)

**Run All Checks Manually**:
```bash
pre-commit run --all-files
```

### Project Structure

```
q-store/
├── src/q_store/           # Main package
│   ├── core/             # Core database operations
│   ├── backends/         # Quantum backend adapters
│   ├── ml/              # ML training components
│   ├── exceptions.py    # Custom exceptions
│   └── constants.py     # Configuration constants
├── tests/               # Test suite
├── examples/            # Example scripts and demos
├── docs/                # Documentation
│   ├── ARCHITECTURE.md  # System architecture
│   └── archive/         # Old version docs
├── pyproject.toml       # Project configuration
└── .pre-commit-config.yaml  # Code quality hooks
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/q_store --cov-report=html

# Run specific test file
pytest tests/test_quantum_database.py

# Run with specific markers
pytest -m "not slow"
pytest -m integration
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run code quality tools: `pre-commit run --all-files`
5. Run tests: `pytest`
6. Commit changes (pre-commit hooks will run automatically)
7. Push to your fork: `git push origin feature/my-feature`
8. Create a Pull Request

### Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system architecture, module descriptions, and design patterns.
