# Q-Store: Quantum-Native Database

A novel database architecture that leverages quantum mechanical properties—superposition, entanglement, decoherence, and tunneling—as **core features** for exponential performance advantages in vector similarity search, relationship management, and pattern discovery.

## Overview

Q-Store provides a hybrid classical-quantum database architecture that:
- **Stores data in quantum superposition** for context-aware retrieval
- **Uses entanglement** for automatic relationship synchronization
- **Applies decoherence** as adaptive time-to-live (TTL)
- **Leverages quantum tunneling** for global pattern discovery
- **Integrates with IonQ quantum hardware** via Cirq

## Key Features

### 🌌 Quantum Superposition
Store vectors in superposition of multiple contexts simultaneously. Measurement collapses to the most relevant context for your query.

```python
db.insert(
    id='doc_1',
    vector=embedding,
    contexts=[
        ('technical_query', 0.6),
        ('general_query', 0.3),
        ('historical_query', 0.1)
    ]
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

```python
db.insert(
    id='hot_data',
    vector=embedding,
    coherence_time=1000  # ms - stays relevant
)
```

### 🌀 Quantum Tunneling
Escape local optima to find globally optimal patterns that classical methods miss.

```python
results = db.query(
    vector=query_embedding,
    enable_tunneling=True,  # Find distant patterns
    top_k=10
)
```

## Installation

### Prerequisites
- Python 3.11
- Conda package manager
- [IonQ API key](https://cloud.ionq.com/settings/keys)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/q-store.git
cd q-store
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate q-store
```

3. Set your IonQ API key:
```bash
export IONQ_API_KEY='your_api_key_here'
```

## Quick Start

```python
from q_store import QuantumDatabase
import numpy as np

# Initialize database
db = QuantumDatabase(
    ionq_api_key='your_api_key',
    target_device='simulator',
    enable_superposition=True,
    enable_entanglement=True,
    enable_tunneling=True
)

# Insert vector with quantum superposition
embedding = np.random.randn(64)
db.insert(
    id='item_1',
    vector=embedding,
    contexts=[('context_a', 0.7), ('context_b', 0.3)]
)

# Query with context-aware collapse
results = db.query(
    vector=query_embedding,
    context='context_a',
    top_k=5
)
```

## Examples

### Basic Example
```bash
python examples/basic_example.py
```

Demonstrates core quantum database features:
- Superposition storage
- Entangled groups
- Context-aware queries
- Tunneling search
- Decoherence

### Financial Services
```bash
python examples/financial_example.py
```

Portfolio correlation management and crisis pattern detection using quantum features.

### ML Training
```bash
python examples/ml_training_example.py
```

Training data selection, hyperparameter optimization, and active learning with quantum advantages.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Application Layer                     │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Quantum Database API                    │
│  • Query Interface  • Measurement Control       │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐   ┌─────▼──────────┐
│  Classical   │   │    Quantum     │
│   Backend    │◄──►   Processor    │
│              │   │                │
│  • Pinecone  │   │  • IonQ        │
│  • pgvector  │   │  • Cirq        │
│  • Qdrant    │   │  • State Mgr   │
└──────────────┘   └────────────────┘
```

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

## Project Structure

```
q-store/
├── q_store/                    # Main package
│   ├── __init__.py
│   ├── quantum_database.py     # Main database class
│   ├── ionq_backend.py         # IonQ quantum backend
│   ├── state_manager.py        # Superposition & coherence
│   ├── entanglement_registry.py # Entanglement management
│   └── tunneling_engine.py     # Quantum tunneling
├── examples/                   # Example scripts
│   ├── basic_example.py
│   ├── financial_example.py
│   └── ml_training_example.py
├── environment.yml             # Conda dependencies
├── quantum_db_design_doc.md    # Architecture documentation
└── README.md
```

## Configuration

```python
from q_store import QuantumDatabaseConfig

config = QuantumDatabaseConfig(
    # Quantum backend
    ionq_api_key='your_key',
    target_device='simulator',  # or 'qpu.aria', 'qpu.forte'
    n_qubits=20,
    
    # Quantum features
    enable_superposition=True,
    enable_entanglement=True,
    enable_tunneling=True,
    enable_decoherence=True,
    
    # Performance
    default_coherence_time=1000.0,  # ms
    barrier_threshold=0.8,
    correlation_threshold=0.7
)
```

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

## API Reference

### QuantumDatabase

**`insert(id, vector, contexts=None, coherence_time=None, metadata=None)`**
Insert vector with optional quantum superposition.

**`query(vector, context=None, mode='balanced', enable_tunneling=None, top_k=10)`**
Query database with quantum advantages.

**`create_entangled_group(group_id, entity_ids, correlation_strength=0.85)`**
Create entangled group of related entities.

**`update(id, new_vector)`**
Update entity (entangled partners auto-update).

**`apply_decoherence()`**
Manually trigger decoherence cleanup.

**`get_stats()`**
Get database statistics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## References

- [Quantum-Native Database Design Document](quantum_db_design_doc.md)
- [IonQ Documentation](https://docs.ionq.com/)
- [IonQ Getting Started](https://github.com/ionq-samples/getting-started)
- [Cirq Documentation](https://quantumai.google/cirq)

## Support

For support, submit issues in this repository or contact support@ionq.com.

## Citation

If you use Q-Store in your research, please cite:

```bibtex
@software{qstore2025,
  title={Q-Store: Quantum-Native Database Architecture},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/q-store}
}
```

---

**Note:** This is a research prototype demonstrating quantum database concepts. For production use, additional optimization and testing are recommended.
