# Fashion MNIST + Quantum Database Integration Examples

## Overview

Two new comprehensive examples have been created that demonstrate the **complete Q-Store workflow**, combining:

1. **Quantum Neural Networks** - Training with quantum circuit layers
2. **Quantum Database** - Vector storage in Pinecone with quantum superposition
3. **Context-Aware Search** - Quantum-enhanced similarity search

## New Examples

### 1. PyTorch Version
**File:** `examples/fashion_mnist_quantum_db.py`

```bash
# Mock mode (no API keys required)
python examples/fashion_mnist_quantum_db.py

# Real backends (requires IONQ_API_KEY + PINECONE_API_KEY)
python examples/fashion_mnist_quantum_db.py --no-mock --samples 500 --epochs 3 --store-items 100
```

### 2. TensorFlow Version
**File:** `examples/fashion_mnist_quantum_db_tf.py`

```bash
# Mock mode (no API keys required for quantum, but needs Pinecone installed)
python examples/fashion_mnist_quantum_db_tf.py

# Real backends (requires IONQ_API_KEY + PINECONE_API_KEY)
python examples/fashion_mnist_quantum_db_tf.py --no-mock --samples 500 --epochs 3 --store-items 100
```

## What These Examples Demonstrate

### Phase 1: Quantum Neural Network Training
- Hybrid quantum-classical model architecture
- `QuantumLayer` for quantum circuit processing
- `AmplitudeEncoding` for quantum state preparation
- Standard neural network training with PyTorch/TensorFlow
- Embedding extraction from trained model

### Phase 2: Quantum Database Storage
- Store learned embeddings in Pinecone vector database
- **Quantum Superposition**: Each embedding stored with multiple contexts:
  - `class_*` - Specific Fashion MNIST class (T-shirt, Trouser, etc.)
  - `category_*` - Broader category (upper_body, footwear, etc.)
  - `style_*` - Style classification (casual, formal, athletic)
- Creates quantum states that exist in superposition across contexts

### Phase 3: Context-Aware Similarity Search
- **Classical Search** - Standard vector similarity (no quantum)
- **Quantum Search with Class Context** - Collapse superposition to class-specific state
- **Quantum Search with Category Context** - Collapse superposition to category-specific state
- Compare results to see quantum enhancement effects

## Key Features

### Multi-Context Quantum Superposition
Each stored embedding exists in quantum superposition across 3 contexts:
```python
contexts = [
    (f"class_{class_name}", 0.6),      # 60% weight - specific class
    (f"category_{category}", 0.3),      # 30% weight - broader category  
    (f"style_{style}", 0.1)             # 10% weight - style type
]
```

### Context-Aware Query
```python
# Search with quantum context collapse
results = await db.query(
    vector=query_embedding,
    context="class_T-shirt/top",  # Collapses superposition to this context
    mode=QueryMode.BALANCED,
    top_k=5
)
```

## Differences from Existing Examples

### Existing Examples (`tensorflow/fashion_mnist.py`, `pytorch/fashion_mnist.py`)
- ✅ Demonstrate quantum layers in neural networks
- ✅ Training with quantum circuits
- ❌ **Do NOT use QuantumDatabase**
- ❌ **No Pinecone integration**
- ❌ **No vector storage**
- ❌ **No quantum superposition features**

**Purpose:** Focus on quantum machine learning (QML) only

### New Examples (`fashion_mnist_quantum_db*.py`)
- ✅ Demonstrate quantum layers in neural networks
- ✅ Training with quantum circuits
- ✅ **Full QuantumDatabase integration**
- ✅ **Pinecone vector storage**
- ✅ **Quantum superposition across contexts**
- ✅ **Context-aware similarity search**
- ✅ **Classical vs quantum comparison**

**Purpose:** Demonstrate the **complete Q-Store system** (Train → Store → Query)

## Requirements

### For Mock Mode (Testing)
```bash
pip install tensorflow  # or torch
# Quantum layers work with mock backend
# Database needs Pinecone SDK
pip install pinecone
```

### For Real Quantum + Database
```bash
pip install tensorflow  # or torch
pip install pinecone

# Configure examples/.env:
IONQ_API_KEY=your_ionq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
```

## Command-Line Options

Both examples support:

- `--no-mock` - Use real IonQ quantum backend + Pinecone
- `--samples N` - Number of training samples (default: 500)
- `--epochs N` - Number of training epochs (default: 3)
- `--batch-size N` - Training batch size (default: 32)
- `--store-items N` - Items to store in database (default: 100)

## Example Output

```
================================================================================
Fashion MNIST with Full Quantum Database Integration (TensorFlow)
================================================================================

Configuration:
  Mode: REAL QUANTUM + PINECONE
  Training samples: 500
  Epochs: 3
  Items to store: 100

✓ Using real IonQ connection
  Backend: ionq_simulator
  Target: simulator

✓ Pinecone: us-east-1

================================================================================
TRAINING WITH QUANTUM LAYERS
================================================================================
Epoch [1/3] Train Loss: 0.6234, Train Acc: 78.50% | Val Loss: 0.5123, Val Acc: 82.00%
Epoch [2/3] Train Loss: 0.4821, Train Acc: 84.25% | Val Loss: 0.4456, Val Acc: 85.00%
Epoch [3/3] Train Loss: 0.4123, Train Acc: 87.00% | Val Loss: 0.4201, Val Acc: 86.50%

================================================================================
STORING EMBEDDINGS IN QUANTUM DATABASE
================================================================================
  Stored 20/100 items...
  Stored 40/100 items...
  Stored 60/100 items...
  Stored 80/100 items...
  Stored 100/100 items...

✓ Successfully stored 100 embeddings in Quantum Database
  - With quantum superposition across 3 contexts per item

================================================================================
QUANTUM-ENHANCED SIMILARITY SEARCH
================================================================================

Query: T-shirt/top

1. Classical Search (no quantum context):
   1. T-shirt/top (score: 0.9234, quantum: False)
   2. Shirt (score: 0.8821, quantum: False)
   3. Pullover (score: 0.8654, quantum: False)
   4. Coat (score: 0.8432, quantum: False)
   5. Dress (score: 0.8123, quantum: False)

2. Quantum Search (context: 'class_T-shirt/top'):
   1. T-shirt/top (score: 0.9456, quantum: True)    # Boosted by quantum collapse
   2. T-shirt/top (score: 0.9234, quantum: True)    # More T-shirts found
   3. T-shirt/top (score: 0.9123, quantum: True)
   4. Shirt (score: 0.8821, quantum: False)
   5. Pullover (score: 0.8654, quantum: False)

3. Quantum Search (context: 'category_upper_body'):
   1. T-shirt/top (score: 0.9345, quantum: True)
   2. Shirt (score: 0.9012, quantum: True)          # Category members boosted
   3. Pullover (score: 0.8876, quantum: True)
   4. Coat (score: 0.8654, quantum: True)
   5. Dress (score: 0.8123, quantum: False)

================================================================================
DATABASE STATISTICS
================================================================================
  Quantum states: 100
  Total queries: 15
  Quantum queries: 10
  Cache hit rate: 0.00%
  Avg latency: 45.23ms

================================================================================
✓ DEMONSTRATION COMPLETE
================================================================================
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fashion MNIST Training                        │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ Classical│→ │   Quantum   │→ │  Embedding   │→ │Classifier│ │
│  │  Layers  │  │   Layers    │  │    Layer     │  │  Head    │ │
│  └──────────┘  └─────────────┘  └──────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Extract Embeddings
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Quantum Database Storage                      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Pinecone Vector Index                       │  │
│  │  • Stores all embeddings as vectors                       │  │
│  │  • Classical similarity search                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Quantum State Manager                          │  │
│  │  • Creates superposition states                           │  │
│  │  • Manages multiple contexts per embedding                │  │
│  │  • Context-aware state collapse                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Similarity Search
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Query Results (Classical + Quantum)                 │
│  • Classical: Standard vector similarity                         │
│  • Quantum: Context-aware collapse + enhanced similarity         │
└─────────────────────────────────────────────────────────────────┘
```

## Use Cases

These examples are ideal for:

1. **Research**: Understanding quantum-enhanced vector search
2. **Prototyping**: Testing quantum database features
3. **Benchmarking**: Comparing classical vs quantum search
4. **Learning**: Complete workflow demonstration
5. **Development**: Template for building quantum-enhanced applications

## Next Steps

1. Install Pinecone SDK: `pip install pinecone`
2. Get API keys from Pinecone and IonQ
3. Configure `examples/.env` with your credentials
4. Run examples with `--no-mock` flag
5. Experiment with different contexts and query strategies
6. Compare quantum vs classical search results

## Summary

These new examples bridge the gap between:
- Quantum machine learning (existing examples)
- Quantum database operations (previously only in tests/docs)

They provide the **only end-to-end demonstration** of the complete Q-Store system in action.
