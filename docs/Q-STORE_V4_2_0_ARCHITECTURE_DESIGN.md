# Q-Store v4.2.0 Architecture Design

## Hybrid GPU-Quantum Performance Acceleration via Quantum Kernel Methods

**Version**: 4.2.0
**Date**: January 8, 2026
**Status**: Draft for Review
**Focus**: Performance Optimization via Quantum Kernel Methods + GPU Acceleration

---

## Executive Summary

Q-Store v4.2.0 introduces a **Quantum Kernel Methods (QKM) architecture with GPU acceleration** to address the critical performance gap identified in v4.1.1, where quantum training was 183-457x slower than classical GPU approaches. This version adopts a fundamentally different approach: using quantum computers as **feature map engines** (not trainable models), combined with GPU-accelerated classical training.

### Key Paradigm Shift

**v4.1.1 Approach**: Variational Quantum Circuits (VQCs) with parameter optimization on QPU
**v4.2.0 Approach**: Quantum Kernel Methods with quantum feature mapping + GPU-based classical training

### Why Quantum Kernel Methods?

| Issue | Variational Quantum Circuits (v4.1.1) | Quantum Kernels (v4.2.0) |
|-------|---------------------------------------|--------------------------|
| Training on QPU | Required | **Not required** |
| Barren plateaus | Common | **Avoided** |
| Gradient noise | High | **None** |
| Optimizer instability | Yes | **No** |
| Network latency impact | 55% overhead | **Minimal** (O(N²) kernel calls) |
| Parameter optimization | On quantum hardware | **On classical GPU** |

### Key Objectives

1. **GPU-Accelerated Preprocessing**: Classical feature extraction and data preparation on GPU
2. **Quantum Kernel Computation**: Use QPU/simulator for quantum feature mapping (O(N²) operations)
3. **GPU-Based Classical Training**: SVM, kernel ridge regression on GPU using quantum kernels
4. **Simplified Architecture**: Single GPU initially, no complex multi-GPU orchestration
5. **Performance Target**: 20-100x speedup vs v4.1.1 for small-to-medium datasets

### No Direct Hardware Dependencies

- No hardcoded TorchQuantum-Dist dependency
- Works with any quantum backend (IonQ, Qiskit, Cirq, PennyLane)
- Graceful fallback to classical kernels if QPU unavailable

---

## Current State Analysis (v4.1.1)

### Performance Benchmarks (Cats vs Dogs - 1,000 images, 5 epochs)

| Platform | Time per Epoch | Total Time | Relative Speed | Approach |
|----------|----------------|------------|----------------|----------|
| NVIDIA A100 GPU | 1.5s | 7.5s | 305x faster | Pure classical |
| **Q-Store v4.1.1** | 457s | 2,288s | Baseline | VQC training |
| Q-Store (no latency) | 204s | 1,020s | 2.2x faster | VQC (theoretical) |

### v4.1.1 Bottlenecks

1. **Network Latency (55%)**: API round-trip to IonQ cloud
2. **Parameter Optimization on QPU (30%)**: Gradient estimation, circuit execution
3. **Barren Plateaus**: Gradient vanishing in deep quantum circuits
4. **Data Serialization (15%)**: Converting circuits to IonQ format

### Why VQCs Struggle

- Training requires many quantum circuit executions with parameter updates
- Each gradient estimation needs multiple circuit evaluations
- Network latency dominates for small circuits
- Barren plateaus make optimization unstable

---

## v4.2.0 Quantum Kernel Methods Architecture

### Core Concept

Instead of training on quantum hardware, we:

1. **Use quantum circuits to compute kernels** (similarity measures between data points)
2. **Train classical models (SVM, kernel methods) on GPU** using the quantum kernel matrix
3. **Eliminate parameter optimization on QPU** - only quantum state preparation and measurement

### Quantum Kernel Definition

For two data points x and x', the quantum kernel is:

```
K(x, x') = |⟨φ(x) | φ(x')⟩|²
```

Where:
- `|φ(x)⟩ = U(x)|0⟩^n` is the quantum feature map
- `U(x)` is a data-dependent quantum circuit
- The kernel measures overlap in quantum Hilbert space

### Practical Kernel Computation (Adjoint Circuit Method)

```
K(x, x') = |⟨0| U†(x') U(x) |0⟩|²
```

**Implementation**:

1. Prepare quantum state `|φ(x)⟩` using circuit U(x)
2. Apply inverse feature map `U†(x')`
3. Measure probability of all-zero state

**Advantages**:
- Shallow circuits (no deep optimization needed)
- No ancilla qubits required
- Robust on noisy NISQ hardware
- Requires only O(N²) quantum executions (not O(N × epochs × batches))

---

## High-Level 4-Layer Hybrid Architecture

**Core Mathematical Principle**: *"Use quantum computers to construct distributions that are classically hard, but keep learning and optimization classical."*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Q-Store v4.2.0 Quantum Kernel Architecture                  │
│                          4-Layer Hybrid System                               │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────┐
                         │   Client / UI        │
                         └──────────┬───────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: GraphQL Orchestration (Optional but Recommended)                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • Job Scheduling: Async quantum kernel job management                 │ │
│  │  • Data Routing: Route data to GPU, QPU, or Vector DB                  │ │
│  │  • Caching: Intelligent kernel caching and reuse                       │ │
│  │  • Cost Tracking: Monitor quantum circuit costs                        │ │
│  │  • Backend Abstraction: Unified API for IonQ, Quantinuum, etc.         │ │
│  │                                                                          │ │
│  │  GraphQL Schema:                                                        │ │
│  │  - submitKernelJob(backend, datasetId, shots)                          │ │
│  │  - kernelValue(x1, x2) → cached lookup                                 │ │
│  │  - jobStatus(jobId) → async tracking                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  LAYER 1: GPU        │   │  LAYER 2: QPU        │   │  LAYER 3: Vector DB  │
│  Classical Training  │   │  Quantum Kernels     │   │  Kernel Storage      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: GPU - Classical Optimization (100-500x speedup vs CPU)             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Mathematics:                                                           │ │
│  │                                                                          │ │
│  │  min_α  ||Kα − y||² + λ||α||²    (Convex optimization)                 │ │
│  │                                                                          │ │
│  │  What happens here:                                                     │ │
│  │  • Data preprocessing (normalization, augmentation)                     │ │
│  │  • Feature extraction (optional CNN)                                    │ │
│  │  • Convex optimization (SVM dual, kernel ridge regression)             │ │
│  │  • Backprop-free: No quantum gradients needed                          │ │
│  │  • Fast linear algebra on GPU (cuBLAS, cuSOLVER)                       │ │
│  │                                                                          │ │
│  │  Supported Models:                                                      │ │
│  │  • Kernel SVM (cuML for GPU acceleration)                              │ │
│  │  • Kernel Ridge Regression (closed-form solution)                      │ │
│  │  • Gaussian Process Models                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: QPU - Quantum Kernel Evaluation (Feature Space Engine)             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Mathematics:                                                           │ │
│  │                                                                          │ │
│  │  |φ(x)⟩ = U(x)|0⟩^n         (Quantum feature map)                      │ │
│  │  K(x, x') = |⟨φ(x)|φ(x')⟩|²  (Quantum kernel)                          │ │
│  │                                                                          │ │
│  │  Implementation (Adjoint Circuit Method):                               │ │
│  │  K(x, x') = |⟨0|U†(x')U(x)|0⟩|²                                        │ │
│  │                                                                          │ │
│  │  Quantum Feature Maps:                                                  │ │
│  │  • Angle Encoding: RY(x_i), RZ(x_i) + entanglement                     │ │
│  │  • IQP Encoding: Diagonal gates + ZZ interactions                      │ │
│  │  • Data Re-uploading: Multiple encoding layers                         │ │
│  │                                                                          │ │
│  │  Backend Options:                                                       │ │
│  │  • IonQ Simulator (free, testing)                                      │ │
│  │  • IonQ Aria QPU (25 qubits, $0.30/circuit)                            │ │
│  │  • Quantinuum (high fidelity, fewer shots)                             │ │
│  │  • Local simulators (Qiskit, PennyLane)                                │ │
│  │                                                                          │ │
│  │  Complexity: O(N²) circuit executions                                  │ │
│  │  Cost: One-time quantum computation, then cached                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Vector Database - Kernel & Support Vector Storage (NEW)            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Purpose: Scalable kernel matrix storage and fast similarity search    │ │
│  │                                                                          │ │
│  │  What's stored:                                                         │ │
│  │  • Kernel matrix rows: K[i,:] for each sample                          │ │
│  │  • Support vectors: x_sv for inference                                 │ │
│  │  • Quantum embeddings: |φ(x)⟩ representations                          │ │
│  │  • Metadata: Dataset info, backend used, shot counts                   │ │
│  │                                                                          │ │
│  │  Supported Vector DBs:                                                  │ │
│  │  • FAISS (Facebook AI Similarity Search) - GPU-accelerated             │ │
│  │  • Milvus - Distributed vector database                                │ │
│  │  • Pinecone - Managed vector DB (cloud)                                │ │
│  │  • Qdrant - Open-source alternative                                    │ │
│  │                                                                          │ │
│  │  Benefits:                                                              │ │
│  │  • Scalability: Handle N > 1,000 samples                               │ │
│  │  • Fast Lookup: O(log N) similarity search for inference               │ │
│  │  • Persistent Cache: Reuse kernels across experiments                  │ │
│  │  • Distributed: Scale across multiple machines                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Hybrid Data Flow: Training                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. GPU: Preprocess data → normalized features                         │ │
│  │  2. QPU: Compute K[i,j] = quantum_kernel(x_i, x_j) for all pairs      │ │
│  │  3. Vector DB: Store kernel matrix K (N×N) with indexing               │ │
│  │  4. GPU: Load K from vector DB → solve convex optimization            │ │
│  │  5. Vector DB: Store support vectors α_i, x_sv                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Hybrid Data Flow: Inference                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Vector DB: Retrieve support vectors x_sv                            │ │
│  │  2. QPU: Compute k_new[i] = quantum_kernel(x_new, x_sv[i])            │ │
│  │  3. Vector DB: Cache k_new for future queries                          │ │
│  │  4. GPU: Prediction = Σ α_i * k_new[i] (fast dot product)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Quantum for Representation, Classical for Training**: QPU computes kernels (exponential feature space), GPU optimizes (convex problem)
2. **One-time Kernel Computation**: O(N²) quantum calls, then cached in vector DB
3. **No Quantum Gradients**: All optimization is classical and convex (no barren plateaus)
4. **Scalable Storage**: Vector DB enables N > 1,000 samples
5. **Orchestration Layer**: GraphQL provides unified API for hybrid workflows

---

## Component Details

### 1. Vector Database Integration (NEW - LAYER 3)

**File**: `q_store/storage/vector_db.py` (NEW)

#### Purpose

Scalable storage for quantum kernel matrices and support vectors, enabling:
- N > 1,000 samples (beyond in-memory limits)
- Fast similarity search (O(log N))
- Persistent kernel caching across experiments
- Distributed storage for production deployments

#### Implementation

```python
import numpy as np
from typing import Optional, List, Dict, Any
import faiss

class QuantumKernelStore:
    """
    Vector database for quantum kernel storage using FAISS.

    Stores:
    - Kernel matrix rows
    - Support vectors
    - Metadata (backend, shots, timestamp)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = 'IVF',
        use_gpu: bool = True,
        cache_dir: str = './kernel_cache'
    ):
        """
        Initialize vector database.

        Args:
            dimension: Feature dimension (matches kernel size)
            index_type: 'Flat' (exact), 'IVF' (fast), 'HNSW' (balanced)
            use_gpu: Use GPU-accelerated FAISS
            cache_dir: Directory for persistent storage
        """
        self.dimension = dimension
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu

        # Create FAISS index
        if index_type == 'Flat':
            # Exact search (brute force)
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IVF':
            # Inverted file index (fast approximate search)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        elif index_type == 'HNSW':
            # Hierarchical Navigable Small World (balanced)
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors

        # Move to GPU if available
        if use_gpu and faiss.get_num_gpus() > 0:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

        self.index = index
        self.metadata_store = {}  # Maps sample_id → metadata

    def store_kernel_row(
        self,
        sample_id: str,
        kernel_row: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Store a single kernel matrix row.

        Args:
            sample_id: Unique identifier for sample
            kernel_row: K[i,:] - kernel values for sample i
            metadata: Additional info (backend, shots, etc.)
        """
        # Add to FAISS index
        kernel_row_2d = kernel_row.reshape(1, -1).astype('float32')
        self.index.add(kernel_row_2d)

        # Store metadata
        self.metadata_store[sample_id] = metadata or {}

    def store_kernel_matrix(
        self,
        K: np.ndarray,
        sample_ids: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Store full kernel matrix.

        Args:
            K: Kernel matrix (N×N)
            sample_ids: List of sample identifiers
            metadata: Shared metadata for all rows
        """
        # Store each row
        for i, sample_id in enumerate(sample_ids):
            self.store_kernel_row(
                sample_id,
                K[i, :],
                {**(metadata or {}), 'row_index': i}
            )

    def retrieve_kernel_row(
        self,
        sample_id: str
    ) -> Optional[np.ndarray]:
        """Retrieve kernel row by sample ID."""
        if sample_id not in self.metadata_store:
            return None

        row_idx = self.metadata_store[sample_id]['row_index']
        # Reconstruct from FAISS
        kernel_row = self.index.reconstruct(row_idx)
        return kernel_row

    def similarity_search(
        self,
        query_kernel: np.ndarray,
        k: int = 5
    ) -> List[tuple]:
        """
        Find k most similar kernel rows.

        Args:
            query_kernel: Query kernel vector
            k: Number of nearest neighbors

        Returns:
            List of (distance, sample_id) tuples
        """
        query_2d = query_kernel.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_2d, k)

        # Map indices to sample IDs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Find sample_id from metadata
            for sample_id, meta in self.metadata_store.items():
                if meta.get('row_index') == idx:
                    results.append((dist, sample_id))
                    break

        return results

    def save(self, path: Optional[str] = None):
        """Save index and metadata to disk."""
        save_path = path or f"{self.cache_dir}/kernel_index.faiss"
        faiss.write_index(self.index, save_path)

        # Save metadata separately
        import pickle
        with open(f"{self.cache_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)

    def load(self, path: Optional[str] = None):
        """Load index and metadata from disk."""
        load_path = path or f"{self.cache_dir}/kernel_index.faiss"
        self.index = faiss.read_index(load_path)

        # Load metadata
        import pickle
        with open(f"{self.cache_dir}/metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)


class SupportVectorStore:
    """
    Store trained SVM support vectors in vector DB.

    Enables fast inference via similarity search.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.support_vectors = []
        self.alpha_coefficients = []
        self.labels = []

    def store_support_vectors(
        self,
        x_sv: np.ndarray,
        alpha: np.ndarray,
        y_sv: np.ndarray
    ):
        """
        Store support vectors from trained SVM.

        Args:
            x_sv: Support vector features
            alpha: Dual coefficients
            y_sv: Support vector labels
        """
        self.support_vectors = x_sv
        self.alpha_coefficients = alpha
        self.labels = y_sv

    def get_support_vectors(self) -> tuple:
        """Retrieve all support vectors."""
        return (
            self.support_vectors,
            self.alpha_coefficients,
            self.labels
        )
```

**Benefits**:
- **Scalability**: Handle datasets with N > 1,000 samples
- **Performance**: O(log N) similarity search vs O(N) linear scan
- **Persistence**: Reuse quantum kernels across experiments
- **GPU-Accelerated**: FAISS supports GPU for faster search

---

### 2. GraphQL Orchestration Layer (NEW - LAYER 4 - Optional)

**File**: `q_store/orchestration/graphql_api.py` (NEW)

#### Purpose

Unified API for managing hybrid quantum-classical workflows, providing:
- Async quantum job management
- Kernel caching and reuse
- Backend abstraction (IonQ, Quantinuum, simulators)
- Cost tracking and budget management

#### GraphQL Schema

```graphql
type QuantumKernelJob {
  id: ID!
  status: String!                # QUEUED, RUNNING, COMPLETED, FAILED
  backend: String!               # ionq.simulator, ionq.aria, quantinuum
  datasetId: ID!
  shots: Int!
  progressPercent: Float
  costEstimateUSD: Float
  actualCostUSD: Float
  createdAt: String
  completedAt: String
}

type KernelValue {
  x1: ID!
  x2: ID!
  value: Float!
  backend: String
  shots: Int
  cachedAt: String
}

type SupportVector {
  id: ID!
  features: [Float!]!
  alpha: Float!
  label: Int!
}

type Query {
  # Retrieve cached kernel value
  kernelValue(x1: ID!, x2: ID!): KernelValue

  # Check job status
  jobStatus(jobId: ID!): QuantumKernelJob

  # List all jobs
  listJobs(status: String, limit: Int): [QuantumKernelJob!]!

  # Get support vectors for trained model
  supportVectors(modelId: ID!): [SupportVector!]!

  # Estimate cost for kernel computation
  estimateCost(datasetId: ID!, backend: String!, shots: Int!): Float
}

type Mutation {
  # Submit quantum kernel computation job
  submitKernelJob(
    backend: String!
    datasetId: ID!
    shots: Int!
    featureMap: String          # angle, iqp, data_reuploading
  ): QuantumKernelJob!

  # Cancel running job
  cancelJob(jobId: ID!): Boolean

  # Clear kernel cache
  clearCache(datasetId: ID): Boolean
}

type Subscription {
  # Real-time job progress updates
  jobProgress(jobId: ID!): QuantumKernelJob!
}
```

#### Implementation (Simplified)

```python
import graphene
from typing import Optional

class QuantumKernelJobType(graphene.ObjectType):
    id = graphene.ID()
    status = graphene.String()
    backend = graphene.String()
    dataset_id = graphene.ID()
    shots = graphene.Int()
    progress_percent = graphene.Float()
    cost_estimate_usd = graphene.Float()
    actual_cost_usd = graphene.Float()

class KernelValueType(graphene.ObjectType):
    x1 = graphene.ID()
    x2 = graphene.ID()
    value = graphene.Float()
    backend = graphene.String()
    shots = graphene.Int()

class Query(graphene.ObjectType):
    kernel_value = graphene.Field(
        KernelValueType,
        x1=graphene.ID(required=True),
        x2=graphene.ID(required=True)
    )

    job_status = graphene.Field(
        QuantumKernelJobType,
        job_id=graphene.ID(required=True)
    )

    def resolve_kernel_value(self, info, x1, x2):
        # Check cache (vector DB)
        cached = kernel_store.retrieve_kernel_value(x1, x2)
        if cached:
            return cached

        # Not cached - return None or trigger computation
        return None

    def resolve_job_status(self, info, job_id):
        # Query job status from backend
        job = job_manager.get_job(job_id)
        return job

class Mutation(graphene.ObjectType):
    submit_kernel_job = graphene.Field(
        QuantumKernelJobType,
        backend=graphene.String(required=True),
        dataset_id=graphene.ID(required=True),
        shots=graphene.Int(required=True)
    )

    def resolve_submit_kernel_job(self, info, backend, dataset_id, shots):
        # Create async job
        job = QuantumKernelJobManager().submit(
            backend=backend,
            dataset_id=dataset_id,
            shots=shots
        )
        return job

schema = graphene.Schema(query=Query, mutation=Mutation)
```

**Benefits**:
- **Unified API**: Single interface for all backends
- **Async Management**: Non-blocking quantum job execution
- **Caching**: Automatic kernel reuse across experiments
- **Cost Tracking**: Real-time cost monitoring

---

### 3. GPU-Accelerated Data Pipeline

**File**: `q_store/gpu/data_pipeline.py` (NEW)

```python
import torch
from torch.utils.data import DataLoader
import numpy as np

class GPUDataPipeline:
    """
    GPU-accelerated data loading and preprocessing.

    Handles:
    - Data loading on GPU
    - Normalization and preprocessing
    - Optional feature extraction via CNN
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        device: str = 'cuda:0',
        feature_extractor: torch.nn.Module = None
    ):
        """
        Initialize GPU data pipeline.

        Args:
            dataset: Dataset from q_store.data
            batch_size: Batch size
            device: GPU device
            feature_extractor: Optional CNN for dimension reduction
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor

        # Convert to PyTorch tensors on GPU
        self.x_train = torch.from_numpy(dataset.x_train).float().to(self.device)
        self.y_train = torch.from_numpy(dataset.y_train).long().to(self.device)

        if dataset.x_val is not None:
            self.x_val = torch.from_numpy(dataset.x_val).float().to(self.device)
            self.y_val = torch.from_numpy(dataset.y_val).long().to(self.device)

        if dataset.x_test is not None:
            self.x_test = torch.from_numpy(dataset.x_test).float().to(self.device)
            self.y_test = torch.from_numpy(dataset.y_test).long().to(self.device)

    def preprocess(self, normalize: bool = True):
        """GPU-accelerated preprocessing."""
        if normalize:
            # Normalize on GPU (100x faster than CPU)
            mean = self.x_train.mean(dim=0, keepdim=True)
            std = self.x_train.std(dim=0, keepdim=True) + 1e-8

            self.x_train = (self.x_train - mean) / std
            if hasattr(self, 'x_val'):
                self.x_val = (self.x_val - mean) / std
            if hasattr(self, 'x_test'):
                self.x_test = (self.x_test - mean) / std

    def extract_features(self):
        """
        Optional: Extract features using GPU CNN.
        Reduces dimensionality for quantum encoding.
        """
        if self.feature_extractor is None:
            return

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        with torch.no_grad():
            self.x_train = self.feature_extractor(self.x_train)
            if hasattr(self, 'x_val'):
                self.x_val = self.feature_extractor(self.x_val)
            if hasattr(self, 'x_test'):
                self.x_test = self.feature_extractor(self.x_test)

    def get_data(self, split: str = 'train'):
        """Get preprocessed data."""
        if split == 'train':
            return self.x_train, self.y_train
        elif split == 'val':
            return self.x_val, self.y_val
        elif split == 'test':
            return self.x_test, self.y_test
```

**Features**:
- All data operations on GPU (10-50x faster than CPU)
- Optional CNN for dimension reduction (28×28 → 8 features)
- Zero-copy data transfer
- Batch processing support

---

### 2. Quantum Kernel Engine

**File**: `q_store/kernels/quantum_kernel.py` (NEW)

```python
import numpy as np
from typing import Callable, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import torch

class QuantumKernel:
    """
    Quantum kernel computation using feature maps.

    Implements kernel K(x, x') = |⟨φ(x)|φ(x')⟩|²
    using adjoint circuit method: K(x, x') = |⟨0|U†(x')U(x)|0⟩|²
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map: str = 'angle',
        backend: str = 'ionq.simulator',
        backend_config: Optional[Dict] = None,
        use_gpu: bool = True
    ):
        """
        Initialize quantum kernel.

        Args:
            n_qubits: Number of qubits
            feature_map: 'angle', 'iqp', 'data_reuploading'
            backend: Quantum backend identifier
            backend_config: Backend configuration
            use_gpu: Use GPU for classical operations
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map
        self.backend_name = backend
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Initialize quantum backend
        from q_store.backends import get_backend
        self.backend = get_backend(backend, backend_config or {})

        # Select feature map
        self.feature_map = self._get_feature_map(feature_map)

    def _get_feature_map(self, map_type: str) -> Callable:
        """Get quantum feature map circuit builder."""
        if map_type == 'angle':
            return self._angle_encoding_map
        elif map_type == 'iqp':
            return self._iqp_map
        elif map_type == 'data_reuploading':
            return self._data_reuploading_map
        else:
            raise ValueError(f"Unknown feature map: {map_type}")

    def _angle_encoding_map(self, x: np.ndarray) -> 'QuantumCircuit':
        """
        Angle encoding feature map.

        Maps features to rotation angles:
        U(x) = ∏_i R_Y(x_i) R_Z(x_i)
        """
        from q_store.circuits import QuantumCircuit

        circuit = QuantumCircuit(self.n_qubits)

        # Hadamard layer for superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # Angle encoding (repeat features if needed)
        features = np.resize(x, self.n_qubits)

        for i in range(self.n_qubits):
            circuit.ry(features[i], i)
            circuit.rz(features[i], i)

        # Entangling layer
        for i in range(self.n_qubits - 1):
            circuit.cnot(i, i + 1)

        return circuit

    def _iqp_map(self, x: np.ndarray) -> 'QuantumCircuit':
        """
        IQP (Instantaneous Quantum Polynomial) encoding.

        Theoretical support for quantum advantage.
        """
        from q_store.circuits import QuantumCircuit

        circuit = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            circuit.h(i)

        # Z rotations (diagonal in computational basis)
        features = np.resize(x, self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rz(features[i], i)

        # ZZ interactions
        for i in range(self.n_qubits - 1):
            for j in range(i + 1, self.n_qubits):
                circuit.rzz(features[i] * features[j], i, j)

        # Final Hadamard
        for i in range(self.n_qubits):
            circuit.h(i)

        return circuit

    def _data_reuploading_map(self, x: np.ndarray) -> 'QuantumCircuit':
        """Data re-uploading: encode data multiple times in circuit."""
        from q_store.circuits import QuantumCircuit

        circuit = QuantumCircuit(self.n_qubits)
        features = np.resize(x, self.n_qubits)

        # 3 layers of data encoding
        for layer in range(3):
            for i in range(self.n_qubits):
                circuit.ry(features[i], i)
                circuit.rz(features[i], i)

            # Entanglement
            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute single kernel element K(x1, x2).

        Uses adjoint circuit method:
        K(x1, x2) = |⟨0|U†(x2)U(x1)|0⟩|²
        """
        # Build circuit: U(x1) followed by U†(x2)
        circuit = self.feature_map(x1)
        circuit_x2 = self.feature_map(x2)

        # Append adjoint (inverse) of U(x2)
        circuit.append(circuit_x2.inverse())

        # Measure probability of all-zero state
        result = self.backend.execute(circuit, shots=1024)

        # Get probability of |00...0⟩ state
        zero_state = '0' * self.n_qubits
        kernel_value = result.get_counts().get(zero_state, 0) / 1024

        return kernel_value

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        parallel: bool = True,
        max_workers: int = 10
    ) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = K(X[i], Y[j]).

        Args:
            X: Training data (n_samples, n_features)
            Y: Test data (m_samples, n_features). If None, use X (training kernel)
            parallel: Execute quantum circuits in parallel
            max_workers: Number of parallel workers

        Returns:
            Kernel matrix (n_samples, m_samples)
        """
        if Y is None:
            Y = X

        n_samples_x = X.shape[0]
        n_samples_y = Y.shape[0]

        # Initialize kernel matrix on GPU if available
        if self.use_gpu:
            K = torch.zeros((n_samples_x, n_samples_y), device='cuda:0')
        else:
            K = np.zeros((n_samples_x, n_samples_y))

        # Compute kernel elements
        if parallel:
            # Parallel execution for faster kernel computation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                indices = []

                for i in range(n_samples_x):
                    for j in range(n_samples_y):
                        # Skip duplicate computations for symmetric kernel
                        if Y is X and j < i:
                            continue

                        future = executor.submit(
                            self.compute_kernel_element,
                            X[i],
                            Y[j]
                        )
                        futures.append(future)
                        indices.append((i, j))

                # Collect results
                for future, (i, j) in zip(futures, indices):
                    kernel_val = future.result()
                    K[i, j] = kernel_val

                    # Symmetric kernel: K[i,j] = K[j,i]
                    if Y is X and i != j:
                        K[j, i] = kernel_val
        else:
            # Sequential execution
            for i in range(n_samples_x):
                for j in range(n_samples_y):
                    K[i, j] = self.compute_kernel_element(X[i], Y[j])

        # Convert to numpy if on GPU
        if self.use_gpu:
            K = K.cpu().numpy()

        return K

    def compute_kernel_matrix_batched(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Compute kernel matrix in batches to handle large datasets.

        Useful when O(N²) is too large to compute at once.
        """
        if Y is None:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]
        K = np.zeros((n_x, n_y))

        # Process in batches
        for i in range(0, n_x, batch_size):
            i_end = min(i + batch_size, n_x)

            for j in range(0, n_y, batch_size):
                j_end = min(j + batch_size, n_y)

                # Compute sub-matrix
                K[i:i_end, j:j_end] = self.compute_kernel_matrix(
                    X[i:i_end],
                    Y[j:j_end],
                    parallel=True
                )

        return K
```

**Features**:
- Multiple quantum feature maps (angle, IQP, data re-uploading)
- Adjoint circuit method (NISQ-friendly)
- Parallel quantum circuit execution
- Batched computation for large datasets
- GPU support for classical operations

---

### 3. GPU-Accelerated Kernel SVM

**File**: `q_store/ml/kernel_svm_gpu.py` (NEW)

```python
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

class KernelSVMGPU:
    """
    GPU-accelerated kernel SVM using precomputed quantum kernel.

    Trains on GPU using the quantum kernel matrix.
    """

    def __init__(
        self,
        C: float = 1.0,
        device: str = 'cuda:0',
        kernel_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize kernel SVM.

        Args:
            C: Regularization parameter
            device: GPU device
            kernel_matrix: Precomputed quantum kernel (N×N)
        """
        self.C = C
        self.device = torch.device(device)
        self.kernel_matrix = None

        if kernel_matrix is not None:
            self.set_kernel_matrix(kernel_matrix)

        self.alpha = None  # Dual coefficients
        self.support_vectors = None
        self.support_labels = None
        self.b = 0.0  # Bias term

    def set_kernel_matrix(self, K: np.ndarray):
        """Set precomputed quantum kernel matrix."""
        self.kernel_matrix = torch.from_numpy(K).float().to(self.device)

    def fit(
        self,
        y: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-3
    ):
        """
        Train SVM using quantum kernel.

        Solves dual SVM optimization on GPU.

        Args:
            y: Labels (N,)
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        if self.kernel_matrix is None:
            raise ValueError("Kernel matrix not set")

        # Convert labels to GPU
        y = torch.from_numpy(y).float().to(self.device)
        n_samples = y.shape[0]

        # Initialize dual variables (alpha)
        self.alpha = torch.zeros(n_samples, device=self.device, requires_grad=True)

        # SMO (Sequential Minimal Optimization) on GPU
        # Simplified version - full implementation would use quadratic programming

        # For now, use scikit-learn-like approach with GPU acceleration
        # In practice, you'd use cuML or implement custom CUDA kernels

        # Placeholder: simple gradient-based optimization
        optimizer = torch.optim.LBFGS([self.alpha], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()

            # Dual objective: maximize -0.5 * α^T Q α + 1^T α
            # where Q[i,j] = y_i * y_j * K[i,j]
            Q = self.kernel_matrix * torch.outer(y, y)

            # Objective (negated for minimization)
            objective = 0.5 * torch.dot(self.alpha, Q @ self.alpha) - torch.sum(self.alpha)

            # Constraints: 0 ≤ α ≤ C and sum(α * y) = 0
            # Penalty for constraint violations
            penalty = 100.0 * torch.sum(torch.relu(-self.alpha)) + \
                     100.0 * torch.sum(torch.relu(self.alpha - self.C)) + \
                     100.0 * (torch.sum(self.alpha * y) ** 2)

            loss = objective + penalty
            loss.backward()
            return loss

        optimizer.step(closure)

        # Extract support vectors
        support_mask = self.alpha > 1e-5
        self.support_vectors = torch.where(support_mask)[0]
        self.support_labels = y[support_mask]

        # Compute bias term
        if len(self.support_vectors) > 0:
            sv_idx = self.support_vectors[0].item()
            self.b = y[sv_idx] - torch.sum(
                self.alpha * y * self.kernel_matrix[:, sv_idx]
            )

        print(f"Training complete. {len(self.support_vectors)} support vectors.")

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict using quantum kernel.

        Args:
            K_test: Test kernel matrix (n_test, n_train)

        Returns:
            Predictions (n_test,)
        """
        K_test = torch.from_numpy(K_test).float().to(self.device)

        # Decision function: f(x) = sum(α_i * y_i * K(x, x_i)) + b
        decision = torch.matmul(
            K_test,
            self.alpha * self.support_labels
        ) + self.b

        # Binary classification: sign(f(x))
        predictions = torch.sign(decision)

        return predictions.cpu().numpy()

    def score(self, K_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict(K_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy


class KernelRidgeRegressionGPU:
    """
    GPU-accelerated kernel ridge regression.

    Closed-form solution: α = (K + λI)^(-1) y
    """

    def __init__(
        self,
        alpha: float = 1.0,
        device: str = 'cuda:0'
    ):
        self.alpha = alpha  # Regularization
        self.device = torch.device(device)
        self.dual_coef = None

    def fit(self, K: np.ndarray, y: np.ndarray):
        """
        Train kernel ridge regression.

        Args:
            K: Kernel matrix (N×N)
            y: Target values (N,)
        """
        K = torch.from_numpy(K).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        n_samples = K.shape[0]

        # Closed-form solution
        K_reg = K + self.alpha * torch.eye(n_samples, device=self.device)
        self.dual_coef = torch.linalg.solve(K_reg, y)

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict using kernel.

        Args:
            K_test: Test kernel (n_test, n_train)

        Returns:
            Predictions (n_test,)
        """
        K_test = torch.from_numpy(K_test).float().to(self.device)
        predictions = torch.matmul(K_test, self.dual_coef)
        return predictions.cpu().numpy()
```

**Features**:
- GPU-accelerated SVM training
- Works with precomputed quantum kernels
- Kernel ridge regression for regression tasks
- 100-500x faster than CPU training

---

### 4. End-to-End Quantum Kernel Workflow

**File**: `q_store/workflows/quantum_kernel_workflow.py` (NEW)

```python
from q_store.data import DatasetLoader
from q_store.gpu import GPUDataPipeline
from q_store.kernels import QuantumKernel
from q_store.ml import KernelSVMGPU
import numpy as np

class QuantumKernelWorkflow:
    """
    End-to-end workflow for quantum kernel methods.

    Orchestrates:
    1. GPU data preprocessing
    2. Quantum kernel computation
    3. GPU-based classical training
    """

    def __init__(
        self,
        n_qubits: int = 8,
        feature_map: str = 'angle',
        quantum_backend: str = 'ionq.simulator',
        gpu_device: str = 'cuda:0'
    ):
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.quantum_backend = quantum_backend
        self.gpu_device = gpu_device

        # Initialize components
        self.data_pipeline = None
        self.quantum_kernel = None
        self.classifier = None

    def prepare_data(self, dataset_config):
        """Load and preprocess data on GPU."""
        # Load dataset
        from q_store.data import DatasetLoader
        dataset = DatasetLoader.load(dataset_config)

        # GPU preprocessing
        self.data_pipeline = GPUDataPipeline(
            dataset,
            device=self.gpu_device
        )
        self.data_pipeline.preprocess(normalize=True)

        print(f"Data loaded on GPU: {dataset.num_samples} samples")

    def compute_quantum_kernel(self, subset_size: int = None):
        """
        Compute quantum kernel matrix.

        Args:
            subset_size: Use subset for faster prototyping (O(N²) scaling)
        """
        # Get training data from GPU
        x_train, y_train = self.data_pipeline.get_data('train')

        # Convert to numpy for quantum circuits
        x_train_np = x_train.cpu().numpy()

        # Use subset if specified
        if subset_size is not None and subset_size < len(x_train_np):
            indices = np.random.choice(len(x_train_np), subset_size, replace=False)
            x_train_np = x_train_np[indices]
            self.y_train_subset = y_train[indices]
        else:
            self.y_train_subset = y_train

        # Initialize quantum kernel
        self.quantum_kernel = QuantumKernel(
            n_qubits=self.n_qubits,
            feature_map=self.feature_map,
            backend=self.quantum_backend,
            use_gpu=True
        )

        print(f"Computing quantum kernel matrix ({len(x_train_np)}×{len(x_train_np)})...")

        # Compute kernel matrix (O(N²) quantum executions)
        self.K_train = self.quantum_kernel.compute_kernel_matrix(
            x_train_np,
            parallel=True,
            max_workers=10
        )

        print(f"Quantum kernel matrix computed. Shape: {self.K_train.shape}")

    def train_classical_model(self, model_type: str = 'svm'):
        """Train classical model on GPU using quantum kernel."""
        y_train = self.y_train_subset.cpu().numpy()

        if model_type == 'svm':
            self.classifier = KernelSVMGPU(
                C=1.0,
                device=self.gpu_device,
                kernel_matrix=self.K_train
            )
            self.classifier.fit(y_train, max_iter=1000)

        elif model_type == 'ridge':
            from q_store.ml import KernelRidgeRegressionGPU
            self.classifier = KernelRidgeRegressionGPU(
                alpha=1.0,
                device=self.gpu_device
            )
            self.classifier.fit(self.K_train, y_train)

        print(f"Classical {model_type} training complete on GPU")

    def evaluate(self):
        """Evaluate on test set."""
        # Get test data
        x_test, y_test = self.data_pipeline.get_data('test')
        x_test_np = x_test.cpu().numpy()

        # Get training data for kernel computation
        x_train, _ = self.data_pipeline.get_data('train')
        x_train_np = x_train.cpu().numpy()

        print("Computing test kernel matrix...")

        # Compute test kernel K_test[i,j] = K(x_test[i], x_train[j])
        K_test = self.quantum_kernel.compute_kernel_matrix(
            x_test_np,
            x_train_np,
            parallel=True
        )

        # Predict on GPU
        y_pred = self.classifier.predict(K_test)

        # Compute accuracy
        accuracy = np.mean(y_pred == y_test.cpu().numpy())

        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def run_full_workflow(self, dataset_config, subset_size: int = 500):
        """
        Run complete quantum kernel workflow.

        Args:
            dataset_config: Dataset configuration
            subset_size: Training subset size (for O(N²) kernel computation)
        """
        print("="*60)
        print("Q-Store v4.2.0 Quantum Kernel Workflow")
        print("="*60)

        # Step 1: GPU data preprocessing
        print("\n[1/4] GPU Data Preprocessing...")
        self.prepare_data(dataset_config)

        # Step 2: Quantum kernel computation
        print("\n[2/4] Quantum Kernel Computation...")
        self.compute_quantum_kernel(subset_size=subset_size)

        # Step 3: GPU-based classical training
        print("\n[3/4] GPU Classical Training...")
        self.train_classical_model(model_type='svm')

        # Step 4: Evaluation
        print("\n[4/4] Evaluation...")
        accuracy = self.evaluate()

        print("\n" + "="*60)
        print(f"Workflow Complete! Test Accuracy: {accuracy:.4f}")
        print("="*60)

        return accuracy
```

---

## Expected Performance Improvements

### Benchmark Scenario: Fashion MNIST (500 training samples, 5 classes)

| Configuration | Kernel Computation | Training Time | Total Time | Speedup vs v4.1.1 | Cost |
|---------------|-------------------|---------------|------------|-------------------|------|
| **v4.1.1 VQC** | N/A | 2,288s | 2,288s | 1x | $0 |
| **v4.2.0 QKM (IonQ Sim)** | 150s (O(N²)) | 5s (GPU) | 155s | **14.8x** | $0 |
| **v4.2.0 QKM (Local Sim)** | 50s (O(N²)) | 5s (GPU) | 55s | **41.6x** | $0 |
| **v4.2.0 QKM (IonQ Aria)** | 300s (O(N²)) | 5s (GPU) | 305s | **7.5x** | $75 |

**Key Insight**: Kernel computation is O(N²) one-time cost, then all training is GPU-accelerated

### Scaling Analysis

| Dataset Size (N) | Kernel Computations (O(N²)) | Quantum Time (IonQ Sim) | GPU Training | Total |
|------------------|----------------------------|------------------------|--------------|-------|
| 100 samples | 10,000 | 30s | 1s | 31s |
| 500 samples | 250,000 | 150s | 5s | 155s |
| 1,000 samples | 1,000,000 | 600s (10min) | 10s | 610s |
| 5,000 samples | 25,000,000 | 15,000s (4h) | 50s | 4h |

**Practical Range**: 100-1,000 samples (where quantum kernels are most effective)

### Performance Breakdown

**GPU Preprocessing**:
- 10-50x faster than CPU
- Negligible time (<1s for 1,000 samples)

**Quantum Kernel Computation**:
- O(N²) circuit executions
- Dominant cost for large N
- Parallelizable (10-20 workers)
- One-time cost (reuse for all epochs)

**GPU Classical Training**:
- 100-500x faster than CPU
- Very fast (<5s for 500 samples)
- Reuses quantum kernel (no additional quantum calls)

---

## Concrete Cost Estimates

### IonQ Pricing (via AWS/Azure)

**Pricing Model**: Shots × Circuit Cost

**Typical Quantum Kernel Circuit**:
- 10-20 two-qubit gates (CNOT for entanglement)
- 8-12 single-qubit gates (RY, RZ rotations)
- ~1-5 ms execution time per circuit
- Cost per circuit: ~$0.30 (IonQ Aria)

### PoC Cost Estimates

| Dataset Size (N) | Kernel Elements (N²) | Shots per Kernel | Total Shots | IonQ Aria Cost | Quantinuum Cost |
|------------------|---------------------|------------------|-------------|----------------|-----------------|
| **20 samples** | 400 | 1,000 | 400K | **$50-80** | **$100-150** |
| **50 samples** | 2,500 | 1,000 | 2.5M | **$120-200** | **$200-350** |
| **100 samples** | 10,000 | 1,000 | 10M | **$300-500** | **$500-800** |
| **500 samples** | 250,000 | 1,000 | 250M | **$7,500-12,000** | **N/A** |

### Cost Optimization Strategies

1. **Start with Simulators** (FREE):
   - IonQ Simulator: $0
   - Qiskit Aer: $0
   - PennyLane default.qubit: $0

2. **Small N for PoC** (N = 20-50):
   - Cost: $50-200
   - Validate quantum advantage
   - Test feature maps

3. **Quantinuum for High Fidelity**:
   - Fewer shots needed (200-500 vs 1,000)
   - Higher quality kernels
   - Best for research-grade results

4. **Cache Kernel Matrix in Vector DB**:
   - One-time quantum cost
   - Reuse across multiple training runs
   - Amortize cost over experiments

### When Quantum Kernels Are Cost-Effective

- **Small datasets** (N < 100): Quantum cost ~$50-500
- **Research projects**: One-time kernel computation, many experiments
- **High-value applications**: Where quantum advantage justifies cost
- **Proof-of-concept**: Validate approach before scaling

### When Classical Kernels Are Better

- **Large datasets** (N > 500): Quantum cost > $5,000
- **Production deployments**: Need low latency and cost
- **Commodity ML**: Where classical RBF/polynomial kernels suffice

---

## 4-Phase PoC Roadmap

### Phase 1: Simulation & Validation (Week 1-2)

**Goal**: Validate quantum kernel methods without QPU cost

**Tasks**:
1. Implement quantum kernel engine with local simulators
2. Test 3 feature maps (angle, IQP, data re-uploading)
3. Compare quantum kernels vs classical kernels (RBF, polynomial)
4. Benchmark on small dataset (N=20, Fashion MNIST)

**Tools**:
- PennyLane + default.qubit simulator (FREE)
- GPU for classical training
- FAISS for kernel storage

**Deliverables**:
- Working quantum kernel computation (simulator-based)
- Performance comparison report
- Feature map selection guide

**Expected Outcome**: Validate that quantum kernels provide different feature space than classical

---

### Phase 2: Real QPU with Small N (Week 3-4)

**Goal**: Run on actual quantum hardware with minimal cost

**Tasks**:
1. Integrate with IonQ Simulator (cloud-based, FREE)
2. Test with N=20-50 samples
3. Measure quantum circuit execution time
4. Cache kernel matrix in FAISS vector DB
5. Train SVM on GPU using quantum kernel

**Backend**: IonQ Simulator (free) → IonQ Aria (paid, small N)

**Cost**: $50-200 for N=20-50 on IonQ Aria

**Deliverables**:
- Quantum kernel matrix from real QPU
- Trained SVM model
- Performance metrics (accuracy, time, cost)

**Expected Outcome**: Demonstrate end-to-end hybrid workflow with real quantum hardware

---

### Phase 3: Hybrid Scaling with Vector DB (Week 5-6)

**Goal**: Scale to N=100-500 with vector DB optimization

**Tasks**:
1. Implement FAISS vector DB integration
2. Batch kernel computation (compute in chunks)
3. Implement GraphQL orchestration layer (optional)
4. Add kernel caching and reuse logic
5. Test with N=100 samples (10K kernel elements)

**Infrastructure**:
- GPU: Single NVIDIA GPU (A100 or V100)
- Vector DB: FAISS (GPU-accelerated)
- QPU: IonQ Aria or Quantinuum

**Cost**: $300-800 for N=100

**Deliverables**:
- Scalable kernel storage system
- GraphQL API for job management
- Benchmark: Training time breakdown (QPU vs GPU vs vector DB)

**Expected Outcome**: Demonstrate scalability to N=100 with manageable cost

---

### Phase 4: Benchmarking & Comparison (Week 7)

**Goal**: Compare quantum kernels vs classical baselines

**Tasks**:
1. Benchmark quantum kernels vs RBF kernels
2. Measure accuracy improvement (if any)
3. Analyze cost-benefit trade-off
4. Document when quantum kernels provide advantage
5. Write research paper or blog post

**Comparison Metrics**:
- **Accuracy**: Quantum kernel SVM vs classical kernel SVM
- **Training Time**: GPU optimization time (same for both)
- **Total Cost**: Quantum kernel cost vs $0 for classical
- **Feature Space**: Visualize quantum vs classical embeddings

**Deliverables**:
- Comprehensive benchmark report
- When-to-use decision matrix
- Publication-ready results

**Expected Outcome**: Clear guidance on when quantum kernels justify the cost

---

### PoC Timeline Summary

| Phase | Duration | N | QPU Cost | Key Milestone |
|-------|----------|---|----------|---------------|
| **1: Simulation** | Weeks 1-2 | 20 | $0 | Validate approach |
| **2: Real QPU** | Weeks 3-4 | 20-50 | $50-200 | End-to-end workflow |
| **3: Scaling** | Weeks 5-6 | 100 | $300-800 | Vector DB integration |
| **4: Benchmark** | Week 7 | 100 | $0 | Publication results |
| **Total** | **7 weeks** | **100** | **$350-1,000** | **Production-ready** |

---

## Implementation Roadmap

### Phase 1: GPU Data Pipeline (Week 1)

**Priority**: High

**Tasks**:

1. Implement `GPUDataPipeline` with PyTorch
2. GPU preprocessing (normalization, augmentation)
3. Optional CNN feature extraction
4. Benchmark GPU vs CPU (expected: 10-50x)
5. Unit tests

**Deliverables**:
- `q_store/gpu/data_pipeline.py`
- GPU preprocessing 10-50x faster than CPU

---

### Phase 2: Quantum Kernel Engine (Weeks 2-3)

**Priority**: Critical

**Tasks**:

1. Implement `QuantumKernel` class
2. Feature maps: angle encoding, IQP, data re-uploading
3. Adjoint circuit method for kernel computation
4. Parallel execution support
5. Batched kernel computation
6. Integration with existing quantum backends
7. Unit tests for each feature map

**Deliverables**:
- `q_store/kernels/quantum_kernel.py`
- Support for 3+ feature maps
- Parallel kernel computation

---

### Phase 3: GPU Kernel SVM (Week 4)

**Priority**: High

**Tasks**:

1. Implement `KernelSVMGPU`
2. Implement `KernelRidgeRegressionGPU`
3. GPU-accelerated training
4. Support for precomputed kernels
5. Unit tests

**Deliverables**:
- `q_store/ml/kernel_svm_gpu.py`
- 100-500x speedup vs CPU SVM

---

### Phase 4: End-to-End Workflow (Week 5)

**Priority**: High

**Tasks**:

1. Implement `QuantumKernelWorkflow`
2. Orchestrate full pipeline
3. Add monitoring and logging
4. Integration tests
5. Example notebooks

**Deliverables**:
- `q_store/workflows/quantum_kernel_workflow.py`
- Complete end-to-end workflow
- Working examples

---

### Phase 5: Testing & Benchmarking (Week 6)

**Priority**: High

**Tasks**:

1. Comprehensive unit tests (95%+ coverage)
2. Integration tests with all backends
3. Performance benchmarks vs v4.1.1
4. Scalability analysis (N=100 to N=5000)
5. Cost analysis

**Deliverables**:
- Complete test suite
- Performance report
- Scaling analysis

---

### Phase 6: Documentation & Examples (Week 7)

**Priority**: Medium

**Tasks**:

1. API reference documentation
2. Migration guide (v4.1.1 → v4.2.0)
3. Quantum kernel methods tutorial
4. Example: Fashion MNIST classification
5. Example: Kernel comparison (quantum vs classical)
6. Example: Feature map selection

**Deliverables**:
- Complete documentation
- 5+ working examples
- Migration guide

---

## File Structure (4-Layer Architecture)

```
q-store/
├── src/q_store/
│   │
│   ├── LAYER 4: Orchestration (Optional)
│   ├── orchestration/                 🆕 NEW
│   │   ├── __init__.py
│   │   ├── graphql_api.py            🆕 GraphQL API for hybrid workflows
│   │   ├── job_manager.py            🆕 Async quantum job management
│   │   └── cost_tracker.py           🆕 QPU cost tracking & budgeting
│   │
│   ├── LAYER 1: GPU - Classical Training
│   ├── gpu/                           🆕 NEW
│   │   ├── __init__.py
│   │   └── data_pipeline.py          🆕 GPU data loading & preprocessing
│   │
│   ├── ml/                            🔧 ENHANCED
│   │   ├── kernel_svm_gpu.py         🆕 GPU-accelerated kernel SVM
│   │   ├── kernel_ridge_gpu.py       🆕 GPU kernel ridge regression
│   │   └── [existing v4.1.1 modules] ✅ Unchanged
│   │
│   ├── LAYER 2: QPU - Quantum Kernel Evaluation
│   ├── kernels/                       🆕 NEW
│   │   ├── __init__.py
│   │   ├── quantum_kernel.py         🆕 Quantum kernel computation
│   │   ├── feature_maps.py           🆕 Angle, IQP, data re-uploading
│   │   └── adjoint_circuit.py        🆕 Adjoint circuit method
│   │
│   ├── LAYER 3: Vector DB - Kernel Storage
│   ├── storage/                       🆕 NEW
│   │   ├── __init__.py
│   │   ├── vector_db.py              🆕 FAISS vector database integration
│   │   ├── kernel_cache.py           🆕 Kernel matrix caching
│   │   └── support_vector_store.py   🆕 SVM support vector storage
│   │
│   ├── Hybrid Workflows
│   ├── workflows/                     🆕 NEW
│   │   ├── __init__.py
│   │   ├── quantum_kernel_workflow.py 🆕 End-to-end QKM workflow
│   │   └── hybrid_trainer.py         🆕 Orchestrates all 4 layers
│   │
│   ├── Existing Components (v4.1.0, v4.1.1)
│   ├── data/                          ✅ FROM v4.1.1
│   ├── tracking/                      ✅ FROM v4.1.1
│   ├── tuning/                        ✅ FROM v4.1.1
│   ├── backends/                      ✅ FROM v4.1.0 (IonQ, Cirq, Qiskit)
│   ├── layers/                        ✅ FROM v4.1.0 (VQC support)
│   └── [other v4.1.0 modules]         ✅ Unchanged (151 files)
│
├── examples/
│   ├── quantum_kernels/               🆕 NEW
│   │   ├── 01_simulation_poc.py      🆕 Phase 1: Local simulation
│   │   ├── 02_real_qpu_small_n.py    🆕 Phase 2: IonQ with N=20-50
│   │   ├── 03_vector_db_scaling.py   🆕 Phase 3: FAISS scaling
│   │   ├── 04_benchmark_classical.py 🆕 Phase 4: Quantum vs classical
│   │   ├── fashion_mnist_qkm.py      🆕 Fashion MNIST with quantum kernels
│   │   ├── kernel_comparison.py      🆕 RBF vs quantum kernels
│   │   ├── feature_map_selection.py  🆕 Compare angle/IQP/data-upload
│   │   ├── cost_optimization.py      🆕 Minimize QPU cost
│   │   └── graphql_api_demo.py       🆕 GraphQL orchestration example
│
├── tests/
│   ├── test_gpu/                      🆕 NEW
│   │   └── test_data_pipeline.py
│   ├── test_kernels/                  🆕 NEW
│   │   ├── test_quantum_kernel.py
│   │   ├── test_feature_maps.py
│   │   └── test_adjoint_circuit.py
│   ├── test_storage/                  🆕 NEW
│   │   ├── test_vector_db.py
│   │   └── test_kernel_cache.py
│   ├── test_orchestration/            🆕 NEW (Optional)
│   │   ├── test_graphql_api.py
│   │   └── test_job_manager.py
│   ├── test_ml/                       🔧 ENHANCED
│   │   ├── test_kernel_svm_gpu.py    🆕 NEW
│   │   └── test_kernel_ridge_gpu.py  🆕 NEW
│   └── integration/                   🔧 ENHANCED
│       ├── test_qkm_workflow.py      🆕 End-to-end test
│       ├── test_4layer_integration.py 🆕 All layers working together
│       └── test_real_qpu.py          🆕 IonQ/Quantinuum integration
│
└── docs/
    ├── Q-STORE_V4_2_0_ARCHITECTURE_DESIGN.md     🆕 THIS FILE
    ├── V4_2_0_MIGRATION_GUIDE.md                 🆕 TODO
    ├── QUANTUM_KERNEL_METHODS_GUIDE.md           🆕 TODO
    ├── VECTOR_DB_INTEGRATION_GUIDE.md            🆕 TODO
    ├── GRAPHQL_API_REFERENCE.md                  🆕 TODO (Optional)
    ├── POC_PHASE_REPORTS/                        🆕 TODO
    │   ├── PHASE1_SIMULATION.md
    │   ├── PHASE2_REAL_QPU.md
    │   ├── PHASE3_SCALING.md
    │   └── PHASE4_BENCHMARK.md
    └── PERFORMANCE_BENCHMARKS_V4_2_0.md          🆕 TODO
```

---

## Dependencies

### New Dependencies (v4.2.0)

```txt
# LAYER 1: GPU acceleration
torch>=2.2.0              # PyTorch with CUDA support (single GPU initially)
cuml>=24.0.0              # NVIDIA RAPIDS for GPU SVM (optional)
scikit-learn>=1.4.0       # Classical kernel methods for comparison

# LAYER 2: Quantum backends (use existing)
# ionq-api-client, cirq, qiskit, pennylane - already in v4.1.0

# LAYER 3: Vector Database
faiss-gpu>=1.7.4          # GPU-accelerated FAISS for kernel storage
# OR faiss-cpu>=1.7.4     # CPU-only version (fallback)
# Optional alternatives:
# milvus>=2.3.0           # Distributed vector DB
# qdrant-client>=1.7.0    # Alternative vector DB

# LAYER 4: GraphQL Orchestration (Optional)
graphene>=3.3.0           # GraphQL framework for Python
graphql-core>=3.2.0       # GraphQL core library
starlette>=0.32.0         # ASGI framework for GraphQL API
# OR fastapi>=0.108.0     # Alternative API framework

# Performance monitoring
GPUtil>=1.4.0             # GPU monitoring
psutil>=5.9.0             # System monitoring
```

### Layer-Specific Dependencies

**LAYER 1 (GPU Classical Training)**:
- torch (PyTorch for GPU operations)
- cuml (NVIDIA RAPIDS for GPU-accelerated SVM - optional)
- scikit-learn (Classical kernel baselines)

**LAYER 2 (QPU Quantum Kernels)**:
- ionq-api-client (IonQ backend - from v4.1.0)
- pennylane (Quantum ML framework - optional)
- cirq, qiskit (Alternative backends - from v4.1.0)

**LAYER 3 (Vector Database)**:
- faiss-gpu (Primary choice for kernel storage)
- milvus or qdrant (Alternative for production deployments)

**LAYER 4 (GraphQL Orchestration - Optional)**:
- graphene (GraphQL schema definition)
- starlette or fastapi (API server)

### Existing Dependencies (from v4.1.0, v4.1.1)

```txt
# Quantum backends
ionq-api-client           # IonQ backend
cirq                      # Google Cirq
qiskit                    # IBM Qiskit
pennylane                 # PennyLane (optional, for quantum ML)

# Data management (v4.1.1)
datasets>=2.16.1
requests>=2.31.0

# Core ML
numpy>=1.24.0
scipy>=1.10.0
```

**Note**: No TorchQuantum-Dist dependency. We implement kernel methods using existing Q-Store quantum backends.

---

## Backward Compatibility

### v4.1.1 Compatibility

All v4.1.1 features work unchanged:
- Data management layer
- Experiment tracking
- Hyperparameter tuning
- VQC training (still available)

### v4.1.0 Compatibility

All v4.1.0 quantum features preserved:
- Quantum layers
- VQC training
- All backends (IonQ, Cirq, Qiskit)

### Migration Path

**From v4.1.1 to v4.2.0**:

```python
# v4.1.1 VQC approach (still works)
from q_store.ml import QuantumTrainer
trainer = QuantumTrainer(...)
trainer.train(x_train, y_train)

# v4.2.0 Quantum Kernel Methods (new, recommended for small datasets)
from q_store.workflows import QuantumKernelWorkflow

workflow = QuantumKernelWorkflow(
    n_qubits=8,
    feature_map='angle',
    quantum_backend='ionq.simulator',
    gpu_device='cuda:0'
)

# Run full workflow
accuracy = workflow.run_full_workflow(
    dataset_config=config,
    subset_size=500  # O(N²) scaling
)
```

**When to use each approach**:
- **Quantum Kernels (v4.2.0)**: Small datasets (N < 1,000), want stable training, avoid barren plateaus
- **VQCs (v4.1.1)**: Large datasets, need end-to-end quantum training, research purposes

---

## Risk Analysis & Mitigation

### Technical Risks

**Risk 1: O(N²) Scaling Limitation**
- Concern: Kernel computation scales as O(N²), limiting dataset size
- Mitigation: Recommend N < 1,000 samples; provide batching for larger datasets
- Alternative: Subset selection methods (Nyström approximation)

**Risk 2: Quantum Kernel Advantage Unclear**
- Concern: Quantum kernels may not always outperform classical RBF kernels
- Mitigation: Provide classical kernel baselines for comparison
- Research: Include feature map selection guide

**Risk 3: Hardware Noise Impact**
- Concern: Noisy quantum hardware can distort kernel values
- Mitigation: Error mitigation techniques, high shot counts (>1024)
- Fallback: Use simulators for prototyping

### Performance Risks

**Risk 1: Kernel Computation Time**
- Concern: O(N²) quantum executions may be slow
- Mitigation: Parallel execution (10-20 workers), caching
- Monitoring: Track kernel computation progress

**Risk 2: GPU Speedup Lower Than Expected**
- Concern: GPU classical training may not achieve 100x speedup
- Mitigation: Benchmarking, optimize hot paths
- Alternative: Use cuML for maximum GPU acceleration

---

## Success Metrics

### Performance Targets

1. **GPU Preprocessing**: 10-50x faster than CPU
2. **Quantum Kernel Computation**: Complete 500×500 matrix in <3 minutes (simulator)
3. **GPU Classical Training**: 100-500x faster than CPU SVM
4. **Overall Speedup**: 10-40x vs v4.1.1 VQC for datasets with N < 1,000

### Quality Targets

1. **Code Coverage**: 95%+ for new modules
2. **Integration Tests**: All feature maps and backends tested
3. **Backward Compatibility**: 100% of v4.1.1 tests pass

### User Experience Targets

1. **Ease of Use**: <10 lines of code for complete workflow
2. **Documentation**: Complete guide on quantum kernel methods
3. **Examples**: 5+ working examples with different datasets

---

## Conclusion

Q-Store v4.2.0 adopts **Quantum Kernel Methods** as a more stable, practical alternative to VQC training. By moving quantum effort from parameter optimization to feature representation, we achieve:

### Key Advantages

1. **No Barren Plateaus**: No parameter optimization on quantum hardware
2. **Stable Training**: Classical GPU-based training is well-understood
3. **Simplified Architecture**: No complex multi-GPU quantum simulation
4. **Practical Near-Term**: Works on today's NISQ devices
5. **Significant Speedup**: 10-40x faster than v4.1.1 VQC approach

### Trade-offs

- **Dataset Size**: Limited to N < 1,000 due to O(N²) scaling
- **One-time Quantum Cost**: Must compute full kernel matrix upfront
- **Feature Map Selection**: Requires choosing appropriate quantum encoding

### Next Steps

1. Review and approve this revised architecture
2. Begin Phase 1: GPU data pipeline
3. Implement quantum kernel engine
4. Benchmark against v4.1.1 and classical baselines
5. Release Q-Store v4.2.0

---

**Document Version**: 2.0 (Revised)
**Status**: Draft for Review
**Target Release**: Q2 2026

**References**:
- Quantum Kernel Methods: `/home/yucelz/Downloads/quantum_kernel_methods_detailed_explanation.md`
- Q-Store v4.1.1 Performance Report: `Q-STORE_PERFORMANCE_REPORT.md`
- Q-Store v4.1.1 Architecture: `Q-STORE_V4_1_1_ARCHITECTURE_DESIGN.md`
