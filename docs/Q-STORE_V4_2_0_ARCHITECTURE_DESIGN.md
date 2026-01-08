# Q-Store v4.2.0 Architecture Design
# Hybrid GPU-Quantum Performance Acceleration

**Version**: 4.2.0
**Date**: January 7, 2026
**Status**: Draft for Review
**Focus**: Performance Optimization via Hybrid GPU-Quantum Architecture

---

## Executive Summary

Q-Store v4.2.0 introduces a **Hybrid GPU-Quantum Architecture** to address the critical performance gap identified in v4.1.1 performance analysis, where quantum training was 183-457x slower than classical GPU approaches. This version strategically combines GPU acceleration for classical preprocessing, postprocessing, and simulation with quantum hardware for specialized quantum advantage workloads.

### Key Objectives

1. **GPU Acceleration Pipeline**: Offload classical preprocessing, feature extraction, and data preparation to GPU
2. **Multi-GPU Quantum Simulation**: Enable distributed quantum statevector simulation across multiple GPUs
3. **Intelligent Workload Routing**: Automatically route tasks to GPU-only, hybrid, or quantum-only execution paths
4. **Hybrid Training Architecture**: Combine classical GPU layers with quantum layers in a unified training framework
5. **Performance Target**: Achieve 10-50x speedup for hybrid workloads compared to v4.1.1

### Inspiration & Technical Foundation

**TorchQuantum-Dist Approach**:
- Multi-GPU quantum statevector simulation using PyTorch DTensor
- Seamless integration with PyTorch's autograd for end-to-end differentiability
- Hardware-agnostic design supporting NVIDIA/AMD GPUs

**D-Wave Hybrid DVAE Approach**:
- Classical neural networks handle high-dimensional pixel-space transformations
- Quantum processor specializes in discrete latent space probability distributions
- Clear separation of classical preprocessing and quantum core processing

---

## Current State Analysis (v4.1.1)

### Performance Benchmarks (Cats vs Dogs - 1,000 images, 5 epochs)

| Platform | Time per Epoch | Total Time | Relative Speed | Cost |
|----------|----------------|------------|----------------|------|
| **NVIDIA H100 GPU** | 1.0s | 5s | **457x faster** | $0.009 |
| **NVIDIA A100 GPU** | 1.5s | 7.5s | **305x faster** | $0.010 |
| **NVIDIA V100 GPU** | 2.5s | 12.5s | **183x faster** | $0.012 |
| Q-Store (no latency) | 204s | 1,020s | 4.5x faster | $0.00 |
| **Q-Store v4.1.1 (current)** | 457s | 2,288s | Baseline | $0.00 |

### Current Bottleneck Analysis

1. **Network Latency (55%)**: API round-trip time to IonQ cloud
2. **Circuit Queue Time (20%)**: Waiting for simulator to process
3. **Data Serialization (15%)**: Converting circuits to IonQ format
4. **Quantum Execution (10%)**: Actual circuit simulation time

### Key Findings

- **Quantum accuracy**: 58.48% (comparable to classical)
- **Primary bottleneck**: Network latency, not quantum computation itself
- **Opportunity**: GPU preprocessing + local quantum simulation can eliminate network overhead
- **Challenge**: Need hybrid approach that leverages both GPU and quantum strengths

---

## v4.2.0 Hybrid Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Q-Store v4.2.0 Hybrid GPU-Quantum System                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  1. GPU-Accelerated Data Pipeline (NEW)                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • GPU Data Loading (CuPy, PyTorch DataLoader)                         │ │
│  │  • GPU Preprocessing (normalization, augmentation on CUDA)             │ │
│  │  • GPU Feature Extraction (classical CNN layers)                       │ │
│  │  • Batched GPU operations (eliminate CPU-GPU transfer overhead)        │ │
│  │  Expected Speedup: 5-10x for data preprocessing                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Intelligent Workload Router (NEW)                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Decision Logic:                                                        │ │
│  │  • Problem Size < 8 qubits → Multi-GPU Quantum Simulation              │ │
│  │  • Problem Size 8-20 qubits → Hybrid (GPU preprocessing + QPU)         │ │
│  │  • Problem Size > 20 qubits → Pure QPU (IonQ Forte 36 qubits)          │ │
│  │  • Cost-sensitive mode → GPU-only classical approximation              │ │
│  │  • Research mode → QPU with full quantum features                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
           ↓                            ↓                            ↓
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Path 1: GPU-Only    │  │  Path 2: Hybrid      │  │  Path 3: QPU-Only    │
│  Classical Training  │  │  GPU + Quantum       │  │  Pure Quantum        │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
         ↓                            ↓                            ↓
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ • Classical CNN      │  │ • GPU Preprocessing  │  │ • IonQ Aria/Forte    │
│ • GPU Training       │  │ • Local Quantum Sim  │  │ • Full Quantum       │
│ • PyTorch/TF native  │  │ • QPU for critical   │  │ • Network overhead   │
│ • 1-5 seconds        │  │ • 30-120 seconds     │  │ • 200-500 seconds    │
│ • $0.01 cost         │  │ • $0.50-5 cost       │  │ • $100-1000 cost     │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  3. Multi-GPU Quantum Simulator (NEW - TorchQuantum-Dist Integration)       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • Distributed statevector simulation across 2-8 GPUs                  │ │
│  │  • PyTorch DTensor for automatic tensor distribution                   │ │
│  │  • Scalable to 12-16 qubits with 4 GPUs (2^16 = 65K states per GPU)   │ │
│  │  • Full gradient support via PyTorch autograd                          │ │
│  │  • Zero network latency (local computation)                            │ │
│  │  Expected Speedup: 50-100x vs cloud QPU for small circuits            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Hybrid Training Engine (ENHANCED)                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Architecture: Classical GPU Layers + Quantum Layers                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Input (28x28x1 MNIST)                                            │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  GPU Conv2D (32 filters) ← GPU accelerated                        │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  GPU MaxPool2D           ← GPU accelerated                        │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  GPU Conv2D (64 filters) ← GPU accelerated                        │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  GPU Flatten → Dense(128) ← GPU accelerated                       │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  Dimension Reduction (128 → 8 features) ← GPU PCA/Autoencoder     │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  ┌──────────────────────────────────────────────────┐             │ │ │
│  │  │  │ Quantum Circuit (8 qubits)                       │             │ │ │
│  │  │  │ • Multi-GPU simulation OR                        │             │ │ │
│  │  │  │ • Cloud QPU (if >16 qubits)                      │             │ │ │
│  │  │  │ • Parameterized gates (trainable)                │             │ │ │
│  │  │  └──────────────────────────────────────────────────┘             │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  GPU Dense(10) → Softmax ← GPU accelerated                        │ │ │
│  │  │         ↓                                                          │ │ │
│  │  │  Output (10 classes)                                              │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  • 70-80% of compute on GPU (classical layers)                        │ │
│  │  • 20-30% on quantum (core entanglement/interference)                 │ │
│  │  • End-to-end differentiable via PyTorch autograd                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  5. Quantum Backend Orchestrator (ENHANCED)                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Backend Selection Logic:                                              │ │
│  │  • Local GPU Simulator (4-14 qubits) ← NEW                            │ │
│  │  • IonQ Simulator (testing, free)                                     │ │
│  │  • IonQ Aria (25 qubits, $0.30/circuit)                               │ │
│  │  • IonQ Forte (36 qubits, reserved pricing)                           │ │
│  │  • Qiskit/Cirq local simulators (fallback)                            │ │
│  │                                                                         │ │
│  │  Selection Criteria:                                                   │ │
│  │  • Qubit count, circuit depth, budget, latency requirements           │ │
│  │  • Automatic fallback if QPU unavailable                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  6. Performance Monitoring & Optimization (ENHANCED)                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • Real-time performance profiling (GPU util, QPU queue time)          │ │
│  │  • Automatic batch size optimization for GPU                           │ │
│  │  • Circuit caching for repeated structures                             │ │
│  │  • Adaptive routing based on runtime metrics                           │ │
│  │  • Cost tracking and budget alerts                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. GPU-Accelerated Data Pipeline

**File**: `q_store/gpu/data_pipeline.py` (NEW)

#### 1.1 GPU Data Loader

```python
import torch
from torch.utils.data import DataLoader
import cupy as cp

class GPUDataPipeline:
    """
    GPU-accelerated data loading and preprocessing pipeline.

    Eliminates CPU-GPU transfer bottlenecks by keeping data on GPU.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        device: str = 'cuda:0',
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        """
        Initialize GPU data pipeline.

        Args:
            dataset: Dataset object from q_store.data
            batch_size: Batch size for GPU processing
            device: CUDA device (cuda:0, cuda:1, etc.)
            num_workers: DataLoader workers for CPU preprocessing
            pin_memory: Pin memory for faster CPU-GPU transfer
            prefetch_factor: Prefetch batches on GPU
        """
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Convert dataset to PyTorch tensors on GPU
        self.train_loader = self._create_gpu_dataloader(
            dataset.x_train, dataset.y_train, batch_size, num_workers, pin_memory
        )

        if dataset.x_val is not None:
            self.val_loader = self._create_gpu_dataloader(
                dataset.x_val, dataset.y_val, batch_size, num_workers, pin_memory
            )

    def _create_gpu_dataloader(self, x, y, batch_size, num_workers, pin_memory):
        """Create PyTorch DataLoader with GPU optimization."""
        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

        # Create DataLoader with GPU optimization
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

        return loader

    def get_batch(self, loader_iter):
        """Get next batch on GPU."""
        x, y = next(loader_iter)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class GPUPreprocessor:
    """GPU-accelerated preprocessing operations."""

    @staticmethod
    def normalize_gpu(data: torch.Tensor, mean: float = 0.0, std: float = 1.0):
        """Normalize data on GPU (100x faster than CPU)."""
        return (data - mean) / std

    @staticmethod
    def augment_gpu(images: torch.Tensor, transforms: dict):
        """GPU-based data augmentation using Kornia."""
        import kornia.augmentation as K

        aug_pipeline = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=10),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )

        return aug_pipeline(images)

    @staticmethod
    def extract_features_gpu(images: torch.Tensor, model: torch.nn.Module):
        """Extract features using GPU-accelerated CNN."""
        with torch.no_grad():
            features = model(images)
        return features

    @staticmethod
    def dimension_reduction_gpu(features: torch.Tensor, target_dim: int):
        """GPU PCA for dimension reduction."""
        # Simplified GPU PCA
        u, s, v = torch.pca_lowrank(features, q=target_dim)
        reduced = torch.matmul(features, v[:, :target_dim])
        return reduced
```

**Expected Performance**:
- Data loading: 10-20x faster than CPU
- Preprocessing: 50-100x faster than CPU (normalization, augmentation)
- Feature extraction: 100-200x faster (GPU CNN vs CPU)

---

### 2. Multi-GPU Quantum Simulator

**File**: `q_store/gpu/quantum_simulator.py` (NEW)

**Inspiration**: TorchQuantum-Dist architecture

#### 2.1 Distributed Quantum Device

```python
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, Replicate

class MultiGPUQuantumDevice:
    """
    Multi-GPU quantum statevector simulator using PyTorch DTensor.

    Distributes quantum statevector across multiple GPUs for scalable simulation.
    Based on TorchQuantum-Dist architecture.
    """

    def __init__(
        self,
        n_qubits: int,
        n_gpus: int = None,
        devices: list = None
    ):
        """
        Initialize multi-GPU quantum device.

        Args:
            n_qubits: Number of qubits (4-16 supported)
            n_gpus: Number of GPUs to use (auto-detect if None)
            devices: List of device IDs (e.g., [0, 1, 2, 3])

        Max qubits by GPU count:
        - 1 GPU: 12 qubits (4096 complex128 states = 32 KB)
        - 2 GPUs: 13 qubits (8192 states = 64 KB)
        - 4 GPUs: 14 qubits (16384 states = 128 KB)
        - 8 GPUs: 15 qubits (32768 states = 256 KB)
        """
        self.n_qubits = n_qubits

        # Auto-detect GPUs if not specified
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()

        if devices is None:
            devices = list(range(n_gpus))

        self.n_gpus = len(devices)
        self.devices = devices

        # Initialize distributed process group
        if not dist.is_initialized() and self.n_gpus > 1:
            dist.init_process_group(backend='nccl')

        # Create device mesh for DTensor
        self.device_mesh = DeviceMesh("cuda", torch.arange(self.n_gpus))

        # Initialize statevector (distributed across GPUs)
        self.state = self._create_distributed_statevector()

    def _create_distributed_statevector(self):
        """Create distributed quantum statevector."""
        state_size = 2 ** self.n_qubits

        if self.n_gpus == 1:
            # Single GPU: regular tensor
            state = torch.zeros(state_size, dtype=torch.complex128, device=f'cuda:{self.devices[0]}')
            state[0] = 1.0 + 0.0j  # |0...0⟩ state
            return state
        else:
            # Multi-GPU: distributed tensor
            # Shard along first dimension (distribute statevector elements)
            local_state = torch.zeros(
                state_size // self.n_gpus,
                dtype=torch.complex128,
                device=f'cuda:{dist.get_rank()}'
            )

            # Initialize |0...0⟩ on rank 0
            if dist.get_rank() == 0:
                local_state[0] = 1.0 + 0.0j

            # Create DTensor with sharding
            state = DTensor.from_local(
                local_state,
                device_mesh=self.device_mesh,
                placements=[Shard(0)]
            )

            return state

    def apply_gate(self, gate_matrix: torch.Tensor, qubit_indices: list):
        """
        Apply quantum gate to statevector.

        Args:
            gate_matrix: 2x2 or 4x4 gate matrix (for 1 or 2 qubit gates)
            qubit_indices: Indices of qubits to apply gate to
        """
        # Convert gate matrix to distributed tensor if needed
        if self.n_gpus > 1 and not isinstance(gate_matrix, DTensor):
            gate_matrix = DTensor.from_local(
                gate_matrix.to(f'cuda:{dist.get_rank()}'),
                device_mesh=self.device_mesh,
                placements=[Replicate()]
            )

        # Apply gate (simplified - actual implementation requires tensor reshaping)
        # This is where TorchQuantum-Dist logic would go
        self.state = self._apply_gate_distributed(gate_matrix, qubit_indices)

    def _apply_gate_distributed(self, gate_matrix, qubit_indices):
        """Distributed gate application (simplified)."""
        # Placeholder for actual distributed gate logic
        # Full implementation requires:
        # 1. Reshape statevector to separate qubit dimensions
        # 2. Apply gate to target qubits
        # 3. Flatten back to 1D statevector
        # 4. Handle distributed tensor operations

        return self.state

    def measure(self, shots: int = 1024):
        """
        Measure all qubits in computational basis.

        Returns:
            dict: Measurement results {bitstring: count}
        """
        # Get probabilities from statevector
        probs = torch.abs(self.state) ** 2

        if self.n_gpus > 1:
            # Gather probabilities from all GPUs
            probs = probs.to_local()
            probs_list = [torch.zeros_like(probs) for _ in range(self.n_gpus)]
            dist.all_gather(probs_list, probs)
            probs = torch.cat(probs_list)

        # Sample from probability distribution
        probs = probs.cpu().numpy()
        indices = np.random.choice(len(probs), size=shots, p=probs)

        # Convert indices to bitstrings
        results = {}
        for idx in indices:
            bitstring = format(idx, f'0{self.n_qubits}b')
            results[bitstring] = results.get(bitstring, 0) + 1

        return results

    def get_statevector(self):
        """Get full statevector (gathered from all GPUs)."""
        if self.n_gpus == 1:
            return self.state
        else:
            # Gather from all GPUs
            local_state = self.state.to_local()
            state_list = [torch.zeros_like(local_state) for _ in range(self.n_gpus)]
            dist.all_gather(state_list, local_state)
            return torch.cat(state_list)


class QuantumGate:
    """Parameterized quantum gates with PyTorch autograd support."""

    @staticmethod
    def rx(theta: torch.Tensor):
        """RX rotation gate."""
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.tensor([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ], dtype=torch.complex128)

    @staticmethod
    def ry(theta: torch.Tensor):
        """RY rotation gate."""
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.tensor([
            [cos, -sin],
            [sin, cos]
        ], dtype=torch.complex128)

    @staticmethod
    def rz(theta: torch.Tensor):
        """RZ rotation gate."""
        exp_neg = torch.exp(-1j * theta / 2)
        exp_pos = torch.exp(1j * theta / 2)
        return torch.tensor([
            [exp_neg, 0],
            [0, exp_pos]
        ], dtype=torch.complex128)

    @staticmethod
    def cnot():
        """CNOT gate."""
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex128)
```

**Expected Performance**:
- Single GPU: 12 qubits, 100-1000x faster than cloud QPU for small circuits
- 2 GPUs: 13 qubits, 500-2000x faster
- 4 GPUs: 14 qubits, 1000-5000x faster
- 8 GPUs: 15 qubits, 2000-10000x faster
- Zero network latency
- Full PyTorch gradient support

**Limitations**:
- Memory-bound: 2^n complex numbers (16 bytes each)
- 16 qubits = 1 GB RAM per GPU (requires 64 GPUs)
- Best for 4-14 qubit circuits

---

### 3. Intelligent Workload Router

**File**: `q_store/orchestration/workload_router.py` (NEW)

```python
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np

class ExecutionMode(Enum):
    """Execution modes for hybrid system."""
    GPU_ONLY = "gpu_only"           # Pure classical on GPU
    GPU_SIMULATION = "gpu_sim"      # Multi-GPU quantum simulation
    HYBRID = "hybrid"               # GPU preprocessing + QPU
    QPU_ONLY = "qpu_only"          # Pure quantum on cloud QPU


class WorkloadRouter:
    """
    Intelligent router for hybrid GPU-Quantum workloads.

    Automatically selects optimal execution path based on:
    - Problem size (qubits, circuit depth, dataset size)
    - Available resources (GPUs, QPU access)
    - Performance requirements (latency, accuracy)
    - Budget constraints
    """

    def __init__(
        self,
        available_gpus: int = None,
        qpu_available: bool = True,
        budget_limit: float = None,
        latency_requirement: str = 'balanced'  # 'low', 'balanced', 'flexible'
    ):
        """
        Initialize workload router.

        Args:
            available_gpus: Number of available GPUs (auto-detect if None)
            qpu_available: Whether cloud QPU is accessible
            budget_limit: Maximum cost per training run ($)
            latency_requirement: Latency sensitivity ('low', 'balanced', 'flexible')
        """
        self.available_gpus = available_gpus or torch.cuda.device_count()
        self.qpu_available = qpu_available
        self.budget_limit = budget_limit
        self.latency_requirement = latency_requirement

    def route(
        self,
        n_qubits: int,
        circuit_depth: int,
        n_samples: int,
        n_epochs: int = 10,
        accuracy_priority: str = 'balanced'  # 'quantum', 'balanced', 'speed'
    ) -> Dict[str, Any]:
        """
        Determine optimal execution mode and configuration.

        Returns:
            dict: {
                'mode': ExecutionMode,
                'backend': backend_config,
                'gpu_config': gpu_config,
                'estimated_time': seconds,
                'estimated_cost': dollars,
                'reasoning': explanation
            }
        """
        # Decision tree for routing
        decision = self._decision_tree(
            n_qubits, circuit_depth, n_samples, n_epochs, accuracy_priority
        )

        return decision

    def _decision_tree(self, n_qubits, circuit_depth, n_samples, n_epochs, accuracy_priority):
        """Execute decision tree logic."""

        # Rule 1: Small circuits (≤14 qubits) with GPUs available → GPU Simulation
        if n_qubits <= 14 and self.available_gpus >= 1:
            if self.latency_requirement == 'low':
                return {
                    'mode': ExecutionMode.GPU_SIMULATION,
                    'backend': 'multi_gpu_simulator',
                    'gpu_config': {
                        'n_gpus': min(2 ** (n_qubits - 12), self.available_gpus),
                        'device_ids': list(range(min(4, self.available_gpus)))
                    },
                    'estimated_time': self._estimate_gpu_sim_time(n_qubits, n_samples, n_epochs),
                    'estimated_cost': 0.0,  # Local GPU compute
                    'reasoning': f'Small circuit ({n_qubits} qubits) fits in multi-GPU simulation. '
                                 f'Zero network latency, 50-100x faster than cloud QPU.'
                }

        # Rule 2: Medium circuits (15-20 qubits) with QPU + budget → Hybrid
        if 15 <= n_qubits <= 20 and self.qpu_available:
            # Check budget
            estimated_cost = self._estimate_qpu_cost(n_qubits, n_samples, n_epochs)

            if self.budget_limit is None or estimated_cost <= self.budget_limit:
                return {
                    'mode': ExecutionMode.HYBRID,
                    'backend': 'ionq_aria',
                    'gpu_config': {
                        'n_gpus': self.available_gpus,
                        'preprocessing': True,
                        'feature_extraction': True
                    },
                    'qpu_config': {
                        'backend': 'ionq.simulator' if estimated_cost > 1000 else 'ionq.qpu.aria',
                        'shots': 1024
                    },
                    'estimated_time': self._estimate_hybrid_time(n_qubits, n_samples, n_epochs),
                    'estimated_cost': estimated_cost,
                    'reasoning': f'Medium circuit ({n_qubits} qubits). GPU preprocessing (5-10x speedup) + '
                                 f'cloud QPU for quantum layers. Cost: ${estimated_cost:.2f}'
                }
            else:
                # Budget exceeded → fallback to GPU-only classical
                return {
                    'mode': ExecutionMode.GPU_ONLY,
                    'backend': 'classical_cnn',
                    'gpu_config': {'n_gpus': 1},
                    'estimated_time': self._estimate_gpu_classical_time(n_samples, n_epochs),
                    'estimated_cost': 0.01,
                    'reasoning': f'Budget limit (${self.budget_limit}) exceeded (need ${estimated_cost:.2f}). '
                                 f'Falling back to classical GPU training.'
                }

        # Rule 3: Large circuits (>20 qubits) → QPU Only (if available and budget allows)
        if n_qubits > 20 and self.qpu_available:
            estimated_cost = self._estimate_qpu_cost(n_qubits, n_samples, n_epochs)

            if self.budget_limit is None or estimated_cost <= self.budget_limit:
                return {
                    'mode': ExecutionMode.QPU_ONLY,
                    'backend': 'ionq_forte',
                    'qpu_config': {
                        'backend': 'ionq.qpu.forte',
                        'shots': 1024,
                        'pricing': 'reserved' if estimated_cost > 5000 else 'pay_as_you_go'
                    },
                    'estimated_time': self._estimate_qpu_time(n_qubits, n_samples, n_epochs),
                    'estimated_cost': estimated_cost,
                    'reasoning': f'Large circuit ({n_qubits} qubits) requires QPU. IonQ Forte (36 qubits max). '
                                 f'Cost: ${estimated_cost:.2f}'
                }

        # Rule 4: Accuracy priority is 'speed' → GPU-only classical
        if accuracy_priority == 'speed':
            return {
                'mode': ExecutionMode.GPU_ONLY,
                'backend': 'classical_cnn',
                'gpu_config': {'n_gpus': min(4, self.available_gpus)},
                'estimated_time': self._estimate_gpu_classical_time(n_samples, n_epochs),
                'estimated_cost': 0.01,
                'reasoning': 'Speed priority selected. Classical GPU training (183-457x faster).'
            }

        # Default fallback: GPU simulation if available, else classical
        if self.available_gpus > 0:
            return {
                'mode': ExecutionMode.GPU_SIMULATION if n_qubits <= 14 else ExecutionMode.GPU_ONLY,
                'backend': 'multi_gpu_simulator' if n_qubits <= 14 else 'classical_cnn',
                'gpu_config': {'n_gpus': min(4, self.available_gpus)},
                'estimated_time': 30.0,
                'estimated_cost': 0.0,
                'reasoning': 'Default fallback to GPU-based processing.'
            }
        else:
            raise RuntimeError("No execution path available. No GPUs and QPU unavailable.")

    def _estimate_gpu_sim_time(self, n_qubits, n_samples, n_epochs):
        """Estimate GPU simulation time."""
        # Rough estimate: 0.1ms per circuit * samples * epochs
        circuits_per_sample = 4  # Typical hybrid architecture
        total_circuits = n_samples * n_epochs * circuits_per_sample
        time_per_circuit = 0.0001 * (2 ** (n_qubits - 10))  # Scales with qubit count
        return total_circuits * time_per_circuit

    def _estimate_qpu_cost(self, n_qubits, n_samples, n_epochs):
        """Estimate QPU cost."""
        circuits_per_sample = 4
        total_circuits = n_samples * n_epochs * circuits_per_sample
        cost_per_circuit = 0.30  # IonQ Aria pricing
        return total_circuits * cost_per_circuit

    def _estimate_hybrid_time(self, n_qubits, n_samples, n_epochs):
        """Estimate hybrid execution time."""
        # GPU preprocessing: 5-10x speedup
        # QPU execution: similar to v4.1.1 but with batching
        return 120.0  # Placeholder

    def _estimate_qpu_time(self, n_qubits, n_samples, n_epochs):
        """Estimate pure QPU time."""
        return 300.0  # Placeholder

    def _estimate_gpu_classical_time(self, n_samples, n_epochs):
        """Estimate classical GPU time."""
        # Based on benchmarks: ~1-5 seconds for 1000 samples, 5 epochs
        return (n_samples / 1000) * (n_epochs / 5) * 2.5
```

---

### 4. Hybrid Training Engine

**File**: `q_store/ml/hybrid_trainer.py` (NEW)

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class HybridQuantumClassicalModel(nn.Module):
    """
    Hybrid quantum-classical neural network.

    Architecture inspired by D-Wave DVAE approach:
    - Classical layers for high-dimensional preprocessing (on GPU)
    - Quantum layers for feature learning (on GPU sim or QPU)
    - Classical layers for postprocessing (on GPU)
    """

    def __init__(
        self,
        input_shape: tuple,
        n_classes: int,
        n_qubits: int = 8,
        quantum_backend: str = 'gpu_simulator',
        gpu_device: str = 'cuda:0'
    ):
        """
        Initialize hybrid model.

        Args:
            input_shape: Input data shape (e.g., (28, 28, 1) for MNIST)
            n_classes: Number of output classes
            n_qubits: Number of qubits for quantum layer
            quantum_backend: 'gpu_simulator', 'ionq_simulator', 'ionq_aria'
            gpu_device: GPU device for classical layers
        """
        super().__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.quantum_backend = quantum_backend
        self.device = torch.device(gpu_device)

        # Classical preprocessing layers (GPU)
        self.classical_encoder = nn.Sequential(
            nn.Conv2d(input_shape[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 2 ** n_qubits)  # Reduce to quantum input size
        ).to(self.device)

        # Quantum layer
        if quantum_backend == 'gpu_simulator':
            self.quantum_layer = MultiGPUQuantumLayer(n_qubits, n_gpus=torch.cuda.device_count())
        else:
            from q_store.layers import QuantumLayer
            self.quantum_layer = QuantumLayer(n_qubits, backend=quantum_backend)

        # Classical decoder layers (GPU)
        self.classical_decoder = nn.Sequential(
            nn.Linear(2 ** n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        ).to(self.device)

    def forward(self, x):
        """Forward pass through hybrid model."""
        # Classical encoding (GPU)
        x = x.to(self.device)
        encoded = self.classical_encoder(x)

        # Quantum processing
        quantum_output = self.quantum_layer(encoded)

        # Classical decoding (GPU)
        output = self.classical_decoder(quantum_output)

        return output


class MultiGPUQuantumLayer(nn.Module):
    """
    Quantum layer with multi-GPU simulation support.

    Wraps MultiGPUQuantumDevice for use in PyTorch models.
    """

    def __init__(self, n_qubits: int, n_gpus: int = 1, circuit_depth: int = 4):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_gpus = n_gpus
        self.circuit_depth = circuit_depth

        # Initialize quantum device
        self.qdevice = MultiGPUQuantumDevice(n_qubits, n_gpus)

        # Trainable parameters for quantum gates
        # RY and RZ gates for each qubit at each layer
        self.params = nn.Parameter(
            torch.randn(circuit_depth, n_qubits, 2) * 0.1  # Small random init
        )

    def forward(self, x):
        """
        Forward pass through quantum layer.

        Args:
            x: Input tensor of shape (batch_size, 2^n_qubits)

        Returns:
            Quantum measurement results (batch_size, 2^n_qubits)
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Encode input into quantum state (amplitude encoding)
            input_state = x[i]
            input_state = input_state / torch.norm(input_state)  # Normalize

            # Set quantum state
            self.qdevice.state = input_state.to(torch.complex128)

            # Apply parameterized quantum circuit
            for layer in range(self.circuit_depth):
                # Apply RY and RZ gates to each qubit
                for qubit in range(self.n_qubits):
                    ry_angle = self.params[layer, qubit, 0]
                    rz_angle = self.params[layer, qubit, 1]

                    ry_gate = QuantumGate.ry(ry_angle)
                    rz_gate = QuantumGate.rz(rz_angle)

                    self.qdevice.apply_gate(ry_gate, [qubit])
                    self.qdevice.apply_gate(rz_gate, [qubit])

                # Apply CNOT entangling gates
                for qubit in range(self.n_qubits - 1):
                    cnot = QuantumGate.cnot()
                    self.qdevice.apply_gate(cnot, [qubit, qubit + 1])

            # Measure quantum state (get probabilities)
            output_state = torch.abs(self.qdevice.get_statevector()) ** 2
            outputs.append(output_state)

        return torch.stack(outputs)


class HybridTrainer:
    """
    Enhanced trainer for hybrid GPU-Quantum models.

    Builds on QuantumTrainer from v4.1.1 with GPU optimizations.
    """

    def __init__(
        self,
        model: HybridQuantumClassicalModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda:0',
        workload_router: Optional[WorkloadRouter] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.router = workload_router or WorkloadRouter()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 10,
        verbose: bool = True
    ):
        """Train hybrid model."""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validation
            self.model.eval()
            val_loss, val_acc = self._validate_epoch(val_loader)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        return history

    def _train_epoch(self, loader):
        """Train one epoch."""
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        return total_loss / len(loader), correct / total

    def _validate_epoch(self, loader):
        """Validate one epoch."""
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        return total_loss / len(loader), correct / total
```

---

### 5. Performance Monitoring & Optimization

**File**: `q_store/monitoring/performance_monitor.py` (NEW)

```python
import time
import torch
from typing import Dict, Any
import psutil
import GPUtil

class PerformanceMonitor:
    """
    Real-time performance monitoring for hybrid GPU-Quantum training.

    Tracks:
    - GPU utilization and memory
    - QPU queue time and execution time
    - Data pipeline throughput
    - Cost accumulation
    """

    def __init__(self):
        self.metrics = {
            'gpu_util': [],
            'gpu_memory': [],
            'qpu_queue_time': [],
            'qpu_exec_time': [],
            'data_throughput': [],
            'cost': 0.0
        }
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()

    def log_gpu_metrics(self):
        """Log GPU utilization and memory."""
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            self.metrics['gpu_util'].append(gpu.load * 100)
            self.metrics['gpu_memory'].append(gpu.memoryUtil * 100)

    def log_qpu_metrics(self, queue_time: float, exec_time: float):
        """Log QPU performance metrics."""
        self.metrics['qpu_queue_time'].append(queue_time)
        self.metrics['qpu_exec_time'].append(exec_time)

    def log_data_throughput(self, samples_per_second: float):
        """Log data pipeline throughput."""
        self.metrics['data_throughput'].append(samples_per_second)

    def add_cost(self, cost: float):
        """Add to cumulative cost."""
        self.metrics['cost'] += cost

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'elapsed_time': elapsed,
            'avg_gpu_util': np.mean(self.metrics['gpu_util']) if self.metrics['gpu_util'] else 0,
            'avg_gpu_memory': np.mean(self.metrics['gpu_memory']) if self.metrics['gpu_memory'] else 0,
            'avg_qpu_queue': np.mean(self.metrics['qpu_queue_time']) if self.metrics['qpu_queue_time'] else 0,
            'avg_qpu_exec': np.mean(self.metrics['qpu_exec_time']) if self.metrics['qpu_exec_time'] else 0,
            'avg_throughput': np.mean(self.metrics['data_throughput']) if self.metrics['data_throughput'] else 0,
            'total_cost': self.metrics['cost']
        }

    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        print(f"Total Time: {summary['elapsed_time']:.2f} seconds")
        print(f"Avg GPU Utilization: {summary['avg_gpu_util']:.1f}%")
        print(f"Avg GPU Memory: {summary['avg_gpu_memory']:.1f}%")
        if summary['avg_qpu_queue'] > 0:
            print(f"Avg QPU Queue Time: {summary['avg_qpu_queue']:.2f}s")
            print(f"Avg QPU Exec Time: {summary['avg_qpu_exec']:.2f}s")
        print(f"Avg Data Throughput: {summary['avg_throughput']:.1f} samples/sec")
        print(f"Total Cost: ${summary['total_cost']:.2f}")
        print("="*60 + "\n")


class AdaptiveOptimizer:
    """
    Adaptive optimization based on runtime performance.

    Automatically adjusts:
    - Batch size based on GPU memory
    - Number of QPU workers based on queue time
    - Circuit caching strategy
    """

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor

    def optimize_batch_size(self, current_batch_size: int) -> int:
        """Optimize batch size based on GPU memory utilization."""
        summary = self.monitor.get_summary()
        gpu_memory = summary['avg_gpu_memory']

        if gpu_memory < 50:
            # Low GPU memory usage - increase batch size
            return int(current_batch_size * 1.5)
        elif gpu_memory > 90:
            # High GPU memory usage - decrease batch size
            return int(current_batch_size * 0.7)
        else:
            return current_batch_size

    def optimize_qpu_workers(self, current_workers: int) -> int:
        """Optimize QPU parallel workers based on queue time."""
        summary = self.monitor.get_summary()
        queue_time = summary['avg_qpu_queue']

        if queue_time > 5.0 and current_workers < 20:
            # High queue time - increase parallelism
            return current_workers + 5
        elif queue_time < 1.0 and current_workers > 5:
            # Low queue time - reduce overhead
            return current_workers - 2
        else:
            return current_workers
```

---

## Expected Performance Improvements

### Benchmark: Cats vs Dogs (1,000 images, 5 epochs)

| Configuration | Time per Epoch | Total Time | Speedup vs v4.1.1 | Cost | Use Case |
|---------------|----------------|------------|-------------------|------|----------|
| **v4.1.1 Baseline** | 457s | 2,288s | 1x | $0.00 | Current |
| **v4.2.0 GPU-Only** | 2s | 10s | **229x** | $0.01 | Speed priority |
| **v4.2.0 GPU Sim (8 qubits, 1 GPU)** | 15s | 75s | **31x** | $0.00 | Balanced |
| **v4.2.0 GPU Sim (8 qubits, 4 GPUs)** | 5s | 25s | **92x** | $0.00 | Performance |
| **v4.2.0 Hybrid (GPU + IonQ Aria)** | 60s | 300s | **7.6x** | $1,152 | Quantum research |
| **v4.2.0 QPU Only (IonQ Forte)** | 200s | 1,000s | **2.3x** | $4,480 | Large circuits |

### Performance Gains Breakdown

1. **GPU Data Pipeline**: 5-10x speedup
   - GPU-accelerated loading, preprocessing, augmentation
   - Eliminates CPU-GPU transfer bottlenecks

2. **Multi-GPU Quantum Simulation**: 50-100x speedup (vs cloud QPU)
   - Zero network latency
   - Local computation with PyTorch DTensor
   - Scales to 14 qubits with 4 GPUs

3. **Intelligent Routing**: 10-50x speedup
   - Automatic selection of optimal execution path
   - Avoids unnecessary QPU costs for small circuits

4. **Hybrid Architecture**: 3-10x speedup
   - Classical layers on GPU (80% of compute)
   - Quantum layers on best available backend (20% of compute)

---

## Implementation Roadmap

### Phase 1: GPU Acceleration (Weeks 1-2)

**Priority**: Critical

**Tasks**:
1. Implement `GPUDataPipeline` with PyTorch DataLoader optimization
2. Implement `GPUPreprocessor` with Kornia for GPU augmentation
3. Add GPU feature extraction with classical CNN
4. Benchmark GPU vs CPU preprocessing (expected: 10-50x speedup)
5. Unit tests for GPU pipeline

**Deliverables**:
- Complete `q_store/gpu/data_pipeline.py`
- GPU preprocessing 10-50x faster than CPU
- Zero-copy GPU data loading

---

### Phase 2: Multi-GPU Quantum Simulator (Weeks 3-4)

**Priority**: High

**Tasks**:
1. Implement `MultiGPUQuantumDevice` with PyTorch DTensor
2. Implement distributed statevector operations
3. Implement `QuantumGate` with autograd support
4. Add gradient computation for quantum circuits
5. Benchmark 1-GPU, 2-GPU, 4-GPU, 8-GPU configurations
6. Unit tests for quantum operations

**Deliverables**:
- Complete `q_store/gpu/quantum_simulator.py`
- Support 4-14 qubits with multi-GPU scaling
- 50-100x speedup vs cloud QPU for small circuits

---

### Phase 3: Workload Router (Week 5)

**Priority**: High

**Tasks**:
1. Implement `WorkloadRouter` decision tree
2. Add cost and performance estimation
3. Add automatic backend selection
4. Integrate with existing backend orchestration
5. Unit tests for routing logic

**Deliverables**:
- Complete `q_store/orchestration/workload_router.py`
- Automatic routing based on problem size, budget, latency

---

### Phase 4: Hybrid Training Engine (Weeks 6-7)

**Priority**: High

**Tasks**:
1. Implement `HybridQuantumClassicalModel`
2. Implement `MultiGPUQuantumLayer` with PyTorch autograd
3. Implement `HybridTrainer` with GPU optimization
4. Add end-to-end training pipeline
5. Integration tests with all execution modes

**Deliverables**:
- Complete `q_store/ml/hybrid_trainer.py`
- End-to-end differentiable hybrid models
- Support for all execution modes (GPU-only, GPU sim, hybrid, QPU-only)

---

### Phase 5: Performance Monitoring (Week 8)

**Priority**: Medium

**Tasks**:
1. Implement `PerformanceMonitor` with GPU/QPU metrics
2. Implement `AdaptiveOptimizer` for runtime optimization
3. Add real-time dashboards (optional)
4. Add cost tracking and budget alerts

**Deliverables**:
- Complete `q_store/monitoring/performance_monitor.py`
- Real-time performance tracking
- Adaptive batch size and worker optimization

---

### Phase 6: Testing & Benchmarking (Weeks 9-10)

**Priority**: High

**Tasks**:
1. Comprehensive unit tests (95%+ coverage)
2. Integration tests for all execution modes
3. Performance benchmarks vs v4.1.1
4. Scalability tests (1-8 GPUs)
5. Cost-performance analysis

**Deliverables**:
- Complete test suite
- Performance report with benchmarks
- Scalability analysis

---

### Phase 7: Documentation & Examples (Week 11)

**Priority**: Medium

**Tasks**:
1. API reference for new modules
2. Migration guide (v4.1.1 → v4.2.0)
3. Example: Fashion MNIST with GPU pipeline
4. Example: Multi-GPU quantum simulation
5. Example: Hybrid training workflow
6. Example: Workload routing tutorial

**Deliverables**:
- Complete documentation
- 6+ working examples
- Migration guide

---

## File Structure

```
q-store/
├── src/q_store/
│   ├── gpu/                           🆕 NEW
│   │   ├── __init__.py
│   │   ├── data_pipeline.py          🆕 GPU data loading & preprocessing
│   │   └── quantum_simulator.py      🆕 Multi-GPU quantum simulator
│   │
│   ├── orchestration/                 🆕 NEW
│   │   ├── __init__.py
│   │   └── workload_router.py        🆕 Intelligent workload routing
│   │
│   ├── ml/                            🔧 ENHANCED
│   │   ├── hybrid_trainer.py         🆕 Hybrid GPU-Quantum trainer
│   │   └── [existing v4.1.1 modules] ✅ Unchanged
│   │
│   ├── monitoring/                    🆕 NEW
│   │   ├── __init__.py
│   │   └── performance_monitor.py    🆕 Performance tracking
│   │
│   ├── data/                          ✅ FROM v4.1.1
│   ├── tracking/                      ✅ FROM v4.1.1
│   ├── tuning/                        ✅ FROM v4.1.1
│   └── [existing modules]             ✅ FROM v4.1.0
│
├── examples/
│   ├── hybrid_workflows/              🆕 NEW
│   │   ├── fashion_mnist_gpu_pipeline.py
│   │   ├── multi_gpu_quantum_sim.py
│   │   ├── hybrid_training_workflow.py
│   │   ├── workload_routing_demo.py
│   │   ├── performance_comparison.py
│   │   └── cost_optimization.py
│
├── tests/
│   ├── test_gpu/                      🆕 NEW
│   │   ├── test_data_pipeline.py
│   │   └── test_quantum_simulator.py
│   ├── test_orchestration/            🆕 NEW
│   │   └── test_workload_router.py
│   ├── test_ml/                       🔧 ENHANCED
│   │   └── test_hybrid_trainer.py    🆕 NEW
│   └── integration/                   🔧 ENHANCED
│       └── test_end_to_end_hybrid.py 🆕 NEW
│
└── docs/
    ├── Q-STORE_V4_2_0_ARCHITECTURE_DESIGN.md     🆕 THIS FILE
    ├── V4_2_0_MIGRATION_GUIDE.md                  🆕 TODO
    ├── HYBRID_TRAINING_GUIDE.md                   🆕 TODO
    ├── GPU_OPTIMIZATION_GUIDE.md                  🆕 TODO
    └── PERFORMANCE_BENCHMARKS_V4_2_0.md           🆕 TODO
```

---

## Dependencies

### New Dependencies

```txt
# GPU acceleration
torch>=2.2.0              # PyTorch with CUDA support
cupy-cuda12x>=12.0.0      # CuPy for GPU arrays
kornia>=0.7.0             # GPU-based image augmentation

# Distributed computing
torch.distributed         # Built-in PyTorch distributed (DTensor)

# Performance monitoring
GPUtil>=1.4.0             # GPU monitoring
psutil>=5.9.0             # System monitoring

# Optional: Multi-node distributed training
horovod>=0.28.0           # Multi-node GPU training (optional)
```

### Existing Dependencies (from v4.1.0, v4.1.1)

```txt
# Quantum backends
ionq-api-client
cirq
qiskit
pennylane

# Data management (v4.1.1)
datasets>=2.16.1
requests>=2.31.0
h5py>=3.10.0

# ML tracking (v4.1.1)
mlflow>=2.9.0
optuna>=3.5.0

# Core ML
numpy>=1.24.0
scipy>=1.10.0
```

---

## Backward Compatibility

### v4.1.1 Compatibility

- All v4.1.1 features preserved
- Data management layer unchanged
- Experiment tracking unchanged
- Hyperparameter tuning unchanged

### v4.1.0 Compatibility

- All v4.1.0 quantum ML features preserved
- Existing QuantumTrainer still works
- All backends compatible (IonQ, Cirq, Qiskit)

### Migration Path

**From v4.1.1 to v4.2.0**:
1. Install new GPU dependencies (`torch`, `cupy`, `kornia`)
2. Update imports: `from q_store.gpu import GPUDataPipeline`
3. Optional: Switch to `HybridTrainer` for GPU acceleration
4. Optional: Enable multi-GPU quantum simulation

**Minimal Changes**:
```python
# v4.1.1 code (still works in v4.2.0)
from q_store.data import DatasetLoader
from q_store.ml import QuantumTrainer

dataset = DatasetLoader.load(config)
trainer = QuantumTrainer(...)
trainer.train(dataset.x_train, dataset.y_train)

# v4.2.0 optimized code (new)
from q_store.data import DatasetLoader
from q_store.gpu import GPUDataPipeline
from q_store.ml import HybridTrainer

dataset = DatasetLoader.load(config)
gpu_pipeline = GPUDataPipeline(dataset, device='cuda:0')
hybrid_trainer = HybridTrainer(model, optimizer, loss_fn, device='cuda:0')
hybrid_trainer.train(gpu_pipeline.train_loader, gpu_pipeline.val_loader)
```

---

## Risk Analysis & Mitigation

### Technical Risks

**Risk 1: Multi-GPU Scaling Complexity**
- Concern: DTensor and distributed training can be complex
- Mitigation: Start with single GPU, progressively add multi-GPU support
- Fallback: Use PyTorch DDP instead of DTensor if needed

**Risk 2: GPU Memory Limitations**
- Concern: Large quantum circuits (>14 qubits) won't fit in GPU memory
- Mitigation: Automatic routing to cloud QPU for large circuits
- Monitoring: Real-time GPU memory tracking with alerts

**Risk 3: TorchQuantum-Dist Integration**
- Concern: Reimplementing TorchQuantum-Dist may be time-consuming
- Mitigation: Start with simplified version, gradually add features
- Alternative: Integrate actual TorchQuantum-Dist as dependency

### Cost Risks

**Risk 1: Unexpected QPU Costs**
- Concern: Budget overruns if routing logic fails
- Mitigation: Hard budget limits in WorkloadRouter
- Monitoring: Real-time cost tracking with alerts

**Risk 2: GPU Compute Costs**
- Concern: Cloud GPU costs (AWS, GCP) for multi-GPU setups
- Mitigation: Support local GPUs (no cloud cost)
- Alternative: Use free tier GPUs (Google Colab, Kaggle)

### Performance Risks

**Risk 1: GPU Speedup Lower Than Expected**
- Concern: GPU acceleration may not achieve 10-50x speedup
- Mitigation: Benchmarking in Phase 6, optimize hot paths
- Fallback: Classical-only mode still 183-457x faster than v4.1.1 quantum

**Risk 2: Multi-GPU Overhead**
- Concern: Communication overhead between GPUs
- Mitigation: Use NCCL backend for efficient GPU-GPU communication
- Monitoring: Track multi-GPU efficiency in performance monitoring

---

## Success Metrics

### Performance Targets

1. **GPU Data Pipeline**: 10-50x speedup vs CPU preprocessing
2. **Multi-GPU Quantum Sim**: 50-100x speedup vs cloud QPU (for 4-14 qubits)
3. **Hybrid Training**: 10-30x speedup vs v4.1.1 baseline
4. **Overall Speedup**: 20-100x improvement for typical workloads

### Quality Targets

1. **Code Coverage**: 95%+ for new modules
2. **Integration Tests**: All execution modes tested end-to-end
3. **Backward Compatibility**: 100% of v4.1.1 tests pass

### User Experience Targets

1. **Ease of Use**: <5 lines of code to enable GPU acceleration
2. **Automatic Optimization**: Workload router requires zero configuration
3. **Documentation**: Complete API reference + 6 examples

---

## Conclusion

Q-Store v4.2.0 transforms Q-Store from a pure quantum ML framework into a **high-performance hybrid GPU-Quantum system** that intelligently leverages the strengths of both classical GPU acceleration and quantum computing.

### Key Innovations

1. **GPU-Accelerated Pipeline**: 10-50x faster data preprocessing
2. **Multi-GPU Quantum Simulation**: 50-100x faster than cloud QPU for small circuits
3. **Intelligent Workload Routing**: Automatic optimization based on problem size and constraints
4. **Hybrid Training Architecture**: Seamless classical-quantum integration inspired by TorchQuantum-Dist and D-Wave DVAE

### Expected Impact

- **Performance**: 20-100x speedup for typical workloads
- **Cost**: Dramatically reduced QPU costs via local GPU simulation
- **Flexibility**: Support for GPU-only, hybrid, and QPU-only execution
- **Scalability**: Multi-GPU scaling for larger quantum circuits

### Next Steps

1. Review and approve this architecture design
2. Begin Phase 1: GPU acceleration pipeline
3. Iterative development with continuous benchmarking
4. Release Q-Store v4.2.0

---

**Document Version**: 1.0
**Status**: Draft for Review
**Target Release**: Q2 2026

**References**:
- TorchQuantum-Dist: https://github.com/ionq/torchquantum-dist
- D-Wave Image Generation: https://github.com/dwave-examples/image-generation
- Q-Store v4.1.1 Architecture: Q-STORE_V4_1_1_ARCHITECTURE_DESIGN.md
- Q-Store v4.1.1 Performance Report: Q-STORE_PERFORMANCE_REPORT.md
