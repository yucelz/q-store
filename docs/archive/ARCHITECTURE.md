# Q-Store Architecture Overview

## System Architecture

Q-Store is a quantum-native database that leverages quantum mechanical properties for enhanced vector search and ML training capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                     Q-Store Database                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Core DB    │  │  ML Training │  │   Backends   │     │
│  │  Operations  │  │   Pipeline   │  │  Abstraction │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                   │             │
│  ┌──────▼────────┬────────▼─────┬────────────▼───────┐    │
│  │  Quantum     │  Gradient    │  Hardware-Agnostic  │    │
│  │  Features    │  Computation │  Backend Interface  │    │
│  └──────────────┴──────────────┴─────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
  │  Pinecone │        │   IonQ    │        │   Cirq    │
  │  (Vector) │        │  (QPU)    │        │  (Sim)    │
  └───────────┘        └───────────┘        └───────────┘
```

## Module Structure

### 1. Core Components (`src/q_store/core/`)

**Purpose**: Core database operations and quantum state management

- **quantum_database.py**: Main database interface
  - QuantumDatabase: Primary API for database operations
  - DatabaseConfig: Configuration dataclass
  - QueryMode: Query execution modes (PRECISE, BALANCED, EXPLORATORY)
  - ConnectionPool: Manages classical and quantum backend connections

- **state_manager.py**: Quantum state lifecycle management
  - StateManager: Tracks and manages quantum states
  - Handles decoherence detection and state cleanup

- **entanglement_registry.py**: Entanglement relationship tracking
  - EntanglementRegistry: Manages entangled vector pairs
  - Tracks correlation strengths and patterns

- **tunneling_engine.py**: Quantum tunneling search
  - TunnelingEngine: Implements quantum tunneling for discovering hidden patterns
  - Allows "tunneling through" classical similarity barriers

### 2. Backend Abstraction (`src/q_store/backends/`)

**Purpose**: Hardware-agnostic quantum backend interface

- **quantum_backend_interface.py**: Abstract interface definitions
  - QuantumBackend (ABC): Base interface for all backends
  - GateType, BackendType enums
  - QuantumCircuit, QuantumGate data structures
  - Utility functions for circuit construction

- **backend_manager.py**: Backend lifecycle and factory
  - BackendManager: Factory and manager for backends
  - MockQuantumBackend: Local simulation backend
  - Backend selection and initialization logic

- **Adapters**:
  - cirq_ionq_adapter.py: Cirq/IonQ integration
  - qiskit_ionq_adapter.py: Qiskit/IonQ integration
  - ionq_backend.py: Legacy direct IonQ integration

**Design Pattern**: Adapter pattern for SDK integration

### 3. ML Training Components (`src/q_store/ml/`)

**Purpose**: Quantum machine learning training pipeline

**Core Training**:
- **quantum_trainer.py**: Main training orchestrator
  - QuantumTrainer: Manages full training lifecycle
  - TrainingConfig: Training hyperparameters
  - Integrates gradient computation, optimization, checkpointing

**Quantum Layers**:
- **quantum_layer.py**: Basic quantum neural network layer
- **quantum_layer_v2.py**: Hardware-efficient quantum layer (v3.3+)
  - HardwareEfficientQuantumLayer: Reduced parameter count

**Gradient Computation**:
- **gradient_computer.py**: Base gradient computation
- **spsa_gradient_estimator.py**: SPSA gradient estimation (v3.3)
- **parallel_spsa_estimator.py**: Parallelized SPSA
- **adaptive_optimizer.py**: Adaptive optimization strategies

**Performance Optimization (v3.3+)**:
- **circuit_cache.py**: Basic circuit caching
- **smart_circuit_cache.py**: Advanced template-based caching (v3.4)
- **circuit_batch_manager.py**: Circuit batching for parallel execution
- **circuit_batch_manager_v3_4.py**: Enhanced batching with native gates

**IonQ-Specific**:
- **ionq_batch_client.py**: Parallel IonQ API client
- **ionq_native_gate_compiler.py**: Compile to IonQ native gates

**Data Encoding**:
- **data_encoder.py**: Encode classical data into quantum states

**Monitoring**:
- **performance_tracker.py**: Training metrics and logging

### 4. Utilities

- **exceptions.py**: Custom exception hierarchy
  - QStoreError (base)
  - Backend errors: QuantumBackendError, CircuitExecutionError
  - Training errors: TrainingError, GradientComputationError
  - Database errors: DatabaseError, VectorStoreError

- **constants.py**: Centralized configuration constants
  - Default timeouts, batch sizes, learning rates
  - Backend limits and capabilities
  - Performance tuning parameters

## Data Flow

### 1. Query Flow (Classical + Quantum Enhancement)

```
User Query
    │
    ├──> Classical Search (Pinecone)
    │         │
    │         └──> Top-K Candidates
    │                   │
    └──> Quantum Enhancement (Optional)
              │
              ├──> Superposition Search
              ├──> Entanglement Discovery
              └──> Tunneling Search
                        │
                        └──> Final Results
```

### 2. Training Flow

```
Training Data
    │
    ├──> Data Encoding (quantum_data_encoder)
    │         │
    │         └──> Quantum States
    │                   │
    │         ┌─────────┴─────────┐
    │         │                   │
    ├──> Forward Pass      ──> Loss Computation
    │    (quantum_layer)
    │         │
    └──> Gradient Estimation
         (SPSA/Parameter Shift)
              │
              └──> Optimizer Update
                        │
                        └──> Next Iteration
```

### 3. Backend Selection Flow

```
User Config
    │
    ├──> SDK Selection (cirq/qiskit/mock)
    │         │
    │         └──> Adapter Selection
    │                   │
    │         ┌─────────┴─────────┐
    │         │                   │
    ├──> Mock Backend      ──> Real QPU/Simulator
    │    (Local)                  │
    │                             │
    │                    ┌────────┴────────┐
    │                    │                 │
    └──────────────> IonQ Aria      IonQ Forte
                   (25 qubits)     (32 qubits)
```

## Key Design Principles

### 1. Hardware Abstraction
- **Backend Interface**: All quantum operations go through QuantumBackend interface
- **Adapter Pattern**: SDK-specific code isolated in adapters
- **Graceful Degradation**: Falls back to mock backend if real hardware unavailable

### 2. Performance Optimization
- **Circuit Caching**: Avoid rebuilding identical circuits
- **Batch Execution**: Group circuits for parallel execution
- **SPSA Gradient**: 24-48x fewer circuit executions vs parameter shift
- **Native Gate Compilation**: Reduce gate count for IonQ hardware

### 3. Production Ready
- **Error Handling**: Custom exceptions for domain errors
- **Logging**: Structured logging throughout (no print statements)
- **Monitoring**: Performance tracking and metrics
- **Configuration**: Centralized constants and config validation

### 4. Extensibility

**Adding a New Backend**:
1. Create adapter implementing `QuantumBackend` interface
2. Register in `BackendManager`
3. Add to `setup_*_backends` factory function

**Adding a New Quantum Layer**:
1. Inherit from `QuantumLayer`
2. Implement `build_circuit()` method
3. Register in training pipeline

**Adding a New Gradient Method**:
1. Inherit from gradient estimator base
2. Implement `estimate_gradients()` method
3. Add to `QuantumTrainer` gradient method selection

## Version History

- **v3.4**: Smart caching, native gate compilation, batch API
- **v3.3**: SPSA gradients, hardware-efficient layers, circuit batching
- **v3.2**: Complete ML training pipeline
- **v3.1**: Hardware abstraction layer
- **v3.0**: Quantum enhancement features
- **v2.0**: Classical vector database integration

## Testing Strategy

### Unit Tests
- Backend adapters (mock mode)
- Quantum layers (circuit construction)
- Gradient estimators (mathematical correctness)
- State managers (lifecycle)

### Integration Tests
- End-to-end query flow
- Training pipeline
- Backend switching
- Error recovery

### Performance Tests
- Circuit cache hit rates
- Batch execution throughput
- Gradient estimation accuracy vs speed

## Configuration

### Environment Variables
```bash
PINECONE_API_KEY=<key>
IONQ_API_KEY=<key>
QSTORE_LOG_LEVEL=INFO
```

### DatabaseConfig
- Pinecone settings (index, dimension, metric)
- Quantum backend (SDK, target, API key)
- Performance tuning (batch sizes, cache sizes)
- Feature flags (enable quantum enhancements)

### TrainingConfig
- Model architecture (qubits, depth, entanglement)
- Training hyperparameters (learning rate, batch size, epochs)
- Optimization (gradient method, optimizer type)
- Performance (circuit caching, batch execution)

## References

- [Quantum Database Design](./Quantum-Native_Database_Architecture_v3_3_1_CORRECTED.md)
- [V3.4 Implementation](./V3_4_IMPLEMENTATION_COMPLETE.md)
- [V3.4 Quick Reference](./V3_4_QUICK_REFERENCE.md)
