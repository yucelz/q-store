# Q-Store Project Folder Structure

## Project Overview
Q-Store is a Quantum-Native Database system with ML capabilities, providing quantum computing backends and quantum machine learning functionality.

## Root Directory Structure

```
q-store/
├── .editorconfig                    # Editor configuration
├── .env.example                     # Environment variables template
├── .git/                            # Git repository data
├── .gitignore                       # Git ignore rules
├── .pre-commit-config.yaml          # Pre-commit hooks configuration
├── LICENCE                          # License file
├── FOLDER_STRUCTURE.md              # This file - project structure documentation
├── Makefile                         # Build automation commands
├── QUICKSTART.md                    # Quick start guide
├── README.md                        # Main project documentation
├── environment.yml                  # Conda environment specification
├── logo.svg                         # Project logo
├── pyproject.toml                   # Python project configuration (PEP 518)
├── config/                          # Configuration files (currently empty)
├── docs/                            # Documentation directory
├── examples/                        # Example applications and demonstrations
├── src/                             # Source code directory
└── tests/                           # Main test suite
```

---

## Source Code (`src/q_store/`)

Main package containing the core quantum database implementation.

```
src/q_store/
├── __init__.py                      # Package initialization
├── constants.py                     # Global constants and configuration
├── exceptions.py                    # Custom exception classes
├── backends/                        # Quantum backend implementations
├── core/                            # Core database functionality
└── ml/                              # Machine learning components
```

### Backends Module (`src/q_store/backends/`)

Quantum computing backend integrations for various platforms.

```
backends/
├── __init__.py                      # Backend module initialization
├── backend_manager.py               # Backend management and selection
├── cirq_ionq_adapter.py             # Cirq framework adapter for IonQ
├── ionq_backend.py                  # IonQ quantum computer backend
├── qiskit_ionq_adapter.py           # Qiskit framework adapter for IonQ
└── quantum_backend_interface.py     # Abstract base class for backends
```

### Core Module (`src/q_store/core/`)

Core quantum database engine components.

```
core/
├── __init__.py                      # Core module initialization
├── entanglement_registry.py         # Quantum entanglement tracking
├── quantum_database.py              # Main database implementation
├── state_manager.py                 # Quantum state management
└── tunneling_engine.py              # Quantum tunneling operations
```

### Machine Learning Module (`src/q_store/ml/`)

Quantum machine learning capabilities and training infrastructure.

```
ml/
├── __init__.py                      # ML module initialization
├── README.md                        # ML module documentation
├── adaptive_optimizer.py            # Adaptive optimization algorithms
├── circuit_batch_manager.py         # Circuit batch processing (stable)
├── circuit_batch_manager_v3_4.py    # Circuit batch processing (v3.4)
├── circuit_cache.py                 # Circuit caching mechanism
├── data_encoder.py                  # Quantum data encoding
├── gradient_computer.py             # Gradient computation
├── ionq_batch_client.py             # IonQ batch job client
├── ionq_native_gate_compiler.py     # IonQ native gate compiler
├── parallel_spsa_estimator.py       # Parallel SPSA gradient estimation
├── performance_tracker.py           # Performance monitoring
├── quantum_layer.py                 # Quantum neural network layer (stable)
├── quantum_layer_v2.py              # Quantum neural network layer (v2)
├── quantum_trainer.py               # Training orchestration
├── smart_circuit_cache.py           # Intelligent circuit caching
└── spsa_gradient_estimator.py       # SPSA gradient estimator
```

---

## Documentation (`docs/`)

Project documentation, architecture specs, and guides.

```
docs/
├── ARCHITECTURE.md                                      # System architecture overview
├── IMPLEMENTATION_GUIDE.md                              # Implementation guide
├── Quantum_Native_Database_Architecture_v3_4_DESIGN.md  # v3.4 design specification
├── V3_4_IMPLEMENTATION_COMPLETE.md                      # v3.4 implementation summary
├── V3_4_QUICK_REFERENCE.md                              # v3.4 quick reference
├── v3_4_ANALYSIS_SUMMARY.md                             # v3.4 analysis summary
├── quantum_benchmark_ui.html                            # Benchmark UI dashboard
├── archive/                                             # Archived documentation
│   ├── ANALYSIS_AND_ISSUES.md
│   ├── DIRECTORY_STRUCTURE.md
│   ├── ML_MODULE_MIGRATION.md
│   ├── PROJECT_STRUCTURE.md
│   ├── quantum_db_analysis.md
│   ├── quantum_db_design_doc.md
│   ├── Quantum-Native Database Architecture v2.0.md
│   ├── Quantum-Native Database Architecture v3.0.md
│   ├── Quantum-Native Database Architecture v3.1.md
│   ├── Quantum-Native_Database_Architecture_v3_2.md
│   ├── Quantum-Native_Database_Architecture_v3_3.md
│   ├── Quantum-Native_Database_Architecture_v3_3_1_CORRECTED.md
│   ├── README_v3_2.md
│   ├── README_v3_3.md
│   ├── RESTRUCTURING_SUMMARY.md
│   ├── V3_2_IMPLEMENTATION_SUMMARY.md
│   ├── V3_3_IMPLEMENTATION_SUMMARY.md
│   ├── V3_3_QUICK_REFERENCE.md
│   ├── V3.1_IMPLEMENTATION_SUMMARY.md
│   └── V3.1_UPGRADE_GUIDE.md
└── sphinx/                                              # Sphinx documentation
    ├── conf.py                                          # Sphinx configuration
    ├── index.rst                                        # Documentation index
    ├── overview.rst                                     # Overview page
    └── api/                                             # API documentation
        └── q_store.rst                                  # Q-Store API reference
```

---

## Examples (`examples/`)

Example applications, demonstrations, and benchmarks.

```
examples/
├── .env.example                     # Example environment variables
├── .gitignore                       # Examples-specific gitignore
├── .venv/                           # Virtual environment (not tracked)
├── LICENSE                          # Examples license
├── MANIFEST.in                      # Package manifest
├── Makefile                         # Examples build commands
├── README.md                        # Examples documentation
├── pytest.ini                       # Pytest configuration
├── quickstart_v3_2.py               # v3.2 quickstart script
├── react_train.jsonl                # React training dataset
├── requirements.txt                 # Python dependencies
├── requirements-minimal.txt         # Minimal dependencies
├── show_config.py                   # Configuration display utility
├── data/                            # Data files
│   └── .gitkeep                     # Keep directory in git
├── docs/                            # Examples documentation
│   ├── README.md
│   └── archive/                     # Archived example docs
├── LOG/                             # Log files and benchmarks
│   ├── benchmark-examples_v3_4-*.json  # Benchmark results
│   └── log-*.txt                    # Execution logs
├── scripts/                         # Utility scripts
│   ├── demo_logging.py
│   ├── demo_v3_3_1_fix.py
│   ├── run_react_training.sh
│   ├── verify_installation.py
│   ├── verify_react_integration.py
│   ├── verify_structure.py
│   └── verify_v3_2.py
├── src/q_store_examples/            # Examples package
│   ├── __init__.py
│   ├── README_UTILS.md              # Utilities documentation
│   ├── basic_example.py             # Basic usage example
│   ├── examples_v31.py              # v3.1 examples
│   ├── examples_v3_2.py             # v3.2 examples
│   ├── examples_v3_3.py             # v3.3 examples
│   ├── examples_v3_3_1.py           # v3.3.1 examples
│   ├── examples_v3_4.py             # v3.4 examples
│   ├── financial_example.py         # Financial use case
│   ├── ml_training_example.py       # ML training demo
│   ├── quantum_db_quickstart.py     # Quickstart guide
│   ├── react_dataset_generator.py   # React dataset generation
│   ├── utils.py                     # Utility functions
│   └── utils_optimized.py           # Optimized utilities
└── tests/                           # Example tests
    ├── __init__.py
    ├── test_basic.py
    ├── test_cirq_adapter_fix.py
    ├── test_fixes.py
    ├── test_logging.py
    ├── test_pinecone_ionq_connection.py
    ├── test_v31.py
    └── test_v3_3.py
```

---

## Tests (`tests/`)

Main test suite for the q_store package.

```
tests/
└── test_quantum_database.py         # Quantum database tests
```

---

## Key Components Description

### Backend System
- **Backend Manager**: Centralized backend selection and lifecycle management
- **IonQ Backend**: Native IonQ quantum computer integration
- **Framework Adapters**: Support for both Cirq and Qiskit frameworks
- **Backend Interface**: Abstract interface for implementing new backends

### Core Database
- **Quantum Database**: Main database API and operations
- **State Manager**: Quantum state lifecycle and persistence
- **Entanglement Registry**: Tracks quantum entanglement relationships
- **Tunneling Engine**: Quantum tunneling-based operations

### Machine Learning
- **Quantum Layer**: Parameterized quantum circuits as neural network layers
- **Circuit Batch Manager**: Efficient batch processing for quantum circuits
- **Data Encoder**: Classical-to-quantum data encoding strategies
- **Gradient Estimators**: SPSA and parallel gradient computation
- **Performance Tracking**: Training metrics and optimization monitoring
- **Circuit Caching**: Intelligent caching for circuit reuse

### Documentation
- **Architecture Docs**: System design and component interactions
- **Implementation Guides**: Step-by-step implementation instructions
- **Version History**: Archived documentation from previous versions
- **Sphinx Docs**: Auto-generated API documentation

### Examples
- **Version Examples**: Examples for each major version (v3.1 - v3.4)
- **Use Case Demos**: Financial, ML training, and basic usage examples
- **Verification Scripts**: Installation and integration verification
- **Benchmarks**: Performance measurement and logging

---

## Development Workflow

1. **Source Code**: Located in `src/q_store/`
2. **Tests**: Main tests in `tests/`, example tests in `examples/tests/`
3. **Documentation**: Keep `docs/` updated with architectural changes
4. **Examples**: Add new examples to `examples/src/q_store_examples/`
5. **Configuration**: Project config in `pyproject.toml`, environment in `environment.yml`

---

## Version Information

This structure represents Q-Store v3.4 architecture as of December 16, 2025.

For detailed implementation information, see:
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [V3_4_QUICK_REFERENCE.md](docs/V3_4_QUICK_REFERENCE.md) - v3.4 quick reference
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
