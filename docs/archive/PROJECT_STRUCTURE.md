# Q-Store Project Structure

This document describes the standardized Python project structure for Q-Store.

## Directory Layout

```
q-store/
├── src/                          # Source code (PEP 420 namespace)
│   └── q_store/                  # Main package
│       ├── __init__.py           # Package initialization and public API
│       ├── core/                 # Core quantum database components
│       │   ├── __init__.py
│       │   ├── quantum_database.py    # Main database implementation
│       │   ├── state_manager.py       # Quantum state management
│       │   ├── entanglement_registry.py  # Entanglement tracking
│       │   └── tunneling_engine.py    # Quantum tunneling implementation
│       ├── backends/             # Quantum backend implementations
│       │   ├── __init__.py
│       │   └── ionq_backend.py   # IonQ quantum hardware interface
│       └── utils/                # Utility functions and helpers
│           └── __init__.py
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_quantum_database.py  # Database tests
│   └── conftest.py               # Pytest configuration (if needed)
│
├── docs/                         # Documentation
│   ├── README.md                 # Documentation index
│   ├── QUICKSTART.md            # Quick start guide
│   ├── quantum_db_design_v2.md  # Design specification v2
│   └── quantum_db_design_doc.md # Original design document
│
├── examples/                     # Example implementations
│   ├── src/                     # Example source code
│   ├── tests/                   # Example tests
│   ├── scripts/                 # Helper scripts
│   └── docs/                    # Example documentation
│
├── config/                       # Configuration files (optional)
│
├── .github/                      # GitHub specific files (workflows, etc.)
│
├── pyproject.toml               # Modern Python project configuration (PEP 621)
├── setup.py                     # Legacy setup (for compatibility)
├── MANIFEST.in                  # Distribution file inclusion rules
├── Makefile                     # Development task automation
├── .editorconfig               # Editor configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
├── LICENCE                     # CC BY-NC 4.0 License
├── environment.yml             # Conda environment specification
├── verify_installation.py      # Installation verification script
├── .env.example               # Environment variables template
└── .env                       # Local environment variables (gitignored)
```

## Structure Rationale

### `src/` Layout (PEP 420)
- **Why**: Prevents accidental imports from working directory
- **Benefit**: Forces proper package installation for testing
- **Standard**: Modern Python packaging best practice

### Package Organization

#### `core/` - Core Components
Contains the fundamental quantum database functionality:
- Database engine
- State management
- Entanglement tracking
- Tunneling algorithms

#### `backends/` - External Integrations
Quantum hardware and classical database backends:
- IonQ quantum backend
- Future: Other quantum providers
- Future: Classical vector databases

#### `utils/` - Utilities
Helper functions, decorators, and common tools

### Configuration Files

#### `pyproject.toml` (Primary)
- Modern standard (PEP 621)
- All configuration in one file
- Tool configuration (black, mypy, pytest, etc.)

#### `setup.py` (Legacy Support)
- Minimal implementation
- Defers to pyproject.toml
- Maintains backward compatibility

#### `MANIFEST.in`
- Controls distribution file inclusion
- Ensures documentation is packaged

#### `.editorconfig`
- Cross-editor consistency
- Enforces coding standards

### Development Files

#### `Makefile`
Common development tasks:
- `make install-dev` - Development setup
- `make test` - Run tests
- `make format` - Auto-format code
- `make verify` - Full validation

#### `environment.yml`
Conda environment specification for reproducibility

## Best Practices Implemented

### 1. **Separation of Concerns**
- Source code in `src/`
- Tests in `tests/`
- Examples in `examples/`
- Documentation in `docs/`

### 2. **Type Safety**
- Type hints throughout codebase
- MyPy configuration in pyproject.toml
- Strict type checking enabled

### 3. **Code Quality**
- Black for formatting (100 char line length)
- isort for import sorting
- flake8 for linting
- pylint for additional checks

### 4. **Testing**
- pytest framework
- Async test support
- Coverage reporting
- Test markers (unit, integration, slow)

### 5. **Documentation**
- README at project root
- Detailed docs in docs/
- Inline docstrings
- Architecture documentation

### 6. **Versioning**
- Semantic versioning (SemVer)
- Version in `__init__.py` and `pyproject.toml`
- Changelog tracking (recommended)

### 7. **Distribution**
- Source distribution support
- Wheel distribution support
- Optional dependencies (backends, dev, docs)
- Proper dependency specification

## Migration Notes

### From Old Structure
```
Old:                          New:
q_store/                  →   src/q_store/core/
├── quantum_database.py   →   ├── quantum_database.py
├── state_manager.py      →   ├── state_manager.py
├── ionq_backend.py       →   src/q_store/backends/ionq_backend.py
```

### Import Changes
```python
# Old imports (still work via __init__.py)
from q_store import QuantumDatabase

# New structure (internal)
from q_store.core.quantum_database import QuantumDatabase
from q_store.backends.ionq_backend import IonQQuantumBackend
```

### Installation
```bash
# Development installation with new structure
pip install -e ".[dev,backends]"

# Or using Make
make install-dev
```

## Future Enhancements

### Planned Additions
1. **CI/CD**: `.github/workflows/` for automated testing
2. **Docker**: `Dockerfile` and `docker-compose.yml`
3. **Benchmarks**: `benchmarks/` directory
4. **Changelog**: `CHANGELOG.md` for version tracking
5. **Contributing**: `CONTRIBUTING.md` guide
6. **Security**: `SECURITY.md` policy

### Additional Backends
- `backends/pinecone_backend.py`
- `backends/qdrant_backend.py`
- `backends/pgvector_backend.py`

### Utilities Expansion
- `utils/logging.py` - Logging configuration
- `utils/metrics.py` - Performance metrics
- `utils/validators.py` - Input validation

## References

- [PEP 621](https://peps.python.org/pep-0621/) - Storing project metadata in pyproject.toml
- [PEP 517](https://peps.python.org/pep-0517/) - Build system interface
- [PEP 518](https://peps.python.org/pep-0518/) - Specifying build dependencies
- [Python Packaging Guide](https://packaging.python.org/)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/structure/)
