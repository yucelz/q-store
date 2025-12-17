# Q-Store Quick Start Guide

Get up and running with Q-Store in 5 minutes!

## Prerequisites

- Python 3.11 or 3.12
- Conda package manager (recommended) or pip
- Pinecone account (free tier available)

## Understanding the Package Structure

Q-Store follows modern Python packaging standards with a `src/` layout:

```
q-store/
├── src/
│   └── q_store/              # Main package
│       ├── __init__.py       # Package exports
│       ├── core/             # Core quantum components
│       │   ├── quantum_database.py
│       │   ├── state_manager.py
│       │   ├── entanglement_registry.py
│       │   └── tunneling_engine.py
│       ├── backends/         # Backend implementations
│       │   └── ionq_backend.py
│       └── utils/            # Utility functions
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── pyproject.toml           # Modern Python package configuration
├── setup.py                 # Backward compatibility
└── environment.yml          # Conda environment specification
```

## Installation Methods

### Method 1: Conda Environment (Recommended)

This method creates an isolated environment with all dependencies:

```bash
# Clone the repository
git clone https://github.com/yucelz/q-store.git
cd q-store

# Create and activate conda environment
conda env create -f environment.yml
conda activate q-store

# Install the package in development mode
pip install -e .
```

### Method 2: Development Installation

For contributors and developers who need all development tools:

```bash
# Clone and navigate
git clone https://github.com/yucelz/q-store.git
cd q-store

# Create conda environment (or use venv)
conda create -n q-store python=3.11
conda activate q-store

# Install with development dependencies
pip install -e ".[dev,backends]"
```

### Method 3: Production Installation

For production deployments without development tools:

```bash
# Using pip directly
pip install git+https://github.com/yucelz/q-store.git

# Or from local clone
git clone https://github.com/yucelz/q-store.git
cd q-store
pip install .

# With backend support (Pinecone, pgvector, Qdrant)
pip install ".[backends]"
```

### Method 4: Minimal Installation

Install only core dependencies (for custom backend implementations):

```bash
pip install git+https://github.com/yucelz/q-store.git
# Core: numpy, scipy, cirq, cirq-ionq, requests
```

## Installation Options Explained

Q-Store uses optional dependencies for flexibility:

| Option | Command | Includes |
|--------|---------|----------|
| **Core** | `pip install .` | numpy, scipy, cirq, cirq-ionq, requests |
| **Backends** | `pip install ".[backends]"` | + pinecone-client, pgvector, qdrant-client |
| **Development** | `pip install ".[dev]"` | + pytest, black, mypy, flake8, etc. |
| **Documentation** | `pip install ".[docs]"` | + sphinx, sphinx-rtd-theme |
| **All** | `pip install ".[all]"` | Everything above |

## Build and Distribution

### For Library Development

```bash
# Install in editable mode (changes reflect immediately)
pip install -e .

# Run tests
make test
# or
pytest tests/ -v

# Run with coverage
make test-cov

# Format code
make format

# Check types
make type-check

# Run all checks
make verify
```

### For Production Build

```bash
# Build distribution packages
make build
# or
python -m build

# This creates:
# - dist/q_store-2.0.0-py3-none-any.whl (wheel)
# - dist/q_store-2.0.0.tar.gz (source)

# Install the built wheel
pip install dist/q_store-2.0.0-py3-none-any.whl
```

### For Testing Different Environments

```bash
# Test in isolated environment
python -m venv test-env
source test-env/bin/activate
pip install dist/q_store-2.0.0-py3-none-any.whl
python -c "from q_store import QuantumDatabase; print('Success!')"
deactivate
```

## Makefile Commands Reference

Q-Store includes a comprehensive Makefile for common tasks:

```bash
make help          # Show all available commands
make install       # Install package (production mode)
make install-dev   # Install with dev dependencies
make clean         # Remove build artifacts and cache
make test          # Run test suite
make test-cov      # Run tests with coverage report
make lint          # Run linters (flake8, pylint)
make format        # Format code (black, isort)
make format-check  # Check formatting without changes
make type-check    # Run mypy type checking
make build         # Build distribution packages
make verify        # Run all checks (CI-ready)
```

## Step 1: Get API Keys

### Pinecone (Required)
1. Sign up at [pinecone.io](https://www.pinecone.io/)
2. Create a free account
3. Go to your dashboard and copy your API key
4. Note your environment (e.g., `us-east-1`)

### IonQ (Optional - for quantum features)
1. Sign up at [cloud.ionq.com](https://cloud.ionq.com/)
2. Go to Settings → API Keys
3. Copy your API key

## Step 2: Configure Environment

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
# Required: Pinecone credentials
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Optional: IonQ for quantum features
IONQ_API_KEY=your_ionq_api_key_here
EOF
```

**Important:** Replace `your_pinecone_api_key_here` with your actual API key!

## Step 3: Verify Installation

**Verify your installation:**

```bash
python verify_installation.py
```

You should see:
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

✓ All checks passed!
```

**Verify package structure:**

```bash
python verify_structure.py
```

**Run the test suite:**

```bash
# Quick test
pytest tests/ -v

# With coverage
make test-cov

# Specific test file
pytest tests/test_quantum_database.py -v
```

## Step 4: Run Your First Demo

```bash
python examples/quantum_db_quickstart.py
```

You should see output like:

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

Initializing database...
INFO:q_store.quantum_database:Pinecone initialized with environment: us-east-1
INFO:q_store.quantum_database:Creating Pinecone index: quantum-demo
✓ Database initialized successfully

=== Example 1: Basic Operations ===
Creating sample embeddings...
Inserting documents...
  ✓ Inserted doc_1
  ✓ Inserted doc_2
  ✓ Inserted doc_3
...
```

## Package Import Structure

After installation, you can import from `q_store`:

```python
# Main database class and configuration
from q_store import QuantumDatabase, DatabaseConfig

# Query types and results
from q_store import QueryMode, QueryResult, Metrics

# Quantum backends
from q_store import IonQQuantumBackend

# State management
from q_store import StateManager, QuantumState, StateStatus

# Advanced features
from q_store import EntanglementRegistry, TunnelingEngine

# Connection management
from q_store import ConnectionPool, MockPineconeIndex
```

All public APIs are exported through `src/q_store/__init__.py` for clean imports.

## Step 5: Your First Program

Create `my_first_qstore.py`:

```python
import asyncio
import numpy as np
from dotenv import load_dotenv
import os
from q_store import QuantumDatabase, DatabaseConfig

# Load environment variables from .env
load_dotenv()

async def main():
    # Configure database
    config = DatabaseConfig(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
        pinecone_index_name='my-first-index',
        pinecone_dimension=128,  # Small dimension for testing
        ionq_api_key=os.getenv('IONQ_API_KEY'),  # Optional
    )
    
    # Initialize database
    db = QuantumDatabase(config)
    
    async with db.connect():
        print("✓ Connected to Q-Store!")
        
        # Insert a vector
        embedding = np.random.rand(128)
        await db.insert(
            id='test_doc_1',
            vector=embedding,
            metadata={'type': 'test', 'timestamp': '2025-01-01'}
        )
        print("✓ Inserted vector")
        
        # Query similar vectors
        results = await db.query(
            vector=embedding,
            top_k=5
        )
        print(f"✓ Found {len(results)} similar vectors")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")

if __name__ == '__main__':
    asyncio.run(main())
```

Run it:

```bash
python my_first_qstore.py
```

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yucelz/q-store.git
cd q-store

# Create conda environment
conda env create -f environment.yml
conda activate q-store

# Install in development mode with all extras
pip install -e ".[dev,backends]"

# Verify installation
python verify_installation.py
make test
```

### Making Changes

```bash
# 1. Make your code changes in src/q_store/

# 2. Format code
make format

# 3. Check types
make type-check

# 4. Run linters
make lint

# 5. Run tests
make test

# 6. Run all checks (recommended before commit)
make verify
```

### Testing GitHub Actions Workflows Locally

Before pushing changes that trigger GitHub Actions, test workflows locally using `act`:

#### Quick Setup

```bash
# Install act (one-time setup)
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or if act is already installed, verify
act --version
```

#### Quick Validation (Recommended)

```bash
# Validate workflow syntax (fast, no Docker required)
./scripts/test_workflow_locally.sh test-syntax

# See what would execute (dry run)
./scripts/test_workflow_locally.sh dry-run

# List all workflows and jobs
./scripts/test_workflow_locally.sh list
```

#### Full Workflow Testing

```bash
# Test Linux build workflow (requires Docker)
./scripts/test_workflow_locally.sh build

# Test specific workflow
./scripts/test_workflow_locally.sh build-wheels.yml

# Test specific job in a workflow
./scripts/test_workflow_locally.sh build-wheels.yml build_wheels
```

#### Available Commands

```bash
./scripts/test_workflow_locally.sh list          # List all workflows and jobs
./scripts/test_workflow_locally.sh test-syntax   # Validate YAML syntax
./scripts/test_workflow_locally.sh dry-run       # Show execution plan
./scripts/test_workflow_locally.sh build         # Test build workflow (Linux)
./scripts/test_workflow_locally.sh windows       # Test Windows workflow (dry run)
```

#### Important Notes

- ✅ **Syntax validation** is fast and doesn't require Docker
- ✅ **Dry runs** show what would execute without running
- ⚠️ **Full builds** require Docker and can take 30+ minutes
- ⚠️ **Windows/macOS** workflows can only be syntax-checked on Linux
- 💡 **Always run** `test-syntax` before pushing to GitHub

#### Typical Development Workflow

```bash
# 1. Make changes to workflow files
vim .github/workflows/build-wheels.yml

# 2. Validate syntax
./scripts/test_workflow_locally.sh test-syntax

# 3. See what would run
./scripts/test_workflow_locally.sh dry-run

# 4. (Optional) Full local test
./scripts/test_workflow_locally.sh build

# 5. Commit and push
git add .github/workflows/
git commit -m "ci: update build workflow"
git push
```

### Testing Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_quantum_database.py -v

# Run with coverage
pytest tests/ --cov=src/q_store --cov-report=html

# Run specific test function
pytest tests/test_quantum_database.py::test_basic_insert -v

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

### Building for Distribution

```bash
# Clean previous builds
make clean

# Build distribution packages
make build

# Check the built packages
ls -lh dist/
# dist/q_store-2.0.0-py3-none-any.whl
# dist/q_store-2.0.0.tar.gz

# Test installation from wheel
pip install dist/q_store-2.0.0-py3-none-any.whl

# Upload to PyPI (maintainers only)
# pip install twine
# twine upload dist/*
```

## Production Deployment

### Docker Deployment (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy package files
COPY pyproject.toml setup.py ./
COPY src/ src/

# Install package with backends
RUN pip install --no-cache-dir ".[backends]"

# Copy application code
COPY your_app/ your_app/

# Set environment variables
ENV PINECONE_API_KEY=""
ENV PINECONE_ENVIRONMENT="us-east-1"

CMD ["python", "your_app/main.py"]
```

### Using in Requirements Files

```txt
# requirements.txt - for pip
q-store @ git+https://github.com/yucelz/q-store.git@v2.0.0
# or after PyPI release:
# q-store==2.0.0

# With extras
q-store[backends] @ git+https://github.com/yucelz/q-store.git@v2.0.0
```

```yaml
# environment.yml - for conda
name: my-app
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
    - q-store @ git+https://github.com/yucelz/q-store.git@v2.0.0
```

## Testing in Different Environments

### Test with Different Python Versions

```bash
# Python 3.11
conda create -n q-store-py311 python=3.11
conda activate q-store-py311
pip install -e ".[dev,backends]"
make test
conda deactivate

# Python 3.12
conda create -n q-store-py312 python=3.12
conda activate q-store-py312
pip install -e ".[dev,backends]"
make test
conda deactivate
```

### Test Production Build

```bash
# Create fresh environment
python -m venv test-prod-env
source test-prod-env/bin/activate

# Install from wheel
pip install dist/q_store-2.0.0-py3-none-any.whl

# Test import
python -c "from q_store import QuantumDatabase; print('✓ Import successful')"

# Run basic test
python examples/src/q_store_examples/basic_example.py

deactivate
rm -rf test-prod-env
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'q_store'`

**Solution:**
```bash
# Make sure package is installed
pip install -e .

# Check installation
pip list | grep q-store

# Verify Python can find it
python -c "import q_store; print(q_store.__version__)"
```

### Development Install Issues

**Problem:** Changes not reflecting after editing code

**Solution:**
```bash
# Ensure you're using editable install
pip install -e .

# NOT: pip install .  (this would require reinstall after each change)
```

### Backend Dependencies

**Problem:** `ModuleNotFoundError: No module named 'pinecone'`

**Solution:**
```bash
# Install backend extras
pip install -e ".[backends]"

# Or install pinecone directly
pip install "pinecone-client>=3.0.0"
```

### Test Failures

**Problem:** Tests fail with import errors

**Solution:**
```bash
# Install development dependencies
pip install -e ".[dev,backends]"

# Make sure pytest-asyncio is installed
pip install pytest-asyncio

# Run with correct async mode
pytest tests/ -v --asyncio-mode=auto
```

### Build Errors

**Problem:** `python -m build` fails

**Solution:**
```bash
# Install build tools
pip install build wheel setuptools

# Clean and rebuild
make clean
make build
```

## What's Next?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 💡 Explore [examples/](examples/) for more use cases
- 🏗️ Review [docs/quantum_db_design_v2.md](docs/quantum_db_design_v2.md) for architecture
- 🧪 Run tests: `make test` or `pytest tests/ -v`
- 📦 Check [pyproject.toml](pyproject.toml) for package configuration
- 🔧 Use [Makefile](Makefile) for development commands

## Common Commands Cheat Sheet

```bash
# Environment Management
conda activate q-store                    # Activate environment
conda deactivate                          # Deactivate environment
conda env update -f environment.yml       # Update environment

# Installation
pip install -e .                          # Install (editable)
pip install -e ".[dev,backends]"          # Install with extras
make install-dev                          # Install for development

# Development
make format                               # Format code
make lint                                 # Run linters  
make type-check                           # Check types
make test                                 # Run tests
make test-cov                             # Tests with coverage
make verify                               # Run all checks
make clean                                # Clean build artifacts

# Workflow Testing (Local)
./scripts/test_workflow_locally.sh test-syntax   # Validate workflows
./scripts/test_workflow_locally.sh dry-run       # Show execution plan
./scripts/test_workflow_locally.sh list          # List all jobs
./scripts/test_workflow_locally.sh build         # Test build locally

# Building
make build                                # Build distribution
python -m build                           # Build manually

# Running Examples
python examples/src/q_store_examples/quantum_db_quickstart.py
python examples/src/q_store_examples/basic_example.py
python verify_installation.py

# Testing
pytest tests/ -v                          # All tests verbose
pytest tests/test_quantum_database.py     # Specific file
pytest -k "test_insert"                   # Tests matching pattern
pytest --asyncio-mode=auto                # Async tests
```

## Package Information

- **Package Name:** `q-store`
- **Version:** 2.0.0
- **Python:** 3.11+
- **License:** CC BY-NC 4.0
- **Repository:** https://github.com/yucelz/q-store
- **Website:** http://www.q-store.tech
- **Package Structure:** `src/` layout (PEP 420)
- **Build System:** setuptools (PEP 517/518)
- **Configuration:** pyproject.toml (PEP 621)

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'q_store'`

**Solution:**
```bash
# Make sure package is installed
pip install -e .

# Check installation
pip list | grep q-store

# Verify Python can find it
python -c "import q_store; print(q_store.__version__)"
```

### Development Install Issues

**Problem:** Changes not reflecting after editing code

**Solution:**
```bash
# Ensure you're using editable install
pip install -e .

# NOT: pip install .  (this would require reinstall after each change)
```

### Backend Dependencies

**Problem:** `ModuleNotFoundError: No module named 'pinecone'`

**Solution:**
```bash
# Install backend extras
pip install -e ".[backends]"

# Or install pinecone directly
pip install "pinecone-client>=3.0.0"
```

### Test Failures

**Problem:** Tests fail with import errors

**Solution:**
```bash
# Install development dependencies
pip install -e ".[dev,backends]"

# Make sure pytest-asyncio is installed
pip install pytest-asyncio

# Run with correct async mode
pytest tests/ -v --asyncio-mode=auto
```

### Build Errors

**Problem:** `python -m build` fails

**Solution:**
```bash
# Install build tools
pip install build wheel setuptools

# Clean and rebuild
make clean
make build
```

### Environment Variables

**Problem:** `PINECONE_API_KEY not found`

**Solution:**
```bash
# Make sure .env file exists in project root
ls -la .env

# Check contents (without exposing keys)
grep -v "^#" .env | grep "PINECONE_API_KEY"

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

### Need Help?

- 📖 Documentation: [docs/](docs/)
- 🐛 GitHub Issues: [github.com/yucelz/q-store/issues](https://github.com/yucelz/q-store/issues)
- 📧 Email: yucelz@gmail.com
- 🌐 Website: [www.q-store.tech](http://www.q-store.tech)

---

**Happy Quantum Computing! 🚀**
