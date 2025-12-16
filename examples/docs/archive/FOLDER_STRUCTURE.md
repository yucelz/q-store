# Python Best Practices Folder Structure

This document explains the folder structure for the Q-Store Examples project, following Python best practices.

## 📁 Directory Structure

```
q-store/examples/
│
├── src/                          # Source code (PEP 420 namespace package)
│   └── q_store_examples/         # Main package
│       ├── __init__.py           # Package initialization
│       ├── basic_example.py      # Basic Q-Store usage
│       ├── financial_example.py  # Financial applications
│       ├── quantum_db_quickstart.py  # Comprehensive tutorial
│       ├── ml_training_example.py    # ML integration
│       ├── tinyllama_react_training.py  # React LLM training
│       └── react_dataset_generator.py   # Dataset generation
│
├── tests/                        # Test suite
│   ├── __init__.py              # Test package marker
│   └── test_basic.py            # Basic tests
│
├── scripts/                      # Utility scripts
│   ├── verify_installation.py   # Installation checker
│   ├── verify_react_integration.py  # React integration checker
│   ├── verify_tinyllama_example.py  # TinyLlama checker
│   └── run_react_training.sh    # React training automation
│
├── docs/                         # Documentation
│   ├── INDEX.md                 # Documentation index
│   ├── README.md                # Copy of main README
│   ├── PROJECT_SEPARATION.md    # Project structure details
│   ├── REACT_QUICK_REFERENCE.md # React quick start
│   ├── REACT_TRAINING_WORKFLOW.md  # React workflow
│   ├── TINYLLAMA_TRAINING_README.md  # TinyLlama guide
│   ├── IMPROVEMENTS_SUMMARY.md  # Code improvements
│   ├── INTEGRATION_COMPLETE.md  # Integration summary
│   └── UPDATE_SUMMARY.md        # Update notes
│
├── data/                         # Data files (gitignored)
│   └── .gitkeep                 # Keep directory in git
│
├── README.md                     # Main documentation
├── LICENSE                       # MIT License
├── pyproject.toml               # Modern Python packaging (PEP 518)
├── setup.py                     # Traditional setup (backward compat)
├── MANIFEST.in                  # Package data inclusion
├── Makefile                     # Automation commands
├── pytest.ini                   # Pytest configuration
├── .gitignore                   # Git exclusions
├── .env.example                 # Environment template
├── requirements.txt             # Full dependencies
├── requirements-minimal.txt     # Minimal dependencies
└── environment.yml              # Conda environment
```

## 🎯 Design Principles

### 1. **src/ Layout (PEP 420)**

The `src/` directory layout prevents accidental usage of the development version:

✅ **Benefits:**
- Ensures tests run against installed package
- Prevents import confusion
- Follows modern Python packaging standards
- Better isolation between source and tests

```python
# Import from installed package
from q_store_examples import basic_example

# Not from local directory
```

### 2. **Separation of Concerns**

Each directory has a clear purpose:

- **src/** - Production code only
- **tests/** - Test code only
- **scripts/** - Utility scripts (not part of package)
- **docs/** - Documentation files
- **data/** - Data files (gitignored)

### 3. **Package Structure**

```
src/q_store_examples/
├── __init__.py          # Package API
├── basic_example.py     # Can be run as: python -m q_store_examples.basic_example
├── financial_example.py # Can be run as: python -m q_store_examples.financial_example
└── ...
```

Each module can be run as:
```bash
python -m q_store_examples.basic_example
```

Or installed as console scripts:
```bash
qstore-basic  # After pip install
```

### 4. **Configuration Files**

#### Modern: pyproject.toml (PEP 518, 517, 621)
- Single source of truth
- Tool configuration
- Build system specification

#### Traditional: setup.py
- Backward compatibility
- Dynamic configuration if needed

#### Testing: pytest.ini
- Test discovery configuration
- Marker definitions
- Plugin settings

## 📦 Installation Methods

### Development Installation

```bash
# Install in editable mode
pip install -e .

# Now you can import from anywhere
python -c "from q_store_examples import basic_example"
```

### User Installation

```bash
# Install from source
pip install .

# Or from git
pip install git+https://github.com/yucelz/q-store.git#subdirectory=examples
```

### Console Scripts

After installation, these commands are available:

```bash
qstore-basic          # Run basic example
qstore-financial      # Run financial example
qstore-quickstart     # Run quickstart
qstore-ml-training    # Run ML training
qstore-react-training # Run React training
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/q_store_examples

# Run specific test
pytest tests/test_basic.py

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

### Test Organization

```
tests/
├── __init__.py
├── test_basic.py           # Unit tests for basic functionality
├── test_financial.py       # Financial example tests
├── test_integration.py     # Integration tests
└── conftest.py             # Pytest fixtures (if needed)
```

## 📝 Documentation Organization

```
docs/
├── INDEX.md                # Documentation index
├── guides/                 # How-to guides
│   ├── quickstart.md
│   └── advanced.md
├── reference/              # API reference
│   └── modules.md
└── tutorials/              # Step-by-step tutorials
    └── first_project.md
```

## 🔧 Scripts Organization

```
scripts/
├── verify_installation.py  # Installation checker
├── verify_*.py            # Other verification scripts
└── run_*.sh               # Automation scripts
```

Scripts are:
- **Not part of the package** (in scripts/, not src/)
- **Executable** (chmod +x)
- **Documented** (docstrings and --help)

## 🗂️ Data Organization

```
data/
├── .gitkeep              # Keep empty dir in git
├── raw/                  # Raw data (not tracked)
├── processed/            # Processed data (not tracked)
└── examples/             # Example data (tracked)
    └── sample.jsonl
```

Data directory is gitignored except for example files.

## 📊 Benefits of This Structure

### ✅ Professional
- Follows PEP standards
- Used by major Python projects
- Familiar to Python developers

### ✅ Maintainable
- Clear separation of concerns
- Easy to navigate
- Logical organization

### ✅ Testable
- src/ layout ensures proper testing
- Clear test organization
- Easy to add new tests

### ✅ Distributable
- Proper package structure
- pip/conda installable
- Console scripts support

### ✅ Documented
- Centralized documentation
- Clear structure
- Easy to update

## 🔄 Migration from Flat Structure

From:
```
examples/
├── basic_example.py
├── financial_example.py
└── ...
```

To:
```
examples/
├── src/
│   └── q_store_examples/
│       ├── basic_example.py
│       └── financial_example.py
└── ...
```

Changes needed:
1. ✅ Move files to src/q_store_examples/
2. ✅ Update imports (already using absolute imports)
3. ✅ Update pyproject.toml (package-dir)
4. ✅ Update setup.py (package_dir, packages)
5. ✅ Update scripts to use `python -m`

## 📚 References

- [PEP 518](https://peps.python.org/pep-0518/) - pyproject.toml
- [PEP 517](https://peps.python.org/pep-0517/) - Build system
- [PEP 621](https://peps.python.org/pep-0621/) - Project metadata
- [PEP 420](https://peps.python.org/pep-0420/) - Namespace packages
- [Python Packaging Guide](https://packaging.python.org/)
- [src layout](https://blog.ionelmc.ro/2014/05/25/python-packaging/)

## 🎯 Next Steps

1. Install in development mode: `pip install -e .`
2. Run tests: `pytest`
3. Check imports: `python -c "import q_store_examples"`
4. Run examples: `python -m q_store_examples.basic_example`
5. Build distribution: `python -m build`

---

This structure follows industry best practices and ensures the project is professional, maintainable, and easy to work with.
