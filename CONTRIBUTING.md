# Contributing to Q-Store

Thank you for your interest in contributing to Q-Store! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/q-store.git
   cd q-store
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/yucelz/q-store.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Conda for environment management

### Installation

1. **Create a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # OR using conda
   conda env create -f environment.yml
   conda activate q-store
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,all]"
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (for testing)
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=q_store --cov-report=html

# Run specific test file
pytest tests/test_quantum_database.py

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Create new examples or improve existing ones
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure

### Contribution Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # OR
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards

3. **Write or update tests** for your changes

4. **Run the test suite** to ensure everything works:
   ```bash
   pytest
   ```

5. **Run code quality checks**:
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/
   ```

6. **Commit your changes** with descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add new quantum circuit optimizer"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding or updating tests
   - `refactor:` code refactoring
   - `perf:` performance improvements
   - `ci:` CI/CD changes

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (configured in pyproject.toml)
- **Formatting**: Use Black for automatic formatting
- **Import sorting**: Use isort with Black profile
- **Type hints**: All public functions must have type hints
- **Docstrings**: Use Google-style docstrings

### Example Function

```python
from typing import List, Optional
import numpy as np


def encode_vector(
    data: List[float],
    encoding_type: str = "amplitude",
    normalize: bool = True
) -> np.ndarray:
    """Encode classical data into quantum-compatible format.

    Args:
        data: Classical data vector to encode.
        encoding_type: Type of encoding ('amplitude' or 'angle').
        normalize: Whether to normalize the input vector.

    Returns:
        Encoded quantum state vector.

    Raises:
        ValueError: If encoding_type is not supported.

    Example:
        >>> vector = [0.5, 0.3, 0.2]
        >>> encoded = encode_vector(vector, encoding_type='amplitude')
        >>> print(encoded.shape)
        (3,)
    """
    if encoding_type not in ["amplitude", "angle"]:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")
    
    arr = np.array(data)
    if normalize:
        arr = arr / np.linalg.norm(arr)
    
    return arr
```

### Code Organization

- Keep functions focused and small (< 50 lines when possible)
- Use meaningful variable and function names
- Avoid deep nesting (max 3 levels)
- Extract complex logic into separate functions
- Add comments for non-obvious code

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Integration tests
├── fixtures/       # Shared test fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

```python
import pytest
from q_store import QuantumDatabase


class TestQuantumDatabase:
    """Test suite for QuantumDatabase."""

    @pytest.fixture
    def db(self):
        """Create a test database instance."""
        return QuantumDatabase(
            pinecone_api_key="test-key",
            dimension=128,
            quantum_backend="mock"
        )

    def test_insert(self, db):
        """Test vector insertion."""
        vector = [0.1] * 128
        db.insert(id="test-1", vector=vector)
        
        # Assertions
        assert db.count() == 1

    @pytest.mark.asyncio
    async def test_async_query(self, db):
        """Test async query operation."""
        result = await db.query_async(vector=[0.1] * 128, top_k=5)
        assert len(result) <= 5
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests
- Integration tests for backend interactions

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted with Black and isort
- [ ] No linting errors (ruff, mypy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for notable changes)
- [ ] Commits follow conventional commit format

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How have you tested this?

## Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, maintainers will merge your PR
4. Delete your branch after merge

## Reporting Bugs

### Before Reporting

- Check existing issues to avoid duplicates
- Verify the bug with the latest version
- Collect relevant information

### Bug Report Template

```markdown
**Describe the bug**
Clear and concise description

**To Reproduce**
Steps to reproduce:
1. Import module '...'
2. Call function '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Q-Store version: [e.g., 3.4.0]
- Backend: [e.g., IonQ, Mock]

**Additional context**
Any other relevant information
```

## Feature Requests

We welcome feature requests! Please:

1. **Search existing issues** for similar requests
2. **Create a detailed proposal** with:
   - Use case and motivation
   - Proposed API/interface
   - Alternative solutions considered
   - Potential impact on existing code
3. **Be open to discussion** and feedback

## Development Guidelines

### Quantum Backend Development

When adding a new quantum backend:

1. Implement the `QuantumBackend` interface
2. Add adapter in `backends/` directory
3. Update `BackendManager`
4. Add comprehensive tests
5. Document backend-specific requirements

### ML Module Development

When adding ML features:

1. Follow hardware-agnostic principles
2. Support multiple quantum SDKs
3. Optimize for batch operations
4. Add performance benchmarks
5. Document training workflows

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Create examples for new features
- Update architecture docs if needed

## Questions?

- **GitHub Discussions**: For general questions
- **Issues**: For bug reports and feature requests
- **Email**: yucelz@gmail.com for private inquiries

## Recognition

Contributors will be recognized in:
- AUTHORS file
- Release notes
- Documentation contributors section

Thank you for contributing to Q-Store! 🚀
