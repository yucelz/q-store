# Q-Store Closed-Source Distribution Guide

## Overview

This guide explains how to distribute Q-Store as a **closed-source library** with public examples.

**Distribution Model:**
- 🔒 **Protected**: All code in `src/q_store/` (compiled to binary)
- ✅ **Public**: Examples in `examples/` directory
- ✅ **Public**: Documentation in `docs/`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Distribution Strategy](#distribution-strategy)
3. [Building Binary Wheels](#building-binary-wheels)
4. [Repository Structure](#repository-structure)
5. [Publishing Options](#publishing-options)
6. [Security Best Practices](#security-best-practices)
7. [Multi-Platform Builds](#multi-platform-builds)

---

## Quick Start

### 1. Install Build Dependencies

```bash
pip install Cython wheel twine build setuptools
```

### 2. Build Binary Distribution

```bash
chmod +x build_binary_distribution.sh
./build_binary_distribution.sh
```

### 3. Test Locally

```bash
pip install dist/q_store-*.whl
python -c "from q_store import QuantumDatabase; print('Success!')"
```

### 4. Publish (Optional)

```bash
# To PyPI
twine upload dist/*.whl

# Or distribute privately to customers
# They can install: pip install q_store-3.4.0-*.whl
```

---

## Distribution Strategy

### What Gets Protected

Everything in `src/q_store/` is compiled to binary (.so/.pyd files):

```
src/q_store/
├── backends/          → Compiled to .so/.pyd
│   ├── backend_manager.py
│   ├── cirq_ionq_adapter.py
│   ├── ionq_backend.py
│   ├── qiskit_ionq_adapter.py
│   └── quantum_backend_interface.py
├── core/              → Compiled to .so/.pyd
│   ├── entanglement_registry.py
│   ├── quantum_database.py
│   ├── state_manager.py
│   └── tunneling_engine.py
└── ml/                → Compiled to .so/.pyd
    ├── adaptive_optimizer.py
    ├── circuit_batch_manager.py
    ├── quantum_trainer.py
    └── ... (all ML modules)
```

**Result**: Users get working binaries but **cannot read or modify your algorithms**.

### What Stays Public

```
examples/                  ✅ Fully readable
docs/                      ✅ Public documentation  
README.md                  ✅ Public
LICENSE                    ✅ Public
```

---

## Building Binary Wheels

### Method 1: Automated Build Script (Recommended)

```bash
./build_binary_distribution.sh
```

This script:
1. ✓ Cleans previous builds
2. ✓ Compiles all Python code to binary
3. ✓ Creates wheel file
4. ✓ Verifies no source code is included
5. ✓ Reports success/failures

### Method 2: Manual Build

```bash
# Clean
rm -rf build/ dist/ *.egg-info

# Build wheel only (no source distribution)
python setup.py bdist_wheel

# Verify
unzip -l dist/*.whl | grep -E "\.py$" | grep -v "__init__.py"
# Should return empty (no .py files except __init__.py)
```

### Critical: Never Build Source Distributions

❌ **NEVER run these commands:**
```bash
python setup.py sdist         # DON'T - creates source distribution!
python -m build               # DON'T - creates both source and wheel!
```

✅ **ONLY run:**
```bash
python setup.py bdist_wheel   # Creates binary wheel only
```

---

## Repository Structure

### Option 1: Single Repository with Access Control

Keep everything in one repo but use `.gitignore` or branch protection:

```
q-store/                    (Private repo)
├── src/q_store/           (Protected - never share)
├── examples/              (Can be public)
└── docs/                  (Can be public)
```

**Distribution:**
- Share only compiled wheels (.whl files)
- Optionally: Create a separate public repo for examples only

### Option 2: Separate Repositories

```
q-store-core/              (Private repo with src/q_store/)
q-store-examples/          (Public repo with examples/)
```

**Workflow:**
1. Develop in `q-store-core` (private)
2. Build wheels from `q-store-core`
3. Share examples via `q-store-examples` (public)
4. Users: `pip install q-store` + clone examples repo

### Option 3: Examples as Separate Package

Create `q-store-examples` as a separate installable package:

```bash
# Users install both
pip install q-store              # Your closed-source library
pip install q-store-examples     # Public examples package
```

---

## Publishing Options

### Option 1: PyPI (Python Package Index)

**Pros:**
- Easy for users: `pip install q-store`
- Automatic version management
- Wide distribution

**Cons:**
- Must be publicly downloadable (but still compiled)
- Anyone can install

**How to Publish:**

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*.whl

# Then publish to real PyPI
twine upload dist/*.whl
```

**Configure `.pypirc`:**
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN

[testpypi]
username = __token__
password = pypi-YOUR-API-TOKEN
```

### Option 2: Private PyPI Server

Host your own package index for paying customers.

**Tools:**
- [devpi](https://devpi.net/)
- [pypiserver](https://github.com/pypiserver/pypiserver)
- [JFrog Artifactory](https://jfrog.com/artifactory/)

**Customer installation:**
```bash
pip install q-store --index-url https://pypi.yourcompany.com/simple/
```

### Option 3: Direct Distribution

Email/download wheels directly to customers:

```bash
# Customers install from file
pip install q_store-3.4.0-cp310-cp310-linux_x86_64.whl
```

### Option 4: License-Protected Distribution

Combine compiled binaries with license checking:

```python
# In your __init__.py (stays as readable Python)
import os
from pathlib import Path

def _check_license():
    """Check for valid license file"""
    license_file = Path.home() / ".q_store" / "license.key"
    if not license_file.exists():
        raise RuntimeError(
            "Q-Store license not found. "
            "Contact sales@yourcompany.com"
        )
    # Add actual license validation logic
    
_check_license()
```

---

## Security Best Practices

### 1. Verify Your Builds

Always check that wheels don't contain source:

```bash
# Extract and inspect
unzip -l dist/*.whl

# Should see:
# ✓ q_store/backends/backend_manager.cpython-310-x86_64-linux-gnu.so
# ✓ q_store/core/quantum_database.cpython-310-x86_64-linux-gnu.so
# ✗ NO .py files (except __init__.py)
```

### 2. Control Distribution Channels

- ✓ Use private PyPI server for sensitive distributions
- ✓ Require authentication for downloads
- ✓ Track who has access to wheels
- ✓ Version wheels with customer IDs if needed

### 3. Add Runtime License Checks

Even compiled code can be distributed. Add license validation:

```python
# Encoded in __init__.py or a compiled module
def validate_license(key):
    # Check license key against server
    # or validate signed license file
    pass
```

### 4. Obfuscation Layer (Optional)

For extra protection, add name obfuscation:

```python
# In setup.py, add to compiler_directives:
compiler_directives={
    'language_level': "3",
    'embedsignature': False,  # Hide function signatures
    'binding': True,           # Make all names private
}
```

Trade-off: Breaks IDE autocomplete and debugging.

### 5. Code Signing

Sign your wheels to prevent tampering:

```bash
# Generate signing key
gpg --gen-key

# Sign wheel
gpg --detach-sign --armor dist/*.whl

# Users verify
gpg --verify dist/*.whl.asc dist/*.whl
```

---

## Multi-Platform Builds

Binary wheels are **platform-specific**. You need separate builds for:

- **Linux**: `*-linux_x86_64.whl`
- **macOS**: `*-macosx_*_x86_64.whl` or `*-arm64.whl`
- **Windows**: `*-win_amd64.whl`

### Option 1: Manual Builds on Each Platform

```bash
# On Linux
python setup.py bdist_wheel

# On macOS
python setup.py bdist_wheel

# On Windows
python setup.py bdist_wheel
```

### Option 2: GitHub Actions (Automated)

Create `.github/workflows/build-wheels.yml`:

```yaml
name: Build Wheels

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install Cython wheel setuptools
    
    - name: Build wheel
      run: python setup.py bdist_wheel
    
    - name: Upload wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: dist/*.whl
```

### Option 3: cibuildwheel (Best for PyPI)

Builds wheels for all platforms automatically:

```bash
pip install cibuildwheel

# Build for all platforms
cibuildwheel --platform linux
cibuildwheel --platform macos
cibuildwheel --platform windows
```

Create `pyproject.toml` section:

```toml
[tool.cibuildwheel]
build = "cp38-* cp39-* cp310-* cp311-*"
skip = "*-win32 *-manylinux_i686"
```

---

## Examples Distribution

### Keep Examples Synchronized

Your users need examples that match your library version.

**Strategy 1: Bundle Examples with Package**

```python
# In setup.py, add:
package_data={
    'q_store': ['examples/*.py', 'examples/README.md'],
}
```

Users access:
```python
import pkg_resources
examples_path = pkg_resources.resource_filename('q_store', 'examples/')
```

**Strategy 2: Separate Examples Repository**

```
# Public repo
github.com/yourorg/q-store-examples

# Tag versions to match library
git tag v3.4.0
```

**Strategy 3: Documentation Site**

Host examples on Read the Docs, GitHub Pages, etc:
```
https://yourorg.github.io/q-store/examples/
```

---

## Testing Your Distribution

### 1. Create Test Environment

```bash
python -m venv test_env
source test_env/bin/activate
pip install dist/*.whl
```

### 2. Verify Imports

```bash
python -c "
from q_store import QuantumDatabase
from q_store.ml import QuantumTrainer
from q_store.backends import BackendManager
print('All imports successful!')
"
```

### 3. Run Examples

```bash
cd examples/
python src/q_store_examples/basic_example.py
```

### 4. Check for Source Leaks

```bash
python -c "
import q_store
import inspect
print(inspect.getsourcefile(q_store.QuantumDatabase))
"
# Should print: None or raise error (no source available)
```

---

## Troubleshooting

### "No module named 'Cython'"

```bash
pip install Cython
```

### "cannot import name 'QuantumDatabase'"

Check `__init__.py` files are properly structured:

```python
# src/q_store/__init__.py
from .core.quantum_database import QuantumDatabase
from .ml.quantum_trainer import QuantumTrainer

__all__ = ['QuantumDatabase', 'QuantumTrainer']
```

### "ImportError: ... undefined symbol"

Recompile with:
```bash
python setup.py build_ext --inplace
python setup.py bdist_wheel
```

### Wheel Contains .py Files

Check your `MANIFEST.in` and ensure `setup.py` isn't including source.

---

## Legal Considerations

### 1. License File

Even with compiled code, include a LICENSE file that:
- Specifies this is proprietary/closed-source software
- Prohibits decompilation/reverse engineering
- Defines usage terms

Example:
```
PROPRIETARY LICENSE

Copyright (c) 2025 Your Company

This software is proprietary and confidential. 
Unauthorized copying, distribution, or reverse 
engineering is strictly prohibited.
```

### 2. Terms of Service

For commercial distribution, add:
- End User License Agreement (EULA)
- Subscription/licensing terms
- Support agreements

---

## Summary Checklist

✅ **Before Building:**
- [ ] Install Cython: `pip install Cython wheel`
- [ ] Review `setup.py` - all modules listed
- [ ] Review `MANIFEST.in` - excludes source
- [ ] Update version number

✅ **Building:**
- [ ] Run `./build_binary_distribution.sh`
- [ ] Verify no .py files in wheel (except `__init__.py`)
- [ ] Test wheel in clean environment
- [ ] Build for all target platforms

✅ **Publishing:**
- [ ] Choose distribution method (PyPI/private/direct)
- [ ] Set up authentication if needed
- [ ] Upload wheels only (never source distributions)
- [ ] Tag git release

✅ **Documentation:**
- [ ] Update examples to match version
- [ ] Document installation process
- [ ] Provide API documentation
- [ ] List supported platforms

---

## Support

For issues with this distribution process:
1. Check wheel contents: `unzip -l dist/*.whl`
2. Verify compilation: Look for `.so`/`.pyd` files
3. Test in isolated environment
4. Review build logs for errors



**Remember:** Your proprietary algorithms in `src/q_store/` will be protected as compiled binaries, while users can still learn from your public `examples/` directory!
