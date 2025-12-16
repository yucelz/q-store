# Q-Store Binary Distribution Setup

This directory contains everything needed to create a closed-source binary distribution of Q-Store.

## 📦 Distribution Files Created

### Core Files
- **setup.py** - Binary compilation configuration using Cython
- **MANIFEST.in** - Controls what gets included in distributions
- **build_binary_distribution.sh** - Automated build and verification script
- **pyproject.toml** - Updated with Cython requirements

### Documentation
- **docs/CLOSED_SOURCE_DISTRIBUTION_GUIDE.md** - Complete 14-page guide
- **docs/QUICK_REFERENCE.md** - Quick command reference
- **docs/IMPLEMENTATION_SUMMARY.md** - Overview of the distribution system

### Configuration
- **.gitignore_distribution** - Git ignore rules for binary builds

---

## 🚀 Quick Start (3 Steps)

### 1. Install Build Tools
```bash
pip install Cython wheel twine setuptools
```

### 2. Build Binary Wheel
```bash
./build_binary_distribution.sh
```

### 3. Test Installation
```bash
# Create clean test environment
python -m venv test_env
source test_env/bin/activate

# Install your wheel
pip install dist/q_store-*.whl

# Verify it works
python -c "from q_store import QuantumDatabase; print('✓ Success!')"

# Verify source is protected
python -c "import inspect; from q_store import QuantumDatabase; inspect.getsource(QuantumDatabase)"
# Should fail with: OSError: could not get source code
```

---

## 🔒 What Gets Protected

All Python source code in `src/q_store/` is compiled to binary:

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
├── ml/                → Compiled to .so/.pyd
│   └── All ML modules
├── constants.py       → Compiled to .so/.pyd
└── exceptions.py      → Compiled to .so/.pyd
```

**Only `__init__.py` files remain readable** for proper Python package structure.

---

## ✅ What Stays Public

```
examples/              ✅ Fully readable (public examples)
docs/                  ✅ Public documentation
README.md              ✅ Public
LICENCE                ✅ Public
```

---

## 🛠️ Build Process Explained

The `build_binary_distribution.sh` script:

1. **Cleans** previous builds
2. **Compiles** all `.py` files (except `__init__.py`) to binary `.so`/`.pyd` files
3. **Creates** wheel distribution (`.whl`)
4. **Verifies** no source code is included
5. **Reports** success with verification details

**Expected Output:**
```
==========================================
Q-Store Binary Distribution Builder
==========================================

Step 1: Cleaning previous builds...
  ✓ Cleaned

Step 2: Building binary wheel distribution...
  ✓ Binary wheel built successfully

Step 3: Verifying binary distribution...
  Checking: dist/q_store-3.4.0-cp311-cp311-linux_x86_64.whl
  - Python source files (non-__init__): 0
  - Compiled binary files (.so/.pyd): 15
  ✓ No source code found - distribution is secure
  ✓ Binary extensions present

==========================================
Build Complete!
==========================================
```

---

## 📤 Distribution Options

### Option 1: PyPI (Public Package Index)
```bash
# Upload to PyPI
twine upload dist/*.whl

# Users install:
pip install q-store
```

### Option 2: Private PyPI Server
```bash
# Host your own package index
# Users install with custom index:
pip install q-store --index-url https://pypi.yourcompany.com/simple/
```

### Option 3: Direct Distribution
```bash
# Send .whl files directly to customers
# They install:
pip install q_store-3.4.0-cp311-cp311-linux_x86_64.whl
```

---

## 🖥️ Multi-Platform Builds

Binary wheels are **platform-specific**. Build on each target platform:

### Linux
```bash
./build_binary_distribution.sh
# Creates: q_store-3.4.0-cp311-cp311-linux_x86_64.whl
```

### macOS
```bash
./build_binary_distribution.sh
# Creates: q_store-3.4.0-cp311-cp311-macosx_11_0_x86_64.whl
```

### Windows
```bash
build_binary_distribution.sh  # Use Git Bash or WSL
# Creates: q_store-3.4.0-cp311-cp311-win_amd64.whl
```

### Automated Multi-Platform (GitHub Actions)
See `docs/QUICK_REFERENCE.md` for GitHub Actions template.

---

## 🔍 Verification Commands

### Check Wheel Contents
```bash
unzip -l dist/*.whl
```

### Verify No Source Files
```bash
unzip -l dist/*.whl | grep -E "\.py$" | grep -v "__init__.py"
# Should return empty
```

### Verify Binary Files Present
```bash
unzip -l dist/*.whl | grep -E "\.(so|pyd)$"
# Should show compiled binary files
```

### Test Source Protection
```bash
python -c "
import inspect
from q_store.core import QuantumDatabase
try:
    source = inspect.getsource(QuantumDatabase)
    print('✗ ERROR: Source code is accessible!')
except (OSError, TypeError):
    print('✓ Source code is protected')
"
```

---

## ⚠️ Critical Rules

### ✅ DO:
- Build ONLY wheels: `python setup.py bdist_wheel`
- Upload ONLY wheels: `twine upload dist/*.whl`
- Test in clean environment before publishing
- Build for each platform separately
- Version your releases: `git tag v3.4.0`

### ❌ DON'T:
- ❌ Never run: `python setup.py sdist` (creates source distribution!)
- ❌ Never run: `python -m build` without flags (creates both!)
- ❌ Never commit: `dist/`, `build/`, `*.egg-info`
- ❌ Never upload: `.tar.gz` files to PyPI
- ❌ Never share: Your `src/` directory publicly

---

## 🔐 Security Features

### Code Protection
- ✅ All algorithms compiled to binary
- ✅ Extremely difficult to reverse engineer
- ✅ No `.py` source files in distribution
- ✅ Automated verification of binary-only distribution

### Access Control Options
- Private PyPI server with authentication
- Direct distribution to verified customers
- License key validation (can be added)
- Download tracking and analytics

### Reverse Engineering Difficulty
- **Python bytecode**: Easy to decompile (~1 hour)
- **Your compiled code**: Very Hard (weeks/months, similar to C extensions)

---

## 📚 Documentation

### Comprehensive Guides
1. **CLOSED_SOURCE_DISTRIBUTION_GUIDE.md** (14 pages)
   - Complete distribution strategy
   - Step-by-step instructions
   - Publishing options
   - Security best practices
   - Troubleshooting

2. **QUICK_REFERENCE.md**
   - Quick commands
   - Common tasks
   - Verification steps
   - Troubleshooting tips

3. **IMPLEMENTATION_SUMMARY.md**
   - Overview of what was created
   - How the system works
   - Next steps

---

## 🐛 Troubleshooting

### "No module named 'Cython'"
```bash
pip install Cython
```

### Import fails after installation
```bash
# Check __init__.py exports
cat src/q_store/__init__.py

# Rebuild from scratch
rm -rf build/ dist/ *.egg-info
./build_binary_distribution.sh
pip install --force-reinstall dist/*.whl
```

### Wheel contains .py source files
```bash
# Inspect what's included
unzip -l dist/*.whl | grep "\.py$"

# Should only see __init__.py files
# If you see other .py files, check:
# 1. setup.py - ext_modules list
# 2. MANIFEST.in - exclusion rules
```

### Platform-specific issues
- **Linux**: Install `python3-dev` or `python3-devel`
- **macOS**: Install Xcode command line tools
- **Windows**: Install Visual Studio Build Tools

---

## 📋 Pre-Release Checklist

Before distributing:
- [ ] Version updated in `setup.py` and `pyproject.toml`
- [ ] Build script runs successfully
- [ ] No `.py` files in wheel (except `__init__.py`)
- [ ] Binary `.so`/`.pyd` files present
- [ ] Test in isolated environment
- [ ] Verify source code inaccessible
- [ ] Test on target platforms
- [ ] Update CHANGELOG
- [ ] Tag git release: `git tag v3.4.0`
- [ ] Documentation updated

---

## 🚢 Publishing Workflow

```
1. DEVELOP
   └─ Code in src/q_store/ (private repo)

2. VERSION
   ├─ Update version in setup.py
   ├─ Update version in pyproject.toml
   └─ Git tag: git tag v3.4.0

3. BUILD
   ├─ Run: ./build_binary_distribution.sh
   ├─ Verify: Check build output
   └─ Test: Install and run tests

4. PUBLISH
   ├─ PyPI: twine upload dist/*.whl
   ├─ Private: Share .whl files
   └─ Examples: Update public examples repo

5. DISTRIBUTE
   └─ Users: pip install q-store
```

---

## 🤝 Support

For questions or issues:
1. Check `docs/CLOSED_SOURCE_DISTRIBUTION_GUIDE.md`
2. Review `docs/QUICK_REFERENCE.md`
3. Verify build with `./build_binary_distribution.sh`
4. Test in clean environment

---

## 📝 License

Q-Store is distributed as closed-source software. The binary distributions protect your intellectual property while allowing users to leverage the quantum database capabilities.

See `LICENCE` for terms.

---

**Ready to build?** Run: `./build_binary_distribution.sh`
