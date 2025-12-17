# Q-Store Binary Distribution - Quick Reference

## 🚀 Quick Commands

### Build Binary Wheel
```bash
./build_binary_distribution.sh
```

### Test Installation
```bash
python -m venv test_env
source test_env/bin/activate  # or: test_env\Scripts\activate on Windows
pip install dist/*.whl
python -c "from q_store import QuantumDatabase; print('✓ Success')"
```

### Publish to PyPI
```bash
twine upload dist/*.whl
```

---

## 📋 Essential Commands

### Setup (First Time)
```bash
# Install build tools
pip install Cython wheel twine setuptools

# Make build script executable
chmod +x build_binary_distribution.sh
```

### Building

```bash
# Clean everything
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "*.so" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Build (automated)
./build_binary_distribution.sh

# Build (manual)
python setup.py bdist_wheel
```

### Verification

```bash
# List wheel contents
unzip -l dist/*.whl

# Check for source files (should be empty)
unzip -l dist/*.whl | grep -E "\.py$" | grep -v "__init__.py"

# Check for compiled files (should see .so or .pyd)
unzip -l dist/*.whl | grep -E "\.(so|pyd)$"

# Test imports
python -c "
import q_store
from q_store import QuantumDatabase
from q_store.ml import QuantumTrainer
print('✓ All imports work')
"

# Verify no source available
python -c "
import inspect
from q_store import QuantumDatabase
try:
    source = inspect.getsource(QuantumDatabase)
    print('✗ ERROR: Source code is accessible!')
except (OSError, TypeError):
    print('✓ Source code is protected')
"
```

### Publishing

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*.whl

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ q-store

# Publish to real PyPI
twine upload dist/*.whl

# Private distribution (direct file sharing)
# Just send the .whl file to customers
# They install: pip install q_store-3.4.0-*.whl
```

---

## ⚠️ Critical Rules

### ✅ DO:
- Build ONLY wheels: `python setup.py bdist_wheel`
- Upload ONLY wheels: `twine upload dist/*.whl`
- Test in clean environment before publishing
- Build for each platform separately (Linux, macOS, Windows)
- Version your releases: `git tag v3.4.0`

### ❌ DON'T:
- ❌ Never run: `python setup.py sdist` (creates source distribution!)
- ❌ Never run: `python -m build` (creates both source and wheel!)
- ❌ Never commit: `dist/`, `build/`, `*.egg-info`
- ❌ Never upload: `.tar.gz` files (source distributions)
- ❌ Never share: Your `src/` directory

---

## 🔍 Troubleshooting

### Build fails with "No module named 'Cython'"
```bash
pip install Cython
```

### Import fails after installation
```bash
# Check __init__.py exports
cat src/q_store/__init__.py

# Rebuild from scratch
rm -rf build/ dist/ *.egg-info
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

### Wheel contains .py source files
```bash
# Check what's included
unzip -l dist/*.whl | grep "\.py$"

# Should only see __init__.py files
# If you see other .py files, check:
# 1. setup.py - ext_modules list
# 2. MANIFEST.in - exclusion rules
# 3. pyproject.toml - package-data settings
```

### Different platforms
```bash
# You need to build on each platform:

# Linux
python setup.py bdist_wheel
# Creates: q_store-3.4.0-cp310-cp310-linux_x86_64.whl

# macOS
python setup.py bdist_wheel
# Creates: q_store-3.4.0-cp310-cp310-macosx_11_0_x86_64.whl

# Windows
python setup.py bdist_wheel
# Creates: q_store-3.4.0-cp310-cp310-win_amd64.whl
```

---

## 📦 Multi-Platform Build (GitHub Actions)

Save as `.github/workflows/build.yml`:

```yaml
name: Build Wheels
on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: pip install Cython wheel setuptools
    
    - name: Build wheel
      run: python setup.py bdist_wheel
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: dist/*.whl
```

---

## 🔐 Security Checklist

Before publishing:
- [ ] No `.py` files in wheel (except `__init__.py`)
- [ ] Binary `.so`/`.pyd` files present
- [ ] Test in isolated environment
- [ ] Verify source code inaccessible: `inspect.getsource()` fails
- [ ] Check wheel on multiple platforms
- [ ] Update version number
- [ ] Tag git release

---

## 📚 File Structure

```
Your Project/
├── setup.py                           # Build configuration
├── pyproject.toml                     # Package metadata
├── MANIFEST.in                        # Distribution control
├── build_binary_distribution.sh       # Build script
├── CLOSED_SOURCE_DISTRIBUTION_GUIDE.md # Full guide
│
├── src/q_store/                       # 🔒 PROTECTED (compiled)
│   ├── __init__.py                    # ✓ Readable
│   ├── backends/                      # 🔒 All .py → .so
│   ├── core/                          # 🔒 All .py → .so
│   └── ml/                            # 🔒 All .py → .so
│
├── examples/                          # ✅ PUBLIC (readable)
│   └── (stays as readable Python)
│
└── docs/                              # ✅ PUBLIC
    └── (public documentation)
```

---

## 🎯 Distribution Workflow

```
1. CODE
   ├─ Develop in src/q_store/
   └─ Update examples/

2. BUILD
   ├─ Run: ./build_binary_distribution.sh
   ├─ Verify: No source in wheel
   └─ Test: Install and import

3. PUBLISH
   ├─ PyPI: twine upload dist/*.whl
   ├─ Private: Share .whl files
   └─ Examples: Separate public repo

4. DISTRIBUTE
   └─ Users: pip install q-store
```

---

## 💡 Tips

1. **Version Everything**: Match wheel version to git tags
   ```bash
   git tag v3.4.0
   git push origin v3.4.0
   ```

2. **Test Before Publishing**: Always test in clean venv
   ```bash
   python -m venv fresh_test
   source fresh_test/bin/activate
   pip install dist/*.whl
   # Run your test suite
   ```

3. **Platform-Specific Wheels**: Build separately or use GitHub Actions

4. **Keep Examples Synced**: Tag examples repo with same version

5. **Documentation**: Host on Read the Docs or GitHub Pages

---

## 📞 Support

If users report issues:
1. Check which platform they're on
2. Verify wheel compatibility: Python version, OS, architecture
3. Ask them to test import: `python -c "from q_store import QuantumDatabase"`
4. Provide platform-specific wheel if needed

---

**Need Help?** See `CLOSED_SOURCE_DISTRIBUTION_GUIDE.md` for complete documentation.
