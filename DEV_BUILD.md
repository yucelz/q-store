# Q-Store Development Build Guide

Quick reference for development workflows when working on Q-Store source code.

## 🚀 Quick Start - Development Setup

### One-Time Setup

```bash
# Navigate to project root
cd /path/to/q-store

# Create and activate virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/macOS
# OR
test_env\Scripts\activate  # Windows

# Install in editable mode (do this once)
pip install -e ".[all]"
```

After this setup, **code changes are automatically available** - no rebuild needed!

---

## 🔄 Development Workflows

### Option 1: Pure Python Mode (RECOMMENDED for Development)

**Fastest development cycle** - changes are instant, no compilation needed.

```bash
# Remove all Cython artifacts to use pure Python
find src/q_store -name "*.c" -delete
find src/q_store -name "*.so" -delete

# Now edit any .py file and run immediately
python examples/async_features/storage_demo.py
```

**Benefits:**
- ✅ Instant code changes (no rebuild)
- ✅ No Cython compilation overhead
- ✅ Faster iteration cycle
- ✅ Easier debugging with pure Python
- ✅ Line numbers match source code

**Use this when:**
- Developing new features
- Debugging issues
- Testing examples
- Running tests

---

### Option 2: Rebuild When Needed (With Cython)

If you need to test with Cython-compiled code:

```bash
# Quick rebuild (only when you change .py files)
pip install -e . --no-deps --force-reinstall

# Then run your code
python examples/async_features/storage_demo.py
```

**Use this when:**
- Testing performance optimizations
- Verifying Cython compatibility
- Before creating production builds

---

### Option 3: Avoid Bytecode Cache Issues

Prevent Python from creating `.pyc` cache files:

```bash
# Set environment variable (per session)
export PYTHONDONTWRITEBYTECODE=1

# Then run normally
python examples/async_features/storage_demo.py
```

Or make it permanent in your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PYTHONDONTWRITEBYTECODE=1' >> ~/.bashrc
source ~/.bashrc
```

---

## 🧹 Cleaning Build Artifacts

### Clear Python Cache

```bash
# Remove __pycache__ and .pyc files
find src/q_store -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find src/q_store -name "*.pyc" -delete 2>/dev/null || true
```

### Clear Cython Artifacts

```bash
# Remove compiled C extensions
find src/q_store -name "*.c" -delete
find src/q_store -name "*.so" -delete
find src/q_store -name "*.cpp" -delete
find src/q_store -name "*.pyx" -delete
```

### Full Clean

```bash
# Remove all build artifacts
rm -rf build/ dist/ *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.so" -delete
find . -name "*.c" -delete
```

---

## 🏗️ Production Builds

### Create Binary Wheel Distribution

```bash
# Use the provided build script
./scripts/build_binary_distribution.sh
```

This script:
1. Cleans previous builds
2. Compiles all Python code to Cython
3. Creates binary wheel (`.whl`)
4. Verifies no source code in distribution
5. Cleans up build artifacts

**Output:** `dist/q_store-*.whl`

### Manual Build Steps

```bash
# 1. Clean previous builds
rm -rf build/ dist/ *.egg-info

# 2. Build wheel
python setup.py bdist_wheel

# 3. Install wheel
pip install dist/q_store-*.whl
```

---

## 📝 Common Development Tasks

### Running Examples

```bash
# Activate environment
source test_env/bin/activate

# Run any example
python examples/basic_usage.py
python examples/async_features/basic_async_usage.py
python examples/ml_frameworks/tensorflow/fashion_mnist_tensorflow.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_async.py

# Run with coverage
pytest --cov=q_store tests/

# Run specific test
pytest tests/test_async.py::test_async_executor -v
```

### Running Validation

```bash
# Run all validation examples
python examples/run_all_validation.py

# Run specific validation
python examples/validation/gradient_validation.py
python examples/validation/simple_classification.py
```

---

## 🐛 Troubleshooting

### Changes Not Taking Effect

**Problem:** Code changes aren't reflected when running examples.

**Solutions:**

1. **Clear Python cache:**
   ```bash
   find src/q_store -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
   find src/q_store -name "*.pyc" -delete
   ```

2. **Remove Cython artifacts:**
   ```bash
   find src/q_store -name "*.c" -delete
   find src/q_store -name "*.so" -delete
   ```

3. **Verify editable install:**
   ```bash
   pip show q-store | grep Location
   # Should point to your source directory
   ```

4. **Reinstall in editable mode:**
   ```bash
   pip uninstall -y q-store
   pip install -e .
   ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'q_store'`

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/q-store

# Activate virtual environment
source test_env/bin/activate

# Install in editable mode
pip install -e .
```

### Cython Compilation Errors

**Problem:** Errors during `pip install -e .` related to Cython compilation.

**Solution:**
```bash
# Install without Cython compilation (use pure Python)
find src/q_store -name "*.c" -delete
pip install -e . --no-build-isolation

# OR install Cython first
pip install Cython
pip install -e .
```

### AttributeError After Code Changes

**Problem:** `AttributeError: module 'zarr' has no attribute 'X'` or similar.

**Solution:**
```bash
# Clear all cache and compiled files
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.so" -delete

# Restart Python interpreter / re-run script
python examples/your_example.py
```

---

## 📊 Development vs Production

| Aspect | Development Mode | Production Build |
|--------|-----------------|------------------|
| Installation | `pip install -e .` | `pip install q-store` or `.whl` |
| Code Changes | Instant | Requires rebuild |
| Performance | Slower (pure Python) | Faster (Cython-compiled) |
| Debugging | Easy (source available) | Harder (compiled) |
| File Size | Small (source only) | Larger (includes compiled) |
| Use Case | Development, testing | Deployment, distribution |

---

## 🎯 Recommended Workflow

### Daily Development

```bash
# 1. Start of day - activate environment
cd ~/yz_code/q-store
source test_env/bin/activate

# 2. Clean Cython artifacts (do once)
find src/q_store -name "*.c" -delete
find src/q_store -name "*.so" -delete

# 3. Make code changes in src/q_store/

# 4. Run examples or tests immediately
python examples/async_features/storage_demo.py

# 5. If needed, clear cache
find src/q_store -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

### Before Committing

```bash
# 1. Run all tests
pytest tests/

# 2. Run validation examples
python examples/run_all_validation.py

# 3. Check code quality
black src/q_store/
ruff src/q_store/

# 4. Verify production build works
./scripts/build_binary_distribution.sh
```

### Before Release

```bash
# 1. Full test suite
pytest tests/ --cov=q_store

# 2. Build distribution
./scripts/build_binary_distribution.sh

# 3. Test wheel installation
pip install dist/q_store-*.whl

# 4. Run validation on installed wheel
python examples/run_all_validation.py

# 5. Tag release
git tag -a v4.1.0 -m "Release v4.1.0"
git push origin v4.1.0
```

---

## 🔗 Related Documentation

- **Examples README:** `examples/README.md` - How to run examples
- **Build Script:** `scripts/build_binary_distribution.sh` - Production builds
- **Contributing:** See GitHub repository for contribution guidelines
- **Main README:** `README.md` - Project overview and features

---

## 💡 Pro Tips

1. **Use editable mode** (`pip install -e .`) for all development work
2. **Remove `.c` files** to avoid Cython compilation during development
3. **Set `PYTHONDONTWRITEBYTECODE=1`** to avoid cache issues
4. **Clear cache** if changes don't take effect
5. **Test with pure Python** first, then verify with Cython
6. **Use the build script** for production builds, not manual commands
7. **Keep `test_env/` local** - don't commit it to git (already in `.gitignore`)

---

**Happy Developing! 🚀**
