#!/bin/bash
# Test script to validate wheel metadata locally before publishing

set -e

echo "=========================================="
echo "Testing Wheel Build and Metadata"
echo "=========================================="
echo ""

# Clean previous builds
echo "1. Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info src/*.egg-info
echo "✓ Cleaned"
echo ""

# Install build dependencies
echo "2. Installing build dependencies..."
pip install --upgrade pip setuptools wheel build twine Cython
echo "✓ Dependencies installed"
echo ""

# Build source distribution to test metadata
echo "3. Building source distribution..."
python -m build --sdist --outdir dist/
echo "✓ Source distribution built"
echo ""

# Check source distribution metadata
echo "4. Checking source distribution with twine..."
python -m twine check dist/*.tar.gz
if [ $? -eq 0 ]; then
    echo "✓ Source distribution metadata is valid"
else
    echo "✗ Source distribution metadata is INVALID"
    exit 1
fi
echo ""

# Build a test wheel (for current Python version)
echo "5. Building wheel for current Python version..."
python -m build --wheel --outdir dist/
echo "✓ Wheel built"
echo ""

# Check wheel metadata
echo "6. Checking wheel metadata with twine..."
python -m twine check dist/*.whl
if [ $? -eq 0 ]; then
    echo "✓ Wheel metadata is valid"
else
    echo "✗ Wheel metadata is INVALID"
    exit 1
fi
echo ""

# Inspect wheel contents
echo "7. Inspecting wheel contents..."
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Wheel file: $WHEEL_FILE"
echo ""
echo "Contents:"
unzip -l "$WHEEL_FILE" | grep -E "(METADATA|WHEEL|LICENSE|__init__)" || true
echo ""

# Extract and display metadata
echo "8. Extracting and displaying METADATA..."
unzip -q -c "$WHEEL_FILE" "*/METADATA" 2>/dev/null || unzip -q -c "$WHEEL_FILE" "*-*.dist-info/METADATA" 2>/dev/null || echo "Could not extract METADATA"
echo ""

# Test import
echo "9. Testing wheel installation and import..."
pip install --force-reinstall "$WHEEL_FILE"
python -c "from q_store import QuantumDatabase; print('✓ Import successful!')"
echo ""

echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "The wheel metadata is valid and ready for GitHub Actions."
