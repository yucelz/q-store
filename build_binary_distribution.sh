#!/bin/bash
# build_binary_distribution.sh
# Builds binary wheel distributions for Q-Store

set -e  # Exit on error

echo "=========================================="
echo "Q-Store Binary Distribution Builder"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src/q_store" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Check if Cython is installed
if ! python -c "import Cython" 2>/dev/null; then
    echo -e "${YELLOW}Cython not found. Installing...${NC}"
    pip install Cython
fi

# Clean previous builds
echo -e "${GREEN}Step 1: Cleaning previous builds...${NC}"
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.so" -delete
find . -type f -name "*.pyd" -delete
echo "  ✓ Cleaned"
echo ""

# Build binary wheels
echo -e "${GREEN}Step 2: Building binary wheel distribution...${NC}"
python setup.py bdist_wheel

if [ $? -eq 0 ]; then
    echo "  ✓ Binary wheel built successfully"
else
    echo -e "${RED}  ✗ Build failed${NC}"
    exit 1
fi
echo ""

# Verify the wheel doesn't contain source code
echo -e "${GREEN}Step 3: Verifying binary distribution...${NC}"
WHEEL_FILE=$(ls dist/*.whl | head -n 1)

if [ -f "$WHEEL_FILE" ]; then
    echo "  Checking: $WHEEL_FILE"

    # Extract wheel to temp directory
    TEMP_DIR=$(mktemp -d)
    unzip -q "$WHEEL_FILE" -d "$TEMP_DIR"

    # Check for .py files (excluding __init__.py)
    PY_FILES=$(find "$TEMP_DIR/q_store" -name "*.py" ! -name "__init__.py" 2>/dev/null | wc -l)

    # Check for compiled files
    SO_FILES=$(find "$TEMP_DIR/q_store" -name "*.so" -o -name "*.pyd" 2>/dev/null | wc -l)

    echo "  - Python source files (non-__init__): $PY_FILES"
    echo "  - Compiled binary files (.so/.pyd): $SO_FILES"

    rm -rf "$TEMP_DIR"

    if [ "$PY_FILES" -gt 0 ]; then
        echo -e "${RED}  ✗ WARNING: Source .py files found in wheel!${NC}"
        echo -e "${YELLOW}  Review your setup.py and MANIFEST.in${NC}"
    else
        echo -e "${GREEN}  ✓ No source code found - distribution is secure${NC}"
    fi

    if [ "$SO_FILES" -gt 0 ]; then
        echo -e "${GREEN}  ✓ Binary extensions present${NC}"
    else
        echo -e "${RED}  ✗ WARNING: No compiled binaries found!${NC}"
    fi
else
    echo -e "${RED}  ✗ Wheel file not found${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${GREEN}=========================================="
echo "Build Complete!"
echo "==========================================${NC}"
echo ""
echo "Distribution files created in: dist/"
ls -lh dist/
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test the wheel: pip install dist/*.whl"
echo "2. Upload to PyPI: twine upload dist/*"
echo "3. Or distribute privately to customers"
echo ""
echo -e "${YELLOW}IMPORTANT SECURITY NOTES:${NC}"
echo "- Never upload source distributions (.tar.gz) to PyPI"
echo "- Only upload binary wheels (.whl)"
echo "- Consider building wheels for multiple platforms:"
echo "  - Linux: Build on Linux machine or use cibuildwheel"
echo "  - macOS: Build on macOS machine"
echo "  - Windows: Build on Windows machine"
echo ""
