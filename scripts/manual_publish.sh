#!/bin/bash
# Manual PyPI Publishing Script for Q-Store
# This script builds, repairs, and uploads wheels to PyPI manually
# Used as a workaround when GitHub Actions workflows encounter issues

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Q-Store Manual PyPI Publishing Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Check dependencies
echo -e "${BLUE}[1/7]${NC} Checking dependencies..."

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi

if ! python -m pip show twine &> /dev/null; then
    echo -e "${YELLOW}⚠ twine not installed. Installing...${NC}"
    python -m pip install twine
fi

if ! command -v auditwheel &> /dev/null; then
    echo -e "${YELLOW}⚠ auditwheel not installed. Installing...${NC}"
    python -m pip install auditwheel
fi

echo -e "${GREEN}✓ All dependencies available${NC}"
echo ""

# Step 2: Check PyPI credentials
echo -e "${BLUE}[2/7]${NC} Checking PyPI credentials..."

if [[ -z "${TWINE_USERNAME}" ]] || [[ -z "${TWINE_PASSWORD}" ]]; then
    echo -e "${YELLOW}⚠ PyPI credentials not found in environment${NC}"
    echo ""

    # Check if .pypirc exists
    if [[ -f ~/.pypirc ]]; then
        echo -e "${GREEN}✓ Found ~/.pypirc configuration file${NC}"
        echo "  Credentials will be read from ~/.pypirc during upload"
        echo ""
    else
        echo "Would you like to enter your PyPI API token now?"
        echo ""
        read -p "Enter credentials interactively? (y/N) " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "Please enter your PyPI API token:"
            echo "(Get your token from: https://pypi.org/manage/account/token/)"
            echo ""

            # Set username to __token__ (PyPI API token standard)
            export TWINE_USERNAME="__token__"

            # Prompt for password securely
            read -s -p "PyPI API Token: " TWINE_PASSWORD
            export TWINE_PASSWORD
            echo ""
            echo ""

            if [[ -z "${TWINE_PASSWORD}" ]]; then
                echo -e "${RED}✗ No token provided${NC}"
                exit 1
            fi

            echo -e "${GREEN}✓ PyPI credentials set for this session${NC}"
        else
            echo ""
            echo -e "${YELLOW}Continuing without credentials...${NC}"
            echo "You can set them before upload using:"
            echo "  export TWINE_USERNAME=__token__"
            echo "  export TWINE_PASSWORD=your_pypi_api_token_here"
            echo ""
            echo "Or create ~/.pypirc with:"
            echo "  [pypi]"
            echo "  username = __token__"
            echo "  password = your_pypi_api_token_here"
            echo ""
        fi
    fi
else
    echo -e "${GREEN}✓ PyPI credentials found in environment${NC}"
fi
echo ""

# Step 3: Clean old builds
echo -e "${BLUE}[3/7]${NC} Cleaning old build artifacts..."

if [[ -d "dist" ]]; then
    echo "  Removing dist/"
    rm -rf dist/
fi

if [[ -d "build" ]]; then
    echo "  Removing build/"
    rm -rf build/
fi

if [[ -d "wheelhouse" ]]; then
    echo "  Removing wheelhouse/"
    rm -rf wheelhouse/
fi

# Remove .egg-info directories
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✓ Build artifacts cleaned${NC}"
echo ""

# Step 4: Build wheel
echo -e "${BLUE}[4/7]${NC} Building wheel..."
echo ""

python setup.py bdist_wheel

if [[ ! -f dist/*.whl ]]; then
    echo -e "${RED}✗ Wheel build failed${NC}"
    exit 1
fi

BUILT_WHEEL=$(ls -1 dist/*.whl | head -n 1)
echo ""
echo -e "${GREEN}✓ Wheel built: $(basename $BUILT_WHEEL)${NC}"
echo ""

# Step 5: Repair wheel with auditwheel
echo -e "${BLUE}[5/7]${NC} Repairing wheel with auditwheel..."
echo ""

mkdir -p wheelhouse/
auditwheel repair "$BUILT_WHEEL" -w wheelhouse/

REPAIRED_WHEEL=$(ls -1 wheelhouse/*.whl | head -n 1)

if [[ ! -f "$REPAIRED_WHEEL" ]]; then
    echo -e "${RED}✗ Wheel repair failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Wheel repaired: $(basename $REPAIRED_WHEEL)${NC}"
echo ""

# Step 6: Validate wheel
echo -e "${BLUE}[6/7]${NC} Validating wheel with twine..."
echo ""

python -m twine check "$REPAIRED_WHEEL"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}✗ Wheel validation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Wheel validation passed${NC}"
echo ""

# Step 7: Upload to PyPI
echo -e "${BLUE}[7/7]${NC} Uploading to PyPI..."
echo ""

# Show file details
ls -lh "$REPAIRED_WHEEL"
echo ""

read -p "Ready to upload to PyPI. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠ Upload cancelled${NC}"
    echo ""
    echo "To upload manually later, run:"
    echo "  python -m twine upload $REPAIRED_WHEEL"
    exit 0
fi

echo ""
python -m twine upload "$REPAIRED_WHEEL"

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Successfully published to PyPI!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Wheel published: $(basename $REPAIRED_WHEEL)"
    echo ""
else
    echo -e "${RED}✗ Upload failed${NC}"
    exit 1
fi
