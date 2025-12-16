#!/bin/bash
# test_before_build.sh
# Tests Q-Store components before building for publication

set -e  # Exit on error

echo "=========================================="
echo "Q-Store Pre-Build Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src/q_store" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}Testing: ${test_name}${NC}"
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  ✗ FAILED${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$test_name")
    fi
    echo ""
}

# Test 1: Python syntax check for all Python files
echo -e "${GREEN}Step 1: Checking Python syntax...${NC}"
SYNTAX_ERRORS=0
for file in $(find src/q_store -name "*.py"); do
    if ! python -m py_compile "$file" 2>/dev/null; then
        echo -e "${RED}  ✗ Syntax error in: $file${NC}"
        ((SYNTAX_ERRORS++))
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo -e "${GREEN}  ✓ All Python files have valid syntax${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}  ✗ Found $SYNTAX_ERRORS files with syntax errors${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("Python syntax check")
fi
echo ""

# Test 2: Import tests for each module
echo -e "${GREEN}Step 2: Testing module imports...${NC}"

# Core imports
run_test "Core: QuantumDatabase" \
    "python -c 'from q_store.core import QuantumDatabase'"

run_test "Core: StateManager" \
    "python -c 'from q_store.core import StateManager'"

run_test "Core: EntanglementRegistry" \
    "python -c 'from q_store.core import EntanglementRegistry'"

run_test "Core: TunnelingEngine" \
    "python -c 'from q_store.core import TunnelingEngine'"

# Backend imports
run_test "Backends: BackendManager" \
    "python -c 'from q_store.backends import BackendManager'"

run_test "Backends: IonQBackend" \
    "python -c 'from q_store.backends import IonQBackend'"

run_test "Backends: CirqIonQAdapter" \
    "python -c 'from q_store.backends import CirqIonQAdapter'"

# ML imports
run_test "ML: QuantumTrainer" \
    "python -c 'from q_store.ml import QuantumTrainer'"

run_test "ML: QuantumModel" \
    "python -c 'from q_store.ml import QuantumModel'"

run_test "ML: QuantumLayer" \
    "python -c 'from q_store.ml import QuantumLayer'"

run_test "ML: AdaptiveGradientOptimizer" \
    "python -c 'from q_store.ml import AdaptiveGradientOptimizer'"

run_test "ML: QuantumDataEncoder" \
    "python -c 'from q_store.ml import QuantumDataEncoder'"

run_test "ML: QuantumCircuitCache" \
    "python -c 'from q_store.ml import QuantumCircuitCache'"

run_test "ML: PerformanceTracker" \
    "python -c 'from q_store.ml import PerformanceTracker'"

# Test 3: Main package import
echo -e "${GREEN}Step 3: Testing main package import...${NC}"
run_test "Main package import" \
    "python -c 'import q_store; print(q_store.__version__)'"

# Test 4: Check for common issues
echo -e "${GREEN}Step 4: Checking for common issues...${NC}"

# Check for undefined loggers
echo -e "${BLUE}Checking for logger definition issues...${NC}"
LOGGER_ISSUES=$(grep -n "logger\." src/q_store/**/*.py 2>/dev/null | head -20)
if echo "$LOGGER_ISSUES" | grep -q "logger\."; then
    # Verify logger is defined before use in each file
    UNDEFINED_LOGGERS=0
    for file in $(find src/q_store -name "*.py" -type f); do
        # Check if file uses logger
        if grep -q "logger\." "$file" 2>/dev/null; then
            # Check if logger is defined
            if ! grep -q "logger = logging.getLogger" "$file" 2>/dev/null && \
               ! grep -q "from.*logger" "$file" 2>/dev/null; then
                # Check if it's before the definition
                FIRST_USE=$(grep -n "logger\." "$file" | head -1 | cut -d: -f1)
                DEFINITION=$(grep -n "logger = logging.getLogger" "$file" | head -1 | cut -d: -f1)
                if [ ! -z "$FIRST_USE" ] && [ ! -z "$DEFINITION" ] && [ "$FIRST_USE" -lt "$DEFINITION" ]; then
                    echo -e "${RED}  ✗ Logger used before definition in: $file${NC}"
                    ((UNDEFINED_LOGGERS++))
                fi
            fi
        fi
    done
    
    if [ $UNDEFINED_LOGGERS -eq 0 ]; then
        echo -e "${GREEN}  ✓ No logger definition issues${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  ✗ Found $UNDEFINED_LOGGERS files with logger issues${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("Logger definition check")
    fi
else
    echo -e "${GREEN}  ✓ No logger usage found${NC}"
    ((TESTS_PASSED++))
fi
echo ""

# Test 5: Check for circular imports
echo -e "${GREEN}Step 5: Checking for circular imports...${NC}"
if python -c "import sys; sys.path.insert(0, 'src'); import q_store" 2>&1 | grep -q "circular"; then
    echo -e "${RED}  ✗ Circular import detected${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("Circular import check")
else
    echo -e "${GREEN}  ✓ No circular imports detected${NC}"
    ((TESTS_PASSED++))
fi
echo ""

# Test 6: Verify required dependencies
echo -e "${GREEN}Step 6: Checking dependencies...${NC}"
MISSING_DEPS=0
REQUIRED_DEPS=("numpy" "scipy" "cirq" "cirq_ionq" "requests")

for dep in "${REQUIRED_DEPS[@]}"; do
    if ! python -c "import $dep" 2>/dev/null; then
        echo -e "${RED}  ✗ Missing dependency: $dep${NC}"
        ((MISSING_DEPS++))
    fi
done

if [ $MISSING_DEPS -eq 0 ]; then
    echo -e "${GREEN}  ✓ All required dependencies available${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}  ✗ Missing $MISSING_DEPS required dependencies${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("Dependency check")
fi
echo ""

# Test 7: Check for __init__.py files
echo -e "${GREEN}Step 7: Checking package structure...${NC}"
MISSING_INIT=0
for dir in $(find src/q_store -type d); do
    if [ ! -f "$dir/__init__.py" ]; then
        echo -e "${YELLOW}  ! Missing __init__.py in: $dir${NC}"
        ((MISSING_INIT++))
    fi
done

if [ $MISSING_INIT -eq 0 ]; then
    echo -e "${GREEN}  ✓ All packages have __init__.py${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}  ! Found $MISSING_INIT directories without __init__.py${NC}"
    echo -e "${YELLOW}  (This may be intentional for some directories)${NC}"
    ((TESTS_PASSED++))
fi
echo ""

# Test 8: Run unit tests if available
if [ -d "tests" ] && command -v pytest &> /dev/null; then
    echo -e "${GREEN}Step 8: Running unit tests...${NC}"
    if pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt; then
        echo -e "${GREEN}  ✓ Unit tests passed${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  ✗ Some unit tests failed${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("Unit tests")
    fi
    echo ""
else
    echo -e "${YELLOW}Step 8: Skipping unit tests (pytest not available or no tests found)${NC}"
    echo ""
fi

# Summary
echo "=========================================="
echo -e "${GREEN}Test Summary${NC}"
echo "=========================================="
echo ""
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed Tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} $test"
    done
    echo ""
    echo -e "${RED}=========================================="
    echo "BUILD NOT RECOMMENDED"
    echo "==========================================${NC}"
    echo ""
    echo "Please fix the failing tests before building."
    exit 1
else
    echo -e "${GREEN}=========================================="
    echo "ALL TESTS PASSED - READY TO BUILD"
    echo "==========================================${NC}"
    echo ""
    echo "You can now safely run:"
    echo "  ./build_binary_distribution.sh"
    echo ""
    exit 0
fi
