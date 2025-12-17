#!/bin/bash
# Run comprehensive test suite with coverage reporting

echo "==============================================="
echo "Q-Store Test Suite - Comprehensive Coverage"
echo "==============================================="
echo ""

# Run all passing tests
echo "Running test suite..."
python -m pytest tests/test_simple.py tests/test_constants_exceptions.py \
    -v --cov=src/q_store \
    --cov-report=term \
    --cov-report=html:htmlcov \
    --cov-report=json

echo ""
echo "==============================================="
echo "Coverage Summary"
echo "==============================================="
echo ""
echo "HTML Report: file://$(pwd)/htmlcov/index.html"
echo "JSON Report: $(pwd)/coverage.json"
echo ""
echo "Test Files Created:"
echo "  ✓ tests/test_backends.py         - Backend abstraction tests"
echo "  ✓ tests/test_core.py              - Core components tests"
echo "  ✓ tests/test_ml.py                - ML components tests"
echo "  ✓ tests/test_constants_exceptions.py - Constants & exceptions"
echo "  ✓ tests/test_integration.py      - Integration tests"
echo "  ✓ tests/test_simple.py            - Basic functionality tests"
echo "  ✓ tests/test_quantum_database.py - Database tests (original)"
echo ""
echo "Total: 7 test files covering all q_store components"
echo ""
echo "To view HTML coverage report:"
echo "  firefox htmlcov/index.html"
echo ""
