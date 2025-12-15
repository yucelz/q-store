#!/usr/bin/env python3
"""
Verification script for Q-Store project structure
Tests that the new organization follows Python best practices
"""

import sys
from pathlib import Path


def verify_structure():
    """Verify the project structure"""
    print("=" * 60)
    print("Q-Store Project Structure Verification")
    print("=" * 60)
    print()

    # Get project root
    project_root = Path(__file__).parent

    # Define expected structure
    expected_dirs = [
        "src/q_store",
        "src/q_store/core",
        "src/q_store/backends",
        "src/q_store/utils",
        "tests",
        "docs",
        "examples",
    ]

    expected_files = [
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "Makefile",
        ".editorconfig",
        "README.md",
        "LICENCE",
        "src/q_store/__init__.py",
        "src/q_store/core/__init__.py",
        "src/q_store/backends/__init__.py",
        "src/q_store/utils/__init__.py",
        "src/q_store/core/quantum_database.py",
        "src/q_store/core/state_manager.py",
        "src/q_store/core/entanglement_registry.py",
        "src/q_store/core/tunneling_engine.py",
        "src/q_store/backends/ionq_backend.py",
        "docs/README.md",
        "docs/PROJECT_STRUCTURE.md",
        "docs/RESTRUCTURING_SUMMARY.md",
    ]

    # Check directories
    print("Checking directory structure...")
    all_dirs_ok = True
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_dirs_ok = False

    print()

    # Check files
    print("Checking required files...")
    all_files_ok = True
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_files_ok = False

    print()

    # Test imports
    print("Testing package imports...")
    try:
        from q_store import (
            QuantumDatabase,
            DatabaseConfig,
            IonQQuantumBackend,
            StateManager,
            QuantumState,
            StateStatus,
            EntanglementRegistry,
            TunnelingEngine,
        )
        print("  ✓ All main imports successful")

        # Check version
        import q_store
        print(f"  ✓ Package version: {q_store.__version__}")

        imports_ok = True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        imports_ok = False

    print()

    # Summary
    print("=" * 60)
    if all_dirs_ok and all_files_ok and imports_ok:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("The project structure follows Python best practices:")
        print("  • src/ layout (PEP 420)")
        print("  • pyproject.toml configuration (PEP 621)")
        print("  • Modular package organization")
        print("  • Proper namespace packages")
        print("  • Development automation (Makefile)")
        print("  • Comprehensive documentation")
        print()
        print("Next steps:")
        print("  1. Run 'make verify' to check code quality")
        print("  2. Run 'make test' to run the test suite")
        print("  3. See docs/PROJECT_STRUCTURE.md for details")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        if not all_dirs_ok:
            print("  • Some directories are missing")
        if not all_files_ok:
            print("  • Some files are missing")
        if not imports_ok:
            print("  • Package imports failed")
        return 1


if __name__ == "__main__":
    sys.exit(verify_structure())
