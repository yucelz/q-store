#!/usr/bin/env python
"""
Quick verification that the TinyLlama training example is ready to run
"""

import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✓ Found: {filepath}")
        return True
    else:
        print(f"✗ Missing: {filepath}")
        return False

def check_import(module_name):
    """Check if module can be imported"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✓ {module_name} is available")
            return True
        else:
            print(f"✗ {module_name} is NOT available")
            return False
    except ImportError:
        print(f"✗ {module_name} is NOT available")
        return False

def main():
    print("=" * 60)
    print("TinyLlama React Training Example - Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check files
    print("Checking files...")
    files_to_check = [
        "examples/tinyllama_react_training.py",
        "examples/TINYLLAMA_TRAINING_README.md",
        "examples/IMPROVEMENTS_SUMMARY.md"
    ]
    
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_checks_passed = False
    
    print()
    
    # Check core dependencies
    print("Checking core dependencies...")
    core_deps = ["numpy", "q_store", "dotenv"]
    
    for dep in core_deps:
        if not check_import(dep):
            all_checks_passed = False
    
    print()
    
    # Check optional dependencies
    print("Checking optional dependencies (for full training)...")
    optional_deps = ["transformers", "peft", "datasets", "torch"]
    
    optional_available = True
    for dep in optional_deps:
        if not check_import(dep):
            optional_available = False
    
    if not optional_available:
        print("\n⚠️  Note: Optional dependencies not installed.")
        print("   The example will work for quantum database demo,")
        print("   but actual model training requires:")
        print("   pip install transformers peft datasets torch")
    
    print()
    
    # Check syntax
    print("Checking Python syntax...")
    try:
        import py_compile
        py_compile.compile("examples/tinyllama_react_training.py", doraise=True)
        print("✓ Python syntax is valid")
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error: {e}")
        all_checks_passed = False
    
    print()
    print("=" * 60)
    
    if all_checks_passed:
        print("✅ All core checks passed!")
        print()
        print("To run the example:")
        print("  1. Set up .env file with PINECONE_API_KEY and PINECONE_ENVIRONMENT")
        print("  2. Optionally add IONQ_API_KEY for quantum features")
        print("  3. Run: python examples/tinyllama_react_training.py")
        print()
        if not optional_available:
            print("Note: Install transformers for full training capability")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
