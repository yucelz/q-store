"""
Quick Test Script for Image Classification Example

This script validates the example can be imported and helps check dependencies.
"""

import sys
from pathlib import Path

def check_dependency(name, import_path):
    """Check if a dependency is available."""
    try:
        module = __import__(import_path)
        print(f"✓ {name} available")
        return True
    except (ImportError, AttributeError) as e:
        print(f"✗ {name} not available: {e}")
        return False

def main():
    print("="*70)
    print("IMAGE CLASSIFICATION EXAMPLE - DEPENDENCY CHECK")
    print("="*70)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    # Check required dependencies
    print("\n📦 Checking dependencies...")
    print("-"*70)

    deps = {
        'NumPy': 'numpy',
        'TensorFlow': 'tensorflow',
        'Keras': 'tensorflow.keras',
        'Matplotlib': 'matplotlib',
        'Python-dotenv': 'dotenv',
        'Q-Store Core': 'q_store.core',
        'Q-Store Layers': 'q_store.layers',
    }

    results = {}
    for name, import_path in deps.items():
        results[name] = check_dependency(name, import_path)

    # Check optional dependencies
    print("\n📦 Checking optional dependencies...")
    print("-"*70)

    optional_deps = {
        'Q-Store TensorFlow': 'q_store.tensorflow',
        'Pillow (PIL)': 'PIL',
    }

    for name, import_path in optional_deps.items():
        results[name] = check_dependency(name, import_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    required = ['NumPy', 'TensorFlow', 'Q-Store Core', 'Q-Store Layers']
    missing_required = [name for name in required if not results.get(name, False)]

    if not missing_required:
        print("\n✓ All required dependencies are available!")
        print("\n🎉 You can run the image classification example:")
        print("   python examples/ml_frameworks/image_classification_from_scratch.py --quick-test")
        return 0
    else:
        print("\n⚠️  Missing required dependencies:")
        for name in missing_required:
            print(f"   - {name}")

        print("\n📥 Install missing dependencies:")
        if 'TensorFlow' in missing_required:
            print("   pip install tensorflow>=2.13.0")
        if 'Q-Store Core' in missing_required or 'Q-Store Layers' in missing_required:
            print("   pip install -e .")

        return 1

if __name__ == "__main__":
    sys.exit(main())
