#!/usr/bin/env python3
"""
Run all Q-Store v4.1 validation examples.

This script runs all validation examples in sequence and reports results.
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path, description):
    """Run a Python script and report results."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            check=False
        )

        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def main():
    """Run all validation scripts."""
    examples_dir = Path(__file__).parent
    validation_dir = examples_dir / "validation"

    print("=" * 70)
    print("Q-Store v4.1 Example Runner")
    print("=" * 70)

    scripts = [
        (validation_dir / "gradient_validation.py", "Gradient Validation"),
        (validation_dir / "simple_classification.py", "Simple Classification"),
    ]

    results = {}
    for script_path, description in scripts:
        if script_path.exists():
            results[description] = run_script(script_path, description)
        else:
            print(f"✗ {description} - script not found: {script_path}")
            results[description] = False

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    for description, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{description}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if all(results.values()):
        print("\nAll validation tests PASSED! 🎉")
        return 0
    else:
        print("\nSome tests failed - please investigate")
        return 1


if __name__ == '__main__':
    sys.exit(main())
