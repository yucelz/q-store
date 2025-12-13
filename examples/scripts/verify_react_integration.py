#!/usr/bin/env python3
"""
Verification script for React Training Integration
Tests that all components are properly integrated
"""

import sys
import os
from pathlib import Path

def print_status(check_name: str, passed: bool, details: str = ""):
    """Print check status with formatting"""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {check_name}")
    if details:
        print(f"  {details}")

def main():
    print("=" * 70)
    print("React Training Integration - Verification")
    print("=" * 70)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Files exist
    print("📁 Checking files...")
    checks_total += 1
    
    required_files = [
        "src/q_store_examples/react_dataset_generator.py",
        "src/q_store_examples/tinyllama_react_training.py",
        "scripts/run_react_training.sh",
        "docs/REACT_TRAINING_WORKFLOW.md",
        "docs/REACT_QUICK_REFERENCE.md"
    ]
    
    all_exist = True
    missing = []
    for filename in required_files:
        if not Path(filename).exists():
            all_exist = False
            missing.append(filename)
    
    if all_exist:
        checks_passed += 1
        print_status("All required files present", True, f"{len(required_files)} files found")
    else:
        print_status("Missing files", False, f"Missing: {', '.join(missing)}")
    
    print()
    
    # Check 2: Import generator
    print("📦 Checking imports...")
    checks_total += 1
    
    try:
        from q_store_examples.react_dataset_generator import ReactDatasetGenerator
        checks_passed += 1
        print_status("ReactDatasetGenerator imports successfully", True)
    except Exception as e:
        print_status("ReactDatasetGenerator import failed", False, str(e))
    
    print()
    
    # Check 3: Import training components
    checks_total += 1
    try:
        # Add parent directory to path for q_store import
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from q_store_examples.tinyllama_react_training import QuantumTrainingDataManager, TrainingConfig
        checks_passed += 1
        print_status("Training components import successfully", True)
    except Exception as e:
        print_status("Training components import failed", False, str(e))
    
    print()
    
    # Check 4: Generator functionality
    print("🔨 Testing generator...")
    checks_total += 1
    
    try:
        gen = ReactDatasetGenerator()
        gen.generate_component_samples(10)
        gen.generate_bug_fixing_samples(5)
        
        if len(gen.samples) >= 1:
            checks_passed += 1
            print_status("Generator creates samples", True, f"{len(gen.samples)} samples created")
        else:
            print_status("Generator creates samples", False, "No samples generated")
    except Exception as e:
        print_status("Generator functionality", False, str(e))
    
    print()
    
    # Check 5: Script is executable
    print("🚀 Checking automation script...")
    checks_total += 1
    
    script_path = Path("scripts/run_react_training.sh")
    if script_path.exists() and os.access(script_path, os.X_OK):
        checks_passed += 1
        print_status("Automation script is executable", True)
    else:
        print_status("Automation script is executable", False, "Run: chmod +x scripts/run_react_training.sh")
    
    print()
    
    # Check 6: Environment variables
    print("🔑 Checking environment setup...")
    checks_total += 1
    
    env_file = Path("../.env")
    if env_file.exists():
        checks_passed += 1
        print_status("Environment file exists", True, str(env_file))
    else:
        print_status("Environment file exists", False, "Create .env file with API keys")
    
    print()
    
    # Summary
    print("=" * 70)
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)
    print()
    
    if checks_passed == checks_total:
        print("✅ All checks passed! Ready to run:")
        print()
        print("   ./scripts/run_react_training.sh")
        print()
        return 0
    else:
        print("⚠️  Some checks failed. Review the issues above.")
        print()
        print("Quick fixes:")
        print("   - Make script executable: chmod +x scripts/run_react_training.sh")
        print("   - Create .env file with API keys")
        print("   - Install dependencies: pip install -r requirements.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
