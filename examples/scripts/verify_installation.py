#!/usr/bin/env python3
"""
Q-Store Examples - Installation Verification Script
Checks that all dependencies are properly installed and configured
"""

import sys
import os
from pathlib import Path

def print_header(text: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_status(check_name: str, passed: bool, details: str = ""):
    """Print check status with formatting"""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status}{reset} {check_name}")
    if details:
        print(f"  {details}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 8)
    passed = version >= required
    
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    if not passed:
        details += f" (requires >= {required[0]}.{required[1]})"
    
    return passed, details

def check_package(package_name: str, import_name: str = None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"{package_name} installed"
    except ImportError:
        return False, f"{package_name} not found"

def check_environment():
    """Check environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        return False, ".env file not found (copy from .env.example)"
    
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check required keys
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key or pinecone_key == "your_pinecone_api_key_here":
            return False, ".env exists but PINECONE_API_KEY not set"
        
        return True, f".env configured with API keys"
    except Exception as e:
        return False, f"Error loading .env: {e}"

def main():
    """Main verification function"""
    print_header("Q-Store Examples - Installation Verification")
    
    checks_passed = 0
    checks_total = 0
    
    # Check Python version
    print("\n📋 System Requirements")
    checks_total += 1
    passed, details = check_python_version()
    if passed:
        checks_passed += 1
    print_status("Python version", passed, details)
    
    # Check core dependencies
    print("\n📦 Core Dependencies")
    
    core_packages = [
        ("q-store", "q_store"),
        ("numpy", "numpy"),
        ("python-dotenv", "dotenv"),
        ("pandas", "pandas"),
    ]
    
    for pkg_name, import_name in core_packages:
        checks_total += 1
        passed, details = check_package(pkg_name, import_name)
        if passed:
            checks_passed += 1
        print_status(pkg_name, passed, details)
    
    # Check ML dependencies (optional)
    print("\n🤖 ML Dependencies (Optional)")
    
    ml_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
    ]
    
    ml_available = True
    for pkg_name, import_name in ml_packages:
        passed, details = check_package(pkg_name, import_name)
        print_status(pkg_name, passed, details)
        if not passed:
            ml_available = False
    
    if not ml_available:
        print("\n  ℹ️  ML packages not installed. Install with:")
        print("     pip install -r requirements.txt")
    
    # Check environment configuration
    print("\n🔑 Environment Configuration")
    checks_total += 1
    passed, details = check_environment()
    if passed:
        checks_passed += 1
    print_status("Environment variables", passed, details)
    
    # Check Q-Store connection
    print("\n🔮 Q-Store Connectivity")
    checks_total += 1
    
    try:
        from q_store import QuantumDatabase
        print_status("Q-Store import", True, "Successfully imported QuantumDatabase")
        checks_passed += 1
        
        # Try to get version
        try:
            import q_store
            version = getattr(q_store, "__version__", "unknown")
            print(f"  Q-Store version: {version}")
        except:
            pass
            
    except Exception as e:
        print_status("Q-Store import", False, str(e))
    
    # Check example files
    print("\n📁 Example Files")
    checks_total += 1
    
    required_files = [
        "src/q_store_examples/basic_example.py",
        "src/q_store_examples/financial_example.py",
        "src/q_store_examples/quantum_db_quickstart.py",
        "src/q_store_examples/tinyllama_react_training.py",
        "src/q_store_examples/react_dataset_generator.py",
    ]
    
    all_exist = all(Path(f).exists() for f in required_files)
    if all_exist:
        checks_passed += 1
        print_status("Example files", True, f"{len(required_files)} files found")
    else:
        missing = [f for f in required_files if not Path(f).exists()]
        print_status("Example files", False, f"Missing: {', '.join(missing)}")
    
    # Summary
    print_header("Summary")
    
    percentage = (checks_passed / checks_total * 100) if checks_total > 0 else 0
    print(f"\n✅ Passed: {checks_passed}/{checks_total} checks ({percentage:.0f}%)\n")
    
    if checks_passed == checks_total:
        print("🎉 Installation verified! You're ready to run the examples.")
        print("\nTry running:")
        print("  python -m q_store_examples.basic_example")
        print("  python -m q_store_examples.quantum_db_quickstart")
        if ml_available:
            print("  python -m q_store_examples.tinyllama_react_training")
        print()
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.\n")
        print("Common fixes:")
        
        if not Path(".env").exists():
            print("  1. Create .env file:")
            print("     cp .env.example .env")
            print("     # Edit .env and add your API keys")
        
        print("\n  2. Install Q-Store:")
        print("     cd ..")
        print("     pip install -e .")
        print("     cd examples")
        
        print("\n  3. Install dependencies:")
        print("     pip install -r requirements.txt")
        
        print("\n  4. For minimal installation (no ML):")
        print("     pip install -r requirements-minimal.txt")
        
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
