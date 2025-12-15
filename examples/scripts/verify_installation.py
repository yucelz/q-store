#!/usr/bin/env python3
"""
Quick verification script to test Q-Store installation
Run this after completing the installation steps
"""

import sys
import os

def check_imports():
    """Check if all required packages can be imported"""
    print("Checking imports...")
    
    required = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('cirq', 'Cirq'),
        ('cirq_ionq', 'Cirq-IonQ'),
        ('pinecone', 'Pinecone'),
        ('dotenv', 'python-dotenv'),
    ]
    
    optional = [
        ('q_store', 'Q-Store'),
    ]
    
    all_good = True
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - NOT INSTALLED")
            print(f"    Error: {e}")
            all_good = False
    
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ⚠ {name} - Install with: pip install -e .")
    
    return all_good

def check_env_file():
    """Check if .env file exists and has required keys"""
    print("\nChecking .env file...")
    
    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("    Create one with: cp .env.example .env")
        return False
    
    print("  ✓ .env file exists")
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    ionq_key = os.getenv('IONQ_API_KEY')
    
    if pinecone_key:
        print(f"  ✓ PINECONE_API_KEY set ({pinecone_key[:8]}...)")
    else:
        print("  ✗ PINECONE_API_KEY not set")
        return False
    
    if pinecone_env:
        print(f"  ✓ PINECONE_ENVIRONMENT set ({pinecone_env})")
    else:
        print("  ⚠ PINECONE_ENVIRONMENT not set (using default: us-east-1)")
    
    if ionq_key:
        print(f"  ✓ IONQ_API_KEY set ({ionq_key[:8]}...)")
    else:
        print("  ⚠ IONQ_API_KEY not set (quantum features will be disabled)")
    
    return True

def test_basic_functionality():
    """Test basic Q-Store functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from q_store import QuantumDatabase, DatabaseConfig
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Create minimal config
        config = DatabaseConfig(
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
            pinecone_index_name='test-verification',
            pinecone_dimension=128,
            ionq_api_key=os.getenv('IONQ_API_KEY'),
        )
        
        print("  ✓ DatabaseConfig created")
        
        db = QuantumDatabase(config)
        print("  ✓ QuantumDatabase instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks"""
    print("="*60)
    print("Q-Store Installation Verification")
    print("="*60 + "\n")
    
    # Check imports
    imports_ok = check_imports()
    
    # Check .env
    env_ok = check_env_file()
    
    # Test basic functionality
    if imports_ok and env_ok:
        func_ok = test_basic_functionality()
    else:
        func_ok = False
    
    print("\n" + "="*60)
    if imports_ok and env_ok and func_ok:
        print("✓ All checks passed!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run the quickstart: python examples/quantum_db_quickstart.py")
        print("  2. Read QUICKSTART.md for your first program")
        print("  3. Explore examples/ directory")
        return 0
    else:
        print("✗ Some checks failed")
        print("="*60)
        print("\nPlease fix the issues above and try again.")
        print("See QUICKSTART.md for detailed installation instructions.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
