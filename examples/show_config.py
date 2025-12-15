#!/usr/bin/env python3
"""
Show current Q-Store configuration from .env file
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def mask_key(key):
    """Mask API key for security"""
    if not key:
        return "Not set"
    if len(key) <= 20:
        return key[:4] + "..." + key[-4:]
    return key[:15] + "..." + key[-4:]

print("\n" + "="*70)
print("Q-Store Configuration (from .env file)")
print("="*70 + "\n")

print("📌 PINECONE Configuration:")
print(f"  API Key:     {mask_key(os.getenv('PINECONE_API_KEY'))}")
print(f"  Environment: {os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')}")
print(f"  Index Name:  {os.getenv('PINECONE_INDEX_NAME', '(will be created as needed)')}")

print("\n🔬 IONQ Quantum Configuration:")
print(f"  API Key:     {mask_key(os.getenv('IONQ_API_KEY'))}")
print(f"  Target:      {os.getenv('IONQ_TARGET', 'simulator')}")

print("\n🤗 HUGGING FACE Configuration:")
print(f"  Token:       {mask_key(os.getenv('HUGGING_FACE_TOKEN'))}")

print("\n⚙️  General Settings:")
print(f"  Log Level:   {os.getenv('LOG_LEVEL', 'INFO')}")

print("\n" + "="*70)
print("Configuration Status")
print("="*70 + "\n")

# Check required settings
pinecone_ok = bool(os.getenv('PINECONE_API_KEY'))
ionq_ok = bool(os.getenv('IONQ_API_KEY'))

if pinecone_ok:
    print("✅ Pinecone: Configured (required)")
else:
    print("❌ Pinecone: NOT configured (REQUIRED)")
    print("   → Set PINECONE_API_KEY in .env file")
    print("   → Get your key from: https://www.pinecone.io/")

if ionq_ok:
    print("✅ IonQ: Configured (optional quantum features enabled)")
else:
    print("⚠️  IonQ: Not configured (quantum features disabled, classical-only mode)")
    print("   → Set IONQ_API_KEY in .env file to enable quantum features")
    print("   → Get your key from: https://ionq.com/")

print("\n" + "="*70)
print("Next Steps")
print("="*70 + "\n")

if not pinecone_ok:
    print("1. Edit .env file and add PINECONE_API_KEY")
    print("2. Run this script again to verify: python show_config.py")
    print()
elif not ionq_ok:
    print("You can now run examples in classical mode:")
    print("  python src/q_store_examples/examples_v3_2.py")
    print()
    print("To enable quantum features:")
    print("  1. Add IONQ_API_KEY to .env file")
    print("  2. Run with --no-mock flag:")
    print("     python src/q_store_examples/examples_v3_2.py --no-mock")
    print()
else:
    print("All configurations are set! You can now:")
    print()
    print("1. Run examples with mock backends (no API usage):")
    print("   python src/q_store_examples/examples_v3_2.py")
    print()
    print("2. Run examples with real Pinecone + IonQ:")
    print("   python src/q_store_examples/examples_v3_2.py --no-mock")
    print()
    print("3. Test connections:")
    print("   python test_pinecone_ionq_connection.py")
    print("   python test_cirq_adapter_fix.py")
    print()

print("="*70 + "\n")
