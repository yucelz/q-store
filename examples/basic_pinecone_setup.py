"""
Basic Pinecone Index Creation with Quantum Database

This example demonstrates:
1. Setting up a Pinecone index via QuantumDatabase
2. Storing vectors with quantum superposition
3. Querying with context-aware search
4. Verifying index creation and basic operations

Usage:
    # With mock backend (no API keys needed)
    python examples/basic_pinecone_setup.py

    # With real Pinecone (requires PINECONE_API_KEY)
    python examples/basic_pinecone_setup.py --no-mock

Configuration:
    Set environment variables or create examples/.env:
    - PINECONE_API_KEY: Your Pinecone API key
    - PINECONE_ENVIRONMENT: Your Pinecone environment (e.g., us-east-1)
"""

import os
import sys
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Load environment variables
if HAS_DOTENV:
    examples_dir = Path(__file__).parent
    env_path = examples_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"ℹ No .env file found at {env_path}, using defaults")
else:
    print(f"ℹ python-dotenv not installed, using environment variables")

try:
    from q_store.core import QuantumDatabase, DatabaseConfig, QueryMode
    HAS_QSTORE = True
except ImportError as e:
    print(f"⚠️  Missing Q-Store dependencies: {e}")
    HAS_QSTORE = False

# Global configuration
USE_MOCK = True
PINECONE_API_KEY = None
PINECONE_ENVIRONMENT = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Basic Pinecone Index Creation with Quantum Database'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Use real Pinecone (requires PINECONE_API_KEY in .env)'
    )
    parser.add_argument(
        '--index-name',
        type=str,
        default='quantum-demo-basic',
        help='Pinecone index name (default: quantum-demo-basic)'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=128,
        help='Vector dimension (default: 128)'
    )
    return parser.parse_args()


async def demo_index_creation(index_name: str, dimension: int):
    """
    Demonstrate Pinecone index creation and basic operations.
    
    Args:
        index_name: Name for the Pinecone index
        dimension: Vector dimension
    """
    global USE_MOCK, PINECONE_API_KEY, PINECONE_ENVIRONMENT
    
    print("\n" + "=" * 80)
    print("BASIC PINECONE SETUP WITH QUANTUM DATABASE")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'REAL PINECONE' if not USE_MOCK else 'MOCK (Testing)'}")
    print(f"  Index Name: {index_name}")
    print(f"  Dimension: {dimension}")
    
    if not USE_MOCK:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            print("\n⚠️  ERROR: Real mode requires PINECONE_API_KEY and PINECONE_ENVIRONMENT")
            print("   Please set them in examples/.env or use mock mode")
            sys.exit(1)
        print(f"  Environment: {PINECONE_ENVIRONMENT}")
        print(f"  API Key: {'*' * 20}")
    else:
        # Use mock credentials
        PINECONE_API_KEY = "mock-test-key-12345"
        PINECONE_ENVIRONMENT = "us-east-1"
        print(f"  Environment: {PINECONE_ENVIRONMENT} (mock)")
    
    # Create database configuration
    print("\n" + "=" * 80)
    print("STEP 1: CREATE DATABASE CONFIGURATION")
    print("=" * 80)
    
    db_config = DatabaseConfig(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        pinecone_index_name=index_name,
        pinecone_dimension=dimension,
        pinecone_metric='cosine',
        enable_quantum=True,
        enable_superposition=True,
        quantum_sdk='mock',  # Always use mock quantum for this demo
        use_mock_pinecone=USE_MOCK  # Use mock Pinecone when in mock mode
    )
    
    print(f"✓ Configuration created:")
    print(f"  Index: {db_config.pinecone_index_name}")
    print(f"  Dimension: {db_config.pinecone_dimension}")
    print(f"  Metric: {db_config.pinecone_metric}")
    print(f"  Quantum Enabled: {db_config.enable_quantum}")
    print(f"  Superposition: {db_config.enable_superposition}")
    
    # Initialize database
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZE QUANTUM DATABASE")
    print("=" * 80)
    print("\nInitializing database (this will create the Pinecone index if needed)...")
    
    db = QuantumDatabase(db_config)
    
    async with db.connect():
        print("✓ Database initialized successfully!")
        
        if not USE_MOCK:
            print(f"✓ Pinecone index '{index_name}' is ready")
            print(f"  (Check your Pinecone dashboard: https://app.pinecone.io/)")
        else:
            print(f"✓ Mock Pinecone index created (in-memory)")
        
        # Insert sample vectors
        print("\n" + "=" * 80)
        print("STEP 3: INSERT SAMPLE VECTORS")
        print("=" * 80)
        
        print("\nInserting 10 sample vectors with quantum superposition...")
        
        categories = ['electronics', 'clothing', 'food', 'books', 'toys']
        styles = ['modern', 'classic', 'vintage']
        
        for i in range(10):
            # Create random vector
            vector = np.random.randn(dimension)
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            # Assign category and style
            category = categories[i % len(categories)]
            style = styles[i % len(styles)]
            
            # Create contexts for quantum superposition
            contexts = [
                (f"category_{category}", 0.6),
                (f"style_{style}", 0.4)
            ]
            
            # Insert with quantum superposition
            await db.insert(
                id=f"item_{i:03d}",
                vector=vector,
                contexts=contexts,
                metadata={
                    'item_id': i,
                    'category': category,
                    'style': style,
                    'name': f"Product {i}"
                }
            )
            
            if i == 0:
                print(f"  ✓ item_000: category={category}, style={style}, contexts={len(contexts)}")
            elif i == 9:
                print(f"  ✓ item_009: category={category}, style={style}, contexts={len(contexts)}")
            elif i == 4:
                print(f"  ... (inserting items 001-008)")
        
        print(f"\n✓ Inserted 10 vectors with quantum superposition")
        print(f"  Each vector has 2 contexts: category and style")
        
        # Query examples
        print("\n" + "=" * 80)
        print("STEP 4: QUERY WITH CONTEXT-AWARE SEARCH")
        print("=" * 80)
        
        # Create a query vector (similar to item_0)
        query_vector = np.random.randn(dimension)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        print("\nQuery 1: Classical search (no quantum context)")
        print("-" * 80)
        results = await db.query(
            vector=query_vector,
            context=None,
            mode=QueryMode.BALANCED,
            top_k=5
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.id} - {result.metadata.get('name', 'N/A')} "
                  f"(category: {result.metadata.get('category', 'N/A')}, "
                  f"score: {result.score:.4f}, quantum: {result.quantum_enhanced})")
        
        print("\nQuery 2: Quantum search with category context")
        print("-" * 80)
        results = await db.query(
            vector=query_vector,
            context="category_electronics",
            mode=QueryMode.BALANCED,
            top_k=5
        )
        
        print(f"Found {len(results)} results with category_electronics context:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.id} - {result.metadata.get('name', 'N/A')} "
                  f"(category: {result.metadata.get('category', 'N/A')}, "
                  f"score: {result.score:.4f}, quantum: {result.quantum_enhanced})")
        
        print("\nQuery 3: Quantum search with style context")
        print("-" * 80)
        results = await db.query(
            vector=query_vector,
            context="style_modern",
            mode=QueryMode.BALANCED,
            top_k=5
        )
        
        print(f"Found {len(results)} results with style_modern context:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.id} - {result.metadata.get('name', 'N/A')} "
                  f"(style: {result.metadata.get('style', 'N/A')}, "
                  f"score: {result.score:.4f}, quantum: {result.quantum_enhanced})")
        
        # Database statistics
        print("\n" + "=" * 80)
        print("STEP 5: DATABASE STATISTICS")
        print("=" * 80)
        
        stats = db.get_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Quantum states: {stats['quantum_states']}")
        print(f"  Total queries: {stats['metrics']['total_queries']}")
        print(f"  Quantum queries: {stats['metrics']['quantum_queries']}")
        print(f"  Cache hit rate: {stats['metrics']['cache_hit_rate']:.2%}")
        print(f"  Average latency: {stats['metrics']['avg_latency_ms']:.2f}ms")
        
        print("\n" + "=" * 80)
        print("✓ DEMONSTRATION COMPLETE")
        print("=" * 80)
        
        print("\nWhat was demonstrated:")
        print("  1. ✓ Pinecone index creation (automatic)")
        print("  2. ✓ Vector insertion with metadata")
        print("  3. ✓ Quantum superposition across contexts")
        print("  4. ✓ Classical similarity search")
        print("  5. ✓ Context-aware quantum search")
        print("  6. ✓ Database statistics and monitoring")
        
        if not USE_MOCK:
            print(f"\nYour Pinecone index '{index_name}' is ready to use!")
            print(f"View it at: https://app.pinecone.io/")
            print(f"\nTo delete the index, use the Pinecone dashboard or API")
        else:
            print(f"\nTo run with real Pinecone, use: --no-mock")
            print(f"Make sure to set PINECONE_API_KEY in examples/.env")


async def main_async():
    """Main async function."""
    global USE_MOCK, PINECONE_API_KEY, PINECONE_ENVIRONMENT
    
    if not HAS_QSTORE:
        print("❌ Cannot run example - missing Q-Store dependencies")
        return
    
    # Parse arguments
    args = parse_args()
    USE_MOCK = not args.no_mock
    
    # Load configuration from environment
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    
    # Run demo
    await demo_index_creation(
        index_name=args.index_name,
        dimension=args.dimension
    )


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
