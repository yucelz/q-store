"""
Quantum Database - Quickstart Guide
Complete example with Pinecone integration
"""

import asyncio
import numpy as np
from typing import List
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the quantum database
from q_store import (
    QuantumDatabase,
    DatabaseConfig,
    QueryMode,
    QueryResult
)


# ============================================================================
# Setup and Configuration
# ============================================================================

def setup_environment():
    """Setup environment variables and configuration"""
    print("=== Quantum Database Setup ===\n")
    
    # Get API keys from environment (loaded from .env via load_dotenv())
    pinecone_key = os.getenv('PINECONE_API_KEY')
    ionq_key = os.getenv('IONQ_API_KEY')  # Optional
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    
    if not pinecone_key:
        raise ValueError(
            "PINECONE_API_KEY is required. Please add it to your .env file.\n"
            "Get your API key from: https://www.pinecone.io/"
        )

    # Create configuration
    config = DatabaseConfig(
        # Pinecone settings
        pinecone_api_key=pinecone_key,
        pinecone_environment=pinecone_environment,
        pinecone_index_name="quantum-demo",
        pinecone_dimension=768,  # Standard for many embeddings
        pinecone_metric="cosine",
        
        # IonQ quantum settings (optional)
        ionq_api_key=ionq_key,
        ionq_target="simulator",  # Use free simulator
        
        # Feature flags
        enable_quantum=True if ionq_key else False,
        enable_superposition=True if ionq_key else False,
        enable_entanglement=True,
        enable_tunneling=True if ionq_key else False,
        
        # Performance tuning
        max_quantum_states=500,
        classical_candidate_pool=1000,
        result_cache_ttl=300,
        
        # Resource limits
        max_connections=25,
        connection_timeout=30
    )
    
    print(f"Configuration:")
    print(f"  - Pinecone Index: {config.pinecone_index_name}")
    print(f"  - Pinecone Environment: {pinecone_environment}")
    print(f"  - Dimension: {config.pinecone_dimension}")
    print(f"  - Quantum Enabled: {config.enable_quantum}")
    print(f"  - Superposition: {config.enable_superposition}")
    if ionq_key:
        print(f"  - IonQ Target: {config.ionq_target}")
    else:
        print(f"  - Note: IONQ_API_KEY not set, quantum features disabled")
    print()
    
    return config


# ============================================================================
# Example 1: Basic Vector Operations
# ============================================================================

async def example_basic_operations(db: QuantumDatabase):
    """Demonstrate basic insert and query operations"""
    print("=== Example 1: Basic Operations ===\n")
    
    # Create sample embeddings (simulating real embeddings)
    print("Creating sample embeddings...")
    
    documents = [
        {
            'id': 'doc_1',
            'text': 'Introduction to quantum computing',
            'embedding': np.random.rand(768),
            'metadata': {
                'category': 'education',
                'level': 'beginner',
                'topic': 'quantum'
            }
        },
        {
            'id': 'doc_2', 
            'text': 'Advanced quantum algorithms',
            'embedding': np.random.rand(768),
            'metadata': {
                'category': 'research',
                'level': 'advanced',
                'topic': 'quantum'
            }
        },
        {
            'id': 'doc_3',
            'text': 'Classical database design',
            'embedding': np.random.rand(768),
            'metadata': {
                'category': 'education',
                'level': 'intermediate',
                'topic': 'databases'
            }
        }
    ]
    
    # Insert documents
    print("Inserting documents...")
    for doc in documents:
        await db.insert(
            id=doc['id'],
            vector=doc['embedding'],
            metadata=doc['metadata']
        )
        print(f"  ✓ Inserted {doc['id']}")
    
    print()
    
    # Query for similar documents
    print("Querying for similar documents...")
    query_embedding = documents[0]['embedding'] + np.random.randn(768) * 0.1
    
    results = await db.query(
        vector=query_embedding,
        top_k=3
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.id}")
        print(f"     Score: {result.score:.4f}")
        print(f"     Metadata: {result.metadata}")
    
    print()


# ============================================================================
# Example 2: Context-Aware Retrieval (Quantum Superposition)
# ============================================================================

async def example_context_aware_retrieval(db: QuantumDatabase):
    """Demonstrate context-aware quantum retrieval"""
    print("=== Example 2: Context-Aware Retrieval ===\n")
    
    # Insert document with multiple contexts
    print("Inserting multi-context document...")
    
    doc_embedding = np.random.rand(768)
    
    await db.insert(
        id='multi_context_doc',
        vector=doc_embedding,
        contexts=[
            ('technical_context', 0.6),
            ('business_context', 0.3),
            ('research_context', 0.1)
        ],
        coherence_time=10000,  # 10 seconds
        metadata={
            'title': 'Quantum Computing in Practice',
            'type': 'multi-perspective'
        }
    )
    
    print("  ✓ Document inserted with 3 contexts\n")
    
    # Query with different contexts
    query_embedding = doc_embedding + np.random.randn(768) * 0.05
    
    for context in ['technical_context', 'business_context', 'research_context']:
        print(f"Querying with context: {context}")
        
        results = await db.query(
            vector=query_embedding,
            context=context,
            mode=QueryMode.BALANCED,
            top_k=3
        )
        
        for result in results:
            if result.quantum_enhanced:
                print(f"  ✓ Found quantum-enhanced result: {result.id}")
                print(f"    Score: {result.score:.4f}")
                print(f"    Source: {result.source}")
    
    print()


# ============================================================================
# Example 3: Batch Operations for Efficiency
# ============================================================================

async def example_batch_operations(db: QuantumDatabase):
    """Demonstrate efficient batch operations"""
    print("=== Example 3: Batch Operations ===\n")
    
    # Generate large batch of documents
    print("Preparing batch of 100 documents...")
    
    batch_docs = [
        {
            'id': f'batch_doc_{i}',
            'vector': np.random.rand(768),
            'contexts': [('general', 1.0)],
            'metadata': {
                'batch': True,
                'index': i,
                'timestamp': datetime.now().isoformat()
            }
        }
        for i in range(100)
    ]
    
    # Batch insert
    import time
    start_time = time.time()
    
    print("Executing batch insert...")
    await db.insert_batch(batch_docs)
    
    duration = time.time() - start_time
    throughput = len(batch_docs) / duration
    
    print(f"  ✓ Inserted {len(batch_docs)} documents")
    print(f"  ✓ Duration: {duration:.2f}s")
    print(f"  ✓ Throughput: {throughput:.1f} docs/sec\n")


# ============================================================================
# Example 4: Advanced Query Modes
# ============================================================================

async def example_query_modes(db: QuantumDatabase):
    """Demonstrate different query modes"""
    print("=== Example 4: Query Modes ===\n")
    
    # Insert reference documents
    reference_docs = [
        {
            'id': f'ref_{i}',
            'vector': np.random.rand(768),
            'metadata': {'category': f'cat_{i % 3}'}
        }
        for i in range(20)
    ]
    
    for doc in reference_docs[:5]:  # Insert subset
        await db.insert(
            id=doc['id'],
            vector=doc['vector'],
            metadata=doc['metadata']
        )
    
    query_vec = np.random.rand(768)
    
    # Test different query modes
    modes = [
        (QueryMode.PRECISE, "High precision, narrow results"),
        (QueryMode.BALANCED, "Balanced precision and coverage"),
        (QueryMode.EXPLORATORY, "Broad exploration, diverse results")
    ]
    
    for mode, description in modes:
        print(f"Query Mode: {mode.value}")
        print(f"Description: {description}")
        
        results = await db.query(
            vector=query_vec,
            mode=mode,
            enable_tunneling=True,
            top_k=5
        )
        
        print(f"  Results: {len(results)}")
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            print(f"  Average Score: {avg_score:.4f}")
        print()


# ============================================================================
# Example 5: Monitoring and Metrics
# ============================================================================

async def example_monitoring(db: QuantumDatabase):
    """Demonstrate monitoring and metrics"""
    print("=== Example 5: Monitoring & Metrics ===\n")
    
    # Perform various operations
    print("Executing operations for metrics...")
    
    for i in range(10):
        # Insert
        await db.insert(
            id=f'metrics_test_{i}',
            vector=np.random.rand(768)
        )
        
        # Query
        await db.query(
            vector=np.random.rand(768),
            top_k=3
        )
    
    # Get metrics
    metrics = db.get_metrics()
    
    print("\nCurrent Metrics:")
    print(f"  Total Queries: {metrics.total_queries}")
    print(f"  Quantum Queries: {metrics.quantum_queries}")
    print(f"  Cache Hits: {metrics.cache_hits}")
    print(f"  Cache Misses: {metrics.cache_misses}")
    print(f"  Cache Hit Rate: {metrics.cache_hits / max(1, metrics.total_queries):.2%}")
    print(f"  Avg Latency: {metrics.avg_latency_ms:.2f}ms")
    print(f"  Active Quantum States: {metrics.active_quantum_states}")
    print(f"  Error Count: {metrics.error_count}")
    
    # Get comprehensive stats
    stats = db.get_stats()
    
    print("\nDatabase Statistics:")
    print(f"  Quantum States: {stats['quantum_states']}")
    print(f"  Configuration:")
    for key, value in stats['config'].items():
        print(f"    - {key}: {value}")
    
    print()


# ============================================================================
# Example 6: Production Patterns
# ============================================================================

async def example_production_patterns():
    """Demonstrate production-ready patterns"""
    print("=== Example 6: Production Patterns ===\n")
    
    config = DatabaseConfig(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_environment=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1'),
        pinecone_index_name="production-index",
        pinecone_dimension=768,
        ionq_api_key=os.getenv('IONQ_API_KEY')
    )
    
    db = QuantumDatabase(config)
    
    print("Pattern 1: Context Manager (Recommended)")
    print("-" * 40)
    
    # Using context manager for automatic cleanup
    async with db.connect():
        await db.insert(
            id='prod_1',
            vector=np.random.rand(768),
            metadata={'env': 'production'}
        )
        
        results = await db.query(
            vector=np.random.rand(768),
            top_k=5
        )
        
        print(f"  ✓ Executed query, found {len(results)} results")
    
    # Database automatically cleaned up
    print("  ✓ Resources cleaned up automatically\n")
    
    print("Pattern 2: Error Handling")
    print("-" * 40)
    
    async with db.connect():
        try:
            # Robust error handling
            await db.insert(
                id='prod_2',
                vector=np.random.rand(768)
            )
            print("  ✓ Operation succeeded")
            
        except Exception as e:
            print(f"  ✗ Error handled: {e}")
            # Implement retry logic, logging, alerting
    
    print("\nPattern 3: Monitoring Integration")
    print("-" * 40)
    
    # Periodic metrics export
    async def export_metrics():
        while True:
            metrics = db.get_metrics()
            # In production: send to monitoring system
            # (Prometheus, Datadog, CloudWatch, etc.)
            print(f"  Metrics exported: {metrics.total_queries} queries")
            await asyncio.sleep(60)
    
    print("  ✓ Metrics export configured\n")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("QUANTUM DATABASE - INTERACTIVE DEMO")
    print("="*60 + "\n")
    
    # Setup
    config = setup_environment()
    
    # Initialize database
    print("Initializing database...")
    db = QuantumDatabase(config)
    
    async with db.connect():
        print("✓ Database initialized successfully\n")
        
        # Run examples
        try:
            await example_basic_operations(db)
            await example_context_aware_retrieval(db)
            await example_batch_operations(db)
            await example_query_modes(db)
            await example_monitoring(db)
            
        except Exception as e:
            print(f"\n✗ Error during demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Production patterns (separate instance)
    await example_production_patterns()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60 + "\n")
    
    print("Next Steps:")
    print("1. Set PINECONE_API_KEY environment variable")
    print("2. (Optional) Set IONQ_API_KEY for quantum features")
    print("3. Customize configuration for your use case")
    print("4. Integrate into your application")
    print("\nDocumentation: See updated design document v2.0")
    print()


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    """Run the demo"""
    
    # Check for API keys
    if not os.getenv('PINECONE_API_KEY'):
        print("\n⚠️  ERROR: PINECONE_API_KEY not found in .env file\n")
        print("Please create a .env file in the project root with:")
        print("  PINECONE_API_KEY=your-pinecone-api-key")
        print("  PINECONE_ENVIRONMENT=us-east-1")
        print("  IONQ_API_KEY=your-ionq-api-key  (optional)\n")
        print("Get your Pinecone API key from: https://www.pinecone.io/")
        print("Get your IonQ API key from: https://cloud.ionq.com/settings/keys\n")
        exit(1)
    
    if not os.getenv('IONQ_API_KEY'):
        print("\n⚠️  Note: IONQ_API_KEY not set in .env file")
        print("Quantum features will be disabled. Only classical vector search will work.\n")
    
    # Run demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
