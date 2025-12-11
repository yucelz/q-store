"""
Basic example demonstrating Q-Store quantum database features
"""

import os
import numpy as np
from getpass import getpass
from q_store import QuantumDatabase, QuantumDatabaseConfig


def main():
    """Basic example of quantum database usage"""
    
    print("=== Q-Store Quantum Database - Basic Example ===\n")
    
    # Get IonQ API key
    api_key = os.getenv('IONQ_API_KEY') or getpass('Enter your IonQ API key: ')
    
    # Initialize database with configuration
    config = QuantumDatabaseConfig(
        ionq_api_key=api_key,
        target_device='simulator',
        enable_superposition=True,
        enable_entanglement=True,
        enable_tunneling=True,
        default_coherence_time=1000.0  # 1 second
    )
    
    db = QuantumDatabase(config=config)
    
    print("✓ Initialized quantum database")
    print(f"  Backend: {config.quantum_backend}")
    print(f"  Target: {config.target_device}\n")
    
    # 1. Insert vectors with superposition (multiple contexts)
    print("1. Inserting vectors with quantum superposition...")
    
    # Example: Document embedding with different contexts
    doc_embedding = np.random.randn(64)  # 64-dimensional embedding
    
    db.insert(
        id='doc_1',
        vector=doc_embedding,
        contexts=[
            ('technical_query', 0.6),
            ('general_query', 0.3),
            ('historical_query', 0.1)
        ],
        coherence_time=2000,  # 2 seconds
        metadata={'title': 'Quantum Computing Introduction'}
    )
    
    print("  ✓ Inserted doc_1 with 3 contexts in superposition\n")
    
    # 2. Create entangled group (correlated documents)
    print("2. Creating entangled group of related documents...")
    
    # Insert related documents
    for i in range(2, 5):
        related_embedding = doc_embedding + np.random.randn(64) * 0.1  # Similar vector
        db.insert(
            id=f'doc_{i}',
            vector=related_embedding,
            metadata={'title': f'Related Document {i}'}
        )
    
    db.create_entangled_group(
        group_id='quantum_computing_docs',
        entity_ids=['doc_1', 'doc_2', 'doc_3', 'doc_4'],
        correlation_strength=0.85
    )
    
    print("  ✓ Created entangled group 'quantum_computing_docs'\n")
    
    # 3. Query with context (superposition collapse)
    print("3. Querying with context (superposition collapse)...")
    
    query_vector = np.random.randn(64)
    
    results = db.query(
        vector=query_vector,
        context='technical_query',  # Collapses to technical context
        mode='balanced',
        enable_tunneling=False,
        top_k=3
    )
    
    print(f"  Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"    {i}. {result.id} (score: {result.score:.4f})")
    print()
    
    # 4. Query with quantum tunneling
    print("4. Querying with quantum tunneling enabled...")
    
    # Note: Tunneling requires quantum backend circuit execution
    # Using simplified classical simulation for demonstration
    results_tunnel = db.query(
        vector=query_vector,
        context='general_query',
        mode='exploratory',
        enable_tunneling=True,  # Find distant patterns
        top_k=3
    )
    
    print(f"  Found {len(results_tunnel)} results with tunneling:")
    for i, result in enumerate(results_tunnel, 1):
        print(f"    {i}. {result.id} (score: {result.score:.4f})")
    print()
    
    # 5. Update entity (entangled partners auto-update)
    print("5. Updating entity (entanglement propagation)...")
    
    new_embedding = np.random.randn(64)
    db.update('doc_1', new_embedding)
    
    # Get entangled partners
    partners = db.entanglement_registry.get_entangled_partners('doc_1')
    print(f"  ✓ Updated doc_1")
    print(f"  ✓ Entangled partners affected: {partners}\n")
    
    # 6. Database statistics
    print("6. Database statistics:")
    stats = db.get_stats()
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Quantum states: {stats['quantum_states']}")
    print(f"  Entangled groups: {stats['entangled_groups']}")
    print(f"  Superposition: {stats['config']['superposition_enabled']}")
    print(f"  Entanglement: {stats['config']['entanglement_enabled']}")
    print(f"  Tunneling: {stats['config']['tunneling_enabled']}\n")
    
    # 7. Decoherence (adaptive TTL)
    print("7. Applying decoherence (adaptive memory cleanup)...")
    import time
    time.sleep(2.5)  # Wait for states to decohere
    
    removed = db.apply_decoherence()
    print(f"  ✓ Removed {len(removed)} decohered states\n")
    
    print("=== Example completed successfully! ===")


if __name__ == '__main__':
    main()
